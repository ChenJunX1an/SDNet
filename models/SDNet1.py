import torch
from torch import nn

from pointnet2_ops import pointnet2_utils
from extensions.chamfer_dist import ChamferDistanceL1,ChamferDistanceL2
from .Transformer3 import PCTransformer
from .build import MODELS


def fps(pc, num):
    fps_idx = pointnet2_utils.furthest_point_sample(pc, num)
    sub_pc = pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return sub_pc

from utils.pmputils import Conv1d, PointNet_FP_Module, PointNet_SA_Module
from seed_utils import vTransformer
from torch import einsum
from seed_utils import vTransformer, PointNet_SA_Module_KNN, MLP_Res, MLP_CONV, fps_subsample, query_knn, \
    grouping_operation, get_nearest_index, indexing_neighbor
from knn_cuda import KNN

knn = KNN(k=16, transpose_mode=False)
def get_graph_feature(coor_q, x_q, coor_k, x_k,k=8):
    # coor: bs, 3, np, x: bs, c, np

    knn_8 = KNN(k, transpose_mode=False)

    batch_size = x_k.size(0)
    num_points_k = x_k.size(2)
    num_points_q = x_q.size(2)

    with torch.no_grad():
        _, idx = knn_8(coor_k, coor_q)  # bs k np
        assert idx.shape[1] == k
        idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
        idx = idx + idx_base
        idx = idx.view(-1)
    num_dims = x_k.size(1)
    x_k = x_k.transpose(2, 1).contiguous()
    feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
    feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
    x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
    feature = feature - x_q
    return feature,idx
def gather_points(points, inds):
    '''
    :param points: shape=(B, N, C)
    :param inds: shape=(B, M) or shape=(B, M, K)
    :return: sampling points: shape=(B, M, C) or shape=(B, M, K, C)
    '''
    device = points.device
    B, N, C = points.shape
    inds_shape = list(inds.shape)
    inds_shape[1:] = [1] * len(inds_shape[1:])
    repeat_shape = list(inds.shape)
    repeat_shape[0] = 1
    batchlists = torch.arange(0, B, dtype=torch.long).to(device).reshape(inds_shape).repeat(repeat_shape)
    return points[batchlists, inds, :]

def get_dists(points1, points2):
    '''
    Calculate dists between two group points
    :param cur_point: shape=(B, M, C)
    :param points: shape=(B, N, C)
    :return:
    '''
    B, M, C = points1.shape
    _, N, _ = points2.shape
    dists = torch.sum(torch.pow(points1, 2), dim=-1).view(B, M, 1) + \
            torch.sum(torch.pow(points2, 2), dim=-1).view(B, 1, N)
    dists -= 2 * torch.matmul(points1, points2.permute(0, 2, 1))
    dists = torch.where(dists < 0, torch.ones_like(dists) * 1e-7, dists) # Very Important for dist = 0.
    return torch.sqrt(dists+ 1e-8).float()

def three_nn(xyz1, xyz2):
    '''

    :param xyz1: shape=(B, N1, 3)
    :param xyz2: shape=(B, N2, 3)
    :return: dists: shape=(B, N1, 3), inds: shape=(B, N1, 3)
    '''
    dists = get_dists(xyz1, xyz2)
    dists, inds = torch.sort(dists, dim=-1)
    dists, inds = dists[:, :, :3], inds[:, :, :3]
    return dists, inds

class FeatureExtractor(nn.Module):
    def __init__(self, out_dim=1024, n_knn=20):
        """Encoder that encodes information of partial point cloud
        """
        super(FeatureExtractor, self).__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.transformer_1 = vTransformer(128, dim=64, n_knn=n_knn)
        self.sa_module_2 = PointNet_SA_Module_KNN(128, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.transformer_2 = vTransformer(256, dim=64, n_knn=n_knn)
        self.sa_module_3 = PointNet_SA_Module_KNN(None, None, 256, [512, out_dim], group_all=True, if_bn=False)

    def forward(self, partial_cloud):
        """
        Args:
             partial_cloud: b, 3, n

        Returns:
            l3_points: (B, out_dim, 1)
        """
        l0_xyz = partial_cloud
        l0_points = partial_cloud

        l1_xyz, l1_points, idx1 = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)
        l1_points = self.transformer_1(l1_points, l1_xyz)
        l2_xyz, l2_points, idx2 = self.sa_module_2(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 128)
        l2_points = self.transformer_2(l2_points, l2_xyz)
        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 1), (B, out_dim, 1)

        return l3_xyz,l3_points, l2_xyz, l2_points,l1_xyz,l1_points
class SkipTransformer(nn.Module):
    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(SkipTransformer, self).__init__()
        self.mlp_v = MLP_Res(in_dim=in_channel*2, hidden_dim=in_channel, out_dim=in_channel)
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(in_channel, dim, 1)
        self.conv_query = nn.Conv1d(in_channel, dim, 1)
        self.conv_value = nn.Conv1d(in_channel, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )

        self.conv_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, pos, key, query, include_self=True):
        """
        Args:
            pos: (B, 3, N)
            key: (B, in_channel, N)
            query: (B, in_channel, N)
            include_self: boolean

        Returns:
            Tensor: (B, in_channel, N), shape context feature
        """
        value = self.mlp_v(torch.cat([key, query], 1))
        identity = value
        key = self.conv_key(key)
        query = self.conv_query(query)
        value = self.conv_value(value)
        b, dim, n = value.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped, include_self=include_self)

        key = grouping_operation(key, idx_knn)  # b, dim, n, n_knn
        qk_rel = query.reshape((b, -1, n, 1)) - key

        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # b, 3, n, n_knn
        pos_embedding = self.pos_mlp(pos_rel)

        attention = self.attn_mlp(qk_rel + pos_embedding)  # b, dim, n, n_knn
        attention = torch.softmax(attention, -1)

        value = value.reshape((b, -1, n, 1)) + pos_embedding  #

        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # b, dim, n
        y = self.conv_end(agg)

        return y + identity
class CrossTransformer(nn.Module):
    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(CrossTransformer, self).__init__()
        self.n_knn = n_knn

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, in_channel, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(in_channel, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, in_channel, 1)
        )

    def forward(self, pcd, feat, pcd_feadb, feat_feadb):
        """
        Args:
            pcd: (B, 3, N)
            feat: (B, in_channel, N)
            pcd_feadb: (B, 3, N2)
            feat_feadb: (B, in_channel, N2)
        Returns:
            Tensor: (B, in_channel, N), shape context feature
        """
        b, _, num_point = pcd.shape

        fusion_pcd = torch.cat((pcd, pcd_feadb), dim=2)
        fusion_feat = torch.cat((feat, feat_feadb), dim=2)

        key_point = pcd
        key_feat = feat

        # Preception processing between pcd and fusion_pcd
        key_point_idx = query_knn(self.n_knn, fusion_feat.transpose(2, 1).contiguous(),
                                  key_feat.transpose(2, 1).contiguous(), include_self=True)

        group_point = grouping_operation(fusion_pcd, key_point_idx)
        group_feat = grouping_operation(fusion_feat, key_point_idx)

        qk_rel = key_feat.reshape((b, -1, num_point, 1)) - group_feat
        pos_rel = key_point.reshape((b, -1, num_point, 1)) - group_point

        pos_embedding = self.pos_mlp(pos_rel)
        sample_weight = self.attn_mlp(qk_rel + pos_embedding)  # b, in_channel + 3, n, n_knn
        sample_weight = torch.softmax(sample_weight, -1)  # b, in_channel + 3, n, n_knn

        group_feat = group_feat + pos_embedding  #
        refined_feat = einsum('b c i j, b c i j -> b c i', sample_weight, group_feat)

        return refined_feat

class UpLayer(nn.Module):
    """
    Upsample Layer with upsample transformers
    """

    def __init__(self, dim, seed_dim, up_factor=2, i=0, radius=1, n_knn=20, interpolate='three', attn_channel=True):
        super(UpLayer, self).__init__()
        self.i = i
        self.up_factor = up_factor
        self.radius = radius
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 5, layer_dims=[256, dim])
        #self.coormlp=MLP_CONV(in_channel=3, layer_dims=[64, 128, 256])
        self.skip_transformer = SkipTransformer(in_channel=dim, dim=64)
        self.mlp_ps = MLP_CONV(in_channel=dim, layer_dims=[128, 64])
        self.ps = nn.ConvTranspose1d(64, 256, up_factor, up_factor, bias=False)  # point-wise splitting

        self.up_sampler = nn.Upsample(scale_factor=up_factor)

        self.mlp_all = MLP_CONV(in_channel=256, layer_dims=[256,256])
        self.mlp_delta_feature = MLP_Res(in_dim=dim * 2, hidden_dim=dim, out_dim=dim)
        self.mlp_delta = MLP_CONV(in_channel=dim, layer_dims=[64, 3])


    def forward(self, pcd_prev, K_prev=None,pcd_feat=None,fine_coor=None,fine_feat=None,all_global=None):
        """
        Args:
            pcd_prev: Tensor, (B, 3, N_prev)
            feat_global: Tensor, (B, feat_dim, 1)
            K_prev: Tensor, (B, 128, N_prev)

        Returns:
            pcd_new: Tensor, upsampled point cloud, (B, 3, N_prev * up_factor)
            K_curr: Tensor, displacement feature of current step, (B, 128, N_prev * up_factor)
        """
        b, _, n_prev = pcd_prev.shape

        # Query mlps

        feat_1 = self.mlp_1(pcd_prev)
        #coorfeat=self.coormlp(torch.cat([pcd_prev,fine_coor],dim=2))
        #coorfeat=torch.max(coorfeat, 2, keepdim=True)[0]
        '''all_global = self.mlp_all(torch.cat([pcd_feat,fine_feat],dim=2))
        all_global=torch.max(all_global, 2, keepdim=True)[0]'''
        feat_1 = torch.cat([feat_1,
                            all_global.repeat((1, 1, feat_1.size(2))),
                            pcd_feat], 1)
                            #coorfeat.repeat((1, 1, feat_1.size(2)))], 1)
        Q = self.mlp_2(feat_1)
        # Upsample Transformers
        H = self.skip_transformer(pcd_prev, K_prev if K_prev is not None else Q, Q)
        #H = self.cross_transformer(pcd_prev,fine_coor,fine_feat,H)
        feat_child = self.mlp_ps(H)
        feat_child = self.ps(feat_child)  # (B, 128, N_prev * up_factor)
        H_up = self.up_sampler(H)
        K_curr = self.mlp_delta_feature(torch.cat([feat_child, H_up], 1))

        delta = torch.tanh(self.mlp_delta(torch.relu(K_curr))) #/ self.radius ** self.i  # (B, 3, N_prev * up_factor)
        pcd_child = self.up_sampler(pcd_prev)
        pcd_child = pcd_child + delta
        return pcd_child, K_curr,feat_child


@MODELS.register_module()
class SDNet1(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.trans_dim = config.trans_dim
        self.knn_layer = config.knn_layer
        self.num_pred = config.num_pred
        self.num_query = config.num_query
        self.fp_module_3 = PointNet_FP_Module(1024, [256, 256], use_points1=True, in_channel_points1=256)
        self.fp_module_2 = PointNet_FP_Module(256, [256, 256], use_points1=True, in_channel_points1=128)
        self.fp_module_1 = PointNet_FP_Module(256, [128, 128, 128], use_points1=True, in_channel_points1=6)
        self.cross_transformer1 = CrossTransformer(in_channel=256, dim=64)
        self.cross_transformer2 = CrossTransformer(in_channel=256, dim=64)
        self.base_model = PCTransformer(in_chans=3, embed_dim=self.trans_dim, depth=[1, 2], drop_rate=0.,
                                        num_query=self.num_query, knn_layer=self.knn_layer)
        self.fineup = self.num_pred//2048-1
        n_knn=16
        global_dim=1024
        up_factors = [4, 4]
        self.feat_extractor = FeatureExtractor(out_dim=global_dim, n_knn=n_knn)
        mlp = [256,128, 64, 3]
        last_channel = 128 + 128*6+32
        mlp_conv = []
        for out_channel in mlp[:-1]:
            mlp_conv.append(Conv1d(last_channel, out_channel, if_bn=True))
            last_channel = out_channel
        mlp_conv.append(Conv1d(last_channel, mlp[-1], if_bn=False, activation_fn=None))
        self.mlp_conv = nn.Sequential(*mlp_conv)
        #self.finemodel= fineModel()
        self.reduce_map = nn.Linear(self.trans_dim + 1027, self.trans_dim)
        if up_factors is None:
            up_factors = [1]
        else:
            up_factors = [1] + up_factors
        radius=1
        uppers = []
        interpolate = 'three'
        attn_channel = True
        for i, factor in enumerate(up_factors):
            uppers.append(UpLayer(dim=256, seed_dim=256, up_factor=factor, i=i, n_knn=n_knn, radius=radius,
                        interpolate=interpolate, attn_channel=attn_channel))
        self.uppers = nn.ModuleList(uppers)
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_3 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.fuse_mlp1=MLP_CONV(in_channel=256+128, layer_dims=[256, 256])
        self.fuse_mlp2 = MLP_CONV(in_channel=256 + 128, layer_dims=[256, 128])
        self.fuse_mlp3 = MLP_CONV(in_channel=256 + 128, layer_dims=[256, 128])
        self.mlpq = MLP_CONV(in_channel=256, layer_dims=[256, 256])
        self.mlpK1 = MLP_CONV(in_channel=256, layer_dims=[256, 256])
        self.mlpK2 = MLP_CONV(in_channel=256, layer_dims=[256, 256])


        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL2()

    def get_loss(self, pcds_pred, gt,partial):
        """loss function
        Args
            pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
        """

        CD = self.loss_func

        P1,P20,P21,P2,P3,P4 = pcds_pred
        G_fine, G_lost, G3 = gt

        # G3 = fps_subsample(G3, 2048)
        cd1 = CD(P1, G_lost)
        cd2 = CD(P2, G_lost)
        cd3 = CD(P3, G_fine)
        cd4 = CD(P4, G3)
        cd5 = CD(P20, G_lost)
        cd6 = CD(P21, G_lost)

        loss_all = (cd1 + cd2+cd3+cd4+cd5+cd6) * 1e3  # +torch.sum(torch.stack(delta_losses)) / 3
        losses = [cd1, cd2,cd3,cd4]

        return loss_all, losses, [G_fine, G_lost, G3]

    def forward(self, xyz):
        l3_xyz,partial_global_feature, l2_xyz, l2_points,l1_xyz,l1_points=self.feat_extractor(xyz.transpose(1,2).contiguous())

        q, coarse_point_cloud = self.base_model(l2_xyz,l2_points)  # B M C and B M 3
        # M=num_pred=96
        B, M, C = q.shape

        #global_feature1=self.globalmlp1(torch.cat([lost_global_feature,partial_global_feature],1).contiguous())
        #global_feature2 = self.globalmlp2(torch.cat([lost_global_feature, partial_global_feature], 1).contiguous())
        l2_points = self.fp_module_3(l2_xyz, l3_xyz, l2_points, partial_global_feature)
        # print('l2_points, prev_s[l2]', l2_points.shape, prev_s['l2'].shape)
        l1_points = self.fp_module_2(l1_xyz, l2_xyz, l1_points, l2_points)
        # print('l1_points, prev_s[l1]', l1_points.shape, prev_s['l1'].shape)
        l0_points = self.fp_module_1(xyz.transpose(1,2).contiguous(), l1_xyz, torch.cat([xyz.transpose(1,2).contiguous(), xyz.transpose(1,2).contiguous()], 1), l1_points)
        # print('l0_points, prev_s[l0]', l0_points.shape, prev_s['l0'].shape)
        l0_points = l0_points.repeat(1, 1, self.fineup)
        b, _, n = l0_points.shape
        qq= self.mlpq(q.permute(0, 2, 1).contiguous())
        coor_emd = self.mlp_1(torch.cat([coarse_point_cloud.transpose(1, 2).contiguous(),l2_xyz],dim=2))
        noise = self.fuse_mlp1(torch.cat([torch.cat([qq,l2_points],dim=2),coor_emd],dim=1))
        all_global=torch.max(noise, 2, keepdim=True)[0]

        # # NOTE: try to rebuild pc
        # coarse_point_cloud = self.refine_coarse(rebuild_feature).reshape(B, M, 3)

        pred_pcds = []
        pcd = coarse_point_cloud  # (B, num_pc, 3)
        glos=[]
        glos.append(all_global.repeat(1,1,n))
        K_prev = None
        pcd_feat = q.transpose(1, 2).contiguous()
        pcd = pcd.permute(0, 2, 1).contiguous()
        for i,upper in enumerate(self.uppers):
            if i==1:
                noise=self.cross_transformer1(pcd,pcd_feat,l2_xyz,l2_points)
                #coor_emd = self.mlp_2(torch.cat([pcd,l2_xyz],dim=2))
                #K_mlp = self.mlpK1(K_prev)
                #noise = self.fuse_mlp2(torch.cat([torch.cat([K_mlp, l2_points], dim=2), coor_emd], dim=1))
                all_global=torch.max(noise, 2, keepdim=True)[0]
                glos.append(all_global.repeat(1, 1, n))
            elif i==2:
                noise=self.cross_transformer2(pcd,pcd_feat,l1_xyz,l1_points)
                #coor_emd = self.mlp_3(torch.cat([pcd, l1_xyz], dim=2))
                #K_mlp = self.mlpK2(K_prev)
                #noise = self.fuse_mlp3(torch.cat([torch.cat([K_mlp, l1_points], dim=2), coor_emd], dim=1))
                all_global = torch.max(noise, 2, keepdim=True)[0]
                glos.append(all_global.repeat(1, 1, n))
            pcd, K_prev,pcd_feat = upper(pcd, K_prev,pcd_feat,None,None,all_global)
            #pcd = self.
            pred_pcds.append(pcd.permute(0, 2, 1).contiguous())
        # NOTE: fc
        # relative_xyz = self.refine(rebuild_feature)  # BM 3S
        # rebuild_points = (relative_xyz.reshape(B,M,3,-1) + coarse_point_cloud.unsqueeze(-1)).transpose(2,3).reshape(B, -1, 3)
        device=xyz.device
        noise = torch.normal(mean=0, std=torch.ones((b, 32, n), device=device))
        delta_xyz = torch.tanh(self.mlp_conv(torch.cat([l0_points, *glos,noise], 1))) * 1.0 / 2
        fine_xyz = xyz.repeat(1,self.fineup,1) + delta_xyz.transpose(1, 2).contiguous()
        fine_xyz = torch.cat([fine_xyz, xyz], dim=1)
        # cat the input
        # rebuild_points = torch.cat([rebuild_points, xyz],dim=1).contiguous()
        all_ret=torch.cat([fine_xyz,pred_pcds[2]],dim=1)

        ret = (coarse_point_cloud, pred_pcds[0], pred_pcds[1], pred_pcds[2],fine_xyz,all_ret)
        return ret

