import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import numpy as np
import cv2
import random
from utils.emd import earth_mover_distance
from utils.config import *
from models.seed_utils import fps_subsample
from scipy import ndimage
import open3d as o3d
#from torchstat import stat
from thop import profile

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
                                                              builder.dataset_builder(args, config.dataset.val)
    # build model

    base_model = builder.model_builder(config.model)

    if args.use_gpu:
        base_model.to(args.local_rank)
        # cov_model.to(args_cov.local_rank)
    #input1 = torch.randn(1, 2048, 3).cuda()
    #flops, params = profile(base_model, inputs=(input1,))
    #print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    #print('Params = ' + str(params / 1000 ** 2) + 'M')
    # from IPython import embed; embed()
    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None
    image_path = os.path.join(args.experiment_path, 'image')
    # resume ckpts
    if args.resume:
        start_epoch = builder.resume_model(base_model, args, logger=logger)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger=logger)
        # builder.load_model(cov_model, args_cov.ckpts, logger=logger)
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model,
                                                         device_ids=[args.local_rank % torch.cuda.device_count()],
                                                         find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    # training
    base_model.zero_grad()

    for epoch in range(start_epoch, config.max_epoch + 1):

        '''metrics = validate(image_path, base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2,
                           val_writer,
                           args, config,
                           logger=logger)
        break'''

        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['SparseLoss', 'DenseLoss', 'pmploss', 'all_loss'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train._base_.N_POINTS
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                #partial = pcn_partial(partial).cuda()
                gt = data[1].cuda()
                if config.dataset.train._base_.CARS:
                    if idx == 0:
                        print_log('padding while KITTI training', logger=logger)
                    partial = misc.random_dropping(partial, epoch)  # specially for KITTI finetune
                gt_16_lost = data[3].cuda()
                gt_16_fine =data[2].cuda()
                #n1=8192
                #r1 = 48
                #r2 = 24
                #gt_16_lost = get_voxel(partial, gt, r2, n1, fine=False).cuda()
                #gt_16_fine = get_voxel(partial, gt, r1, n1, fine=True).cuda()
            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                partial, lost = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1 / 4), int(npoints * 3 / 4)],
                                                      fixed_points=None)
                partial = partial.cuda()
                n1 = 4096
                r1 = 16
                gt_16_fine, gt_16_lost = get_shapenet_voxel(partial,lost, gt, r1, n1)
                #gt_16_fine=gt_16_lost=gt
                gt_16_fine = gt_16_fine.cuda()
                gt_16_lost = gt_16_lost.cuda()
            else:
                partial = data[0].float().cuda()
                # partial = pcn_partial(partial).cuda()
                gt = data[1].float().cuda()

                gt_16_lost = data[3].float().cuda()
                gt_16_fine = data[2].float().cuda()
                #raise NotImplementedError(f'Train phase do not support {dataset_name}')
            # print(partial)

            '''
            ######################cov############################

            aaa=val_grid(image_path, base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2,
                     val_writer,
                     args, config,
                     logger=logger)
            break
            '''
            num_iter += 1
            ret = base_model(partial)
            gt_all = [gt_16_fine, gt_16_lost, gt]
            loss_all, loss_es, gt_sp = base_model.module.get_loss(ret, gt_all,partial)
            _loss = loss_all
            _loss.backward()

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss_es[0] = dist_utils.reduce_tensor(loss_es[0], args)
                loss_es[1] = dist_utils.reduce_tensor(loss_es[1], args)
                loss_es[2] = dist_utils.reduce_tensor(loss_es[2], args)
                loss_es[3] = dist_utils.reduce_tensor(loss_es[3], args)

                losses.update([loss_es[0].item() * 1000, loss_es[1].item() * 1000, loss_es[2].item() * 1000,
                               loss_es[3].item() * 1000])
            else:
                losses.update([loss_es[0].item() * 1000, loss_es[1].item() * 1000, loss_es[2].item() * 1000,
                               loss_es[3].item() * 1000])

            if args.distributed:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx
            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Sparse', loss_es[0].item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/Dense', loss_es[1].item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/pmploss', loss_es[2].item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/all_loss', loss_es[3].item() * 1000, n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 100 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                          (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                           ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger=logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Sparse', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Dense', losses.avg(1), epoch)
            train_writer.add_scalar('Loss/Epoch/pmploss', losses.avg(2), epoch)
            train_writer.add_scalar('Loss/Epoch/all_loss', losses.avg(3), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
                  (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger=logger)
        valn=10
        if dataset_name=="MVP":
            valn=30
        if epoch % valn == 0 and epoch != 0:
            # Validate the current model

            metrics = validate(image_path, base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2,
                               val_writer,
                               args, config,
                               logger=logger)

            # Save ckeckpoints
            # Save ckeckpoints
        if epoch % 5 == 0 and epoch != 0 and (config.max_epoch - epoch) > 10:
            builder.save_checkpoint(base_model, optimizer, epoch, f'ckpt-epoch-{epoch:03d}', args,
                                    logger=logger)
        builder.save_checkpoint(base_model, optimizer, epoch, 'ckpt-last', args, logger=logger)
        if (config.max_epoch - epoch) < 10:
            builder.save_checkpoint(base_model, optimizer, epoch, f'ckpt-epoch-{epoch:03d}',
                                    args, logger=logger)
    train_writer.close()
    val_writer.close()

    ###pointr


def validate(image_path, base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args,
             config,
             logger=None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger=logger)
    base_model.eval()  # set model to eval mode
    image_row = True
    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader)  # bs is 1
    image_merged = None
    a = random.sample(range(0, 10517), 6)
    b = 0
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.val._base_.N_POINTS
            dataset_name = config.dataset.val._base_.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                #partial = pcn_partial(partial).cuda()

                gt = data[1].cuda()
                gt_32 = data[3].cuda()
                gt32_fine = data[2].cuda()
                '''n1 = 8192
                r1 = 48
                r2 = 24
                gt_32 = get_voxel(partial, gt, r2, n1, fine=False).cuda()
                gt32_fine = get_voxel(partial, gt, r1, n1, fine=True).cuda()'''
            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                partial, lost = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1 / 4), int(npoints * 3 / 4)],
                                                      fixed_points=None)
                partial = partial.cuda()
                n1 = 4096
                r1 = 16
                gt32_fine, gt_32 = get_shapenet_voxel(partial,lost, gt, r1, n1)
                gt32_fine = gt32_fine.cuda()
                gt_32 = gt_32.cuda()
            else:
                partial = data[0].float().cuda()
                # partial = pcn_partial(partial).cuda()

                gt = data[1].float().cuda()
                gt_32 = data[3].float().cuda()
                gt32_fine = data[2].float().cuda()
                #raise NotImplementedError(f'Train phase do not support {dataset_name}')
            ret = base_model(partial)
            # loss_all, loss_es, gt_sp = base_model.module.get_loss(ret, partial, gt)
            coarse_points = ret[3]
            pointr_points = ret[0]
            pmp_points = ret[-2]
            # coarse=torch.cat([partial,coarse_points],dim=1)
            dense_points = ret[-1]

            # print(partial)
            # print(gt_32)
            # gt_32_all=torch.cat([partial,gt_32],dim=1)
            # gt_16 = get_voxel(partial, gt, r2, n1).cuda()

            sparse_loss_l1 = ChamferDisL1(coarse_points, gt_32)
            sparse_loss_l2 = ChamferDisL2(coarse_points, gt_32)
            dense_loss_l1 = ChamferDisL1(dense_points, gt)
            dense_loss_l2 = ChamferDisL2(dense_points, gt)

            if args.distributed:
                sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, args)
                sparse_loss_l2 = dist_utils.reduce_tensor(sparse_loss_l2, args)
                dense_loss_l1 = dist_utils.reduce_tensor(dense_loss_l1, args)
                dense_loss_l2 = dist_utils.reduce_tensor(dense_loss_l2, args)

            test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000,
                                dense_loss_l2.item() * 1000])

            # dense_points_all = dist_utils.gather_tensor(dense_points, args)
            # gt_all = dist_utils.gather_tensor(gt, args)

            # _metrics = Metrics.get(dense_points_all, gt_all)
            _metrics = Metrics.get(dense_points, gt)
            # _metrics = [dist_utils.reduce_tensor(item, args) for item in _metrics]

            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            category_metrics[taxonomy_id].update(_metrics)

            if val_writer is not None and idx % 1000 == 0:
                input_pc = partial.squeeze().detach().cpu().numpy()
                input_pc = misc.get_ptcloud_img(input_pc, tit='partial')
                val_writer.add_image('Model%02d/Input' % idx, input_pc, epoch, dataformats='HWC')

                sparse = coarse_points.squeeze().cpu().numpy()
                sparse_img = misc.get_ptcloud_img(sparse, tit='sparse')
                val_writer.add_image('Model%02d/Sparse' % idx, sparse_img, epoch, dataformats='HWC')
                dense = dense_points.squeeze().cpu().numpy()
                dense_img = misc.get_ptcloud_img(dense, tit='dense')
                val_writer.add_image('Model%02d/Dense' % idx, dense_img, epoch, dataformats='HWC')

                gt_ptcloud = gt.squeeze().cpu().numpy()
                gt_ptcloud_img = misc.get_ptcloud_img(gt_ptcloud, tit='gt')
                val_writer.add_image('Model%02d/DenseGT' % idx, gt_ptcloud_img, epoch, dataformats='HWC')

            if image_row and idx <= 6:
                input_pc = partial.squeeze().detach().cpu().numpy()
                input_pc = misc.get_ptcloud_img(input_pc, tit='partial')
                sparse = coarse_points.squeeze().cpu().numpy()
                sparse_img = misc.get_ptcloud_img(sparse, tit='lostpcds')

                dense = dense_points.squeeze().cpu().numpy()
                dense_img = misc.get_ptcloud_img(dense, tit='dense')

                gt_ptcloud = gt.squeeze().cpu().numpy()
                gt_ptcloud_img = misc.get_ptcloud_img(gt_ptcloud, tit='gt')

                gt32_ptcloud = gt_32.squeeze().cpu().numpy()
                gt32_ptcloud_img = misc.get_ptcloud_img(gt32_ptcloud, tit='lost')

                fine_ptcloud = gt32_fine.squeeze().cpu().numpy()
                fine_ptcloud_img = misc.get_ptcloud_img(fine_ptcloud, tit='fine')

                gt32_ptcloud_all = pmp_points.squeeze().cpu().numpy()
                gt32_ptcloud__all_img = misc.get_ptcloud_img(gt32_ptcloud_all, tit='pmp')
                '''
                coarse_ptcloud = coarse.squeeze().cpu().numpy()
                coarse_ptcloud_img = misc.get_ptcloud_img(coarse_ptcloud, tit='coarse')
                '''
                # gt16_ptcloud = gt_16.squeeze().cpu().numpy()
                # gt16_ptcloud_img = misc.get_ptcloud_img(gt16_ptcloud, tit='gt4')
                image_m = np.concatenate(
                    [input_pc, sparse_img, gt32_ptcloud__all_img,dense_img], axis=0)
                image_m1 = np.concatenate(
                    [gt_ptcloud_img, gt32_ptcloud_img, fine_ptcloud_img,gt_ptcloud_img], axis=0)
                image_m = np.concatenate([image_m, image_m1], axis=1)
                if image_merged is None:
                    image_merged = image_m
                else:
                    image_merged = np.concatenate([image_merged, image_m], axis=1)
                    if val_writer is not None and idx == 6:
                        val_writer.add_image('Merged/merged', image_merged, epoch, dataformats='HWC')
                        if os.path.exists(image_path):
                            cv2.imwrite(os.path.join(image_path, f'{epoch}_merged.jpg'), image_merged)
                        else:
                            os.mkdir(image_path)
                            cv2.imwrite(os.path.join(image_path, f'{epoch}_merged.jpg'), image_merged)

            if idx in a:
                input_pc = partial.squeeze().detach().cpu().numpy()
                input_pc = misc.get_ptcloud_img(input_pc)
                sparse = coarse_points.squeeze().cpu().numpy()
                sparse_img = misc.get_ptcloud_img(sparse)

                dense = dense_points.squeeze().cpu().numpy()
                dense_img = misc.get_ptcloud_img(dense)
                gt_ptcloud = gt.squeeze().cpu().numpy()
                gt_ptcloud_img = misc.get_ptcloud_img(gt_ptcloud)
                image_m = np.concatenate([input_pc, sparse_img, dense_img, gt_ptcloud_img], axis=0)
                if b == 0:
                    image_me = image_m
                else:
                    image_me = np.concatenate([image_me, image_m], axis=1)
                b = b + 1
                if b == 6:
                    val_writer.add_image('Merged_M/merged', image_me, epoch, dataformats='HWC')
                    break

            if (idx + 1) % 200 == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                          (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()],
                           ['%.4f' % m for m in _metrics]), logger=logger)
                if dataset_name == 'ShapeNet':
                    break
                elif dataset_name == 'MVP':
                    break
        for _, v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]),
                  logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================', logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (str(taxonomy_id) + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        if dataset_name=='MVP':
            msg += str(taxonomy_id) + '\t'
        else:
            msg += str(shapenet_dict[taxonomy_id]) + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/Sparse', test_losses.avg(0), epoch)
        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)

    return Metrics(config.consider_metric, test_metrics.avg())


crop_ratio = {
    'easy': 1 / 4,
    'median': 1 / 2,
    'hard': 3 / 4
}


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger=logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger=logger)
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP
    if args.distributed:
        raise NotImplementedError()

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger=logger)


def test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger=None):
    base_model.eval()  # set model to eval mode
    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader)  # bs is 1

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.test._base_.N_POINTS
            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                gt = data[1].cuda()

                ret = base_model(partial)
                coarse_points = ret[-2]
                dense_points = ret[-1]

                sparse_loss_l1 = ChamferDisL1(coarse_points, gt)
                sparse_loss_l2 = ChamferDisL2(coarse_points, gt)
                dense_loss_l1 = ChamferDisL1(dense_points, gt)
                dense_loss_l2 = ChamferDisL2(dense_points, gt)

                test_losses.update(
                    [sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000,
                     dense_loss_l2.item() * 1000])

                _metrics = Metrics.get(dense_points, gt)
                test_metrics.update(_metrics)

                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)

            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                choice = [torch.Tensor([1, 1, 1]), torch.Tensor([1, 1, -1]), torch.Tensor([1, -1, 1]),
                          torch.Tensor([-1, 1, 1]),
                          torch.Tensor([-1, -1, 1]), torch.Tensor([-1, 1, -1]), torch.Tensor([1, -1, -1]),
                          torch.Tensor([-1, -1, -1])]
                num_crop = int(npoints * crop_ratio[args.mode])
                for item in choice:
                    partial, _ = misc.seprate_point_cloud(gt, npoints, num_crop, fixed_points=item)
                    # NOTE: subsample the input
                    partial = misc.fps(partial, 2048)
                    ret = base_model(partial)
                    coarse_points = ret[-2]
                    dense_points = ret[-1]

                    sparse_loss_l1 = ChamferDisL1(coarse_points, gt)
                    sparse_loss_l2 = ChamferDisL2(coarse_points, gt)
                    dense_loss_l1 = ChamferDisL1(dense_points, gt)
                    dense_loss_l2 = ChamferDisL2(dense_points, gt)

                    test_losses.update(
                        [sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000,
                         dense_loss_l2.item() * 1000])

                    _metrics = Metrics.get(dense_points, gt)

                    # test_metrics.update(_metrics)

                    if taxonomy_id not in category_metrics:
                        category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                    category_metrics[taxonomy_id].update(_metrics)
            elif dataset_name == 'KITTI':
                partial = data.cuda()
                ret = base_model(partial)
                dense_points = ret[1]
                target_path = os.path.join(args.experiment_path, 'vis_result')
                if not os.path.exists(target_path):
                    os.mkdir(target_path)
                misc.visualize_KITTI(
                    os.path.join(target_path, f'{model_id}_{idx:03d}'),
                    [partial[0].cpu(), dense_points[0].cpu()]
                )
                continue
            else:
                partial = data[0].float().cuda()
                gt = data[1].float().cuda()

                ret = base_model(partial)
                coarse_points = ret[-2]
                dense_points = ret[-1]

                sparse_loss_l1 = ChamferDisL1(coarse_points, gt)
                sparse_loss_l2 = ChamferDisL2(coarse_points, gt)
                dense_loss_l1 = ChamferDisL1(dense_points, gt)
                dense_loss_l2 = ChamferDisL2(dense_points, gt)

                test_losses.update(
                    [sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000,
                     dense_loss_l2.item() * 1000])

                _metrics = Metrics.get(dense_points, gt)
                test_metrics.update(_metrics)

                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)
                #raise NotImplementedError(f'Train phase do not support {dataset_name}')

            if (idx + 1) % 200 == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                          (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()],
                           ['%.4f' % m for m in _metrics]), logger=logger)
        if dataset_name == 'KITTI':
            return
        for _, v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[TEST] Metrics = %s' % (['%.4f' % m for m in test_metrics.avg()]), logger=logger)

    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================', logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (str(taxonomy_id) + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        if dataset_name == 'MVP':
            msg += str(taxonomy_id) + '\t'
        else:
            msg += str(shapenet_dict[taxonomy_id]) + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall \t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)
    return


def get_shapenet_voxel(partial,lost,gt,reso,npoints=4096):
    b, n, c = partial.shape
    normalize = False
    eps = 0
    reso=32
    fine_ret = None
    lost_ret = None
    for p,l, g in zip(partial,lost, gt):
        resolution = reso
        fine_error = 0
        lost_error = 0
        p_coords = p
        p_norm_coords = p
        # p_norm_coords = p_coords - p_coords.mean(1, keepdim=True)
        if normalize:
            p_norm_coords = p_norm_coords / (
                    p_norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + eps) + 0.5
        else:
            p_norm_coords = p_norm_coords + 0.5
        p_norm_coords = torch.clamp(p_norm_coords * resolution, 0, resolution - 1)
        vox_coords = torch.round(p_norm_coords).to(torch.int32)

        # print("voxsize:",vox_coords.size())
        univox_p = np.unique(vox_coords.cpu(), axis=0)  # b
        # print("univox_p:",univox_p)
        # print("univox_p:", univox_p.size)

        g_norm_coords = g
        # p_norm_coords = p_coords - p_coords.mean(1, keepdim=True)
        if normalize:
            g_norm_coords = g_norm_coords / (
                    g_norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + eps) + 0.5
        else:
            g_norm_coords = g_norm_coords + 0.5
        g_norm_coords = torch.clamp(g_norm_coords * resolution, 0, resolution - 1)
        g_vox_coords = torch.round(g_norm_coords).to(torch.int32).cpu()  # a
        univox_g = np.unique(g_vox_coords.cpu(), axis=0)  # b

        l_norm_coords = l
        l_norm_coords = l_norm_coords + 0.5
        l_norm_coords = torch.clamp(l_norm_coords * resolution, 0, resolution - 1)
        l_vox_coords = torch.round(l_norm_coords).to(torch.int32).cpu()  # a
        univox_l = np.unique(l_vox_coords.cpu(), axis=0)  # b

        p_index_matrix = np.zeros((resolution, resolution, resolution))
        g_index_matrix = np.zeros((resolution, resolution, resolution))
        l_index_matrix = np.zeros((resolution, resolution, resolution))

        p_index_matrix[tuple(univox_p.T)] = 1
        g_index_matrix[tuple(univox_g.T)] = 1
        l_index_matrix[tuple(univox_l.T)] = 1
        p_index_matrix = ndimage.binary_dilation(p_index_matrix,structure=np.ones((3, 3, 3)))
        fine_x, fine_y, fine_z = np.nonzero(p_index_matrix)
        l_index_matrix=g_index_matrix-g_index_matrix*p_index_matrix
        lost_index_matrix = ndimage.binary_dilation(l_index_matrix,structure=np.ones((3, 3, 3)))
        lost_x, lost_y, lost_z = np.nonzero(lost_index_matrix)
        fine_res = fine_x * resolution * resolution + fine_y * resolution + fine_z
        lost_res = lost_x * resolution * resolution + lost_y * resolution + lost_z

        g_res = (g_vox_coords.numpy()[:, 0] * resolution * resolution + g_vox_coords.numpy()[:,
                                                                        1] * resolution + g_vox_coords.numpy()[:,
                                                                                          2]).flatten()
        fine_idx = np.isin(g_res, fine_res)
        fine = g[fine_idx]

        fine_gt = torch.reshape(fine, (1, -1, 3))
        fine_n = fine.shape[0]

        lost_idx = np.isin(g_res, lost_res)
        lost = g[lost_idx]

        lost_gt = torch.reshape(lost, (1, -1, 3))
        lost_n = lost.shape[0]
        if (fine_gt.size(1) > npoints):
            # ret_gt=torch.from_numpy(test)
            fine_gt = fps_subsample(fine_gt, npoints)
            '''
            #down=test.shape[0]-npoints
            downcoor=test[:npoints]
            ret_gt = torch.reshape(downcoor, (1, -1, 3))
            '''
        else:
            fine_up = npoints - fine_n
            while fine_up > fine_n:
                fine_error = fine_error + 1
                fine = torch.cat([fine, fine], dim=0)
                fine_n = fine.shape[0]
                fine_up = npoints - fine_n
                if fine_error > 5:
                    print('fineerror:', fine_error)
                    print(fine_n)

            upcoor = fine[:fine_up]
            fine = torch.cat([upcoor, fine], dim=0)
            fine_n = fine.shape[0]
            fine_gt = torch.reshape(fine, (1, -1, 3))
        if (fine_ret == None):
            fine_ret = fine_gt
        else:
            fine_ret = torch.cat([fine_ret, fine_gt], dim=0)

        if (lost_gt.size(1) > npoints):
            # ret_gt=torch.from_numpy(test)
            lost_gt = fps_subsample(lost_gt, npoints)
            '''
            #down=test.shape[0]-npoints
            downcoor=test[:npoints]
            ret_gt = torch.reshape(downcoor, (1, -1, 3))
            '''
        else:
            lost_up = npoints - lost_n
            while lost_up > lost_n:
                lost_error = lost_error + 1
                lost = torch.cat([lost, lost], dim=0)
                lost_n = lost.shape[0]
                lost_up = npoints - lost_n
                if lost_error > 5:
                    print('losterror:', lost_error)
                    print(lost_n)
                    if lost.shape[0] <= 1024:
                        lost = g
                        lost_up=0
            lost_coor = lost[:lost_up]
            lost = torch.cat([lost_coor, lost], dim=0)
            lost_n = lost.shape[0]
            lost_gt = torch.reshape(lost, (1, -1, 3))
        if (lost_gt.size(1) > npoints):
            # ret_gt=torch.from_numpy(test)
            lost_gt = fps_subsample(lost_gt, npoints)
        if (lost_ret == None):
            lost_ret = lost_gt
        else:
            lost_ret = torch.cat([lost_ret, lost_gt], dim=0)
    return fine_ret, lost_ret

def get_voxel(partial, gt, reso, npoints=4096, fine=True):
    b, n, c = partial.shape
    normalize = False
    eps = 0
    ret = None
    for p, g in zip(partial, gt):
        resolution = reso
        # print("p:",p.size())
        # print("g:",g.size())
        error=0
        p_coords = p
        p_norm_coords = p
        # p_norm_coords = p_coords - p_coords.mean(1, keepdim=True)
        if normalize:
            p_norm_coords = p_norm_coords / (
                    p_norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + eps) + 0.5
        else:
            p_norm_coords = p_norm_coords +0.5
        p_norm_coords = torch.clamp(p_norm_coords * resolution, 0, resolution - 1)
        vox_coords = torch.round(p_norm_coords).to(torch.int32)

        # print("voxsize:",vox_coords.size())
        univox_p = np.unique(vox_coords.cpu(), axis=0)  # b
        # print("univox_p:",univox_p)
        # print("univox_p:", univox_p.size)

        g_norm_coords = g
        # p_norm_coords = p_coords - p_coords.mean(1, keepdim=True)
        if normalize:
            g_norm_coords = g_norm_coords / (
                    g_norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + eps) + 0.5
        else:
            g_norm_coords = g_norm_coords +0.5
        g_norm_coords = torch.clamp(g_norm_coords * resolution, 0, resolution - 1)
        g_vox_coords = torch.round(g_norm_coords).to(torch.int32).cpu()  # a
        univox_g = np.unique(g_vox_coords.cpu(), axis=0)  # b
        if not fine:
            index_matrix = np.zeros((resolution, resolution, resolution))
            index_matrix[tuple(univox_p.T)] = 1
            index_matrix = ndimage.binary_dilation(index_matrix, structure=np.ones((3, 3, 3)))

            g_index_matrix = np.zeros((resolution, resolution, resolution))
            g_index_matrix[tuple(univox_g.T)] = 1
            index_matrix = g_index_matrix - index_matrix*g_index_matrix
            index_matrix = ndimage.binary_dilation(index_matrix, structure=np.ones((3, 3, 3)))
            p_x, p_y, p_z = np.nonzero(index_matrix)
            p_res = p_x * resolution * resolution + p_y * resolution + p_z

        elif fine:
            index_matrix = np.zeros((resolution, resolution, resolution))
            index_matrix[tuple(univox_p.T)] = 1
            index_matrix = ndimage.binary_dilation(index_matrix, structure=np.ones((3, 3, 3)))
            p_x, p_y, p_z = np.nonzero(index_matrix)
            p_res = p_x * resolution * resolution + p_y * resolution + p_z
        g_res = (g_vox_coords.numpy()[:, 0] * resolution * resolution + g_vox_coords.numpy()[:,
                                                                        1] * resolution + g_vox_coords.numpy()[:,
                                                                                          2]).flatten()
        res = np.isin(g_res, p_res)
        test = g[res]
        ret_gt = torch.reshape(test, (1, -1, 3))

        while test.shape[0] < npoints:

            if not fine:
                resolution = resolution + 8
                if (resolution > 48):
                    up = npoints - test.shape[0]
                    while up > test.shape[0]:
                        error = error+1
                        test = torch.cat([test, test], dim=0)
                        up = npoints - test.shape[0]
                        if error>5:
                            print('error:',error)
                            print(test.shape[0])
                            if test.shape[0]<=1024:
                                test=g
                    upcoor = test[:up]
                    test = torch.cat([upcoor, test], dim=0)
                    ret_gt = torch.reshape(test, (1, -1, 3))
                    break
            elif fine:
                resolution = resolution - 4
            if (resolution <= 0):
                test = g
                ret_gt = torch.reshape(g, (1, -1, 3))
                break
            p_coords = p
            p_norm_coords = p
            if normalize:
                p_norm_coords = p_norm_coords / (
                        p_norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + eps) + 0.5
            else:
                p_norm_coords = p_norm_coords + 0.5
            p_norm_coords = torch.clamp(p_norm_coords * resolution, 0, resolution - 1)
            vox_coords = torch.round(p_norm_coords).to(torch.int32)

            # print("voxsize:",vox_coords.size())
            univox_p = np.unique(vox_coords.cpu(), axis=0)  # b
            # print("univox_p:",univox_p)
            # print("univox_p:", univox_p.size)

            g_norm_coords = g
            # p_norm_coords = p_coords - p_coords.mean(1, keepdim=True)
            if normalize:
                g_norm_coords = g_norm_coords / (
                        g_norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + eps) + 0.5
            else:
                g_norm_coords = g_norm_coords + 0.5
            g_norm_coords = torch.clamp(g_norm_coords * resolution, 0, resolution - 1)
            g_vox_coords = torch.round(g_norm_coords).to(torch.int32).cpu()  # a
            univox_g = np.unique(g_vox_coords.cpu(), axis=0)  # b
            if not fine:
                index_matrix = np.zeros((resolution, resolution, resolution))
                index_matrix[tuple(univox_p.T)] = 1
                index_matrix = ndimage.binary_dilation(index_matrix, structure=np.ones((3, 3, 3)))

                g_index_matrix = np.zeros((resolution, resolution, resolution))
                g_index_matrix[tuple(univox_g.T)] = 1
                index_matrix = g_index_matrix - index_matrix * g_index_matrix
                index_matrix = ndimage.binary_dilation(index_matrix, structure=np.ones((3, 3, 3)))
                p_x, p_y, p_z = np.nonzero(index_matrix)
                p_res = p_x * resolution * resolution + p_y * resolution + p_z

            elif fine:
                index_matrix = np.zeros((resolution, resolution, resolution))
                index_matrix[tuple(univox_p.T)] = 1
                index_matrix = ndimage.binary_dilation(index_matrix, structure=np.ones((3, 3, 3)))
                p_x, p_y, p_z = np.nonzero(index_matrix)
                p_res = p_x * resolution * resolution + p_y * resolution + p_z
            g_res = (g_vox_coords.numpy()[:, 0] * resolution * resolution + g_vox_coords.numpy()[:,
                                                                            1] * resolution + g_vox_coords.numpy()[:,
                                                                                              2]).flatten()
            res = np.isin(g_res, p_res)
            test = g[res]
            ret_gt = torch.reshape(test, (1, -1, 3))


        if (ret_gt.size(1) > npoints):
            # ret_gt=torch.from_numpy(test)
            pcd2 = o3d.geometry.PointCloud()
            test_np=test.cpu().numpy()
            pcd2.points = o3d.utility.Vector3dVector(test_np)
            res=pcd2.voxel_down_sample_and_trace(0.01, min_bound=pcd2.get_min_bound() - 0.5,
                                             max_bound=pcd2.get_max_bound() + 0.5, approximate_class=True)
            pcd = res[0]
            xyz2 = np.asarray(pcd.points).astype(np.float32)
            grid=torch.from_numpy(xyz2)
            grid=torch.reshape(grid, (1, -1, 3)).cuda()

            if grid.size(1)< npoints:
                ret_gt = fps_subsample(ret_gt, npoints-grid.size(1))
                ret_gt = torch.cat([grid,ret_gt],dim=1)
            elif grid.size(1)> npoints:
                ret_gt = fps_subsample(grid, npoints)
            else:
                ret_gt =grid
            '''
            #down=test.shape[0]-npoints
            downcoor=test[:npoints]
            ret_gt = torch.reshape(downcoor, (1, -1, 3))
            '''
        if (ret == None):
            ret = ret_gt
        else:
            ret = torch.cat([ret, ret_gt], dim=0)
        '''
        # print("voxsize:", g_vox_coords.size())
        univox_g = np.unique(g_vox_coords.cpu(), axis=0)
        # print("univox_g:", univox_g)
        # print("univox_g:", univox_g.size)
        # print("iou:",(univox_p.size/univox_g.size))
        iou.append((univox_p.size / univox_g.size))
    iou = np.array(iou)
    iou = torch.from_numpy(iou)
    # print(iou.size())
    iou_data = torch.unsqueeze(iou, dim=1)
    '''
    # print(ret.size())
    # print(ret)
    return ret

def get_voxel1(partial, gt, reso, npoints=4096, fine=True):
    b, n, c = partial.shape
    normalize = False
    eps = 0
    ret = None
    for p, g in zip(partial, gt):
        resolution = reso
        # print("p:",p.size())
        # print("g:",g.size())
        error=0
        p_coords = p
        p_norm_coords = p
        # p_norm_coords = p_coords - p_coords.mean(1, keepdim=True)
        if normalize:
            p_norm_coords = p_norm_coords / (
                    p_norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + eps) + 0.5
        else:
            p_norm_coords = p_norm_coords +0.5
        p_norm_coords = torch.clamp(p_norm_coords * resolution, 0, resolution - 1)
        vox_coords = torch.round(p_norm_coords).to(torch.int32)

        # print("voxsize:",vox_coords.size())
        univox_p = np.unique(vox_coords.cpu(), axis=0)  # b
        # print("univox_p:",univox_p)
        # print("univox_p:", univox_p.size)

        g_norm_coords = g
        # p_norm_coords = p_coords - p_coords.mean(1, keepdim=True)
        if normalize:
            g_norm_coords = g_norm_coords / (
                    g_norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + eps) + 0.5
        else:
            g_norm_coords = g_norm_coords +0.5
        g_norm_coords = torch.clamp(g_norm_coords * resolution, 0, resolution - 1)
        g_vox_coords = torch.round(g_norm_coords).to(torch.int32).cpu()  # a
        univox_g = np.unique(g_vox_coords.cpu(), axis=0)  # b
        if not fine:
            index_matrix = np.zeros((resolution, resolution, resolution))
            index_matrix[tuple(univox_p.T)] = 1
            index_matrix = ndimage.binary_dilation(index_matrix, structure=np.ones((3, 3, 3)))

            g_index_matrix = np.zeros((resolution, resolution, resolution))
            g_index_matrix[tuple(univox_g.T)] = 1
            index_matrix = g_index_matrix - index_matrix*g_index_matrix
            index_matrix = ndimage.binary_dilation(index_matrix, structure=np.ones((3, 3, 3)))
            p_x, p_y, p_z = np.nonzero(index_matrix)
            p_res = p_x * resolution * resolution + p_y * resolution + p_z

        elif fine:
            index_matrix = np.zeros((resolution, resolution, resolution))
            index_matrix[tuple(univox_p.T)] = 1
            index_matrix = ndimage.binary_dilation(index_matrix, structure=np.ones((3, 3, 3)))
            p_x, p_y, p_z = np.nonzero(index_matrix)
            p_res = p_x * resolution * resolution + p_y * resolution + p_z
        g_res = (g_vox_coords.numpy()[:, 0] * resolution * resolution + g_vox_coords.numpy()[:,
                                                                        1] * resolution + g_vox_coords.numpy()[:,
                                                                                          2]).flatten()
        res = np.isin(g_res, p_res)
        test = g[res]
        ret_gt = torch.reshape(test, (1, -1, 3))

        while test.shape[0] < npoints:

            if not fine:
                resolution = resolution + 8
                if (resolution > 40):
                    up = npoints - test.shape[0]
                    while up > test.shape[0]:
                        error = error+1
                        test = torch.cat([test, test], dim=0)
                        up = npoints - test.shape[0]
                        if error>5:
                            print('error:',error)
                            print(test.shape[0])
                            if test.shape[0]<=1024:
                                test=g
                    upcoor = test[:up]
                    test = torch.cat([upcoor, test], dim=0)
                    ret_gt = torch.reshape(test, (1, -1, 3))
                    break
            elif fine:
                resolution = resolution - 4
            if (resolution <= 0):
                test = g
                ret_gt = torch.reshape(g, (1, -1, 3))
                break
            p_coords = p
            p_norm_coords = p
            if normalize:
                p_norm_coords = p_norm_coords / (
                        p_norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + eps) + 0.5
            else:
                p_norm_coords = p_norm_coords + 0.5
            p_norm_coords = torch.clamp(p_norm_coords * resolution, 0, resolution - 1)
            vox_coords = torch.round(p_norm_coords).to(torch.int32)

            # print("voxsize:",vox_coords.size())
            univox_p = np.unique(vox_coords.cpu(), axis=0)  # b
            # print("univox_p:",univox_p)
            # print("univox_p:", univox_p.size)

            g_norm_coords = g
            # p_norm_coords = p_coords - p_coords.mean(1, keepdim=True)
            if normalize:
                g_norm_coords = g_norm_coords / (
                        g_norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + eps) + 0.5
            else:
                g_norm_coords = g_norm_coords + 0.5
            g_norm_coords = torch.clamp(g_norm_coords * resolution, 0, resolution - 1)
            g_vox_coords = torch.round(g_norm_coords).to(torch.int32).cpu()  # a
            univox_g = np.unique(g_vox_coords.cpu(), axis=0)  # b
            if not fine:
                index_matrix = np.zeros((resolution, resolution, resolution))
                index_matrix[tuple(univox_p.T)] = 1
                index_matrix = ndimage.binary_dilation(index_matrix, structure=np.ones((3, 3, 3)))

                g_index_matrix = np.zeros((resolution, resolution, resolution))
                g_index_matrix[tuple(univox_g.T)] = 1
                index_matrix = g_index_matrix - index_matrix * g_index_matrix
                index_matrix = ndimage.binary_dilation(index_matrix, structure=np.ones((3, 3, 3)))
                p_x, p_y, p_z = np.nonzero(index_matrix)
                p_res = p_x * resolution * resolution + p_y * resolution + p_z

            elif fine:
                index_matrix = np.zeros((resolution, resolution, resolution))
                index_matrix[tuple(univox_p.T)] = 1
                index_matrix = ndimage.binary_dilation(index_matrix, structure=np.ones((3, 3, 3)))
                p_x, p_y, p_z = np.nonzero(index_matrix)
                p_res = p_x * resolution * resolution + p_y * resolution + p_z
            g_res = (g_vox_coords.numpy()[:, 0] * resolution * resolution + g_vox_coords.numpy()[:,
                                                                            1] * resolution + g_vox_coords.numpy()[:,
                                                                                              2]).flatten()
            res = np.isin(g_res, p_res)
            test = g[res]
            ret_gt = torch.reshape(test, (1, -1, 3))
        '''
        #crop 2022-10-13
        crop_all =torch.cat([p_coords,test],dim=0)
        #print(crop_all.size())
        ret_gt_sorted,idx = torch.unique(crop_all,return_inverse=True,dim=0)

        idx2=torch.unique(idx.cpu(),sorted=False)
        ret_gt = ret_gt_sorted[idx2,:]
        ret_gt =ret_gt[:ret_gt.size(0)-2048,:]
        '''
        # print(ret_gt)
        '''
        crop_all_np =crop_all.cpu().numpy()
        _,idx = np.unique(crop_all_np,axis=0,return_index=True)
        croped = crop_all_np[np.sort(idx),:]
        print(idx)

        ret_gt = torch.from_numpy(croped[:2048,:])
        '''
        # ret_gt = torch.reshape(ret_gt, (1, -1, 3))
        # print(ret_gt)
        # -----------------------
        # print(test.shape[0])
        '''
        if(test.shape[0]<npoints):
            up=npoints-test.shape[0]
            upcoor=g[:up]
            ret_gt=torch.cat([test, upcoor], dim=0)
            ret_gt = torch.reshape(ret_gt, (1, -1, 3))
        '''
        if (ret_gt.size(1) > npoints):
            # ret_gt=torch.from_numpy(test)
            ret_gt = fps_subsample(ret_gt, npoints)
            '''
            #down=test.shape[0]-npoints
            downcoor=test[:npoints]
            ret_gt = torch.reshape(downcoor, (1, -1, 3))
            '''
        if (ret == None):
            ret = ret_gt
        else:
            ret = torch.cat([ret, ret_gt], dim=0)
        '''
        # print("voxsize:", g_vox_coords.size())
        univox_g = np.unique(g_vox_coords.cpu(), axis=0)
        # print("univox_g:", univox_g)
        # print("univox_g:", univox_g.size)
        # print("iou:",(univox_p.size/univox_g.size))
        iou.append((univox_p.size / univox_g.size))
    iou = np.array(iou)
    iou = torch.from_numpy(iou)
    # print(iou.size())
    iou_data = torch.unsqueeze(iou, dim=1)
    '''
    # print(ret.size())
    # print(ret)
    return ret


def val_grid(image_path, base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args,
             config,
             logger=None):
    print_log(f"[VALIDATION] Start val_grid epoch {epoch}", logger=logger)
    base_model.eval()  # set model to eval mode
    image_row = True
    image_path = '/home/M2021-CJX/ml_hw/PoinTr-master/train_image/'
    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2', 'r5'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader)  # bs is 1
    image_merged = None
    a = random.sample(range(0, 10517), 6)
    b = 0
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.val._base_.N_POINTS
            dataset_name = config.dataset.val._base_.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                gt = data[1].cuda()
            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1 / 4), int(npoints * 3 / 4)],
                                                      fixed_points=None)
                partial = partial.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')
            r1 = 32
            r2 = 16
            r3 = 12
            r4 = 8
            r5 = 4
            n1 = 4096
            gt_32, num_32, gt_32_raw = get_grid(partial, gt, r1, n1)
            gt_16, num_16, gt_16_raw = get_grid(partial, gt, r2, n1)
            gt_12, num_12, gt_12_raw = get_grid(partial, gt, r3, n1)
            gt_8, num_8, gt_8_raw = get_grid(partial, gt, r4, n1)
            gt_4, num_4, gt_4_raw = get_grid(partial, gt, r5, n1)

            num_4 = num_4.cuda()
            num_8 = num_8.cuda()
            num_12 = num_12.cuda()
            num_16 = num_16.cuda()
            num_32 = num_32.cuda()

            dense_points = gt_4_raw.cuda()

            test_losses.update([num_32, num_16, num_12, num_8, num_4])

            # dense_points_all = dist_utils.gather_tensor(dense_points, args)
            # gt_all = dist_utils.gather_tensor(gt, args)

            # _metrics = Metrics.get(dense_points_all, gt_all)
            _metrics = Metrics.get(dense_points, gt, num_4)
            # _metrics = [dist_utils.reduce_tensor(item, args) for item in _metrics]

            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            category_metrics[taxonomy_id].update(_metrics)

            if image_row and idx <= 500:
                input_pc = partial.squeeze().detach().cpu().numpy()
                input_pc = misc.get_ptcloud_img(input_pc, tit='partial')
                sparse = gt_32.squeeze().cpu().numpy()
                sparse_img = misc.get_ptcloud_img(sparse, tit='32_dilation')

                dense = gt_32_raw.squeeze().cpu().numpy()
                dense_img = misc.get_ptcloud_img(dense, tit='32_raw')
                gt_ptcloud = gt_16.squeeze().cpu().numpy()
                gt_ptcloud_img = misc.get_ptcloud_img(gt_ptcloud, tit='16_dilation')
                gt32_ptcloud = gt_16_raw.squeeze().cpu().numpy()
                gt32_ptcloud_img = misc.get_ptcloud_img(gt32_ptcloud, tit='16_raw')
                gt16_ptcloud = gt_8.squeeze().cpu().numpy()
                gt16_ptcloud_img = misc.get_ptcloud_img(gt16_ptcloud, tit='8_dilation')
                gt8_raw = gt_8_raw.squeeze().cpu().numpy()
                gt8_raw_img = misc.get_ptcloud_img(gt8_raw, tit='8_raw')
                gt4_dilation = gt_4.squeeze().cpu().numpy()
                gt4_dilation_image = misc.get_ptcloud_img(gt4_dilation, tit='4_dilation')
                gt4_raw = gt_4_raw.squeeze().cpu().numpy()
                gt4_raw_img = misc.get_ptcloud_img(gt4_raw, tit='4_raw')
                gt_raw = gt.squeeze().cpu().numpy()
                gt_img = misc.get_ptcloud_img(gt_raw, tit='gt')
                image_m0 = np.concatenate(
                    [input_pc, dense_img, gt32_ptcloud_img, gt8_raw_img, gt4_raw_img], axis=0)
                image_m1 = np.concatenate(
                    [gt_img, sparse_img, gt_ptcloud_img, gt16_ptcloud_img, gt4_dilation_image], axis=0)
                image_m = np.concatenate([image_m0, image_m1], axis=1)
                '''
                if idx == 0:
                    image_merged = image_m
                else:
                    image_merged = np.concatenate([image_merged, image_m], axis=1)
                '''

                if os.path.exists(image_path):
                    cv2.imwrite(os.path.join(image_path, f'{idx}_merged.jpg'), image_m)
                else:
                    os.mkdir(image_path)
                    cv2.imwrite(os.path.join(image_path, f'{idx}_merged.jpg'), image_merged)

            if (idx + 1) % 200 == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                          (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()],
                           ['%.4f' % m for m in _metrics]), logger=logger)
        for _, v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]),
                  logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================', logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/Sparse', test_losses.avg(0), epoch)
        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)

    return Metrics(config.consider_metric, test_metrics.avg())


def get_grid(partial, gt, resolution, npoints=4096):
    b, n, c = partial.shape
    normalize = False
    eps = 0
    ret = None
    for p, g in zip(partial, gt):
        # print("p:",p.size())
        # print("g:",g.size())
        p_coords = p
        p_norm_coords = p
        # p_norm_coords = p_coords - p_coords.mean(1, keepdim=True)
        if normalize:
            p_norm_coords = p_norm_coords / (
                    p_norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + eps) + 0.5
        else:
            p_norm_coords = (p_norm_coords + 1) / 2.0
        p_norm_coords = torch.clamp(p_norm_coords * resolution, 0, resolution - 1)
        vox_coords = torch.round(p_norm_coords).to(torch.int32)
        # print("voxsize:",vox_coords.size())
        univox_p = np.unique(vox_coords.cpu(), axis=0)  # b
        # print("univox_p:",univox_p)
        # print("univox_p:", univox_p.size)

        g_norm_coords = g
        # p_norm_coords = p_coords - p_coords.mean(1, keepdim=True)
        if normalize:
            g_norm_coords = g_norm_coords / (
                    g_norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + eps) + 0.5
        else:
            g_norm_coords = (g_norm_coords + 1) / 2.0
        g_norm_coords = torch.clamp(g_norm_coords * resolution, 0, resolution - 1)
        g_vox_coords = torch.round(g_norm_coords).to(torch.int32).cpu()  # a
        index_matrix = np.zeros((resolution, resolution, resolution))
        index_matrix[tuple(univox_p.T)] = 1
        if resolution >= 16:
            index_matrix = ndimage.binary_dilation(index_matrix, structure=np.ones((5, 5, 5)))
        elif resolution >= 8 and resolution < 16:
            index_matrix = ndimage.binary_dilation(index_matrix, structure=np.ones((3, 3, 3)))
        else:
            index_matrix = ndimage.binary_dilation(index_matrix, structure=np.ones((2, 2, 2)))
        p_x, p_y, p_z = np.nonzero(index_matrix)
        p_res = p_x * resolution * resolution + p_y * resolution + p_z
        # res = (g_vox_coords.numpy()[:, None] == univox_p).all(-1).any(-1)
        p_res2 = (univox_p[:, 0] * resolution * resolution + univox_p[:, 1] * resolution + univox_p[:, 2]).flatten()
        g_res = (g_vox_coords.numpy()[:, 0] * resolution * resolution + g_vox_coords.numpy()[:,
                                                                        1] * resolution + g_vox_coords.numpy()[:,
                                                                                          2]).flatten()
        res = np.isin(g_res, p_res)
        res2 = np.isin(g_res, p_res2)
        test2 = g[res2]
        test = g[res]
        ret_gt = torch.reshape(test, (1, -1, 3))
        ret_gt2 = torch.reshape(test2, (1, -1, 3))
        ret_num = test2.shape[0]
        num = []
        num.append(ret_num)
        # print(test.shape[0])
        '''
        if(test.shape[0]<npoints):
            up=npoints-test.shape[0]
            upcoor=g[:up]
            ret_gt=torch.cat([test, upcoor], dim=0)
            ret_gt = torch.reshape(ret_gt, (1, -1, 3))
        elif(test.shape[0]>npoints):
            #ret_gt=torch.from_numpy(test)
            ret_gt=torch.reshape(test,(1,-1,3))
            ret_gt=fps_subsample(ret_gt,npoints)


            #down=test.shape[0]-npoints
            downcoor=test[:npoints]
            ret_gt = torch.reshape(downcoor, (1, -1, 3))
            '''
        if (ret == None):
            ret = ret_gt
            ret2 = ret_gt2
        else:
            ret = torch.cat([ret, ret_gt], dim=0)
            ret2 = torch.cat([ret2, ret_gt2], dim=0)
        '''
        # print("voxsize:", g_vox_coords.size())
        univox_g = np.unique(g_vox_coords.cpu(), axis=0)
        # print("univox_g:", univox_g)
        # print("univox_g:", univox_g.size)
        # print("iou:",(univox_p.size/univox_g.size))
        iou.append((univox_p.size / univox_g.size))
    iou = np.array(iou)
    iou = torch.from_numpy(iou)
    # print(iou.size())
    iou_data = torch.unsqueeze(iou, dim=1)
    '''
    ret_num = np.array(num)
    ret_num = torch.from_numpy(ret_num)
    # print(iou.size())
    ret_num = torch.unsqueeze(ret_num, dim=1)
    # print(ret.size())
    return ret, ret_num, ret2

def pcn_partial(partial):
    b, n, c = partial.shape
    normalize = False
    eps = 0
    ret = None
    for p in partial:
        error=0
        p_unique=np.unique(p.cpu(), axis=0)  # b
        #p_unique=p.cpu()
        if p_unique.shape[0]<2048:
            p_nozero=p[:(p_unique.shape[0]-1)]
        else:
            p_nozero=p
        #p_nozero = p_unique[[not np.all(p_unique[i] == 0) for i in range(p_unique.shape[0])], :]
        up = 2048 - p_nozero.shape[0]
        while up > p_nozero.shape[0]:
            error = error + 1
            p_nozero = torch.cat([p_nozero, p_nozero], dim=0)
            up = 2048 - p_nozero.shape[0]
            if error > 5:
                print('error:', error)
                print(p_nozero.shape[0])
        if p_nozero.shape[0]<2048:
            p_nozero = torch.cat([p_nozero, p_nozero], dim=0)

        if (p_nozero.shape[0] > 2048):
            # ret_gt=torch.from_numpy(test)
            p_nozero = p_nozero[:2048]

            '''
            #down=test.shape[0]-npoints
            downcoor=test[:npoints]
            ret_gt = torch.reshape(downcoor, (1, -1, 3))
            '''
        ret_p = torch.reshape(p_nozero, (1, -1, 3))
        if (ret == None):
            ret = ret_p
        else:
            ret = torch.cat([ret, ret_p], dim=0)
    return ret