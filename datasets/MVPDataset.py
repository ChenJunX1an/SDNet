import os
import torch
import numpy as np
import torch.utils.data as data
from .build import DATASETS
import logging
import h5py
@DATASETS.register_module()
class MVP(data.Dataset):
    #def __init__(self, train=True, npoints=2048, novel_input=True, novel_input_only=False):
    def __init__(self,config):
        self.subset = config.subset
        self.npoints = config.N_POINTS
        novel_input = True
        novel_input_only = False
        if self.subset == 'train':
            self.input_path = './data/MVP/mvp_train_input.h5'
            self.gt_path = './data/MVP/mvp_train_gt_%dpts.h5' % self.npoints
            self.split_path='./data/MVP/mvp_train_split_%d_new.h5' % self.npoints
        else:
            self.input_path = './data/MVP/mvp_test_input.h5'
            self.gt_path = './data/MVP/mvp_test_gt_%dpts.h5' % self.npoints
            self.split_path = './data/MVP/mvp_test_split_%d_new.h5' %self.npoints
        self.train = self.subset

        input_file = h5py.File(self.input_path, 'r')
        self.input_data = np.array((input_file['incomplete_pcds'][()]))
        self.labels = np.array((input_file['labels'][()]))
        self.novel_input_data = np.array((input_file['novel_incomplete_pcds'][()]))
        self.novel_labels = np.array((input_file['novel_labels'][()]))
        input_file.close()

        gt_file = h5py.File(self.gt_path, 'r')
        self.gt_data = np.array((gt_file['complete_pcds'][()]))
        self.novel_gt_data = np.array((gt_file['novel_complete_pcds'][()]))
        gt_file.close()

        f=h5py.File(self.split_path, 'r')
        self.fine =  np.array((f['fine'][()]))
        self.lost =  np.array((f['lost_pcds'][()]))
        f.close()

        if novel_input_only:
            self.input_data = self.novel_input_data
            self.gt_data = self.novel_gt_data
            self.labels = self.novel_labels
        elif novel_input:
            self.input_data = np.concatenate((self.input_data, self.novel_input_data), axis=0)
            self.gt_data = np.concatenate((self.gt_data, self.novel_gt_data), axis=0)
            self.labels = np.concatenate((self.labels, self.novel_labels), axis=0)

        print(self.input_data.shape)
        print(self.gt_data.shape)
        print(self.labels.shape)
        print(self.fine.shape)
        self.len = self.input_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        sample={}
        data={}
        partial = torch.from_numpy((self.input_data[index]))
        complete = torch.from_numpy((self.gt_data[index // 26]))
        label = (self.labels[index])
        sample['taxonomy_id']=label
        sample['model_id']=index
        data['partial'] = partial
        data['gt'] = complete

        fine =torch.from_numpy((self.fine[index]))
        lost = torch.from_numpy((self.lost[index]))

        data['fine']=fine
        data['lost']=lost
        return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'],data['fine'],data['lost'])

'''
if __name__ == '__main__':
    pcn=MVP()
    print(len(pcn))
    for idx in range(len(pcn)):
        a,b,c=pcn.__getitem__(idx)
        print(a)
        print(b)
        print(c)
        if idx>10:
            break
'''