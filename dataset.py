import os, cv2
import pdb
import numpy as np
import scipy.io as sio
from skimage import io
import skimage.transform as SkT
from PIL import Image
import glob
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T

import np_transforms as NP_T
from utils import density_map, gaussian_filter_density
from scipy.ndimage.filters import gaussian_filter


class CrowdDataset(Dataset):
    def __init__(self,
                 mode,
                 path,
                 out_shape=None,
                 transform=None,
                 gamma=5,
                 max_len=None,
                 adaptive=False,
                 k_nearest=3,
                 load_all=False):
        self.k = k_nearest
        self.adaptive = adaptive
        self.path = path
        self.out_shape = np.array(out_shape)
        self.transform = transform
        self.gamma = gamma
        self.load_all = load_all

        self.mode = mode
        self.img_path = os.path.join(self.path, self.mode, 'images')
        self.label_path = os.path.join(self.path, self.mode, 'ground_truth')

        self.image_files = glob.glob(os.path.join(self.img_path, '*.jpg'))

        if self.load_all:
            if self.mode == 'train':
                self.images, self.gts, self.densities = [], [], []
                for img_f in self.image_files:
                    X, density, gt = self.load_example(img_f, self.mode)
                    self.images.append(X)
                    self.densities.append(density)
                    self.gts.append(gt)
            else:
                self.images, self.gts = [], []
                for img_f in self.image_files:
                    X, gt = self.load_example(img_f, self.mode)
                    self.images.append(X)
                    self.gts.append(gt)

    def load_example(self, img_f, mode='train'):
        img = cv2.imread(img_f)
        points = np.array(sio.loadmat(img_f.replace('jpg', 'mat').replace('images', 'ground_truth').replace('img', 'GT_img'))['locations'])
        points = points[:,:-1]

        H_orig, W_orig = img.shape[:2]
        if H_orig != self.out_shape[0] or W_orig != self.out_shape[1]:
            # img = img.resize((self.out_shape[1], self.out_shape[0]), Image.BILINEAR)
            img = cv2.resize(img, (self.out_shape[1], self.out_shape[0]), cv2.INTER_LINEAR)
            ratio = self.out_shape / np.array([H_orig, W_orig])
            print(ratio.shape, points.shape)
            points = np.round(points*ratio)
        img = np.array(img, np.float32)
        points = np.array(points, np.int32)
        points = np.minimum(points, self.out_shape - 1)
        gt = np.zeros(self.out_shape)
        gt[points[:, 0], points[:, 1]] = 1
        gt = gt[:, :, np.newaxis].astype('float32')
        
        if mode == 'train':
            density = gaussian_filter_density(gt, self.gamma, self.k, self.adaptive)
            density = cv2.resize(density, (density.shape[1] // 8, density.shape[0] // 8), interpolation=cv2.INTER_LINEAR) * 64
            density = density[:, :, np.newaxis].astype('float32')
            
            return img, density, gt
        
        return img, gt

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, i):
        if self.load_all:
            img_f = self.image_files[i]
            X = self.images[i]
            gt = self.gts[i]
            if self.mode == 'train':
                density = self.densities[i]
        else:
            img_f = self.image_files[i]
            if self.mode == 'train':
                X, density, gt = self.load_example(img_f)
            X, gt = self.load_example(img_f, mode='test')

        if self.transform:
            if self.mode == 'train':
                X, density, gt = self.transform([X, density, gt])
            else:
                X, gt = self.transform([X, gt])

        if self.mode == 'train':
            return X, density, gt
        return X, gt, img_f


class CrowdSeq(CrowdDataset):
    def __init__(self,
                 train=True,
                 path='../../ds/dronebird',
                 out_shape=[1080, 1920],
                 transform=None,
                 gamma=5,
                 adaptive=False,
                 k_nearest=3,
                 max_len=None,
                 load_all=False):
        super(CrowdSeq, self).__init__(train=train,
                                       path=path,
                                       out_shape=out_shape,
                                       transform=transform,
                                       gamma=gamma,
                                       adaptive=adaptive,
                                       k_nearest=k_nearest,
                                       max_len=max_len,
                                       load_all=load_all)

        self.img2idx = {img: idx for idx, img in enumerate(self.image_files)}
        self.seqs = []
        prev_seq = None
        cur_len = 0
        for img_f in self.image_files:
            seq_name, img_name = os.path.basename(img_f)[3:6], os.path.basename(img_f)[6:9]
            if (seq_name == prev_seq) and ((max_len is None) or
                                           (cur_len < max_len)):
                self.seqs[-1].append(img_f)
                cur_len += 1
            else:
                self.seqs.append([img_f])
                cur_len = 1
                prev_seq = seq_name

        if max_len is None:
            self.max_len = max([len(seq) for seq in self.seqs])
        else:
            self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        seq = self.seqs[i]
        seq_len = len(seq)

        # randomize the (random) transformations applied to the first image of the sequence
        # and then apply the same transformations to the remaining images of the sequence
        if isinstance(self.transform, T.Compose):
            for transf in self.transform.transforms:
                if hasattr(transf, 'rand_state'):
                    transf.reset_rand_state()
        elif hasattr(self.transform, 'rand_state'):
            self.transform.reset_rand_state()

        # build the sequences
        X = torch.zeros(self.max_len, 3, self.out_shape[0], self.out_shape[1])
        gt = torch.zeros(self.max_len, 1, self.out_shape[0], self.out_shape[1])
        if self.mode == 'train':              
            density = torch.zeros(self.max_len, 1, self.out_shape[0]//8, self.out_shape[1]//8)
        names = []
        for j, img_f in enumerate(seq):
            idx = self.img2idx[img_f]
            if self.mode == 'train':
                X[j], density[j], gt[j] = super().__getitem__(idx)
            else:
                X[j], gt[j], name = super().__getitem__(idx)
                names.append(name)
        if self.mode == 'train':
            return X, density, gt, seq_len
        return X, gt, seq_len, names


if __name__ == '__main__':
    train_transf = T.Compose([
        NP_T.RandomHorizontalFlip(0.5, keep_state=True),
        NP_T.ToTensor()
    ])
    # data = CrowdDataset(train=False,
    #                     path='../FDST/FDST',
    #                     load_all=False,
    #                     max_len=1,
    #                     transform=train_transf,
    #                     out_shape=[240, 320],
    #                     adaptive=False,
    #                     k_nearest=4,
    #                     gamma=100)
    # train_loader = DataLoader(data, batch_size=10, shuffle=True, num_workers=1)
    # for i, (X, density, gt, count) in enumerate(train_loader):
    #     aa = 1
    #     print('Image {}: count={}, density_sum={:.3f}'.format(
    #         i, count.sum(), density.sum()))
    path = '../../ds/dronebird'
    
    data = CrowdSeq(train=True,
                    path=path,
                    load_all=False,
                    max_len=1,
                    transform=train_transf,
                    out_shape=[240, 320],
                    adaptive=False,
                    k_nearest=2,
                    gamma=100)
    train_loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=0)
    for i, (X, density, gt, seq_len) in enumerate(train_loader):
        # print(i)
        print('count={}, density_sum={:.3f}'.format(gt.sum(), density.sum()))
