import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import os.path as osp
import torchvision.transforms as T
from helper.utils import CircularList
import mrcfile as mrc
import warnings

warnings.simplefilter('ignore')  # to mute some warnings produced when opening the tomos
import re
from skimage import io


class cryoDataset(Dataset):
    def __init__(self, root_dir, use_labels=True, num_crops=100, transform=None):
        """
        Inputs - 
            root_dir : Root directory of data
            num_crops : Number of crops to extract from each image
            transform : Augmentations to perform on each crop (for positive/negative sample setup)
            use_labels : Whether to use labels to pick only valid slices with particles

        Output -
            [ x_aug1, x_aug2 ] : Pair of augmented views of same image

        Lookup tables -
            1) scan_z_idx - {'scan_id' : {'z_id' : glbl_idx } }
        """
        self.root_dir = root_dir
        self.transform = transform
        self.use_labels = use_labels
        self.data_dir = osp.join(root_dir, 'data')
        self.lbl_dir = osp.join(root_dir, 'labels')
        self.scan_z_idx = {}
        # To handle calls for indices > len(dataset) : new_index = index % len(dataset)
        self.scan_ids = os.listdir(self.data_dir)
        self.lbl_ids = os.listdir(self.lbl_dir)
        self.image_paths = [osp.join(self.data_dir, curr_path) for curr_path in self.scan_ids]
        self.lbl_paths = [osp.join(self.lbl_dir, curr_path) for curr_path in self.lbl_ids]

        self.actual_dataset = self.read_dataset()
        self.dataset_len = num_crops * len(self.actual_dataset)
        self.resize_cropper = T.RandomResizedCrop(size=(128, 128), scale=(0.01, 0.03))

    def get_scan_id(self, img_path):
        img_name = img_path.split('/')[-1]
        return re.findall('\d+', img_name)[-1]

    def get_z_range(self, particle_path):
        ## Pick only those slices where particles are present
        z_max = -1
        z_min = float('inf')
        with open(particle_path, 'rU') as f:
            for line in f:
                pdb_id, X, Y, Z, *_ = line.rstrip('\n').split()
                z_max = max(z_max, int(Z))
                z_min = min(z_min, int(Z))

        return z_max, z_min

    def read_dataset(self):
        """
        Read each scan, store the all slices as separate images, populate lookup table
        """
        out_list = []
        glbl_idx = 0
        for img_path in self.image_paths:
            scan_id = self.get_scan_id(img_path)
            with mrc.open(img_path, permissive=True) as f:
                tomo_data = f.data

            if self.use_labels:
                particle_path = osp.join(self.lbl_dir, 'particle_{}.txt'.format(scan_id))
                z_max, z_min = self.get_z_range(particle_path)
                tomo_data = tomo_data[z_min:z_max + 1]

            self.scan_z_idx[scan_id] = [-1 for _ in range(tomo_data.shape[0])]

            for sl in range(tomo_data.shape[0]):
                out_list.append(tomo_data[sl])
                self.scan_z_idx[scan_id][sl] = glbl_idx
                glbl_idx += 1

        return out_list

    def __getitem__(self, index):
        index = index % len(self.actual_dataset)
        img = self.resize_cropper(torch.from_numpy(self.actual_dataset[index]).unsqueeze(0))
        if self.transform is not None:
            img = self.transform(img)
        return img  # Returns [ x_aug1, x_aug2 ]

    def __len__(self):
        return self.dataset_len


class cryoDataset2(Dataset):
    def __init__(self, root_dir, use_labels=True, num_crops=100, transform=None, n_channels=80):
        """
        Inputs -
            root_dir : Root directory of data
            num_crops : Number of crops to extract from each image
            transform : Augmentations to perform on each crop (for positive/negative sample setup)
            use_labels : Whether to use labels to pick only valid slices with particles
            n_channels: Number of slices to be extracted from each side of the tomogram

        Output -
            [ x_aug1, x_aug2 ] : Pair of augmented views of same image

        Lookup tables -
            1) scan_z_idx - {'scan_id' : {'z_id' : glbl_idx } }
        """
        self.root_dir = root_dir
        self.transform = transform
        self.use_labels = use_labels
        self.data_dir = osp.join(root_dir, 'data')
        self.mask_dir = osp.join(root_dir, 'mask')

        self.nch = n_channels
        self.n_crops = num_crops

        # To handle calls for indices > len(dataset) : new_index = index % len(dataset)
        self.scan_ids = os.listdir(self.data_dir)

        self.mrc_paths = [osp.join(self.data_dir, curr_path) for curr_path in self.scan_ids]

        self.actual_dataset, self.actual_nch = self.read_dataset()

        self.dataset_len = num_crops * len(self.actual_dataset) * self.actual_nch

        # self.resize_cropper = T.RandomResizedCrop(size=(128, 128), scale=(0.01, 0.03))

    def read_dataset(self):
        """
        Read each scan, store the all slices as separate images, populate lookup table
        """
        mrc_list = []

        # print(self.image_paths) # path to the mrc files

        for img_path in self.mrc_paths:
            with mrc.open(img_path, permissive=True) as f:
                tomo_data = f.data
                # select only nch slices arround the zero coordinate
                mrc_list.append(tomo_data[256 - self.nch:256 + self.nch + 1])

            # print(tomo_data.shape)
            # print(len(coord))
            # coord = np.array(coord)
            # print(coord.shape)
            # # exit()
            #
            # # plot images and particles
            # for i in range(0, len(tomo_data)):
            #     img = tomo_data[i]
            #     plt.imshow(img)
            #
            #     map = coord[:, 0] == (i + z_min)
            #
            #     c = coord[map, :]
            #
            #     print(c.shape)
            #     # exit()
            #
            #     plt.plot(c[:, 1], c[:, 2], 'xr')
            #
            #     plt.show()

        return mrc_list, len(mrc_list[0])

    def __getitem__(self, index):
        # index = index % len(self.actual_dataset)

        mrc_idx = index // (self.n_crops * self.actual_nch)

        a = index - mrc_idx * (self.n_crops * self.actual_nch)

        slice_idx = (a // (self.n_crops))
        slice_idx_actual = slice_idx  + 256 - self.nch

        # print(index, mrc_idx, slice_idx)

        mrc_path = self.mrc_paths[mrc_idx]

        mrc_idx_actual = mrc_path[-5]

        mask_address = self.mask_dir + '/%d_%d.jpg' % (int(mrc_idx_actual), slice_idx_actual)

        # print(mask_address)

        mask = io.imread(mask_address)

        # correct mask distortion during saving
        mask[mask > 200] = 255
        mask[mask < 255] = 0


        w = 16 # half width of the patch to extract
        # zero out border of the masks
        mask[:w, :] = 0
        mask[:, :w] = 0
        mask[len(mask)-w:, :] = 0
        mask[:, len(mask)-w:] = 0



        idx = np.where(mask.reshape(-1) == 255)[0]


        rnd = np.random.randint(low= 0, high= len(idx))

        idx = idx[rnd]

        # location of the crop based on the mask
        x = idx // 512
        y = idx % 512


        slice = self.actual_dataset[mrc_idx][slice_idx]

        crop = slice[x-w:x+w, y-w:y+w]


        # plot results for debugging
        # plt.subplot(1,3,1)
        # plt.imshow(slice)
        # plt.plot(y, x, '+r')
        # plt.subplot(1,3, 2)
        # plt.imshow(mask)
        # plt.plot(y, x, '+r')
        # plt.subplot(1,3, 3)
        # plt.imshow(crop)
        # plt.show()




        crop = torch.from_numpy(crop).unsqueeze(0)
        # img = self.resize_cropper(torch.from_numpy(self.actual_dataset[index]).unsqueeze(0))
        if self.transform is not None:
            crop = self.transform(crop)
        return crop  # Returns [ x_aug1, x_aug2 ]

    def __len__(self):
        return self.dataset_len
