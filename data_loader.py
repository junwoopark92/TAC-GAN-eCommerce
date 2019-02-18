import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy.random as random
import h5py

class ImTextDataset(Dataset):
    '''
    data_dir   : path to the directory that contains the dataset files
    dataset    : name of the dataset (eg. flowers/coco)
    train      : determines which part of the dataset to use. By default:train
    image_size : intented image size. By default: 128x128
    '''
    def __init__(self, data_dir, dataset='products', train=True, image_size=128, cap_size_per_img=1):
        super(ImTextDataset, self).__init__()
        
        self.train = train  # determines whether to return train or validation images
        self.data_dir = data_dir
        self.dataset = dataset
        self.image_dir = os.path.join(self.data_dir, dataset, 'images')
        self.data = h5py.File(os.path.join(data_dir, dataset, 'train/data.h5py'), 'r')['train']
        self.trans_img = transforms.Compose([transforms.Scale(image_size), transforms.CenterCrop(image_size),
                                             transforms.ToTensor(),])# transformation for output image
        self.cap_size_per_img = cap_size_per_img
        for key in self.data.keys():
            print(key)

    def __getitem__(self, index):
        asin = self.data['asin'][index].decode("utf-8")
        vec = self.data['docvec'][index]
        cate = self.data['cate'][index]
        # load the image and apply the transformation to it
        image_path = os.path.join(self.image_dir, asin)
        image =Image.open(image_path)
        image = self.trans_img(image)
        # pick a random encoded caption
        return image, cate, vec

    def __len__(self):
        if self.train:
            return self.data['asin'].shape[0]
