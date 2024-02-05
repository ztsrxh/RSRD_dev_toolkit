import PIL.Image
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class RSRD(Dataset):
    def __init__(self, dataset_path):
        super(RSRD, self).__init__()
        self.dataset_path = dataset_path  # 'XXX/train/'
        self.file_name_list = self.read_file_name()
        self.dataset_size = len(self.file_name_list)

        self.transform_jpg = transforms.Compose([
                transforms.ToTensor(),  # image --> [0, 1]
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [0, 1] --> [-1, 1]
            ])

    def read_file_name(self):
        file_name_list = []
        sub_folders = os.listdir(self.dataset_path)
        for folder in sub_folders:
            imgs = os.listdir(self.dataset_path + folder + '/left/')
            imgs_stamp = [(folder, i.replace('.jpg', '')) for i in imgs]  # retain the time stamp in the file name
            file_name_list += imgs_stamp

        return file_name_list

    def __getitem__(self, index):
        (folder, stamp) = self.file_name_list[index]
        path_left = self.dataset_path + folder + '/left/' + stamp + '.jpg'
        path_right = self.dataset_path + folder + '/right/' + stamp + '.jpg'
        path_dis = self.dataset_path + folder + '/disparity/' + stamp + '.png'
        path_depth = self.dataset_path + folder + '/depth/' + stamp + '.png'
        # path_pcd = self.dataset_path + folder + '/pcd/' + stamp + '.pcd'

        img_left = PIL.Image.open(path_left)
        img_left = self.transform_jpg(img_left)
        img_right = PIL.Image.open(path_right)
        img_right = self.transform_jpg(img_right)
        dis_map = PIL.Image.open(path_dis)
        dis_map = torch.from_numpy(np.array(dis_map, dtype=np.float32)).div(256)
        depth_map = PIL.Image.open(path_depth)
        depth_map = torch.from_numpy(np.array(depth_map, dtype=np.float32)).div(256)

        return img_left, img_right, dis_map, depth_map

    def __len__(self):
        return self.dataset_size


if __name__ == '__main__':

    train_set = RSRD('/dataset/RSRD-dense/train/')
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    for i, (img_left, img_right, dis_map, depth_map) in enumerate(train_loader):
        pass