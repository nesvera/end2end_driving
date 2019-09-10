import torch
from torch.utils.data import Dataset, DataLoader

import os
import cv2
import numpy as np

class Image2SteeringDataset(Dataset):
    '''

        label format = 'image_name:speed:steering_front:steering_rear'

    '''

    def __init__(self, dataset_path):

        self.dataset_path = dataset_path
        self.label_path = dataset_path + "/label.txt"

        if os.path.exists(self.label_path) == False:
            return None

        label_file = open(self.label_path, 'r')

        self.label = list()

        while True:
            label_line = label_file.readline()

            if label_line == '':
                break

            label_line = label_line.split(':')

            image_name = label_line[0]
            front_steering = int(label_line[2])

            self.label.append((image_name, front_steering))

        label_file.close()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.label[idx]

        image_path = self.dataset_path + '/' + label[0] + '.jpg'

        # load image
        image = cv2.imread(image_path)
        image = image.astype('float')
        image /= 255
        image = np.clip(image, 0, 1)
        image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]]) 

        # normalize steering
        steering = float(label[1])
        steering /= 127
        steering = np.clip(steering, -1, 1)

        sample = {'image': image, 'steering': steering}

        return sample
