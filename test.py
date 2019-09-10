import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
from torch.utils.data import DataLoader

import model
import dataset

import argparse
import os
import cv2
import random
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        dest='dataset_path',
        required=True,
        help="Dataset path"
    )
    parser.add_argument(
        '-m',
        dest='model_path',
        required=False,
        help="Path to the trained model"
    )
    
    args = parser.parse_args()

    dataset_path = args.dataset_path
    model_path = args.model_path

    # device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load datataset and dataloaders
    if os.path.exists(dataset_path) == False:
        print("Error: input path is not a foler")
        exit(1)

    if os.path.exists(model_path) == False:
        print("Error: Model not found!")
        exit(1)

    train_dataset = dataset.Image2SteeringDataset(dataset_path)

    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

    model = torch.load(model_path)
    model.eval()

    while True:
        
        image_np = random.choice(train_dataset)['image']
        image = np.expand_dims(image_np.copy(), axis=0)
        image = torch.from_numpy(image)
        image = image.to(device)

        output = model.forward(image.float())

        print(output)

        image_np = np.reshape(image_np, (image_np.shape[1], image_np.shape[2], image_np.shape[0]))
        cv2.imshow('image', image_np)
        cv2.waitKey(0)


  



