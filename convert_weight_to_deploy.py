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
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        dest='model_path',
        required=True,
        help="Path to the trained model"
    )
    
    args = parser.parse_args()

    # device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load model
    model = model.PilotNet(input_shape=(66, 200))
    model.eval()
      
    if os.path.exists(args.model_path) == False:
        print("Error: model not found!")
        exit(1)
    else:
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
    # save to train -> .pth
    # save to deloy -> .pthd
    output_model_path = args.model_path + "d"
    torch.save(model, output_model_path)
