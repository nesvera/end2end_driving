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

learning_rate = 0.0001
epochs = 20
batch_size = 32

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        dest='dataset_path',
        required=True,
        help="Dataset path"
    )
    parser.add_argument(
        '-o',
        dest='output_model_path',
        required=False,
        help="Folder to store the model and weights"
    )
    parser.add_argument(
        '-m',
        dest='model_path',
        required=False,
        default=None,
        help="Path to the trained model"
    )
    
    args = parser.parse_args()

    dataset_path = args.dataset_path
    output_model_path = args.output_model_path

    # device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load datataset and dataloaders
    if os.path.isdir(dataset_path) == False:
        print("Error: input path is not a folder")
        exit(1)

    if os.path.isdir(output_model_path) == False:
        print("Error: path to save the model is not a folder")
        exit(1)

    train_dataset = dataset.Image2SteeringDataset(dataset_path)

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # load model
    model = model.PilotNet(input_shape=(66, 200))

    if torch.cuda.is_available():
        model  = model.cuda()
      
    # loss and optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fcn = nn.MSELoss()

    checkpoint_epoch = 0

    # train model
    total_step = len(dataloader)
    training_logger = list()

    if args.model_path is not None:
        if os.path.exists(args.model_path) == False:
            print("Error: model not found!")
            exit(1)
        else:
            checkpoint = torch.load(args.model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            checkpoint_epoch = checkpoint['epoch']
            loss_fcn = checkpoint['loss_fcn']
            training_logger = checkpoint['training_logger']

    for epoch in range(checkpoint_epoch, checkpoint_epoch+epochs):
        
        verbose_logger_sum = 0
        batch_logger_sum = 0
        for i_batch, sample_batched in enumerate(dataloader):

            images = sample_batched['image'].to(device)
            labels = sample_batched['steering'].to(device)

            # Foward pass
            outputs = model(images.float())
            outputs = outputs.view(outputs.size()[0])
            
            loss = loss_fcn(outputs, labels)
            batch_logger_sum += loss
            verbose_logger_sum += loss

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i_batch+1) % 100 == 0:
                verbose_logger_sum /= 100.
                print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}'
                      .format(epoch+1, checkpoint_epoch+epochs, i_batch+1, total_step, verbose_logger_sum))

                verbose_logger_sum = 0
    # save model


        batch_logger_avg = batch_logger_sum/len(dataloader)

        training_logger.append(batch_logger_avg)

    output_model_path += "/model_" + str(epoch+1) + ".pth"
    torch.save({'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_fcn': loss_fcn,
                'training_logger': training_logger}, output_model_path)

    plt.plot(training_logger)
    plt.show()
