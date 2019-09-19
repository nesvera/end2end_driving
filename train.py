#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras.models import model_from_json, load_model
import numpy as np


# ### Load Model

# In[2]:


# load model
json_file = open('autopilot_model.json')

model_json = json_file.read()
json_file.close()

model = model_from_json(model_json)
model.load_weights('/home/nvidia/Documents/nesvera/neural_nets/lane_following/train_results/2019_05_31-t1/weights/19_06_03-13_21-400.h5')

model.summary()


# ### Load dataset

# In[3]:


x_train_path = "/home/nvidia/Documents/nesvera/neural_nets/lane_following/dataset/2019_05_25_x_train.npy"
y_train_path = "/home/nvidia/Documents/nesvera/neural_nets/lane_following/dataset/2019_05_25_y_train.npy"

x_validation_path = "/home/nvidia/Documents/nesvera/neural_nets/lane_following/dataset/02019_05_25_x_validation.npy"
y_validation_path = "/home/nvidia/Documents/nesvera/neural_nets/lane_following/dataset/02019_05_25_y_validation.npy"

x_train = np.load(x_train_path)
y_train = np.load(y_train_path)

# ### Compile Model

# In[8]:


from keras import optimizers

optimizer = optimizers.Nadam(lr = 0.002, schedule_decay=0.0001)
model.compile(loss='mse',
              optimizer=optimizer,
              metrics=['mse', 'accuracy'])

import csv


# ### Train model and Save Weights

# In[ ]:


import datetime
import pickle
#import pandas as pd
#import csv

weight_directory = "/home/nvidia/Documents/nesvera/neural_nets/lane_following/weights"

time_start_training = datetime.datetime.now().strftime('%y_%m_%d-%H_%M')

epochs_by_save = 100
total_epochs = 100000
train_times = int(total_epochs/epochs_by_save)

batch_size = 32

total_history = 0

hist_loss = []
hist_acc = []
hist_val_loss = []
hist_val_acc = []

for i in range(train_times):
    file_path = weight_directory + "/" + time_start_training + "-" + str((i+1)*epochs_by_save)
    
    
    # Training
    history = model.fit(x_train,
                      y_train, 
                      validation_split = 0.2,
                      epochs = epochs_by_save, 
                      batch_size = batch_size,
                      verbose = 1)
    
    print(history)
    
    # Save
    for key, values in history.history.items():
        
        for value in values:
            if key == 'loss':
                hist_loss.append(value)

            elif key == 'acc':
                hist_acc.append(value)

            elif key == 'val_loss':
                hist_val_loss.append(value)

            elif key == 'val_acc':
                hist_val_acc.append(value)
        
    # Save plot data
    hist_path = file_path + "_hist.csv"
    with open(hist_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(hist_loss)
        writer.writerow(hist_acc)
        writer.writerow(hist_val_loss)
        writer.writerow(hist_val_acc)
        
    # Save weight
    weight_path = file_path + ".h5"
    model.save_weights(weight_path)

