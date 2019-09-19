#!/usr/bin/env python
# coding: utf-8

import keras
from keras.models import model_from_json, load_model

import numpy as np
import cv2

# load model
json_file = open('autopilot_model.json')

model_json = json_file.read()
json_file.close()

model = model_from_json(model_json)
model.load_weights('/home/nvidia/Documents/nesvera/neural_nets/lane_following/train_results/2019_05_31-t1/weights/19_06_03-13_21-400.h5')

model.summary()

x_test_path = "/home/nvidia/Documents/nesvera/neural_nets/lane_following/dataset/2019_05_25_x_validation.npy"
y_test_path = "/home/nvidia/Documents/nesvera/neural_nets/lane_following/dataset/2019_05_25_y_validation.npy" 

x_test = np.load(x_test_path)
y_test = np.load(y_test_path)

print("dataset shape")
print(x_test.shape)
print(y_test.shape)
input()

# Read test file and predict steering
for i in range(len(x_test_path)):

    image = x_test[i]
    
    steering = model.predict(image.reshape(1, image.shape[0], image.shape[1], image.shape[2]))
    
    print("Steering - Ground truth: " + str(y_test[i]) + " Predicted: " + str(steering))
    
    cv2.imshow("input", image)
    key = cv2.waitKey(0)
    
    if key == ord('q'):
        break
