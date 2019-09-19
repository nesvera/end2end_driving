import numpy as np
import cv2

x_train_path = "/home/nvidia/Documents/nesvera/neural_nets/lane_following/dataset/2019_05_25_x_train.npy"
y_train_path = "/home/nvidia/Documents/nesvera/neural_nets/lane_following/dataset/2019_05_25_y_train.npy"

x_validation_path = "/home/nvidia/Documents/nesvera/neural_nets/lane_following/dataset/02019_05_25_x_validation.npy"
y_validation_path = "/home/nvidia/Documents/nesvera/neural_nets/lane_following/dataset/02019_05_25_y_validation.npy"


if __name__ == "__main__":

    x_train = np.load(x_train_path)
    y_train = np.load(y_train_path)
    
    for i in range(x_train.shape[0]):
    
        image = x_train[i]
        command = y_train[i]
        
        print("Steering: " + str(command))
        
        cv2.imshow("image", image)
        key = cv2.waitKey(0)
        
        if key == ord('q'):
            exit(0)
