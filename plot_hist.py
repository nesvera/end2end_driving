import csv
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

if __name__ == "__main__":

    csv_file = "/home/nvidia/Documents/nesvera/neural_nets/lane_following/weights/19_06_03-13_21-400_hist.csv"
    
    my_data = np.genfromtxt(csv_file, delimiter=',')
    
    hist_loss = my_data[0]
    hist_acc = my_data[1]
    hist_val_loss = my_data[2]
    #hist_val_acc = my_data[3]
    
    #print(hist_loss.shape)
    
    plt.figure()
    #plt.plot(hist_loss)
    plt.plot(hist_val_loss)
    
    #plt.plot(hist_acc)
    
    plt.show()
    
