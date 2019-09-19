import matplotlib
import matplotlib.pyplot as plt
import pickle

hist_file = "/home/nvidia/Documents/nesvera/neural_nets/lane_following/weights/19_06_03-13_21-400_hist.csv"

if __name__ == "__main__":

    file = open(hist_file, 'rb')
    history = pickle.load(file)
    file.close()

    plt.figure()
    #plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    #plt.legend(['train', 'validation'], loc='upper left')

    #lt.plot(history.history['acc'])
    #lt.plot(history.history['val_acc'])

    #lt.plot(history.history['mean_squared_error'])
    #lt.plot(history.history['val_mean_squared_error'])


    plt.show()
