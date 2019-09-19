import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# input and output directories
input_dataset_path = "/home/nvidia/Documents/nesvera/neural_nets/lane_following/dataset/all"
output_dataset_path = "/home/nvidia/Documents/nesvera/neural_nets/lane_following/dataset"

output_dataset_name = "2019_05_25"

test_size = 0.2       # x percent of the dataset goes to validation

# input image size
img_w = 220
img_h = 66
img_ch = 3

target_output_size = 1      # just front_steering

image_extensions = {".jpg", ".JPG"}

if __name__ == "__main__":
    
    # path to the label file of a folder    
    label_file = input_dataset_path + "/label.txt"
    input_label_file = open(label_file, "r")
        
    # get all files of the folder
    dataset_list = os.listdir(input_dataset_path)
    dataset_list.remove("label.txt")
    
    # sort images by name eg. (0.jpg, 1.jpg)
    dataset_list = sorted(dataset_list, key=lambda a: int(a.split(".")[0]))
         
    # create numpy array
    dataset_image_input = np.zeros((len(dataset_list), img_h, img_w, img_ch), dtype="uint8")
                                            
    dataset_label_output = np.ndarray(shape=(len(dataset_list), 
                                            target_output_size))
                                            
    # read line from label file
    command_line = input_label_file.readline()
    command = command_line.split(";")
           
    # build array for input and output
    for i, file in enumerate(dataset_list):
        
        # get file extension
        filename, extension = os.path.splitext(file)

        #print(int(filename), int(command[0]))      
        if int(filename) != int(command[0]):
            print("Error! Mismatch between images and labels")
            input()
            continue
       
        else:
            # read line from label file
            command_line = input_label_file.readline()
            command = command_line.split(";") 
        

        
        # read image
        image_path = input_dataset_path + "/" + file
        image = cv2.imread(image_path)
        
        # convert to numpy array
        #image_np = np.reshape(image, (img_h, img_w, img_ch))
        
        if command_line != "":
            dataset_image_input[i] = image
            dataset_label_output[i] = command[2]
           
        else:
            pass
        
        print(str(i) + " of " + str(len(dataset_list)))
            
    print("Spliting data ...")
    
    # split in training and validation set  
    x_train, x_validation, y_train, y_validation = train_test_split(dataset_image_input, dataset_label_output, 
                                                                    test_size=test_size, shuffle=True)
                                                        
    # output files
    x_train_path = output_dataset_path + "/" + output_dataset_name + "_x_train"
    y_train_path = output_dataset_path + "/" + output_dataset_name + "_y_train"
    x_validation_path = output_dataset_path + "/" + output_dataset_name + "_x_validation"
    y_validation_path = output_dataset_path + "/" + output_dataset_name + "_y_validation"
    
    print("Saving ...")
    
    # save files
    np.save(x_train_path, x_train)
    np.save(y_train_path, y_train)
    np.save(x_validation_path, x_validation)
    np.save(y_validation_path, y_validation)

          
                    
    
