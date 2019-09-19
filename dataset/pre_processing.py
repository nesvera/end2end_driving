import cv2
import numpy as np
import os

# Input label format = 'time_in_milliseconds:speed:steering_front:steering_rear'
min_steering_value = -127
max_steering_value = 127

output_min_steering = -1
output_max_steering = +1

# input and output directories
original_dataset_path = "/home/nvidia/Documents/nesvera/neural_nets/lane_following/dataset/original"

output_dataset_directory = "/home/nvidia/Documents/nesvera/neural_nets/lane_following/dataset/all"
output_dataset_image_name = 0       # dataset will use numers (0,1,2,3....) to reference image and command
# eg. image: 1.jpg  command: 1;velocity;front_steering;rear_steering

# filter properties
input_fps = 30
output_fps = 10
skip_frames = np.ceil(input_fps/output_fps)
skiped_frames = -1

print("Skip : " + str(skip_frames) )
raw_input()

# delay between image and command in milliseconds
# negative delay means the command comes before the image
# positive delay means the command comes after the image
# if the delay is 0, it will get the command with the same or greater timestamp than the image
image_command_delay = 0 

image_extensions = {".jpg", ".JPG"}

def image_processing(image):

    # roi
    new_roi = image[0:180,:,:]
    
    # resize
    new_width = 220
    new_height = 66
    new_image = cv2.resize(new_roi, (new_width, new_height))
    
    # change color
    
    return new_image
    
def map_value(input_value,
              input_min_value, input_max_value,
              output_min_value, output_max_value):
              
    value = (output_min_value + 
             ((float)(output_max_value-output_min_value)/
              (float)(input_max_value-input_min_value))
            *(input_value-input_min_value));
       
    return value
    
    
if __name__ == "__main__":

    output_dataset_label_file = output_dataset_directory + "/label.txt"
    
    output_label_file = open(output_dataset_label_file, "w")

    # open all directories inside the dataset path
    for folder in os.listdir(original_dataset_path):
    
        # complete path of the folder
        folder_path = original_dataset_path + "/" + folder;

        # path to the label file of a folder    
        label_file = folder_path + "/label.txt"
        input_label_file = open(label_file, "r")
        
        input_label_header = input_label_file.readline()

        # get all files of the folder
        all_files = os.listdir(folder_path)
        all_files.sort()
        
        for file in all_files:
        
            # get file extension
            filename, extension = os.path.splitext(file)
            
            # is it a image?
            if extension in image_extensions:
            
                # control fps of dataset
                if (skiped_frames == -1) or (skiped_frames >= skip_frames):
                            
                    # filename is the time in millisenconds when the image was capture
                    image_time_ms = int(filename)
                    command_time_ms = image_time_ms + image_command_delay
                    
                    # read image
                    image_path = folder_path + "/" + file
                    image = cv2.imread(image_path)
                    
                    # process the image
                    ret = image_processing(image.copy())
                    
                    # save image
                    output_image_path = output_dataset_directory + "/" + str(output_dataset_image_name) + ".jpg"
                    cv2.imwrite(output_image_path, ret)
                    
                    #print("image name: " + str(output_dataset_image_name))
                    
                    # read the label.txt of the folder and find a suitable command
                    while True:
                    
                        label_command = input_label_file.readline()
                        label_command_fields =  label_command.split(":")

                        # if read blank lines
                        if label_command_fields[0] == "":
                            break

                        # find a command suitable with the frame time and delay
                        if int(label_command_fields[0]) >= command_time_ms:
                        
                            speed = label_command_fields[1];
                            front_steering = map_value(int(label_command_fields[2]),
                                                       min_steering_value, max_steering_value,
                                                       output_min_steering, output_max_steering)
                            rear_steering = map_value(int(label_command_fields[3]),
                                                       min_steering_value, max_steering_value,
                                                       output_min_steering, output_max_steering)
                            
                            # build command to be saved
                            output_command = str(output_dataset_image_name) + \
                                             ";" + speed + \
                                             ";" + str(front_steering) + \
                                             ";" + str(rear_steering) + "\n";
                                             
                            # save command to the label file
                            output_label_file.write(output_command)
                            
                            break     
                    
                    cv2.imshow("original", image)
                    cv2.imshow("processed", ret)
                    key = cv2.waitKey(1)
                    
                    if key == ord('q'):
                        output_label_file.close()
                        exit(1)
                    
                    output_dataset_image_name += 1
                    skiped_frames = 0
            
                else:
                    skiped_frames += 1
          
         
        # close input label file of the open folder 
        input_label_file.close()            
        #cv2.waitKey(0)
        
        # go to next folder
 
    # close output file
    output_label_file.close()

