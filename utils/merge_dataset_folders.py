import argparse
import os
from datetime import datetime
import cv2
from shutil import copyfile

'''

    input label format label    timestamp:speed:steering_front:steering_rear'
    output label format label   image_name:speed:steering_front:steering_rear'

'''

period_btw_frames = 50         # period between frames in milliseconds
dif_frame_command = -100         # get the command delta milliseconds before the image
error_dif_frame_command = 50     # 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge several records in one dataset')
    parser.add_argument('-i', '--input_dataset', required=True,
                        help='Path to the driectory that contains recorded runs')
    parser.add_argument('-o', '--output_dataset', required=False,
                        help='Path to the directory that will receive the dataset')
    
    args = parser.parse_args()

    input_path = args.input_dataset
    if os.path.isdir(input_path) == False:
        print('Input path is not a directory')
        exit(1)

    output_path = args.output_dataset
    if os.path.isdir(output_path) == False:
        print('Output path is not a directory')
        exit(1)

    # dataset name will be the time
    current_time = datetime.now().strftime('%Y_%m_%d_%H_%H_%M_%S')

    # output label file
    output_path = output_path + "/" + current_time
    
    if os.path.exists(output_path) == False:
        os.mkdir(output_path)

    output_label_path = output_path + "/label.txt"
    output_label_file = open(output_label_path, 'w')

    #
    dataset_index = 0

    # loop through the folders inside the input directory
    for record in os.listdir(input_path):

        # read all files inside a record file (images, label.txt)
        record_path = input_path + "/" + record
        
        if os.path.isdir(record_path) == False:
            continue

        record_files = os.listdir(record_path)

        #delete label.txt
        record_files.remove('label.txt')
        input_label_path = record_path + "/label.txt"
        input_label_file = open(input_label_path, 'r')

        # ascending sort the images
        record_files.sort(key=lambda x: int(x.split('.')[0]))

        last_command_stamp = None

        # read record label
        while True:
            
            label_line = input_label_file.readline()

            # end of the file
            if label_line == '':
                break
            
            else:
                # remove '\n'
                label_line = label_line[:-1]
        
                # break command
                label_line = label_line.split(":")
        
                # skip head
                if label_line[0].isdigit() == False:
                    continue

                # if is the first command from the record
                if last_command_stamp is None:
                    last_command_stamp = int(label_line[0])

                # skip some frames
                if (last_command_stamp + period_btw_frames) <= int(label_line[0]):
                    last_command_stamp = int(label_line[0])
                
                else:
                    continue

                # find image to the command
                min_image_timestamp = int(label_line[0]) + dif_frame_command - error_dif_frame_command
                max_image_timestamp = int(label_line[0]) + dif_frame_command + error_dif_frame_command

                image_index = None

                for i, image_record in enumerate(record_files):

                    image_record_stamp = int(image_record.split('.')[0])

                    if (image_record_stamp > min_image_timestamp) and \
                       (image_record_stamp < max_image_timestamp):

                       image_index = i
                       break

                if image_index is not None:
                    # original image path
                    image_path = record_path + "/" + record_files[image_index]

                    # new image path
                    output_image_path = output_path + "/" + str(dataset_index)
                    output_image_path += "." + record_files[image_index].split('.')[-1]

                    # new label
                    label_data = str(dataset_index) + "." + image_path.split('.')[-1]
                    for i in range(1, len(label_line)):
                        label_data += ":" + label_line[i]

                    label_data += '\n'
                

                    # copy image to the dataset folder
                    copyfile(image_path, output_image_path)

                    # append command to the label file
                    output_label_file.write(label_data)

                    dataset_index += 1

                    im = cv2.imread(image_path)
                    print(label_line)
                    cv2.imshow("image", im)
                    cv2.waitKey(1)

        # close record file
        input_label_file.close()

    #close dataset file
    output_label_file.close()
