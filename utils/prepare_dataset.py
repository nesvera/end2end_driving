import argparse
import os
import cv2
import random

img_extension = ['png', 'jpg']

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Preprocessing dataset (change image size, crop)')
    parser.add_argument(
        '-x0', 
        dest='x0', 
        type=int,
        required=False,
        help='Crop top-left x point of the image')
    parser.add_argument(
        '-y0',
        dest='y0', 
        type=int,
        required=False,
        help='Crop top-left y point of the image')
    parser.add_argument(
        '-x1', 
        dest='x1', 
        type=int,
        required=False,
        help='Crop bottom-right x point of the image')
    parser.add_argument(
        '-y1', 
        dest='y1', 
        type=int,
        required=False,
        help='Crop bottom-right y point of the image')
    parser.add_argument(
        '-rw', 
        dest='w', 
        type=int,
        required=False,
        help='Output image size -> width')
    parser.add_argument(
        '-rh', 
        dest='h', 
        type=int,
        required=False,
        help='Output image size -> height')
    parser.add_argument(
        '-i', 
        dest='dataset_path', 
        required=False,
        help='Path to the dataset')
    parser.add_argument(
        '-d', '--debug',
        dest='debug',
        type=int,
        required=False,
        help='(True=1, False=0) If true, it will only display the new images, without save it'
    )

    args = parser.parse_args()

    crop_p0 = (int(args.x0), int(args.y0))
    crop_p1 = (int(args.x1), int(args.y1))
    reshape_size = (int(args.w), int(args.h))
    dataset_input_path = args.dataset_path

    if (crop_p0[0] < 0) or (crop_p0[1] < 0) or \
       (crop_p1[0] < 0) or (crop_p1[1] < 0):
        print('Error: (x0,y0,x1,y0) must be positive')
        exit(1)

    if (crop_p0[0] >= crop_p1[0]) or (crop_p0[1] >= crop_p1[1]):
        print('Error: (x1,y1) must be greater than (x0,y0)')
        exit(1)

    if (reshape_size[0] <= 0) or (reshape_size[1] <= 0):
        print('Error: new shape (rw,rh) must be greater than 0')
        exit(1)

    if os.path.isdir(dataset_input_path) == False:
        print('Error: input path is not a directory')
        exit(1)

    if args.debug == False:
        while True:
            print("This process will overide the images!")
            print("Use debug mode (-d 1) to make sure that the format is correct!")
            print("Do you want to change the images? [y]es or [n]o")

            r = input()
            if r == 'y':
                break
            elif r == 'n':
                exit(0)
    else:
        print("Press 'q' to exit!")

    # list files inside the directory
    folder_files = os.listdir(dataset_input_path)

    if os.path.exists(output_train_folder) == False:
        os.mkdir(output_train_folder)

    if os.path.exists(output_test_folder) == False:
        os.mkdir(output_test_folder)

    # loop through the files
    for i, filename in enumerate(folder_files):

        if filename.split('.')[-1] in img_extension:

            image_path = dataset_input_path + "/" + filename
            image = cv2.imread(image_path)

            cropped = image[crop_p0[1]:crop_p1[1], crop_p0[0]:crop_p1[0]]
            resized = cv2.resize(cropped, reshape_size)

            if args.debug == True:
                print("Original image shape: " + str(image.shape))
                print("Cropped shape: " + str(cropped.shape))
                print("Resized shape:" + str(resized.shape))
                print()

                cv2.imshow('image', image)
                cv2.imshow('cropped', cropped)
                cv2.imshow('resized', resized)
                
                if cv2.waitKey(0) == ord('q'):
                    exit(0)

            else:
                print("Image " + str(i) + " of " + str(len(folder_files)))                
                cv2.imwrite(image_path, resized)

                cv2.imshow('image', image)
                cv2.imshow('resized', resized)
                cv2.waitKey(1)
                
                