import numpy as np
import cv2
import argparse
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        dest='model_path',
        required=True,
        help="Path to the trained model"
    )
    
    args = parser.parse_args()

    model_path = args.model_path

    if os.path.exists(model_path) == False:
        print("Error: Model not found!")
        exit(1)

    model = cv2.dnn.readNetFromONNX(model_path)