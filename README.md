Requirements:

    sudo pip3 install torchsummary  #Print a summary (similar keras) of the model

    sudo pip3 install onnx onnxruntime

How to use:
1. Place recorded runs inside "records" folder as following

    ├── record_1\
    |   ├── image_1.jpg\
    |   ├── image_2.jpg\
    |   ├── image_3.jpg\
    |   ├── image_4.jpg\
    |   ├── image_5.jpg\
    |   ├── label.txt\
    ├── record_2\
    |   ├── image_1.jpg\
    |   ├── image_2.jpg\
    |   ├── image_3.jpg\
    |   ├── image_4.jpg\
    |   ├── image_5.jpg\
    |   ├── label.txt\

2. Merge the recorded runs into a single one

    python3 utils/merge_dataset_folders.py -i records/ -o dataset/

3. Prepare (crop, resize, and split) the current dataset

    e.g.,\
    record image shape: (240, 376, 3)\
    crop:\
        p0 -> (0, 0)\
        p1 -> (190, 376)\
    cropped image shape: (190, 376, 3)\
    resized shape: (66, 200, 3)\
    split: 80% to train, 20% to test\

    python3 utils/prepare_dataset.py -i dataset/2019_09_08_21_21_07_45/ -x0 0 -y0 0 -x1 376 -y1 190 -rh 66 -rw 200 -d 0 -s 20

4. Train

    python3 train.py -i dataset/2019_09_08_21_21_07_45/ -o model/ -m model/model_44.pth

5. Convert trained weight to deploy

    python3 convert_weight_to_deploy.py -m model/model_44.pth

6. Test

    python3 test.py -i dataset/2019_09_08_21_21_07_45/ -m model/model_44.pthd
