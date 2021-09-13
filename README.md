# Experiments with convolutional neural networks for emotion recognition
## Introduction
This work based on [Kaggle competition](https://www.kaggle.com/c/skillbox-computer-vision-project/overview).
I've tried to use various strategy in data preparation and model training for reaching a maximum score on a private dataset.
## Versions of used libraries
Python 3.8.5
- tensorflow 2.4.1
- numpy 1.19.2
- tensorflow-hub 0.12.0
## Preparation
Before starting do these steps:
- Download dataset files from [Kaggle competition](https://www.kaggle.com/c/skillbox-computer-vision-project/data) and unzip to ./Data folder. 
  After that you will get a folder structure like this:
  ```
   ./dataset
     /train
       /anger
       /contempt
       /disgust
       etc.
     /test_kaggle
       <unstructured images>
  ```
  Also you can do this inside EDA.ipynb
- Download openCV model files for face detector:
  - [model](https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel) and rename as `opencv_face_detector.caffemodel`
  - [config](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/opencv_face_detector.pbtxt) file and place both files to `./Data` folder.
## Data analysis
Data analysis was placed in [EDA.ipynb](https://github.com/lugrenl/Emotion-Recognition_model/blob/main/EDA.ipynb). In these part I explore the data. Get structure and data statistics. I've cleaned data from outliers and prepeared to datasets: 
- a full-image dataset 
- a dataset with cropped faces from source image.
  
    


