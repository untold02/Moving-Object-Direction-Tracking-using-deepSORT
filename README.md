# Moving-Object-Direction-Tracking-using-deepSORT
CPU による"ウロウロ行動"の検知(Loitering Behavior Detection using CPU)

Here I am working on detecting moving objects direction using YOLOv4, deepSORT and Tensorflow.

# Raw Demo Video↓

![ezgif com-video-to-gif (1)](https://user-images.githubusercontent.com/84290745/227867156-aad39051-0f4c-4737-8301-ee1945127d92.gif)


# Getting Started
To get started, install the dependencies either via Anaconda or Pip. 

## download python

https://www.python.org/downloads/

## install tensorflow

https://www.tensorflow.org/install/pip?hl=ja

## conda
**#Tensorflow CPU**

`conda env create -f conda-cpu.yml`

`conda activate yolov4-cpu`

## pip

(TensorFlow 2 packages require a pip version >19.0.)

**#TensorFlow CPU**

`pip install -r requirements.txt`

## Downloading YOLOv4 Pre-trained Weights

I am using YOLOv4 to make the object detections and deepsort to track. There exists an pre-trained YOLOv4 object detector model that is able to detect 80 classes. For easy demo purposes I am using the pre-trained weights for the tracker. Download pre-trained yolov4.weights file: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

move yolov4.weights from your downloads folder into the 'data' folder of this repository(folder).

you can also use yolov4-tiny.weights, a smaller model that is faster at running detections but less accurate, for download: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights

## Running the object_tracker with YOLOv4

To implement the object tracking using YOLOv4, first we convert the .weights into the corresponding TensorFlow model which will be saved to a checkpoints folder. Then all we need to do is run the **object_tracker.py** script to run our object tracker with YOLOv4, DeepSort and TensorFlow.

**Convert darknet weights to tensorflow model**

`python save_model.py --model yolov4 `

**Run yolov4 deepSORT object tracker on video**

`python object_tracker.py --video ./data/video/test.mp4 --output ./outputs/demo.avi --model yolov4`

 **Run yolov4 deepSORT object tracker on webcam (set video flag to 0)**

`python object_tracker.py --video 0 --output ./outputs/webcam.avi --model yolov4`

## Running the Tracker with YOLOv4-Tiny
The following commands will allow you to run yolov4-tiny model. Yolov4-tiny allows you to obtain a higher speed (FPS) for the tracker at a slight cost to accuracy. Make sure that you have downloaded the tiny weights file and added it to the 'data' folder in order for commands to work!

**save yolov4-tiny model**

`python save_model.py --weights ./data/yolov4-tiny.weights --output ./checkpoints/yolov4-tiny-416 --model yolov4 --tiny
`
**Run yolov4-tiny object tracker**

`python object_tracker.py --weights ./checkpoints/yolov4-tiny-416 --model yolov4 --video ./data/video/test.mp4 --output ./outputs/tiny.avi --tiny
`

# Result demo Video


![ezgif com-video-to-gif](https://user-images.githubusercontent.com/84290745/227864521-20f90e73-febe-4abb-a7a6-925b206fd0c4.gif)

## References
1) https://github.com/hunglc007/tensorflow-yolov4-tflite

2) https://github.com/nwojke/deep_sort
