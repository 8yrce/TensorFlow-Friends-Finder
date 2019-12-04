# Friends-Finder
## Always wanted your computer to recognize someone specific? Problem solved!
Using a custom TensorFlow model for computer vision object detection, this application detects specific characters from the show Friends. This effectively simulates using this software on say a security camera to track certain individuals passing through an area. Should you want to detect other people this repo also features scripts to quickly gather and annotate your own data.

## Why make this project?
The main motivation behind this project was twofold. To create an application to assist with the time consuming process of annotating images for detection datasets, and to showcase some of my work in computer vision outside of my office in a more casual setting.

## Requirements:
- TensorFlow 
- PIL
- numpy


## How to use: - coming shorty
For using it with the (soon to be) included model
```
python3 people_finder.py --model modelname
```
Current Status:
- [x] Finish data collection program
- [x] Create XML generator for collected data
- [X] Complete assisted auto annotation program
- [ ] Specify network, train model on collected data
- [x] Make testing program for new model

## If you want to capture your own data:

First download/extract a model to use to pull people out of frames of the show

```
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar -xzvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
```

Run the data collection program
```
python3 people-capture.py --model OptionalPathToYourOwnModel
```
	  
This will take a screen capture anytime your model detects anything on screen. Keep in mind if you use your own model you will have to edit the index at which it looks for as default it is '1' so it only grabs people with coco.


