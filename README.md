# Friends-Finder

![](media/vid1.gif)

## Always wanted your computer to recognize someone specific? Problem solved!
Using a custom TensorFlow model for computer vision object detection, this application detects specific characters from the show Friends. This effectively simulates using this software on say a security camera to track certain individuals passing through an area. Should you want to detect other people this repo also features scripts to quickly gather and annotate your own data.

![](media/vid2.gif)

## Why make this project?
The main motivation behind this project was twofold. To create an application to assist with the time consuming process of annotating images for detection datasets, and to showcase some of my work in computer vision outside of my office in a more casual setting.

## Requirements:
- TensorFlow=1.14 
- OpenCv=Latest 
- PIL 
- resizeimage 
- numpy 
- mss 
- glob
- tkinter


## How to use:

#Using with provided Friends model:

- First, open up an episode of Friends. As of now the model is only trained on S6:E6 ( Ross hugs Rachel ) as I wanted to provide a working example as soon as possible. Check the build status for the latest included episodes. Should still work on episodes not trained on, however the accuracy may be slightly lower.

```
python3 friends_finder.py --model model/friends_inference.pb
```

For best experience please be sure to:
- Make the viewing media as large as you can without using a "Full-screen" option.
- Ensure to view box ( as of current version, will be fixed in later build ) is not blocking any characters

## If you want to capture your own data:

- First download/extract a model to use to pull people out of frames of the show, I recommend the one below.

```
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar -xzvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
```

- Pull up an episode of Friends ( or something else ) and ensure its window is maximized ( not full screen )

- Run the data collection program

```
python3 people-capture.py || python3 people-capture.py --model OptionalPathToYourOwnModel
```

This will take a screen capture anytime your model detects anything on screen. Keep in mind if you use your own model you will have to edit the index at which it looks for as default it is '1' so it only grabs people with coco.

## Build status:
Current Status:
- [X] Finish data collection program
- [X] Create XML generator for collected data
- [X] Complete assisted auto annotation program
- [X] Specify network, train model on collected data
- [X] Make testing program for new model 
- [ ] Optimize program for use with other lower resolution displays
- [ ] Update model for usage on more episodes


