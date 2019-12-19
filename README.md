# Friends-Finder

![](media/vid1.gif)

## Always wanted your computer to recognize someone specific? Problem solved!
Using a custom TensorFlow model for computer vision object detection, this application detects specific characters from the show Friends. This effectively simulates using this software on say a security camera to track certain individuals passing through an area. Should you want to detect other people this repo also features scripts to quickly gather and annotate your own data.

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

- First, open up an episode of Friends. As of now the model is only trained on S6:E2 ( Ross hugs Rachel ) as I wanted to provide a working example as soon as possible. Check the build status for the latest included episodes. Should still work on episodes not trained on, however the accuracy may be slightly lower.

```
python3 friends_finder.py
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

- When finished, run the annotation program ( if using Friends )

'''
python3 annotation_assist.py
'''

- Train data with method of your choosing! Depending on what you choose you may need .jpg not .png.
'''
mogrify -format png /path/*.jpg
'''

## What each py file does:
- annotation_assist.py: Does just that, assists you with annotating the bounding boxes provided by people_capture.py

- friends_finder.py: Finds characters from the show friends using the included friends_inference.pb model

- people_capture.py: Captures our training data of people from the show us to annotate with annotation_assist.py

- people_finder.py: Fun testing application for testing a model against an episode of Friends. Similair to friends_finder.py, but more generic so you can more easily use it with different models/media.

- people_xml.py: Creates the xml annotations for the data gathered by people_capture.py to be used by annotation_assist.py during the annotation process.


## Build status:
Current Status:
- [X] Finish data collection program
- [X] Create XML generator for collected data
- [X] Complete assisted auto annotation program
- [X] Specify network, train model on collected data
- [X] Make testing program for new model 
- [ ] Optimize program for use with other lower resolution displays / multi-monitor setups
- [ ] Update model for usage on more episodes


