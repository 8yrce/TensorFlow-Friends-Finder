# Friends-Finder
## Don't know who is who in Friends? Problem solved!

Requirements:
- TensorFlow 
- PIL
- numpy

## To capture your own data:

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



## To test out on some data ( like an episode of friends ): - coming shorty
```
$
```

