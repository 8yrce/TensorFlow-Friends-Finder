Friends-Finder

Be sure you have TensorFlow, OpenCV4, and numpy installed.

To capture your own data:

* wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

* tar -xzvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz

* $python3 people-capture.py --model OptionalPathToYourOwnModel
	  
This will capture a screen capture anytime your model detects anything on screen. Keep in mind if you use your own model you will have to edit the index at which it looks for as default it is '1' so it only grabs people with coco.



To test out on some data ( like an episode of friends ): - coming shorty

*$
