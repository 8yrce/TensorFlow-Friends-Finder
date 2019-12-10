# lets detect on one half of the screen and display the detections on the other ( L/R )
"""
people_finder is the testing application for the model we created with the data from people_capture
it is essentially a repurposed people_capture.py using cv2 primarily instead of just pil.

to get this to work we just need to make the image reduction not as great. cant see anything.
also the image we display should probably be a little larger

"""

import cv2
from PIL import Image
import mss
import mss.tools
import tkinter as tk
import numpy as np
from resizeimage import resizeimage


"""
Setting up arg parse
"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', default="ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb",help="Path to the model")
args = parser.parse_args()

"""
#Importing an setting the tensorflow vals for RTX series architecture ( if you dont have an RTX card its fine this wont affect anything )
"""
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

"""
Import the model into a graph
"""
detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	try:
		print("\n\n[INFO] Trying with: {}\n\n".format(args.model))
		with tf.io.gfile.GFile(args.model, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')
		
	except Exception as e:
		print("[ERROR]: {}".format(e))

#Handles any kind of detection event
"""
PARAMS: boxes - bb cords of the detect, classes - class of the detect, scores - score of detection
RETURNS: bool - true if photo contains our class, false if not
"""
def detection_handler(classes, scores, THRESHOLD):
	if classes > 1 and classes < 8 and scores > THRESHOLD: # aka people
		return True
	else:
		return False

#Decodes the class numbers to strings for us to read
"""
PARAMS: classes[0][i] = the current class number
RETURNS: class - the corrosponding string
"""
def class_decoder(current_class):

	if current_class == 1:
		character = "Person"
	elif current_class == 2:
		character = "Ross"
	elif current_class == 3:
		character = "Monica"
	elif current_class == 4:
		character = "Joey"
	elif current_class == 5:
		character = "Chandler"
	elif current_class == 6:
		character = "Pheobe"
	elif current_class == 7:
		character = "Rachel"
	else: 
		"Invalid choice, ignoring"
		character = "ERROR out of bounds"
	return character

#grabs screen size and returns us the workable area
"""
RETURNS: Our monitor object to grab the screen with
"""
def gather_screen_info():
	#gather screen info
	screen = tk.Tk()
	screen_width = screen.winfo_screenwidth()
	screen_height = screen.winfo_screenheight()
	print("Screen width:height", screen_width, ":", screen_height, "\n\n")

	#Make the full screen is our viewing zone
	TOP = 0
	LEFT = 0
	WIDTH = int(screen_width) 
	HEIGHT = int(screen_height)
	monitor = {"top": TOP, "left": LEFT, "width": WIDTH, "height": HEIGHT}
	return monitor

#handles the operations necessary for our monitor to both give us an image / use our model to see if it sees anything
"""
PARAMS: monitor - our monitor confines, sees - the current tensorflow inference session
"""
def image_operations(monitor, sess, detection_graph, WIDTH, HEIGHT):
	#Grab image from the monitor
	with mss.mss() as sct:
		sct_img = sct.grab(monitor)
		sct.close()
	
	image = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

	
	cover = resizeimage.resize_thumbnail(image, [WIDTH,HEIGHT])
	padded_img = Image.new('RGB', (WIDTH,HEIGHT), (255,255,255))
	padded_img.paste(cover, cover.getbbox())
	padded_img.save ("image_to_check.png", image.format, quality=100)
	image.close()
	padded_img.close()
	cover.close()

	image = cv2.imread("image_to_check.png")#cv2.imread("image_to_check.png")

	# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
	image_expanded = np.expand_dims(image, axis=0)
	
	image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
	# Each box represents a part of the image where a particular object was detected.
	boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
	# Each score represent how level of confidence for each of the objects.
	scores = detection_graph.get_tensor_by_name('detection_scores:0')
	classes = detection_graph.get_tensor_by_name('detection_classes:0')
	num_detections = detection_graph.get_tensor_by_name('num_detections:0')

	(boxes, scores, classes, num_detections) = sess.run(
					[boxes, scores, classes, num_detections],
					feed_dict={image_tensor: image_expanded})

	return boxes, scores, classes, num_detections

#our wonderful little main loop
def main():
	#make sure these match your model
	WIDTH = 300 * 2
	HEIGHT = 150 * 2
	THRESHOLD = 0.7

	input("Press enter when you are ready to start inferencing, please make sure to have the media maximized to take up as much of the screen without using a 'full-screen' option")

	with detection_graph.as_default():
		with tf.Session(graph=detection_graph) as sess:
			monitor = gather_screen_info()
			while True:
				# Actual detection
				try:
					boxes, scores, classes, num_detections = image_operations(monitor, sess, detection_graph, WIDTH, HEIGHT)
					#Feeding into the detection logic handler
					#also we pass the top detection for this ( box[0][0] and classes[0][0] ) to make sure we have at least one hit
					image = cv2.imread("image_to_check.png")

					if (detection_handler(classes[0][0], scores[0][0], THRESHOLD)):
						#print("\n\n\nBB: ", boxes[0][0], "\nClass: ", classes[0][0], "\nScore: ",scores[0][0], "\n\n\n")
						
						for i in range(1): # we really dont want to label a whole crowd, top 5 is more than enough
							if classes[0][i] > 1 and classes[0][i] < 8 and scores[0][i] >= THRESHOLD:
								box = boxes[0][i]
								
								#Drawing bounding box
								cv2.rectangle(image, ( int(box[1]*WIDTH), int(box[0]*HEIGHT) ), ( int(box[3]*WIDTH), int(box[2]*HEIGHT) ),(255,255,255), 1) # cv2 takes in image, left/top, right/bottom, color, line thickness

								cv2.putText( image, "{}".format(class_decoder(classes[0][i])), ( int(box[0]*WIDTH), int(box[1]*HEIGHT) ), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255) )

					window = cv2.namedWindow("Friends-Finder", cv2.WINDOW_NORMAL)
					cv2.resizeWindow("Friends-Finder", 1200,700)
					cv2.imshow("Friends-Finder", image)
					cv2.waitKey(1)

				except Exception as e:
					print(e)
					exit()

if __name__ == "__main__":
	main()