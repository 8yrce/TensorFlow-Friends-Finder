"""
 People capture is the application that takes our generic model and uses it to help us find / annotate data

the optional model flag may be used if you would like to use a model other than the one I have for development
"""

from PIL import Image
import cv2
import mss
import mss.tools
import os
import tkinter as tk
import numpy as np
from resizeimage import resizeimage
import people_xml
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
	if classes == 1 and scores > THRESHOLD: # aka people
		return True
	else:
		return False

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
def image_operations_capture(monitor, sess, detection_graph, WIDTH, HEIGHT):
	#Grab image from the monitor
    with mss.mss() as sct:
        sct_img = sct.grab(monitor)
        sct.close()
    
    #Convert sct image to bytes
    image = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
    
    #modifying image to correct size, that way screen size doesnt effect our detection
    cover = resizeimage.resize_cover(image, [WIDTH,HEIGHT])
    padded_img = Image.new('RGB', (WIDTH,HEIGHT), (255,255,255))
    padded_img.paste(cover, cover.getbbox())
    #saving and closing our image operations
    padded_img.save ("image_to_check.png", image.format, quality=100)
    image.close()
    padded_img.close()
    cover.close()
    # reading back in image to use in detection, imread formatting seems to be the most reliable way to format for detection
    image = cv2.imread("image_to_check.png")

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_expanded = np.expand_dims(image, axis=0)
    
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    # getting our classes detected
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    # gather number of detections made
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    # run detection
    (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_expanded})

    return boxes, scores, classes, num_detections, image


#our wonderful little main loop
def main():
	#make sure this size is what you want your model to train on
	WIDTH = 300
	HEIGHT = 300
	THRESHOLD = 0.9


	input("Press enter when you are ready to start capturing, application will capture full screen")
	try:
		os.mkdir("screen_captures")
	except Exception as e:
		print("Directory already exits, overwriting")

	with detection_graph.as_default():
		with tf.Session(graph=detection_graph) as sess:
			gather_images = True
			file_counter = 0
			monitor = gather_screen_info()
			while gather_images:
				# Actual detection
				try:
					boxes, scores, classes, num_detections, _ = image_operations_capture(monitor, sess, detection_graph, WIDTH, HEIGHT)
					#Feeding into the detection logic handler
					#also we pass the top detection for this ( box[0][0] and classes[0][0] ) to make sure we have at least one hit
					if (detection_handler(classes[0][0], scores[0][0], THRESHOLD)):
						pic_name = "picture-{}".format(file_counter)
						os.rename("image_to_check.png", pic_name)

						image = Image.open(pic_name)
						padded_img = Image.new('RGB', (WIDTH,HEIGHT), (255,255,255))
						padded_img.paste(image, image.getbbox())
						padded_img.save ("{}/screen_captures/{}.png".format(os.getcwd(),pic_name), image.format, quality=100)
						image.close()
						padded_img.close()
						os.remove(pic_name)
						file_counter += 1
						print("Images captured: {}".format(file_counter), end="\r")
						
						# so now if the most prominent thing in the picture is what we are looking for
						#	we should probably record that info so we dont have to annotate it later
						annotations = []
						for i in range(5): # we really dont want to label a whole crowd, top 5 is more than enough
							if classes[0][i] == 1 and scores[0][i] >= THRESHOLD:
								box = boxes[0][i]
								
								annotations.append( [(box[0]),box[1],(box[2]),box[3], classes[0][i]] )
						people_xml.generate_xml(annotations, pic_name, WIDTH, HEIGHT)

				except Exception as e:
					print(e)

if __name__ == "__main__":
	main()