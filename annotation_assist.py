"""
 run me in the screen_captures folder from people_capture

 annotation assist uses the premade bounding boxes taken from our generic network
 and allows us to reassign the label if we see the object is what we want to identify specifically
"""


import fileinput
import glob
import os
import cv2

os.chdir(".")

def check_annotation():
	print("e")

def main():
	#for each xml file
	for xml_file in glob.glob("*.xml"):
		#find and open image associated with it
		image_file = xml_file.replace(".xml", ".png")
		image = cv2.imread(image_file)

		#open xml
		xml = open(xml_file, 'r')

		#go through xml and grab all the bb cordinates
		bb_cords = []
		cur_bb = []
		for line in xml:
			if "<xmin>" in line:
				xmin = line.replace("<xmin>","")
				xmin = xmin.replace("</xmin>","")
				xmin = xmin.join(xmin.split())
				cur_bb.append([xmin])
			elif "<ymin>" in line:
				ymin = line.replace("<ymin>","")
				ymin = ymin.replace("</ymin>","")
				ymin = ymin.join(ymin.split())
				cur_bb.append([ymin])
			elif "<xmax>" in line:
				xmax = line.replace("<xmax>","")
				xmax = xmax.replace("</xmax>","")
				xmax = xmax.join(xmax.split())
				cur_bb.append([xmax])
			elif "<ymax>" in line:
				ymax = line.replace("<ymax>","")
				ymax = ymax.replace("</ymax>","")
				ymax = ymax.join(ymax.split())
				cur_bb.append([ymax])
				bb_cords.append([cur_bb])
				cur_bb = []

		print(bb_cords[0])
		print(bb_cords[0][0])

		#new_xml = []
		#for line in xml:
			#if "person" in line:
				#display and approve of the bounding box before changing to label

			#new_xml.append(line)

if __name__ == "__main__":
	main()