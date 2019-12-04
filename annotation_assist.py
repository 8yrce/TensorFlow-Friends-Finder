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

def check_annotation(bb_cords, line, image):
	cv2.rectangle(image, ( bb_cords[0], bb_cords[1] ), (bb_cords[2], bb_cords[3]),(255,255,255), 2) # cv2 takes in image, left/top, right/bottom, color, line thickness
	cv2.imshow("Annotation assist", image)
	cv2.waitKey(1)
	r = "Ross"
	m = "Monica"
	j = "Joey"
	c = "Chandler"
	p = "Pheobe"
	a = "Rachel"

	print("1:	Ross\n2:	Monica\n3:	Joey:\n4:	Chandler\n5:	Pheobe\n6:	Rachel\n0:	None")
	choice = int(input("What character is this?		"))
	
	if choice == 0:
		return line
	elif choice == 1:
		character = r
	elif choice == 2:
		character = m
	elif choice == 3:
		character = j
	elif choice == 4:
		character = c
	elif choice == 5:
		character = p
	elif choice == 6:
		character = a
	else: 
		"Invalid choice, ignoring"
		return line

	line = line.replace("person", character)
	return line


def main():
	#for each xml file
	for xml_file in glob.glob("*.xml"):
		#find and open image associated with it
		image_file = xml_file.replace(".xml", ".png")
		image = cv2.imread(image_file)
		#cv2.imshow("test", image)

		#open xml
		xml = open(xml_file, 'r')

		#go through xml and grab all the bb cordinates
		bb_cords = []
		cur_bb = []
		old_xml = []
		for line in xml:
			if "<xmin>" in line:
				xmin = line.replace("<xmin>","")
				xmin = xmin.replace("</xmin>","")
				xmin = xmin.join(xmin.split())
				cur_bb.append(int(xmin))
			elif "<ymin>" in line:
				ymin = line.replace("<ymin>","")
				ymin = ymin.replace("</ymin>","")
				ymin = ymin.join(ymin.split())
				cur_bb.append(int(ymin))
			elif "<xmax>" in line:
				xmax = line.replace("<xmax>","")
				xmax = xmax.replace("</xmax>","")
				xmax = xmax.join(xmax.split())
				cur_bb.append(int(xmax))
			elif "<ymax>" in line:
				ymax = line.replace("<ymax>","")
				ymax = ymax.replace("</ymax>","")
				ymax = ymax.join(ymax.split())
				cur_bb.append(int(ymax))
				bb_cords.append(cur_bb)
				cur_bb = []
			old_xml.append(line)
		xml.close()
		new_xml = []
		object_counter = 0
		for line in old_xml:
			if "person" in line:
				#display and approve of the bounding box for each object before changing to label
			 	image = cv2.imread(image_file)
			 	line = check_annotation(bb_cords[object_counter], line, image)
			 	object_counter += 1

			new_xml.append(line)
		xml = open(xml_file, 'w')
		xml.writelines(new_xml)
		xml.close()

if __name__ == "__main__":
	main()