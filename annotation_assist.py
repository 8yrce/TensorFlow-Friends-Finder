"""
 annotation assist uses the premade bounding boxes taken from our generic network
 and allows us to reassign the label if we see the object is what we want to identify specifically


"""


import fileinput
import glob
import os
import cv2
import time

os.chdir("screen_captures")


#Check our annotation to see if it needs to be modified
"""
PARAMS: bb_cords: bounding box cordinates of detection, line: current line of xml, image: current image to draw bounding box on, xml_file: the current xml file being modified
RETURNS: line: the new line to place in the xml
"""
def check_annotation(bb_cords, line, image, xml_file):

	cv2.rectangle(image, ( ( bb_cords[0] ), (bb_cords[1] ) ) , ( ( bb_cords[2] ) , ( bb_cords[3]) ) ,(255,255,255), 2) # cv2 takes in image, left/top, right/bottom, color, line thickness
	window = cv2.namedWindow("Annotation assist", cv2.WINDOW_NORMAL)
	
	# if your resolution makes this display size too large/small modify the below values
	cv2.resizeWindow("Annotation assist", 1800,900) # *** If you have issues with screen size change this ***

	cv2.imshow("Annotation assist", image)

	r = "Ross"
	m = "Monica"
	j = "Joey"
	c = "Chandler"
	p = "Pheobe"
	a = "Rachel"

	print("1:	Ross\n2:	Monica\n3:	Joey:\n4:	Chandler\n5:	Pheobe\n6:	Rachel\n0:	None\nOR 'q' to quit")
	print("What character is this?		")

	choice = cv2.waitKey(10000) & 0xFF

	if choice == ord('q'):
		exit()

	print("Saved to: {}\n\n".format(xml_file))


	if choice == ord('0'):
		character = "Person" # this ensures that if you mark someone as not a main character they dont pop back up in the annotation process as unmarked
	elif choice == ord('1'):
		character = r
	elif choice == ord('2'):
		character = m
	elif choice == ord('3'):
		character = j
	elif choice == ord('4'):
		character = c
	elif choice == ord('5'):
		character = p
	elif choice == ord('6'):
		character = a
	else: 
		print( "Invalid choice '",choice,"'", " ignoring")
		return line

	line = line.replace("person", character)

	return line


def main():
	try:
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
			
			object_counter = 0
			line_counter = 0
			
			for line in old_xml:
				if "person" in line:
					#display and approve of the bounding box for each object before changing to label
					image = cv2.imread(image_file)
					print("object counter: {}".format(object_counter))
					
					new_line = check_annotation(bb_cords[object_counter], line, image, xml_file)
					object_counter += 1
					old_xml[line_counter] = new_line

				line_counter += 1

			xml = open(xml_file, 'w')
			xml.writelines(old_xml)
			xml.close()

	except Exception as e:
		print("Please ensure you have data in your screen_captures folder")
		print(e)

if __name__ == "__main__":
	main()