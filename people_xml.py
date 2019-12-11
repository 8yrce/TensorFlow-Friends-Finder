"""
	Generates the xml file for our annotations based on the output of our generic model during people_capture
"""

import os
import shutil
import fileinput

#generates xml from the detections passed to it from the model
"""
PARAMS: box - the bounding box, classe - the class 
"""
def generate_xml(annotations, pic_name, WIDTH, HEIGHT):
	#make copy of base
	shutil.copy("base.xml", "{}/screen_captures/{}.xml".format(os.getcwd(),pic_name))
	#assign filename to boiler plate
	xml_read = open(("{}/screen_captures/{}.xml".format(os.getcwd(),pic_name)), 'r')
	old_lines = xml_read.readlines()
	xml_read.close()

	new_lines = []
	object_lines = []
	ol = False
	for line in old_lines:
		if "*FOLDER*" in line:
			line = line.replace("*FOLDER*", "screen_captures")

		elif "*FILENAME*" in line:
			line = line.replace("*FILENAME*", ("{}.xml".format(pic_name)))

		elif "*PATH*" in line:
			line = line.replace("*PATH*", "{}/{}.xml".format(os.getcwd(),pic_name).rstrip())

		elif "*WIDTH*" in line:
			line = line.replace("*WIDTH*", "{}".format(WIDTH))
		
		elif "*HEIGHT*" in line:
			line = line.replace("*HEIGHT*", "{}".format(HEIGHT))

		elif "<object>" in line:
			ol = True

		elif "</annotation>" in line:
			break # making sure we dont write the last line until we are done

		if ol == False:
			new_lines.append(line)
		else:
			object_lines.append(line)

	xml_write = open(("{}/screen_captures/{}.xml".format(os.getcwd(),pic_name)), 'w')
	xml_write.writelines(new_lines)
	xml_write.close()

	xml_append = open(("{}/screen_captures/{}.xml".format(os.getcwd(),pic_name)), 'a')
	for a in annotations: #start appending in the objects we found
		#box is left,right,top,bottom
		an_object_lines = []
		for line in object_lines:
			if "*NAME*" in line:
				line = line.replace("*NAME*", "{}".format("person"))#int(a[4])))

			elif "*YMIN*" in line:
				line = line.replace("*YMIN*", "{}".format(int(a[0] * HEIGHT)))
			
			elif "*XMIN*" in line:
				line = line.replace("*XMIN*", "{}".format(int(a[1] * WIDTH)))

			elif "*YMAX*" in line:
				line = line.replace("*YMAX*", "{}".format(int(a[2] * HEIGHT)))
			
			elif "*XMAX*" in line:
				line = line.replace("*XMAX*", "{}".format(int(a[3] * WIDTH)))
			an_object_lines.append(line)

		xml_append.writelines(an_object_lines)

	xml_append.write("</annotation>")
	xml_append.close()