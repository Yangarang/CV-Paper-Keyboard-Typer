import cv2
import numpy as np

# determine if key was clicked if finger coordinate has rested in one place
def checkKeyClick(fingerCoords, listSize, bufferCoord = 30, bufferFrames=5):
	# if finger coords is not yet fully populated, return false
	if len(fingerCoords)<listSize:
		return False
	# initialize output boolean to false
	boolKeyClick = False
	# use first coordinate as baseline to if key has been clicked
	init_x, init_y = fingerCoords[0]
	# count number of frames that are too far away from initial coordinates
	offFrames = 0
	for coord in fingerCoords:
		# if the x y coordinates are too far away from the init x y coordinates
		if (abs(coord[0]-init_x) > bufferCoord and abs(coord[1]-init_y)> bufferCoord):
			# add to number of off frames
			offFrames= offFrames+1
			# if the number of frames that were too far from expected passes the buffer frames allowed
			if offFrames > bufferFrames:
				# break loop and set key click to false
				boolKeyClick = False
				break
	# if there were less than # of buffer frames that were off, key is considered clicked
	if offFrames <= bufferFrames:
		boolKeyClick = True
	return boolKeyClick

def determineKeyClicked(fingerCoords, keyCoords, bufferCoord = 100):
	# initialize as invalid key
	key = "Invalid"
	# use first coordinate as the overall finger coordinate
	finger_x, finger_y = fingerCoords[0]
	# initialize list of distances
	list_distances = []
	# loop through all defined key coordinates
	for keyCoord in keyCoords:
		# distance equation
		keyDistance = (((finger_x - keyCoord[1])**2) + ((finger_y-keyCoord[2])**2))**0.5
		# append to list of distances
		list_distances.append([keyCoord[0], keyDistance])
	# get smallest distance out of all calculated distances from each key
	closest_key = min(list_distances, key=lambda x: x[1])
	# if the closest key distance is smaller than the buffer allowed
	if closest_key[1] < bufferCoord:
		key = closest_key[0]
	return key
