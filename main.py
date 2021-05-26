import numpy as np
import cv2
import mediapipe as mp
import time
from matplotlib import pyplot as plt
from transform import getTransformationMatrices, getEndpoints, getContour, transformCoordinate
from keyboard import checkKeyClick, determineKeyClicked
from keycoordinates import keyboardKeys
from filters import convolution, gaussian_filter, magnitude_gradient
from cannyedge import non_max_suppression,threshold,hysteresis

def runCamera(listKeyCoordinates):
	# Instantiate VideoCapture object
	cap = cv2.VideoCapture(0)
	# Instantiate hands object from Media Pipe
	mpHands = mp.solutions.hands
	hands = mpHands.Hands()
	# Instantiate Media Pipe hands drawing utilities
	mpDraw = mp.solutions.drawing_utils
	# initialize list of fingertip top down (x,y) coordinates of index finger for first hand
	HandA_IndexCoordinates = []
	# initialize list of fingertip top down (x,y) coordinates of index finger for second hand
	HandB_IndexCoordinates = []
	# instantiate counter per frame
	counter = 0
	# instantiate clicked key as empty value
	clickedKey = ""

	# Keep displaying most updated frame from webcam
	while(True):
	    # Capture frame-by-frame
	    ret, frame = cap.read()
	    # flip frame vertically and horizontally
	    frame = cv2.flip(frame, 0)
	    frame = cv2.flip(frame, 1)
	    # Convert current frame to RGB image
	    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	    # Find hands on frame
	    handProcess = hands.process(imgRGB)
	    # get the camera frame's height and width
	    frameHeight, frameWidth, _ = frame.shape

	    # Get keyboard matrix information
	    if (clickedKey == "GETKBRD" or cv2.waitKey(1) == ord('c') or counter==0):
	    	# Print getting keyboard layout
	    	print("Getting keyboard layout...")
	    	# make output frame 1400x1000 pixels, standard paper size for 8.5"x11"
	    	output_width = 1400
	    	output_height = 1000
	    	# get matrix and inverse matrix of keyboard coordinates
	    	P, inv_P = getTransformationMatrices(frame, output_width, output_height)
	    	# print("Perspective Transformation Matrix: ")
	    	# print(P)
	    	# print("Inverse Perspective Transformation Matrix: ")
	    	# print(inv_P)
	    	# get topdown image and reverted image to frame using the found transformation matrix and inverse transformationmatrix	    	
	    	topdown_img = cv2.warpPerspective(frame,P,(output_width,output_height))
	    	reverted_img = cv2.warpPerspective(topdown_img,inv_P,(frameWidth,frameHeight))
	    	cv2.imwrite("testkeyboardimage.png", topdown_img)

	    	# Show top down image with keys and key points overlayed
	    	fig = plt.figure(figsize=(12, 6))
	    	fig.add_subplot(1,3,1)
	    	plt.imshow(frame, cmap='gray')
	    	fig.add_subplot(1,3,2)
	    	plt.imshow(topdown_img, cmap='gray')
	    	# save list of already previously annotated key names
	    	prevAnnotatedList = []
	    	for pt in listKeyCoordinates:
	    		plt.scatter(pt[1], pt[2], s=10, c='red', marker='.')
	    		if pt[0] not in prevAnnotatedList:
	    			plt.annotate(pt[0], (pt[1], pt[2]))
	    			prevAnnotatedList.append(pt[0])
	    	# Show reverted image with keys and key points overlayed
	    	fig.add_subplot(1,3,3)
	    	plt.imshow(reverted_img, cmap='gray')
	    	for pt in listKeyCoordinates:
	    		new_x, new_y = transformCoordinate(pt[1], pt[2],inv_P)
	    		plt.scatter(new_x, new_y, s=10, c='red', marker='.')
	    	plt.show()
	    	# reset clicked key to empty value
    		clickedKey = ""

	    # Draw hand coordinates on frame 
	    if handProcess.multi_hand_landmarks:
	    	# loop through each hand on the screen
	    	for handid, handLms in enumerate(handProcess.multi_hand_landmarks):
	    		# loop through each hand landmark (joints on the hand) 
	    		for fingerid, point in enumerate(mpHands.HandLandmark):
	    			# convert hand landmark to pixel xy coordinates in frame
	    			normalizedLandmark = handLms.landmark[point]
	    			pixelCoordinatesLandmark = mpDraw._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, frameWidth, frameHeight)
	    			# Get Top Down x y coordinate of first hand's index finger
	    			if (fingerid == 8 and handid == 0 and pixelCoordinatesLandmark is not None):
		    			currentTopDownCoordinate = transformCoordinate(pixelCoordinatesLandmark[0], pixelCoordinatesLandmark[1], P)
		    			HandA_IndexCoordinates.insert(0,currentTopDownCoordinate)
		    			# print("Hand 1 Index Finger Coordinates: ")
		    			# print(currentTopDownCoordinate)
	    			# Get Top Down x y coordinate of second hand's index finger
	    			if (fingerid == 8 and handid == 1 and pixelCoordinatesLandmark is not None):
		    			currentTopDownCoordinate = transformCoordinate(pixelCoordinatesLandmark[0], pixelCoordinatesLandmark[1], P)
		    			HandB_IndexCoordinates.insert(0,currentTopDownCoordinate)
		    			# print("Hand 2 Index Finger Coordinates: ")
		    			# print(currentTopDownCoordinate)
		    	# draw the hand landmarks on the frame
	    		mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

	    # Cleanup hand coordinate lists to save only the last 75 frames (remove oldest coordinate)
	    if len(HandA_IndexCoordinates) > 75:
	    	HandA_IndexCoordinates.pop()
	    if len(HandB_IndexCoordinates) > 75:
	    	HandB_IndexCoordinates.pop()
	    # check if a key is considered clicked based on coordinate list
	    if checkKeyClick(HandA_IndexCoordinates, 75, 30, 5):
	    	# determine which key was clicked using distance
	    	clickedKey = determineKeyClicked(HandA_IndexCoordinates, listKeyCoordinates, 100)
	    	# show key if valid key was pressed
	    	if clickedKey != "Invalid":
	    		print("Key was clicked: " + clickedKey)
	    	# clear list of index finger coordinates
	    	HandA_IndexCoordinates.clear()
	    if checkKeyClick(HandB_IndexCoordinates, 75, 30, 5):
	    	# determine which key was clicked using distance
	    	clickedKey = determineKeyClicked(HandB_IndexCoordinates, listKeyCoordinates, 100)
	    	# show key if valid key was pressed
	    	if clickedKey != "Invalid":
	    		print("Key was clicked: " + clickedKey)
	    	# clear list of index finger coordinates
	    	HandB_IndexCoordinates.clear()

	    # update the frame to output window
	    cv2.imshow('Camera',frame)
	    # Iterate counter
	    counter = counter + 1

	  	# Quit if QUIT Key was pressed
	    if cv2.waitKey(1) & 0xFF == ord('q') or clickedKey == "QUIT":
	        break

	# Release capture and close windows
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	# Print starting keyboard typer
	print("Keyboard Typer Started...")
	# get coordinates of keyboard keys
	listKeyCoordinates = keyboardKeys()
	# Start camera process
	runCamera(listKeyCoordinates)