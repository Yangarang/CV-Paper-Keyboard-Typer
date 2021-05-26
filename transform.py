import cv2
import numpy as np
from matplotlib import pyplot as plt
from filters import convolution, gaussian_filter, magnitude_gradient
from cannyedge import non_max_suppression,threshold,hysteresis


def getContour(frame):	
	# cleanup by filtering the frame using bilateral and gaussian blur
	# convert frame to grayscale
	grayimg = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  

	# apply bilateral filtering
	medianimg = cv2.medianBlur(grayimg,5)
	bilateralimg = cv2.bilateralFilter(medianimg,5,75,75)

	# apply gaussian blur with 5 by 5 kernel
	gaussian = gaussian_filter(5,1)
	gaussianimg = convolution(gaussian, bilateralimg)
	gaussianimg = np.uint8(gaussianimg)

	# get edges using magnitude gradient
	maggradimg, theta = magnitude_gradient(gaussianimg)
	nonMaxSuppImg = non_max_suppression(maggradimg,theta)
	thresholdImg, weak, strong = threshold(nonMaxSuppImg, 30, 50, 0.05, 0.09)
	cannyimg = hysteresis(thresholdImg, weak, strong)
	cannyimg = cannyimg.astype(np.uint8)

	# close small holes of the contour to create closed line
	kernel = np.ones((3,3))
	outimg = cv2.morphologyEx(cannyimg,cv2.MORPH_CLOSE, kernel)

	# display median bilateral and gaussian filters
	fig = plt.figure()
	fig.add_subplot(1, 3, 1)
	plt.title("Median Blur")
	plt.imshow(medianimg, cmap='gray')
	fig.add_subplot(1, 3, 2)
	plt.title("Bilateral Filter")
	plt.imshow(bilateralimg, cmap='gray')
	fig.add_subplot(1, 3, 3)
	plt.title("Gaussian Blur")
	plt.imshow(gaussianimg, cmap='gray')
	plt.show()
	# display gradient magnitutde, non Max, Thresholding, Canny, and closed morphological transformation
	fig = plt.figure()	
	fig.add_subplot(2, 3, 1)
	plt.title("Magnitude Gradient")
	plt.imshow(maggradimg, cmap='gray')
	fig.add_subplot(2, 3, 2)
	plt.title("Non-Max Suppression")
	plt.imshow(nonMaxSuppImg, cmap='gray')
	fig.add_subplot(2, 3, 3)
	plt.title("Threshold")
	plt.imshow(thresholdImg, cmap='gray')
	fig.add_subplot(2, 3, 4)
	plt.title("Canny Edge Detection")
	plt.imshow(cannyimg, cmap='gray')
	fig.add_subplot(2, 3, 5)
	plt.title("Closed Morphological")
	plt.imshow(outimg, cmap='gray')
	plt.show()

	## using libraries instead ##
	# gaussianimg =cv2.GaussianBlur(bilateralimg,(5,5),0)
	# cannyimg = cv2.Canny(gaussianimg,30,50)

	# get contours, the curve joining all continous points using source image, contour retrieval mode, and contour approximation
	contours,hierarchy = cv2.findContours(outimg,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	# sort the list of contours
	contours = sorted(contours, key=cv2.contourArea, reverse=True)

	# loop over list of contours
	for contour in contours:
		# approximate the contour by calculating the perimeter
	    perimeter = cv2.arcLength(contour,True)
	    # loop over contours to get the rectangle
	    approx=cv2.approxPolyDP(contour,0.02*perimeter,True)
	    # if our contour has 4 points, we can say that the contour is the rectangle
	    if len(approx) == 4:
	        sheetContour = approx
	        break
	# return the contour sheet
	return sheetContour

def getEndpoints(sheetContour):
	# get the endpoints of inputted frame by reshaping input contour
    sheetContour = sheetContour.reshape((4,2))
    # initialize new contour output with zeroes
    endpoints = np.zeros((4,2),dtype = np.float32)
    # get end points by getting minimum and maximum of contour
    add = sheetContour.sum(1)
    endpoints[2] = sheetContour[np.argmax(add)]
    endpoints[0] = sheetContour[np.argmin(add)]
    # get end points by getting minimum and maximum of contour
    diff = np.diff(sheetContour,axis = 1)
    endpoints[3] = sheetContour[np.argmax(diff)]
    endpoints[1] = sheetContour[np.argmin(diff)]
    
    return endpoints

def getTransformationMatrices(frame, output_width, output_height):
	# apply some cleanup filtering and extract contours
	sheetContour = getContour(frame)
	# get the endpoints of the sheet
	sheetEndpoints = getEndpoints(sheetContour)
	# make output frame 1400x1000 pixels, standard paper size for 8.5"x11"
	output_pts = np.float32([[0,0],[output_width,0],[output_width,output_height],[0,output_height]])
	# get the transformation matrix by using the output frame's endpoints and the found sheet's endpoints
	P = cv2.getPerspectiveTransform(sheetEndpoints,output_pts) 	
	## get inverse matrix
	inv_P = np.linalg.pinv(P)

	return P, inv_P

def transformCoordinate(in_x, in_y, matrix):
	# given x y coordinate, transform point using inputted matrix
	out_x = 0
	out_y = 0
	np_pts = np.float32([[[in_x, in_y]]])
	transformed_pts = cv2.perspectiveTransform(np_pts, matrix)
	out_x, out_y = transformed_pts[0][0]

	return out_x, out_y