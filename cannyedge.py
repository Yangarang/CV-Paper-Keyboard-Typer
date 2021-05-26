import numpy as np
import math
import cv2


def non_max_suppression(img, theta):
	## loop through all points on the gradient magnitude and find max pixel values based on direction ##
	# instantiate image x y size
    img_x, img_y = img.shape
    # setup output and initialize all to 0
    output = np.zeros((img_x,img_y))
    # uzing theta, determine degrees by multiplying theta by 180 and dividing by pi
    degrees = theta*180/np.pi
    # instantiate degree list and convert all degrees less than 0 convert to positive
    degrees[degrees<0] = degrees[degrees<0] + 180

    # loop through each pixel of the input image
    for i in range(1, img_x-1):
        for j in range(1, img_y-1):
        	#instantiate neighboring intensity values
            first_intensity = 255
            second_intensity = 255
            # if degree in range of 0
            if (0 <= degrees[i,j] < 22.5) or (157.5 <= degrees[i,j] <= 180):
            	# get the surrounding pixels at 0 or 180 degree line direction intensities
                first_intensity = img[i, j+1]
                second_intensity = img[i, j-1]
            # if degree in range of 45
            elif (22.5 <= degrees[i,j] < 67.5):
            	# get the surrounding pixels at 45 degree direction intensities
                first_intensity = img[i+1, j-1]
                second_intensity = img[i-1, j+1]
            # if degree in range of 90
            elif (67.5 <= degrees[i,j] < 112.5):
            	# get the surrounding pixels at 90 degree direction intensities
                first_intensity = img[i+1, j]
                second_intensity = img[i-1, j]
            # if degree in range of 135
            elif (112.5 <= degrees[i,j] < 157.5):
            	# get the surrounding pixels at 135 degree direction intensities
                first_intensity = img[i-1, j-1]
                second_intensity = img[i+1, j+1]
            # if a pixel in the same direction has a higher intensity than the current pixel's
            if ((img[i,j]>=first_intensity) and (img[i,j]>=second_intensity)):
            	# set output pixel to input pixel's intensity
                output[i,j] = img[i,j]
            else:
            	# set output pixel to 0
                output[i,j] = 0
    return output

def threshold(img, weak, strong, lowRatio, highRatio):
	## identify strong and weak pixels ##
	# instantiate image x y size
    img_x, img_y = img.shape
    # setup output and initialize all to 0
    output = np.zeros((img_x,img_y))

    # instantiate weak and strong pixels based on input
    weak_pixel = np.int32(weak)
    strong_pixel = np.int32(strong)

    # determine ratios of high/low thresholds based on max input image value
    high_Threshold = img.max()*highRatio
    low_Threshold = high_Threshold*lowRatio
    
    # get all x and y coordinates of strong pixel values above threshold
    strong_pixel_i, strong_pixel_j = np.where(img >= low_Threshold)
    # get all x y coordinates of weak pixel values below threshold 
    weak_pixel_i, weak_pixel_j = np.where((img <= high_Threshold) & (img >= low_Threshold))    
    # output the two pixel intensity values based on if they are within weak or strong range
    output[strong_pixel_i, strong_pixel_j] = strong_pixel
    output[weak_pixel_i, weak_pixel_j] = weak_pixel
    
    return output, weak_pixel, strong_pixel

def hysteresis(img, weak, strong):
	## convert weak pixels to strong pixels if surrounding pixels are strong ##
	# instantiate image x y size
    img_x, img_y = img.shape
    # setup output and initialize all to 0
    output = np.copy(img)

    # loop through each pixel of the input image    
    for i in range(1, img_x-1):
        for j in range(1, img_y-1):
        	# if at vertical edges, automatically set to black pixels
        	if i < 5 or i > img_x-5:
        		output[i, j] = 0
        	# if at horizontal edges, automatically set to black pixels
        	elif j < 5 or j > img_y-5:
        		output[i, j] = 0
        	else:
        		# if the current intensity is weak
	            if output[i,j] == weak:
	            	# determine if one of the neighboring pixels are strong, if so set to strong value
	            	if img[i, j-1] == strong:
	            		output[i, j] = strong
	            	elif img[i, j+1] == strong:
	            		output[i, j] = strong
	            	elif img[i-1, j] == strong:
	            		output[i, j] = strong
	            	elif img[i+1, j] == strong:
	            		output[i, j] = strong
	            	elif img[i-1, j-1] == strong:
	            		output[i, j] = strong
	            	elif img[i-1, j+1] == strong:
	            		output[i, j] = strong
	            	elif img[i+1, j-1] == strong:
	            		output[i, j] = strong
	            	elif img[i+1, j+1] == strong:
	            		output[i, j] = strong
	            	# if none of the surrounding pixels is strong, set to current pixel to 0
	            	else:
	            		output[i, j] = 0
    return output