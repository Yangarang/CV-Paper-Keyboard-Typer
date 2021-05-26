import numpy as np
import math
import cv2
from matplotlib import pyplot as plt

def convolution(f, I): 
	# Handle boundary of I, e.g. pad I according to size of f 
	# Compute im_conv = f*I 

	# flip the kernel for convolution
	f = np.flipud(np.fliplr(f))
	# setup height and width of image and kernel
	I_row, I_col = I.shape
	f_row, f_col = f.shape
	# add padding based on half of the size of kernel
	pad_row = int(f_row/2)
	pad_col = int(f_col/2)
	# create padded image template with 0s added to all sides
	img_padded = np.zeros((int(I_row+(2*pad_row)),int(I_col+(2*pad_col))))
	# get height and width of padded image template
	I_row_padded, I_col_padded = img_padded.shape
	# add original image into the middle of the padded image template
	for idx_i, i in enumerate(range(pad_row,(I_row_padded-pad_row))):
		for idx_j, j in enumerate(range(pad_col,(I_col_padded-pad_col))):
			img_padded[i][j] = I[idx_i][idx_j]	
	# setup height and width of output convolution image
	im_conv = np.zeros((I_row, I_col))
	# loop through height and length of input image range
	for i in range(I_row):
		for j in range(I_col):
			# setup kernel sized section of image at current index
			section_img = img_padded[i:i+f_row, j:j+f_col]
			# set output convolution image to the sum of section image * kernel
			im_conv[i][j] = np.sum(f*section_img)
	return im_conv

# class demo implementation of Gaussian filter
def gaussian_filter(size, sigma):    
    # Precompute sigma*sigma
    sigma2 = sigma*sigma    
    # Create a coordinate sampling from -n/2 to n/2 so that (0,0) will be at the center of the filter
    x = np.linspace(-size/2.0, size/2.0, 1)
    y = np.linspace(-size/2.0, size/2.0, 1)    
    # Blank array for the Gaussian filter
    gaussian_filter = np.zeros((size,size))
    # Loop over all elements of the filter
    for i in range(0, len(x)):
        for j in range(0, len(y)):            
            # Use the x and y coordinate sampling as the inputs to the 2D Gaussian function
            gaussian_filter[i,j] = (1/2*math.pi*sigma2)*math.exp(-(x[i]*x[i]+y[j]*y[j])/(2*sigma2))      
    # Normalize so the filter sums to 1
    return gaussian_filter/np.sum(gaussian_filter.flatten())

def magnitude_gradient(img):
	# declare vertical and horizontal sobels
	vert_Sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
	hori_Sobel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
	# convolution on input image using the sobels
	vert_img = convolution(vert_Sobel, img)
	hori_img = convolution(hori_Sobel, img)
	# compute gradient magnitude image G = sqrt(Gx^2 + Gy^2)
	gradientmag_img = np.sqrt(np.square(vert_img) + np.square(hori_img))
	# normalize gradient magnitude to scale of 0 to 255q
	gradientmag_img = gradientmag_img*(255/gradientmag_img.max())
	# get slope theta of gradient
	theta = np.arctan2(hori_img, vert_img)
	return gradientmag_img, theta
