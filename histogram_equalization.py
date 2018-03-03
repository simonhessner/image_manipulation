import cv2
import sys
import numpy as np
from collections import defaultdict

"""
	Decides if image is grayscale or RGB and calls the corresponding function
"""
def histogram_equalization(img, minv=0, maxv=255):
	if len(img.shape) == 2:		
		return histogram_equalization_gray(img, minv, maxv)
	return histogram_equalization_rgb(img, minv, maxv)

"""
	Expects a single-channel image and performs histogram equalization on it
	The resulting image has a higher contrast than the input
"""
def histogram_equalization_gray(img, minv=0, maxv=255):
	assert(len(img.shape) == 2)

	# Count the occurences of each intensity value (0-255)
	histogram = defaultdict(int)
	for x in img.flatten():
		histogram[x] += 1

	# Compute the cumulative histogram, for each intensity count how
	# many pixels are darker or equally bright
	cum_hist = defaultdict(float)
	for i in range(256):
		cum_hist[i] = cum_hist[i-1] + histogram[i]

	# replace the intensity of every pixel in the input image by the probability of a pixel being as bright or darker as the current pixel
	# multiplied by the size of the desired intensity range
	# In other words, close intensity values that occur often will have remarkable different probabilities and will thus be mapped to different
	# intensities, resulting in a higher contrast in regions where common intensities occur
	result = np.zeros(img.shape, dtype=np.uint8)
	for y in range(result.shape[0]):
		for x in range(result.shape[1]):
			result[y,x] = int(cum_hist[img[y,x]] / (img.shape[0]*img.shape[1]) * (maxv-minv) + minv)

	return result

"""
	Expects a bgr/rgb image, converts it to HSV and applies single-channel
	histogram equalization on the luminance channel. The result is transformed back
	to RGB
"""
def histogram_equalization_rgb(img, minv=0, maxv=255):
	assert(len(img.shape) == 3)

	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	luminance_channel = hsv_img[:,:,2]
	hsv_img[:,:,2] = histogram_equalization_gray(luminance_channel, minv, maxv)	
	return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

if __name__ == "__main__":
	img = cv2.imread(sys.argv[1])	
	result = histogram_equalization(img)
	cv2.imwrite(sys.argv[2], result)	