import numpy as np
import cv2
import sys
import os

"""
	Image must be 1 channel! This function is called by fold_image for every channel
"""
def fold_pixel(img, x, y, kernel):
	img_height, img_width = np.array(img).shape
	kernel_height, kernel_width = np.array(kernel).shape

	start_y = max(0, y - int(kernel_height/2))
	end_y   = min(img_height, y + int(kernel_height/2) + 1)

	start_x = max(0, x - int(kernel_width/2))
	end_x   = min(img_width, x + int(kernel_width/2) + 1)

	result = 0

	for j in range(start_y, end_y):
		for i in range(start_x, end_x):
			result += img[j][i] * kernel[j-start_y][i-start_x]
	
	return result

"""
	Image must be 3-channel! (RGB)
"""
def fold_image(img, kernel):	
	height, width, channels = img.shape
	new_img = np.zeros((height,width,channels), dtype=np.uint8)

	for y in range(height):
		for x in range(width):
			for c in range(channels):
				new_img[y][x][c] = fold_pixel(img[:,:,c], x, y, kernel)

	return new_img

"""
	This function expects a RGB image file (filename) as first parameter and the number of iterations as second
"""
def simple_blur(img_file, iterations):
	orig_img = cv2.imread(img_file)	

	kernel = [[1,1,1],
			  [1,1,1],
			  [1,1,1]]
	fact = 1.0/np.sum(np.array(kernel).flatten())
	kernel = [[x*fact for x in row] for row in kernel]	
	
	new_img = np.copy(orig_img)

	for iteration in range(iterations):	
		print("Iteration %d" % (iteration+1))
		new_img = fold_image(new_img, kernel)

	cv2.imshow("new_img", new_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == "__main__":
	if not os.path.exists(sys.argv[1]):
		print("%s not found")
		exit()

	iterations = 1
	if len(sys.argv) > 2:		
		iterations = int(sys.argv[2])

	simple_blur(sys.argv[1], iterations)