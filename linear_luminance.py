import cv2
import sys
import numpy as np

def linear_luminance(img, axis, increasing):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	for y in range(hsv.shape[0]):
		for x in range(hsv.shape[1]):
			if axis == "x":
				hsv[y,x,2] = int(x/hsv.shape[1]*hsv[y,x,2] if increasing else hsv[y,x,2]-x/hsv.shape[1]*hsv[y,x,2])
			elif axis == "y":
				hsv[y,x,2] = int(y/hsv.shape[0]*hsv[y,x,2] if increasing else hsv[y,x,2]-y/hsv.shape[0]*hsv[y,x,2])
			else:
				assert False, "Invalid axis. x,y are allowed"

	return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


if __name__ == "__main__":
	print(len(sys.argv))
	if len(sys.argv) < 5:
		print("Usage: linear_luminance.py input.jpg output.jpg (axis: x/y) (increasing: 0/1)")
		exit()

	img = cv2.imread(sys.argv[1])
	result = linear_luminance(img, sys.argv[3], sys.argv[4] == "1")
	cv2.imwrite(sys.argv[2], result)
	