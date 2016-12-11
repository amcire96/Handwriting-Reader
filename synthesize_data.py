import cv2
import numpy as np
import os

def squish_image(img, squish_factor):
	image = img
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	height, width = np.shape(image)
	new_image = cv2.resize(image, dsize=(int(round(squish_factor*width)), height))
	padding = width-round(squish_factor*width)
	left_padding = int(padding/2)
	right_padding = int(round(padding/2))
	final_image = cv2.copyMakeBorder(new_image,0,0,left_padding,right_padding,cv2.BORDER_CONSTANT, value=[255,255,255])
	return final_image

def slide_row(row, displacement, direction):
	temp_row = row
	padding = [255]*displacement
	if "left" in direction:
		temp_row = row[displacement:len(row)]
		return np.concatenate((temp_row, padding), axis=0)
	elif "right" in direction:
		temp_row = row[0:(len(row)-displacement)]
		return np.concatenate((padding, temp_row), axis=0)
	else:
		print("error: invalid direction")
		return []

def slant_image(img, slant_factor, direction):
	image = img
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	height, width = np.shape(image)
	slant = slant_factor+1
	slant_steps = height/slant + 1
	print(height)
	print(slant_steps)
	for i in range(height):
		if i%slant_steps == 0:
			slant -= 1
		#print(np.shape(image[i]))
		#print(image[i])
		image[i] = slide_row(image[i], slant, direction)
	return image


def main():
	test = cv2.imread("img001-001.png")
	squish_test = squish_image(test, 0.7)
	slant_left_test = slant_image(test, 200, "left")
	slant_right_test = slant_image(test, 200, "right")
	cv2.imwrite("squish_test.jpg", squish_test)
	cv2.imwrite("slant_left_test.jpg", slant_left_test)
	cv2.imwrite("slant_right_test.jpg", slant_right_test)


if __name__ == '__main__':
	main()