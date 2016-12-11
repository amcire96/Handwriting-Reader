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
	slant_steps = height/slant+1
	# print(height)
	# print(slant_steps)
	for i in range(height):
		if i%slant_steps == 0:
			slant -= 1
		#print(np.shape(image[i]))
		#print(image[i])
		image[i] = slide_row(image[i], slant, direction)
	return image

def transform_and_write(current_dir, new_dir):
	i = 0
	for sample_dir in os.listdir(current_dir):
		# Weird random file inside dir
		if "txt" in sample_dir or ".DS_Store" in sample_dir:
			continue
		joined_sample_dir = os.path.join(current_dir, sample_dir)
		j = 0
		for image_file in os.listdir(joined_sample_dir):
			if ".DS_Store" not in image_file:
				full_filename = os.path.join(joined_sample_dir, image_file)
				image = cv2.imread(full_filename)
				#trimmed_image = trim_training_data(full_filename, dims)
				squished_image = squish_image(image, 0.7)
				slant_left_image = slant_image(image, 150, "left")
				slant_right_image = slant_image(image, 150, "right")
				kernel = np.ones((30,30), np.uint8)
				thin_image = cv2.dilate(image, kernel, iterations=1)

				#full_output_filename = os.path.join(new_dir, sample_dir, image_file)
				squished_filename = os.path.join(new_dir, sample_dir, "squished"+str(j)+".png")
				slant_left_filename = os.path.join(new_dir, sample_dir, "slant_left"+str(j)+".png")
				slant_right_filename = os.path.join(new_dir, sample_dir, "slant_right"+str(j)+".png")
				thin_filename = os.path.join(new_dir, sample_dir, "thin"+str(j)+".png")

				cv2.imwrite(squished_filename, squished_image)
				cv2.imwrite(slant_left_filename, slant_left_image)
				cv2.imwrite(slant_right_filename, slant_right_image)
				cv2.imwrite(thin_filename, thin_image)

				j += 1
				i += 1
				print(i)

def main():
	# test = cv2.imread("img001-001.png")
	# squish_test = squish_image(test, 0.7)
	# slant_left_test = slant_image(test, 250, "left")
	# slant_right_test = slant_image(test, 250, "right")
	# kernel = np.ones((30,30), np.uint8)
	# erosion_test = cv2.dilate(test, kernel, iterations=1)
	# cv2.imwrite("squish_test.jpg", squish_test)
	# cv2.imwrite("slant_left_test.jpg", slant_left_test)
	# cv2.imwrite("slant_right_test.jpg", slant_right_test)
	# cv2.imwrite("erosion_test.jpg", erosion_test)
	transform_and_write("character_data_clean/Hnd/Img", "synthetic_data_clean/Hnd/Img")



if __name__ == '__main__':
	main()