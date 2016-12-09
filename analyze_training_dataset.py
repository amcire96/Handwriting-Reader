import os
import cv2
import numpy as np


def get_bounding_box(image_file):
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = np.shape(image)

    minrow = 0
    maxrow = 0
    mincol = 0
    maxcol = 0

    # Simplistic BB is just finding first/last row/col we find a black (0-valued) pixel
    for i in range(h):
        # There is a single black pizel in row i
        if np.min(image[i, :]) == 0:
            minrow = i
            break

    for i in range(i, h):
        # print(str(i) + " " + str(h))
        # There are only white pixels in row i
        if np.min(image[i, :]) == 255:
            maxrow = i
            break
    if maxrow == 0:
        maxrow = h

    for j in range(w):
        # There is a single black pizel in row i
        if np.min(image[:, j]) == 0:
            mincol = j
            break

    for j in range(j, w):
        # There are only white pixels in row i
        if np.min(image[:, j]) == 255:
            maxcol = j
            break
    if maxcol == 0:
        maxcol = w

    return minrow, maxrow, mincol, maxcol


def analyze_char_bounding_boxes(image_dir):
    for sample_dir in os.listdir(image_dir):
        # Weird random file inside dir
        if "txt" in sample_dir:
            continue
        joined_sample_dir = os.path.join(image_dir, sample_dir)

        heights = []
        widths = []

        for image_file in os.listdir(joined_sample_dir):
            full_filename = os.path.join(joined_sample_dir, image_file)
            # print(full_filename)
            minrow, maxrow, mincol, maxcol = get_bounding_box(full_filename)

            widths.append(maxcol - mincol)
            heights.append(maxrow - minrow)

            # cv2.rectangle(image, (mincol, minrow), (maxcol, maxrow), (0,0,0), 2)
            #
            # cv2.imshow("img", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        print("%s: height has avg %d and std of %d and width has avg of %d and std of %d" %
              (sample_dir, np.average(heights), np.std(heights), np.average(widths), np.std(widths)))


def trim_training_data(image_file):
    minrow, maxrow, mincol, maxcol = get_bounding_box(image_file)
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    no_white_space_image = image[minrow:maxrow, mincol:maxcol]

    height, width = np.shape(no_white_space_image)
    if height>width:
        ratio = float(width)/height
        unpadded_image = cv2.resize(no_white_space_image, dsize=(int(round(20*ratio)), 20))
        padding = 20-round(20*ratio)
        left_padding = int(padding/2)
        right_padding = int(round(padding/2))
        final_image = cv2.copyMakeBorder(unpadded_image,0,0,left_padding,right_padding,cv2.BORDER_CONSTANT, value=[255,255,255])
    else:
        ratio = float(height)/width
        unpadded_image = cv2.resize(no_white_space_image, dsize=(20, int(round(20*ratio))))
        padding = 20-round(20*ratio)
        top_padding = int(padding/2)
        bottom_padding = int(round(padding/2))
        final_image = cv2.copyMakeBorder(unpadded_image,top_padding,bottom_padding,0,0,cv2.BORDER_CONSTANT, value=[255,255,255])
    # cv2.imshow("cut down", no_white_space_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return final_image


def trim_and_write(current_dir, new_dir):
    i = 0
    for sample_dir in os.listdir(current_dir):
        # Weird random file inside dir
        if "txt" in sample_dir or ".DS_Store" in sample_dir:
            continue
        joined_sample_dir = os.path.join(current_dir, sample_dir)

        for image_file in os.listdir(joined_sample_dir):
            if ".DS_Store" not in image_file:
                full_filename = os.path.join(joined_sample_dir, image_file)
                trimmed_image = trim_training_data(full_filename)

                full_output_filename = os.path.join(new_dir, sample_dir, image_file)

                cv2.imwrite(full_output_filename, trimmed_image)

                i += 1
                print(i)


def main():
    # analyze_char_bounding_boxes("character_data/Hnd/Img")
    # trim_training_data("character_data/Hnd/Img/Sample001/img001-001.png")
    trim_and_write("character_data/Hnd/Img", "character_data_trim/Hnd/Img")


if __name__ == "__main__":
    main()