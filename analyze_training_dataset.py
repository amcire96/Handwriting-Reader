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

    w,h = np.shape(no_white_space_image)
    scaling_factor = min(20 / float(w), 20 / float(h))

    new_width = scaling_factor * w
    new_height = scaling_factor * h

    print(new_width)
    print(new_height)

    resized_image = cv2.resize(image, dsize=(int(round(new_width)), int(round(new_height))), interpolation=cv2.INTER_CUBIC)
    w, h = np.shape(resized_image)
    print(np.shape(resized_image))

    width_diff = 20 - w
    height_diff = 20 - h

    print(width_diff)
    print(height_diff)

    leftBorder = width_diff / 2
    rightBorder = width_diff / 2
    topBorder = height_diff / 2
    bottomBorder = height_diff / 2

    if width_diff % 2 == 1:
        rightBorder += 1

    if height_diff % 2 == 1:
        bottomBorder += 1

    image = cv2.copyMakeBorder(image, topBorder, bottomBorder, leftBorder, rightBorder,
                               cv2.BORDER_CONSTANT, value=[255,255,255])

    print(np.shape(image))

    cv2.imshow("cut down", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return no_white_space_image


def trim_and_write(current_dir, new_dir):
    i = 0
    for sample_dir in os.listdir(current_dir):
        # Weird random file inside dir
        if "txt" in sample_dir:
            continue
        joined_sample_dir = os.path.join(current_dir, sample_dir)

        for image_file in os.listdir(joined_sample_dir):
            full_filename = os.path.join(joined_sample_dir, image_file)
            trimmed_image = trim_training_data(full_filename)

            full_output_filename = os.path.join(new_dir, sample_dir, image_file)

            # cv2.imwrite(full_output_filename, trimmed_image)

            i += 1
            print(i)


def main():
    # analyze_char_bounding_boxes("character_data/Hnd/Img")
    # trim_training_data("character_data/Hnd/Img/Sample001/img001-001.png")
    trim_and_write("character_data/Hnd/Img", "character_data_trim/Hnd/Img")


if __name__ == "__main__":
    main()