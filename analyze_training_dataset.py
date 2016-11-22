import os
import cv2
import numpy as np

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
            image = cv2.imread(full_filename)
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

            # print(minrow)
            # print(maxrow)
            # print(mincol)
            # print(maxcol)

            widths.append(maxcol - mincol)
            heights.append(maxrow - minrow)

            # cv2.rectangle(image, (mincol, minrow), (maxcol, maxrow), (0,0,0), 2)
            #
            # cv2.imshow("img", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        print("%s: height has avg %d and std of %d and width has avg of %d and std of %d" %
              (sample_dir, np.average(heights), np.std(heights), np.average(widths), np.std(widths)))


def main():
    analyze_char_bounding_boxes("character_data/Hnd/Img")


if __name__ == "__main__":
    main()