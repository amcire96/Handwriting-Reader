import cv2
import numpy as np
from matplotlib import pyplot as plt

import segmentation
import preprocess_image


# Tried to get top answer of SO working
# http://stackoverflow.com/questions/23506105/extracting-text-opencv
# Doesn't work right now...
def detect_letters(image):
    segmented_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # image_sobel = cv2.Sobel(image, cv2.CV_8U, 1, 0, 3, 1, 0, cv2.BORDER_DEFAULT)
    # cv2.imshow("image_sobel", image_sobel)

    _, image_threshold = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
    # _, image_threshold = cv2.threshold(image_sobel, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    cv2.imshow("image_threshold", image_threshold)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (30, 30))
    image_threshold = cv2.morphologyEx(image_threshold, cv2.MORPH_CLOSE, element)

    cv2.imshow("morphologyex", image_threshold)

    contours, _ = cv2.findContours(image_threshold, 0, 1)
    for contour in contours:
        approx_contour = cv2.approxPolyDP(contour, 3, True)

        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(approx_contour)

        # discard areas that are too large
        if h > 300 and w > 300:
            continue

        # discard areas that are too small
        if h < 40 or w < 40:
            continue

        # draw rectangle around contour on original image
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)

    cv2.imshow("Image", image)
    # cv2.imshow("Segmented Image", segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Actually is able to draw bounding boxes
def detect_letters_bounding_boxes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grayscale
    print(gray)
    _, thresh = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY_INV)  # threshold

    cv2.imshow("threshold", thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=13)  # dilate

    cv2.imshow("dilated", dilated)

    # dilated = thresh

    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours

    # for each contour found, draw a rectangle around it on original image
    for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        # These "if" statements are subject to change depending on
        # the sizes of the images we use

        # Likely too small
        if h < 40 and w < 40:
            continue

        # discard areas that are too large
        if h > 300 and w > 300:
            continue
        #
        # # discard areas that are too small
        # if h < 40 or w < 40:
        #     continue

        # draw rectangle around contour on original image
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)

    # write original image with added contours to disk
    # cv2.imshow("Segmented", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return image


def main():
    # demo_digits()
    # demo_alphabet()
    # mser()

    # image_file = "business_card.jpg"
    # image_file = "street-sign.jpg"
    image_file = "test.jpg"


    image = cv2.imread(image_file)
    # print(image.shape)
    # detect_letters(image)
    # detect_letters_bounding_boxes(image)

    new_image = detect_letters_bounding_boxes(image)
    cv2.imwrite("test_segmented.jpg", new_image)
    cv2.imshow("Segmented", new_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
