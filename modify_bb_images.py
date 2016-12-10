import cv2
import numpy as np


# Character dataset iamges are 1200x900 and not all are centered
# Goal is to change image dimensions to match these and probably dilate?
def process_image(bounding_box_image, dimensions=30, show_intermediate=False):
    # image = cv2.imread(bounding_box_image_filename)
    image = cv2.cvtColor(bounding_box_image, cv2.COLOR_BGR2GRAY)
    # print(np.shape(image))
    h, w = np.shape(image)

    scaling_factor = min(350/float(w), 300/float(h))
    # print(scaling_factor)

    image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation = cv2.INTER_CUBIC)
    if show_intermediate:
        cv2.imshow("resized", image)

    _, image_threshold = cv2.threshold(image, 175, 255, cv2.THRESH_BINARY_INV)
    if show_intermediate:
        cv2.imshow("thres", image_threshold)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(image_threshold, kernel, iterations=7)  # dilate
    if show_intermediate:
        cv2.imshow("dilated",dilated)

    image = (255-dilated)
    if show_intermediate:
        cv2.imshow("final", image)

    final_width = dimensions
    final_height = dimensions

    h, w = np.shape(image)
    scaling_factor = min(final_width / float(w), final_height / float(h))
    new_width = int(round(w * scaling_factor))
    new_height = int(round(h * scaling_factor))

    # print(new_width)
    # print(new_height)

    resized = cv2.resize(image, (new_width, new_height))
    # resized = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_CUBIC)
    if show_intermediate:
        cv2.imshow("resized", resized)

    width_diff = final_width - new_width
    height_diff = final_height - new_height
    # print(width_diff)
    # print(height_diff)

    leftBorder = width_diff / 2
    rightBorder = width_diff / 2
    topBorder = height_diff / 2
    bottomBorder = height_diff / 2
    if width_diff % 2 == 1:
        rightBorder += 1

    if height_diff % 2 == 1:
        bottomBorder += 1

    # print(leftBorder)
    # print(rightBorder)

    resized = cv2.copyMakeBorder(resized, topBorder, bottomBorder, leftBorder, rightBorder,
                                cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # print(np.shape(resized))
    if show_intermediate:
        cv2.imshow("20x20", resized)
    #
    # need to pad the image with white to make it 1200x900
    # widthDiff = 1200-np.shape(image)[1]
    # heightDiff = 900-np.shape(image)[0]
    #
    # leftBorder = widthDiff / 2
    # rightBorder = widthDiff / 2
    # topBorder = heightDiff / 2
    # bottomBorder = heightDiff / 2
    #
    # if widthDiff % 2 == 1:
    #     rightBorder += 1
    #
    # if heightDiff % 2 == 1:
    #     bottomBorder += 1
    #
    # image = cv2.copyMakeBorder(image, topBorder, bottomBorder, leftBorder, rightBorder,
    #                            cv2.BORDER_CONSTANT, value=[255,255,255])
    #
    # if show_intermediate:
    #     cv2.imshow("final-rescaled", image)

    if show_intermediate:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return resized


def main():
    filename = "bounding_box3.jpg"
    image = cv2.imread(filename)
    image = process_image(image, 20, True)


if __name__ == "__main__":
    main()