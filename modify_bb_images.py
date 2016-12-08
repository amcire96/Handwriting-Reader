import cv2
import numpy as np

# Character dataset iamges are 1200x900 and not all are centered
# Goal is to change image dimensions to match these and probably dilate?

def process_image(bounding_box_image, show_intermediate=False):
    # image = cv2.imread(bounding_box_image_filename)
    image = cv2.cvtColor(bounding_box_image, cv2.COLOR_BGR2GRAY)
    # print(np.shape(image))
    (w, h) = np.shape(image)

    # ultimately want to scale to 1200x900
    # in training dataset, BB of images are about 350 by 300
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

    # print(np.shape(image))

    # need to pad the image with white to make it 1200x900
    widthDiff = 1200-np.shape(image)[1]
    heightDiff = 900-np.shape(image)[0]

    leftBorder = widthDiff / 2
    rightBorder = widthDiff / 2
    topBorder = heightDiff / 2
    bottomBorder = heightDiff / 2

    if widthDiff % 2 == 1:
        rightBorder += 1

    if heightDiff % 2 == 1:
        bottomBorder += 1

    image = cv2.copyMakeBorder(image, topBorder, bottomBorder, leftBorder, rightBorder,
                               cv2.BORDER_CONSTANT, value=[255,255,255])

    if show_intermediate:
        cv2.imshow("final-rescaled", image)

    if show_intermediate:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return image

def main():
    filename = "bounding_box1.jpg"
    image = cv2.imread(filename)
    image = process_image(image, True)


if __name__ == "__main__":
    main()