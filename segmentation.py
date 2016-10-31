import cv2


def mser(image):
    mser = cv2.MSER()
    regions = mser.detect(image, None)
    print(regions)


def contour(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
    dilated = cv2.dilate(image, kernel, iterations=9)  # dilate , more the iteration more the dilation

    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours

    return contours