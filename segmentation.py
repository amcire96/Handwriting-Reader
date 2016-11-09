import cv2


def mser(image):
    mser = cv2.MSER()
    regions = mser.detect(image, None)
    # print(regions)
    return regions


def contour(image):
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
    # dilated = cv2.dilate(image, kernel, iterations=9)  # dilate , more the iteration more the dilation

    thresh = cv2.dilate(image, None, iterations=3)
    thresh = cv2.erode(thresh, None, iterations=2)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)  # get contours

    print(len(contours))

    return contours