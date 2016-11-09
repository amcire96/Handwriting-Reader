import cv2
import numpy as np
from matplotlib import pyplot as plt


# Adaptive threshold to preprocess the image
# Takes in a gray-scale image and returns binary image (B+W)
def adaptive_threshold(image_file):
    img = cv2.imread(image_file, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    img = cv2.medianBlur(img, 5)

    threshold_image = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0)

    return threshold_image


def sobel_preprocessing(image_file):
    img = cv2.imread(image_file, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    img_sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 5)
    img_sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    return img_sobelx, img_sobely


def laplacian_preprocessing(image_file):
    img = cv2.imread(image_file, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    return laplacian


def main():
    image_file = "test.png"
    image = cv2.imread(image_file, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    print(np.shape(image))

    threshold_image = adaptive_threshold(image_file)

    images = [image, threshold_image]
    titles = ["original", "gaussian threshold"]

    for i in xrange(2):
        plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()



if __name__ == "__main__":
    main()