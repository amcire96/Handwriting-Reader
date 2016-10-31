import cv2
import numpy as np

from matplotlib import pyplot as plt

import preprocess_image
import segmentation


def demo_digits():
    img = cv2.imread("digits.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

    x = np.array(cells)

    train = x[:,:50].reshape(-1,400).astype(np.float32)
    test = x[:,50:100].reshape(-1,400).astype(np.float32)

    k = np.arange(10)
    train_labels = np.ndarray.flatten(np.repeat(k,250)[:,np.newaxis])
    test_labels = train_labels.copy()

    # print(np.shape(train))
    # print(np.shape(train_labels))

    knn = cv2.KNearest()
    knn.train(train, train_labels)
    ret, result, neighbors, dist = knn.find_nearest(test, k=5)
    result = np.ndarray.flatten(result)

    # print result
    # print test_labels

    matches = (result == test_labels)

    correct = np.count_nonzero(matches)
    accuracy = correct*100.0/result.size

    print accuracy


def demo_alphabet():
    data = np.loadtxt('letter-recognition.data', dtype='float32', delimiter=',',
                      converters = {0: lambda ch: ord(ch) - ord('A')})

    train, test = np.vsplit(data, 2)

    responses, trainData = np.hsplit(train, [1])
    labels, testData = np.hsplit(test, [1])

    # print labels
    #
    # print np.shape(trainData)
    # print np.shape(testData)
    #
    # print np.shape(responses)
    # print np.shape(labels)

    knn = cv2.KNearest()
    knn.train(trainData, responses)
    ret, result, neighbors, dist = knn.find_nearest(testData, k=5)

    # result = np.ndarray.flatten(result)

    # print(result)

    correct = np.count_nonzero(result == labels)
    accuracy = correct * 100.0 / 10000
    print accuracy



def main():
    # demo_digits()
    # demo_alphabet()
    # mser()

    image_file = "test.png"
    image = cv2.imread(image_file, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    image2 = cv2.imread(image_file, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    print(np.shape(image))

    threshold_image = preprocess_image.adaptive_threshold(image_file)

    images = [image, threshold_image, image2]
    titles = ["original", "gaussian threshold", "contours"]

    # segmentation.mser(threshold_image)
    contours = segmentation.contour(threshold_image)
    for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        # Don't plot small false positives that aren't text
        if w < 35 and h < 35:
            continue

        # draw rectangle around contour on original image
        cv2.rectangle(image2, (x, y), (x + w, y + h), (255, 0, 255), 2)

        #you can crop image and send to OCR  , false detected will return no text :)
        # cropped = img_final[y :y +  h , x : x + w]

        # s = file_name + '/crop_' + str(index) + '.jpg'
        # cv2.imwrite(s , cropped)
        # index = index + 1

    for i in xrange(3):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()




if __name__ == "__main__":
    main()
