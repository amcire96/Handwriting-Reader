import cv2
import numpy as np

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
    demo_digits()
    demo_alphabet()


if __name__ == "__main__":
    main()
