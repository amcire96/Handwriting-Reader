from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Dropout, Flatten
from collections import namedtuple
import cv2
import numpy as np
import os


Params = namedtuple("Params", ["input_dimsh", "input_dimsw", "output_dims",
                               "num_hidden_layers", "hidden_layer_size",
                               "activation_fcn", "num_filters", "batch_size", "nb_epoch"])


def build_model(model_params):
    model = Sequential()
    model.add(Convolution2D(model_params.num_filters, 3, 3, border_mode='valid',
                            input_shape=(model_params.input_dimsw, model_params.input_dimsh, 1),
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(output_dim=model_params.hidden_layer_size,
                    input_dim=model_params.num_filters))
    model.add(Activation(model_params.activation_fcn))

    for _ in range(model_params.num_hidden_layers - 1):
        model.add(Dense(output_dim=model_params.hidden_layer_size))
        model.add(Activation(model_params.activation_fcn))

    model.add(Dense(output_dim=model_params.output_dims))
    model.add(Activation("softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adagrad", metrics=["accuracy"])
    return model


# image_size can only be 30 or 20
def get_digits_data(full, image_size=30):
    imglist = []
    trainlist = []
    testlist = []

    if image_size == 30:
        chardir = 'character_data_trim_clean/Hnd/Img'
    elif image_size == 20:
        chardir = 'character_data_trim_20/Hnd/Img'
    else:
        print("ERROR")
        return
    for directory in sorted(os.listdir(chardir)):
        counter = 1
        dirpath = os.path.join(chardir, directory)
        if ".DS_Store" not in directory and ".txt~" not in directory:
            for filename in sorted(os.listdir(dirpath)):
                if ".DS_Store" not in filename:
                    filepath = os.path.join(dirpath, filename)
                    img = cv2.imread(filepath)
                    #height, width = np.shape(img)
                    #img = resize(image, None, fx=20/width, fy=20/height)
                    #print(filename)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    imglist.append(gray)
                    if counter <= 39:
                        trainlist.append(gray)
                    else:
                        testlist.append(gray)
                    counter += 1

    # img = cv2.imread("digits.png")
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

    x = np.array(imglist)
    train = np.array(trainlist)
    test = np.array(testlist)

    print(x.shape)

    # train = x[:, :70]
    # test = x[:, 70:100]

    print(train.shape)
    print(test.shape)

    # train = train.reshape(3500, 20, 20)
    # test = test.reshape(1500, 20, 20)

    # print(train.shape)

    x = x.reshape(x.shape[0], image_size, image_size, 1)
    train = train.reshape(train.shape[0], image_size, image_size, 1)
    test = test.reshape(test.shape[0], image_size, image_size, 1)

    x = x.astype('float')
    train = train.astype('float32')
    test = test.astype('float32')
    x /= 255
    train /= 255
    test /= 255

    sample_size = 39
    if full:
        sample_size = 55

    k = np.arange(62)
    flattened_labels = np.ndarray.flatten(np.repeat(k, sample_size)[:, np.newaxis])
    train_labels = np.zeros((flattened_labels.size, 62))
    train_labels[np.arange(flattened_labels.size), flattened_labels] = 1

    flattened_labels = np.ndarray.flatten(np.repeat(k, 16)[:, np.newaxis])
    test_labels = np.zeros((flattened_labels.size, 62))
    test_labels[np.arange(flattened_labels.size), flattened_labels] = 1

    # print(train_labels[0])
    # print(train_labels[3499])
    # print(train_labels)

    if full:
        return x, test, train_labels, test_labels
    else:
        return train, test, train_labels, test_labels


def run_on_seventy_thirty_split(model_params):

    model = build_model(model_params)

    train, test, train_labels, test_labels = get_digits_data(full=False, image_size=model_params.input_dimsw)
    print(np.shape(train))
    print(np.shape(test))
    print(np.shape(train_labels))
    print(np.shape(test_labels))

    validation_data = (test, test_labels)
    model.fit(train, train_labels, batch_size=64, nb_epoch=50, validation_data=validation_data)

    scores = model.evaluate(test, test_labels)
    print("Error = %.2f%%" % (100 - scores[1] * 100))


def generate_model(model_params, model_name):
    model = build_model(model_params)

    train, test, train_labels, test_labels = get_digits_data(full=True, image_size=model_params.input_dimsw)
    print(np.shape(train))
    print(np.shape(test))
    print(np.shape(train_labels))
    print(np.shape(test_labels))

    model.fit(train, train_labels, batch_size=model_params.batch_size, nb_epoch=model_params.nb_epoch)
    model.save(model_name)
    # scores = model.evaluate(test, test_labels)
    # print("Error = %.2f%%" % (100 - scores[1] * 100))


def cross_validation(model_params, k=5):
    num_images_of_each_char = 55
    size_of_partition = int(num_images_of_each_char / k)

    train, _, train_labels, _ = get_digits_data(full=True, image_size=model_params.input_dimsw)

    scores = []

    for i in range(k):
        curr_interval = []
        compliment_of_interval = []

        model = build_model(model_params)

        for j in range(62):
            curr_interval += range(j * num_images_of_each_char + i * size_of_partition, j * num_images_of_each_char + (i + 1) * size_of_partition)

        for j in range(62 * 55):
            if j not in curr_interval:
                compliment_of_interval.append(j)

        print(curr_interval)
        print(compliment_of_interval)

        curr_validation_features = train[curr_interval, :, :, :]
        curr_training_features = train[compliment_of_interval, :, :, :]

        curr_validation_labels = train_labels[curr_interval, :]
        curr_training_labels = train_labels[compliment_of_interval, :]

        print(np.shape(curr_training_features))
        print(np.shape(curr_validation_features))
        print(np.shape(curr_training_labels))
        print(np.shape(curr_validation_labels))

        model.fit(curr_training_features, curr_training_labels, batch_size=model_params.batch_size, nb_epoch=model_params.nb_epoch)
        loss, accuracy = model.evaluate(curr_validation_features, curr_validation_labels)
        scores.append(accuracy)
    for i in range(k):
        print("\n" + str(scores[i]))
    print("\n"+str(np.average(np.array(scores))))


def main():
    print("START")

    model_params = Params(input_dimsh=30, input_dimsw=30, output_dims=62,
                          num_hidden_layers=2, hidden_layer_size=128,
                          activation_fcn="tanh",
                          num_filters=32, batch_size=64, nb_epoch=50)

    # run_on_seventy_thirty_split(model_params)
    generate_model(model_params, 'id30_nhl2_hls128_nf32_tanh_cleaned.h5')
    # cross_validation(model_params, k=5)






if __name__ == "__main__":
    main()