from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Dropout, Flatten
from collections import namedtuple
import cv2
import numpy as np


Params = namedtuple("Params", ["input_dims", "output_dims",
                               "num_hidden_layers", "hidden_layer_size",
                               "activation_fcn", "num_filters"])


def build_model(model_params):
    model = Sequential()
    model.add(Convolution2D(model_params.num_filters, 3, 3, border_mode='valid',
                            input_shape=(model_params.input_dims, model_params.input_dims, 1),
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

    model.add(Dense(output_dim=10))
    model.add(Activation("softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adagrad", metrics=["accuracy"])
    return model


def get_digits_data():
    img = cv2.imread("digits.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

    x = np.array(cells)

    print(x.shape)

    train = x[:, :70]
    test = x[:, 70:100]

    print(train.shape)
    print(test.shape)

    train = train.reshape(3500, 20, 20)
    test = test.reshape(1500, 20, 20)

    print(train.shape)

    train = train.reshape(train.shape[0], 20, 20, 1)
    test = test.reshape(test.shape[0], 20, 20, 1)

    train = train.astype('float32')
    test = test.astype('float32')
    train /= 255
    test /= 255

    k = np.arange(10)
    flattened_labels = np.ndarray.flatten(np.repeat(k, 350)[:, np.newaxis])
    train_labels = np.zeros((flattened_labels.size, 10))
    train_labels[np.arange(flattened_labels.size), flattened_labels] = 1

    flattened_labels = np.ndarray.flatten(np.repeat(k, 150)[:, np.newaxis])
    test_labels = np.zeros((flattened_labels.size, 10))
    test_labels[np.arange(flattened_labels.size), flattened_labels] = 1

    print(train_labels[0])
    print(train_labels[3499])
    # print(train_labels)

    return train, test, train_labels, test_labels


def main():
    print("START")


    model_params = Params(input_dims=20, output_dims=10,
                          num_hidden_layers=2, hidden_layer_size=128,
                          activation_fcn="relu",
                          num_filters=32)
    model = build_model(model_params)

    train, test, train_labels, test_labels = get_digits_data()

    print(np.shape(train))
    print(np.shape(test))
    print(np.shape(train_labels))
    print(np.shape(test_labels))

    model.fit(train, train_labels, batch_size=64, nb_epoch=30,
              validation_data=(test, test_labels))
    scores = model.evaluate(test, test_labels)
    print("Error = %.2f%%" %(100-scores[1]*100))


if __name__ == "__main__":
    main()