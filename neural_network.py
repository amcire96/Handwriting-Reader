from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Dropout, Flatten
from collections import namedtuple
import cv2
import numpy as np

Params = namedtuple("Params", ["input_dims", "output_dims",
                               "num_hidden_layers", "hidden_layer_size",
                               "activation_fcn"])


def build_model(model_params):
    model = Sequential()
    model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(output_dim=model_params.hidden_layer_size,
                    input_dim=model_params.input_dims))
    model.add(Activation(model_params.activation_fcn))

    # for _ in range(model_params.num_hidden_layers - 1):
    #     model.add(Dense(output_dim=model_params.hidden_layer_size))
    #     model.add(Activation(model_params.activation_fcn))

    model.add(Dense(output_dim=model_params.output_dims))
    model.add(Activation("softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adagrad", metrics=["accuracy"])
    return model


def get_digits_data():
    img = cv2.imread("digits.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

    x = np.array(cells)

    train = x[:, :50].reshape(-1, 400).astype(np.float32)
    test = x[:, 50:100].reshape(-1, 400).astype(np.float32)

    k = np.arange(10)
    train_labels = np.ndarray.flatten(np.repeat(k, 250)[:, np.newaxis])
    test_labels = train_labels.copy()

    return train, test, train_labels, test_labels


def main():
    print("START")
    model_params = Params(400, 62, 5, 5, "relu")
    model = build_model(model_params)

    train, test, train_labels, test_labels = get_digits_data()

    print(np.shape(train))
    print(np.shape(test))
    print(np.shape(train_labels))
    print(np.shape(test_labels))

    # model.fit(train, train_labels, batch_size=32, num_epoch=10)
    # scores = model.evaluate(test, test_labels)
    # print("Error = %.2f%%" %(100-scores[1]*100))


if __name__ == "__main__":
    main()