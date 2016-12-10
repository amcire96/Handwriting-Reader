import numpy as np


def convert_to_char(prediction):
    if 0 <= prediction <= 9:
        prediction_char = int(prediction)
    elif 10 <= prediction <= 35:
        prediction_char = chr(ord('A') + prediction - 10)
    elif 36 <= prediction <= 61:
        prediction_char = chr(ord('a') + prediction - 36)
    return prediction_char


def predict_letter(image, model):
    h, w = np.shape(image)
    # print(np.shape(image))
    image = image.reshape(1, h, w, 1)
    [prediction] = model.predict(image)
    print(prediction.tolist())
    prediction_val = np.argmax(prediction)
    # print(prediction_val)
    return convert_to_char(prediction_val)