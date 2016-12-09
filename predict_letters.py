

def convert_to_char(prediction):
    if 0 <= prediction <= 9:
        prediction_char = int(prediction)
    elif 10 <= prediction <= 35:
        prediction_char = chr(ord('A') + prediction - 10)
    elif 36 <= prediction <= 61:
        prediction_char = chr(ord('a') + prediction - 36)
    return prediction_char

def predict_letter(image, model):

    prediction = model.predict(image)
    print(convert_to_char(prediction))