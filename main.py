from keras import models
import cv2

import modify_bb_images
import extract_letters_from_image
import get_letter_ordering
import predict_letters


model = models.load_model("id20_nhl2_hls128_nf32_tanh_cleaned.h5")


def main():
    image_file = "test2.jpg"
    image, bounding_boxes, bounding_boxes_dimensions = extract_letters_from_image.detect_letters_bounding_boxes(image_file)
    processed_bounding_boxes = []
    for bounding_box in bounding_boxes:
        processed_bb = modify_bb_images.process_image(bounding_box, dimensions=20)
        processed_bounding_boxes.append(processed_bb)
        # cv2.imshow("bb", processed_bb)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    print(len(processed_bounding_boxes))

    ordering = get_letter_ordering.get_letter_ordering(bounding_boxes_dimensions)
    print(ordering)
    final_text = []
    for line in ordering:
        curr_line = []
        for word in line:
            curr_word = ""
            for letter_index in word:
                image = processed_bounding_boxes[letter_index]
                prediction = predict_letters.predict_letter(image, model)
                print(prediction)
                cv2.imshow("bb", processed_bounding_boxes[letter_index])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                curr_word += str(prediction)
            curr_line.append(curr_word)
        curr_line = " ".join(curr_line)
        final_text.append(curr_line)
    print("\n".join(final_text))

if __name__ == "__main__":
    main()