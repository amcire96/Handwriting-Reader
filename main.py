import cv2

import modify_bb_images
import extract_letters_from_image
import get_letter_ordering


def main():
    image_file = "test.jpg"
    image, bounding_boxes, bounding_boxes_dimensions = extract_letters_from_image.detect_letters_bounding_boxes(image_file)
    processed_bounding_boxes = []
    for bounding_box in bounding_boxes:
        processed_bb = modify_bb_images.process_image(bounding_box)
        processed_bounding_boxes.append(processed_bb)
        # cv2.imshow("bb", processed_bb)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    ordering = get_letter_ordering.get_letter_ordering(bounding_boxes_dimensions)
    for line in ordering:
        for word in line:
            for letter_index in word:
                cv2.imshow("bb", processed_bounding_boxes[letter_index])
                cv2.waitKey(0)
                cv2.destroyAllWindows()




if __name__ == "__main__":
    main()