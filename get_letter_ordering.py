from intervaltree import Interval, IntervalTree
import numpy as np
import extract_letters_from_image

# each bounding box coordinate will have x/y of top left pt and width/height
def get_letter_ordering(bounding_box_coordinates):
    intervals = IntervalTree()
    for i, bounding_box_coord in enumerate(bounding_box_coordinates):
        [x, y, w, h] = bounding_box_coord
        intervals[y:y+h] = i
    # intervals of heights
    intervals.merge_overlaps()
    # print(intervals)

    interval_to_bounding_boxes = {}
    for interval in intervals:
        interval_to_bounding_boxes[interval] = []
    for i, bounding_box_coord in enumerate(bounding_box_coordinates):
        [x, y, w, h] = bounding_box_coord
        overlapping_set = intervals[y:y+h]

        if len(overlapping_set) != 1:
            raise Exception("Should not happen since we merged all intervals previously.")

        overlapping_interval = list(overlapping_set)[0]

        interval_to_bounding_boxes[overlapping_interval].append(i)

    # print(interval_to_bounding_boxes)

    # order is ordered list of lines with each line
    # a list of ordered indices of bounding boxes
    order = []

    # distances is 2D list of distances between each letter in each line
    distances = []

    # widths of letters
    widths = []

    for i, interval in enumerate(sorted(interval_to_bounding_boxes)):
        bounding_box_index_list = interval_to_bounding_boxes[interval]
        interval_for_line = IntervalTree()
        order.append([])
        distances.append([])
        widths.append([])

        for bounding_box_index in bounding_box_index_list:
            bounding_box_coordinate = bounding_box_coordinates[bounding_box_index]
            [x, y, w, h] = bounding_box_coordinate
            widths[i].append(w)
            # print(bounding_box_index)
            interval_for_line[x:x+w] = bounding_box_index
        interval_for_line = sorted(interval_for_line)
        # print(interval_for_line)

        for interval in interval_for_line:
            index = interval[2]
            if order[i] != []:
                # print(order[i][-1])
                [x2, y2, w2, h2] = bounding_box_coordinates[order[i][-1]]
                # print(interval[0])
                distances[i].append(interval[0]-(x2+w2))
            order[i].append(index)

    final_ordering = []
    # print(widths)
    # print(distances)
    # print(order)
    for i, line in enumerate(distances):
        final_ordering.append([])
        curr_word = [order[i][0]]
        max_width_letter = sorted(widths[i])[int(len(widths[i]) * .75)]
        for j, distance in enumerate(line):
            # print(curr_word)
            if distance < max_width_letter:
                curr_word.append(order[i][j+1])
            else:
                final_ordering[i].append(curr_word)
        if curr_word != []:
            final_ordering[i].append(curr_word)

    # print(final_ordering)

    return final_ordering


def main():
    image_file = "test.jpg"
    image, bounding_boxes, bounding_boxes_dimensions = extract_letters_from_image.detect_letters_bounding_boxes(
        image_file)
    final_ordering = get_letter_ordering(bounding_boxes_dimensions)
    print(final_ordering)


if __name__ == "__main__":
    main()