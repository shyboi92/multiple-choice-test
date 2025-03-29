#!/usr/bin/env python3

import json
from pathlib import Path
import sys
import imutils
import numpy as np
import cv2
from math import ceil
from model import CNN_Model
from collections import defaultdict


def get_x(s):
    return s[1][0]


def get_y(s):
    return s[1][1]


def get_h(s):
    return s[1][3]


def get_x_ver1(s):
    s = cv2.boundingRect(s)
    return s[0] * s[1]


def crop_image(img):
    # convert image from BGR to GRAY to apply canny edge detection algorithm
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray_img = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    # remove noise by blur image
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    
    # apply canny edge detection algorithm
    #img_canny = cv2.Canny(blurred, 100, 200)
    img_canny = cv2.Canny(blurred, 50, 150)

    # find contours
    cnts = cv2.findContours(img_canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    ans_blocks = []
    x_old, y_old, w_old, h_old = 0, 0, 0, 0

    # ensure that at least one contour was found
    if len(cnts) > 0:
        # sort the contours according to their size in descending order
        cnts = sorted(cnts, key=get_x_ver1)

        # loop over the sorted contours
        for i, c in enumerate(cnts):
            x_curr, y_curr, w_curr, h_curr = cv2.boundingRect(c)
            area = w_curr * h_curr
            if area > 100000:
                # print(f"Contour {i + 1}: x={x_curr}, y={y_curr}, w={w_curr}, h={h_curr}")
                cv2.rectangle(img, (x_curr, y_curr), (x_curr + w_curr, y_curr + h_curr), (0, 255, 0), 2)

            if 200000> w_curr * h_curr > 110000:
                # check overlap contours
                check_xy_min = x_curr * y_curr - x_old * y_old
                check_xy_max = (x_curr + w_curr) * (y_curr + h_curr) - (x_old + w_old) * (y_old + h_old)

                # if list answer box is empty
                if len(ans_blocks) == 0:
                    ans_blocks.append(
                        (gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr], [x_curr, y_curr, w_curr, h_curr]))
                    # update coordinates (x, y) and (height, width) of added contours
                    x_old = x_curr
                    y_old = y_curr
                    w_old = w_curr
                    h_old = h_curr
                elif check_xy_min > 20000 and check_xy_max > 20000:
                    ans_blocks.append(
                        (gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr], [x_curr, y_curr, w_curr, h_curr]))
                    # update coordinates (x, y) and (height, width) of added contours
                    x_old = x_curr
                    y_old = y_curr
                    w_old = w_curr
                    h_old = h_curr
        # print(f"Number of contours found: {len(cnts)}")
        # sort ans_blocks according to x coordinate
        sorted_ans_blocks = sorted(ans_blocks, key=get_x)
        # cv2.imshow("Contours", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(f"Number of answer blocks detected: {len(ans_blocks)}")
        # for idx, block in enumerate(ans_blocks):
            # x, y, w, h = block[1]
            # print(f"Ans_block {idx + 1}: x={x}, y={y}, w={w}, h={h}")
        return sorted_ans_blocks


def process_ans_blocks(ans_blocks):
    """
        this function process 2 block answer box and return a list answer has len of 200 bubble choices
        :param ans_blocks: a list which include 2 element, each element has the format of [image, [x, y, w, h]]
    """
    list_answers = []

    # Loop over each block ans in
    for ans_block in ans_blocks:
        ans_block_img = np.array(ans_block[0])

        offset1 = ceil(ans_block_img.shape[0] / 6)
        # Loop over each box in answer block
        for i in range(6):
            box_img = np.array(ans_block_img[i * offset1:(i + 1) * offset1, :])
            # height_box = box_img.shape[0]

            # box_img = box_img[14:height_box - 14, :]
            offset2 = ceil(box_img.shape[0] / 5)

            # loop over each line in a box
            for j in range(5):
                list_answers.append(box_img[j * offset2:(j + 1) * offset2, :])
    
    # print(f"Total number of answer areas: {len(list_answers)}")
    return list_answers


def process_list_ans(list_answers):
    list_choices = []
    offset = 35
    start = 35

    for answer_img in list_answers:
        for i in range(4):
            bubble_choice = answer_img[:, start + i * offset:start + (i + 1) * offset]
            bubble_choice = cv2.threshold(bubble_choice, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            if bubble_choice.size == 0:
                # print("Error: Empty image passed to resize.")
                continue  # Bỏ qua vòng lặp hiện tại nếu hình ảnh rỗng

            bubble_choice = cv2.resize(bubble_choice, (28, 28), cv2.INTER_AREA)
            bubble_choice = bubble_choice.reshape((28, 28, 1))
            list_choices.append(bubble_choice)
    
    if len(list_choices) != 360:
        print(f"Expected 360 choices but got {len(list_choices)}")
        raise ValueError("Invalid number of choices detected")
    return list_choices


def map_answer(idx):
    if idx % 4 == 0:
        answer_circle = "A"
    elif idx % 4 == 1:
        answer_circle = "B"
    elif idx % 4 == 2:
        answer_circle = "C"
    else:
        answer_circle = "D"
    return answer_circle


def get_answers(list_answers):
    results = defaultdict(list)
    
    path_to_script = Path(__file__)
    if path_to_script.is_symlink():
        path_to_script = path_to_script.readlink()
    weight_path = path_to_script.with_name('weight.h5')
    model = CNN_Model(weight_path).build_model(rt=True)
    
    list_answers = np.array(list_answers)
    scores = model.predict_on_batch(list_answers / 255.0)
    for idx, score in enumerate(scores):
        question = idx // 4

        # score [unchoiced_cf, choiced_cf]
        if score[1] > 0.9:  # choiced confidence score > 0.9
            chosed_answer = map_answer(idx)
            results[question + 1].append(chosed_answer)

    return results


if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])
    # Resize ảnh trước khi hiển thị
    scale_percent = 30  # Tỷ lệ kích thước (50% kích thước gốc)
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Thay đổi kích thước ảnh
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    list_ans_boxes = crop_image(resized_img)
    #list_ans_boxes2 = crop_image(img2)
    list_ans = process_ans_blocks(list_ans_boxes)
    list_ans = process_list_ans(list_ans)
    answers = get_answers(list_ans)
    print(json.dumps(answers))