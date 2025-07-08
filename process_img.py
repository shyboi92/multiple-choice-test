#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import sys
import imutils
import numpy as np
import cv2
import logging
import base64
from math import ceil
from model import CNN_Model
from collections import defaultdict
from ultralytics import YOLO
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Tắt log TensorFlow
# Đặt biến này thành True
# để hiện các kết quả và thông tin phục vụ debug
DEBUG=False

if DEBUG:
	logging.basicConfig(level=logging.DEBUG)
else:
	logging.basicConfig(level=logging.CRITICAL)

def get_x(s):
	return s[1][0]


def get_y(s):
	return s[1][1]


def get_h(s):
	return s[1][3]


def get_x_ver1(s):
	s = cv2.boundingRect(s)
	return s[0] * s[1]

# def get_contour_corners(contour):
# 	peri = cv2.arcLength(contour, True)
# 	approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
# 	if len(approx) == 4:
# 		return approx.reshape(4, 2)
# 	return None

def order_corners(corners):
	rect = np.zeros((4, 2), dtype="float32")
	s = corners.sum(axis=1)
	rect[0] = corners[np.argmin(s)]  # top-left
	rect[2] = corners[np.argmax(s)]  # bottom-right
	diff = np.diff(corners, axis=1)
	rect[1] = corners[np.argmin(diff)]  # top-right
	rect[3] = corners[np.argmax(diff)]  # bottom-left
	return rect

def perspective_transform(img, corners):
	rect = order_corners(corners)
	(tl, tr, br, bl) = rect
	widthA = np.linalg.norm(br - bl)
	widthB = np.linalg.norm(tr - tl)
	maxWidth = max(int(widthA), int(widthB))
	heightA = np.linalg.norm(tr - br)
	heightB = np.linalg.norm(tl - bl)
	maxHeight = max(int(heightA), int(heightB))
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]
	], dtype="float32")
	M = cv2.getPerspectiveTransform(rect, dst)
	# return cv2.warpPerspective(img, M, (maxWidth, maxHeight))
	warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
	return warped, M

def scale_image(img, target_height=1000):
	h, w = img.shape[:2]
	scale = target_height / h
	new_size = (int(w * scale), target_height)
	return cv2.resize(img, new_size), scale

def is_similar(x, y, w, h, rx, ry, rw, rh, tolerance=0.4, min_h=600):
	# So sánh kích thước w, h trước
	w_match = abs(w - rw) / rw <= tolerance
	h_match = abs(h - rh) / rh <= tolerance

	# So sánh area cho chắc chắn
	area = w * h
	ref_area = rw * rh
	area_match = abs(area - ref_area) / ref_area <= tolerance

	# Chỉ cần w,h,area đều match
	return w_match and h_match and area_match and h >= min_h

def detect_and_warp_yolo4points(img, model_path=None):
    if model_path is None:
        # Luôn lấy theo vị trí file script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'runs', 'detect', 'train7', 'weights', 'best.pt')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy YOLO model tại: {model_path}")
    model = YOLO(model_path, verbose= False)
    results = model(img, verbose = False )[0]
    points = {'top_left': None, 'top_right': None, 'bottom_right': None, 'bottom_left': None}

    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        label = model.names[int(cls)]
        points[label] = (cx, cy)
        if DEBUG:
            logging.debug(f" Detected {label}: ({cx}, {cy})")

    if all(points.values()):
        corners = np.array([
            points['top_left'],
            points['top_right'],
            points['bottom_right'],
            points['bottom_left']
        ], dtype="float32")
        return perspective_transform(img, corners)

    return None, None

def enhance_for_pencil(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Làm mịn nhẹ để giảm noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # CLAHE để tăng tương phản nét tô mờ
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # Ngưỡng OTSU — giữ màu gốc (vùng tô đen, nền trắng)
    _, threshed = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return threshed

def determine_answer_blocks(img):
	warped_img, M = detect_and_warp_yolo4points(img)
	if warped_img is None:
		raise ValueError("Không tìm thấy đủ 4 góc A4 để warp.")

	img, scale = scale_image(warped_img)
	enhanced = enhance_for_pencil(img)
	gray_img = enhanced.copy()
	
	# enhanced = enhance_for_pencil(warped_img)
	# img, scale = scale_image(enhanced)
	# gray_img = img.copy()
	
	blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
	img_canny = cv2.Canny(blurred, 50, 150)

	cnts = cv2.findContours(img_canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	# # 4. Tìm 5 contour lớn nhất
	# cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

	# img_contours = img.copy()

	# for i, c in enumerate(cnts):
	# 	x, y, w, h = cv2.boundingRect(c)
	# 	area = cv2.contourArea(c)
	# 	cv2.rectangle(img_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)
	# 	cv2.putText(img_contours, f"#{i+1} ({w}x{h}) {area:.0f}", (x, y - 10),
	# 				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

	# 	logging.debug(f"Contour {i+1}: x={x}, y={y}, w={w}, h={h}, area={area:.0f}")

	# # Show hình có 5 contour lớn nhất
	# cv2.imshow("Top 5 Contours", img_contours)
	# cv2.imshow("Canny edges", img_canny)
	# cv2.waitKey(0)

	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	ans_blocks = []
	mssv_block = []

	# Đây là 3 block mẫu mới
	ref_blocks = [
	(580, 388, 227, 691),
	(308, 390, 224, 689),
	(34, 391, 227, 688)]

	ref_blocks2 = (400, 7, 286, 360)

	found = set()
	enhanced = gray_img  

	for c in cnts:
		x, y, w, h = cv2.boundingRect(c)
		for i, (rx, ry, rw, rh) in enumerate(ref_blocks):
			if i in found:
				continue
			if is_similar(x, y, w, h, rx, ry, rw, rh, tolerance=0.5, min_h=500):
				ans_blocks.append((enhanced[y:y + h, x:x + w], [x, y, w, h]))
				found.add(i)
				break
		# Kiểm tra block MSSV
		if is_similar(x, y, w, h, *ref_blocks2, tolerance=0.35, min_h=200) and x>300:
			mssv_block.append((enhanced[y:y + h, x:x + w], [x, y, w, h]))

	# Fallback nếu thiếu block
	if len(ans_blocks) == 1:
		x1, y1, w1, h1 = ans_blocks[0][1]
		distances = [abs(x1 - rx) for rx, _, _, _ in ref_blocks]
		idx = distances.index(min(distances))
		ref_x, _, _, _ = ref_blocks[idx]
		dx = x1 - ref_x

		# Nếu là cột trái (idx == 2), tính 2 cái còn lại theo kiểu đều nhau
		if idx == 2:
			step = int(w1 / 5 * 6.2)
			x2 = x1 + step
			x3 = x2 + step
			for x_new in [x2, x3]:
				ans_blocks.append((enhanced[y1:y1+h1, x_new:x_new+w1], [x_new, y1, w1, h1]))
		else:
			# fallback cách cũ cho trường hợp khác
			for i, (rx, ry, rw, rh) in enumerate(ref_blocks):
				if i == idx:
					continue
				x_new = rx + dx
				y_new = y1
				ans_blocks.append((enhanced[y_new:y_new+rh, x_new:x_new+rw], [x_new, y_new, rw, rh]))

	elif len(ans_blocks) == 2:
		matched = {}
		for block in ans_blocks:
			x_real = block[1][0]
			distances = [abs(x_real - rx) for rx, _, _, _ in ref_blocks]
			best = distances.index(min(distances))
			matched[best] = block

		missing = [i for i in range(3) if i not in matched][0]
		i1, i2 = list(matched.keys())
		x1, _ = matched[i1][1][:2]
		x2, _ = matched[i2][1][:2]
		r1 = ref_blocks[i1][0]
		r2 = ref_blocks[i2][0]
		rm = ref_blocks[missing][0]

		# scale theo độ chênh ref và thực tế
		scale = (x2 - x1) / (r2 - r1)
		x_new = int(x1 + (rm - r1) * scale)
		y_new = int((matched[i1][1][1] + matched[i2][1][1]) / 2)
		w_new = int((matched[i1][1][2] + matched[i2][1][2]) / 2)
		h_new = int((matched[i1][1][3] + matched[i2][1][3]) / 2)
		ans_blocks.append((enhanced[y_new:y_new+h_new, x_new:x_new+w_new], [x_new, y_new, w_new, h_new]))

	elif len(ans_blocks) == 0:
		raise AssertionError('Không tìm thấy vùng đáp án.')
	if len(ans_blocks) == 0:
		raise AssertionError('Không tìm thấy vùng đáp án.')

	sorted_ans_blocks = sorted(ans_blocks, key=get_x)

	# Vẽ khung vùng đáp án
	for x, y, w, h in [b[1] for b in sorted_ans_blocks]:
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# cv2.imshow("Answer Blocks", img)
	# cv2.imshow("Canny edges", img_canny)
	# cv2.waitKey(0)

	logging.debug(f"Số vùng đáp án: {len(sorted_ans_blocks)}")
	for idx, block in enumerate(sorted_ans_blocks):
		x, y, w, h = block[1]
		logging.debug(f"Block {idx + 1}: x={x}, y={y}, w={w}, h={h}")
	
	if mssv_block:
		x, y, w, h = mssv_block[0][1]
		logging.debug(f"Vùng MSSV: x={x}, y={y}, w={w}, h={h}")
		cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # màu xanh dương
	else:
		raise AssertionError("Không tìm thấy vùng MSSV.")
				

	return sorted_ans_blocks, img, scale, M, mssv_block

def process_ans_blocks(ans_blocks):
	list_answers = []
	list_positions = []

	for block_idx, ans_block in enumerate(ans_blocks):
		ans_block_img = np.array(ans_block[0])
		block_x, block_y, block_w, block_h = ans_block[1]

		# Chuyển sang ảnh màu nếu đang là ảnh xám
		if len(ans_block_img.shape) == 2:
			vis_img = cv2.cvtColor(ans_block_img.copy(), cv2.COLOR_GRAY2BGR)
		else:
			vis_img = ans_block_img.copy()

		offset1 = ceil(ans_block_img.shape[0] / 4)
		for i in range(4):
			box_img = np.array(ans_block_img[i * offset1:(i + 1) * offset1, :])
			offset2 = ceil(box_img.shape[0] / 5)

			for j in range(5):
				answer_img = box_img[j * offset2:(j + 1) * offset2, :]
				list_answers.append(answer_img)

				# Tính toán vị trí trên ảnh warp gốc
				y1 = block_y + (i * offset1 + j * offset2)
				y2 = block_y + (i * offset1 + (j + 1) * offset2)
				x1 = block_x
				x2 = block_x + ans_block_img.shape[1]
				list_positions.append((x1, y1, x2, y2))

				# In tọa độ ô đang xử lý
				logging.debug(f'Block {block_idx + 1}, Group {i + 1}, Answer {j + 1}: Position = ({x1}, {y1}, {x2}, {y2})')
				
				cv2.rectangle(vis_img, (0, j * offset2), (ans_block_img.shape[1] - 1, (j + 1) * offset2), (255, 0, 0), 1)
		# cv2.imshow(f'Block {block_idx + 1} - Split lines', vis_img)
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	logging.debug(list_positions)
	return list_answers, list_positions

def process_list_ans(list_answers, list_positions):
	list_choices = []
	choice_positions = []

	for idx, (answer_img, pos) in enumerate(zip(list_answers, list_positions)):
		h, w = answer_img.shape[:2]
		offset = w // 5
		start = offset  

		for i in range(4):  
			x1 = start + i * offset
			x2 = start + (i + 1) * offset
			bubble_choice = answer_img[:, x1:x2]
			# Nếu chưa là grayscale thì chuyển
			if bubble_choice.ndim == 3 and bubble_choice.shape[2] == 3:
				bubble_choice = cv2.cvtColor(bubble_choice, cv2.COLOR_BGR2GRAY)

			# Đảm bảo đúng kiểu
			if bubble_choice.dtype != np.uint8:
				bubble_choice = bubble_choice.astype(np.uint8)
			bubble_choice = cv2.threshold(bubble_choice, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

			if bubble_choice.size == 0:
				continue

			bubble_choice = cv2.resize(bubble_choice, (28, 28), cv2.INTER_AREA)
			bubble_choice = bubble_choice.reshape((28, 28, 1))
			list_choices.append(bubble_choice)

			# tính vị trí tuyệt đối trên ảnh warp
			absolute_x1 = pos[0] + x1
			absolute_x2 = pos[0] + x2
			absolute_y1 = pos[1]
			absolute_y2 = pos[3]
			choice_positions.append((absolute_x1, absolute_y1, absolute_x2, absolute_y2))

	if len(list_choices) != 240:
		logging.critical(f"Expected 240 choices but got {len(list_choices)}")
		raise ValueError("Invalid number of choices detected")
	
	# logging.debug("Passed check, choice_positions:")
	# logging.debug(choice_positions)
	return list_choices, choice_positions

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
	model = CNN_Model('weight.h5').build_model(rt=True)
	list_answers = np.array(list_answers)
	scores = model.predict_on_batch(list_answers / 255.0)
	for idx, score in enumerate(scores):
		question = idx // 4

		# score [unchoiced_cf, choiced_cf]
		if score[1] > 0.9:  # choiced confidence score > 0.9
			chosed_answer = map_answer(idx)
			results[question + 1].append(chosed_answer)

	return results

def process_mssv_block(mssv_block):
	
	mssv_img = mssv_block[0]
	if mssv_img.ndim == 3 and mssv_img.shape[2] == 3:
		mssv_img = cv2.cvtColor(mssv_img, cv2.COLOR_BGR2GRAY)
	h, w = mssv_img.shape[:2]

	row_h = h // 10  # vẫn chia 10 hàng

	# Tỉ lệ chiều ngang cho từng cột (dựa theo tỷ lệ bạn cung cấp)
	# ratios = [0.9] + [0.88] * 5 + [0.83] * 3  # tổng 9 phần
	ratios = [0.6, 0.6, 0.5, 0.45, 0.5, 0.65, 0.55, 0.5, 0.55]
	total_ratio = sum(ratios)
	normalized_ratios = [r / total_ratio for r in ratios]

	# Tính vị trí x1, x2 của từng cột bằng cộng dồn
	col_positions = [0]
	for ratio in normalized_ratios:
		col_positions.append(col_positions[-1] + int(ratio * w))

	# Cắt cột 1 → 8 (bỏ cột 0 hiển thị số)
	digits_matrix = []

	for col in range(1, 9):
		x1 = col_positions[col]
		x2 = col_positions[col + 1]
		digit_col = []

		for row in range(10):
			y1 = row * row_h
			y2 = (row + 1) * row_h
			cell = mssv_img[y1:y2, x1:x2]

			# print(f"Cell {col},{row} shape:", cell.shape)

			if cell is None or cell.size == 0 or cell.shape[0] < 5 or cell.shape[1] < 5:
				print(f" Skipping invalid cell {col},{row}")
				continue
			if cell.ndim == 3 and cell.shape[2] != 1:
				cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

			if cell.dtype != np.uint8:
				cell = cell.astype(np.uint8)
			cell = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
			digit_col.append(cell)

		digits_matrix.append(digit_col)
	# print("Cell shape:", cell.shape)
	# print("dtype:", cell.dtype)
	# Load model CNN
	model = CNN_Model('weight.h5').build_model(rt=True)

	mssv_digits = ""
	ma_de_digits = ""

	for col_idx, column in enumerate(digits_matrix):
		column_imgs = [cv2.resize(digit, (28, 28)).reshape(28, 28, 1) for digit in column]
		column_imgs = np.array(column_imgs) / 255.0
		if len(column_imgs) == 0:
			print(f"Bỏ qua cột {col_idx + 1} vì không có ảnh hợp lệ")
			if col_idx < 5:
				mssv_digits += "?"
			else:
				ma_de_digits += "?"
			continue
		preds = model.predict_on_batch(column_imgs)

		# Lấy số có xác suất tô cao
		selected = [i for i, p in enumerate(preds) if p[1] > 0.90]

		if len(selected) == 1:
			if col_idx < 5:
				mssv_digits += str(selected[0])
			else:
				ma_de_digits += str(selected[0])
		else:
			if col_idx < 5:
				mssv_digits += "?"
			else:
				ma_de_digits += "?"

	# Hiển thị toàn bộ ô MSSV để debug
	# if DEBUG:
	# 	for col_idx in range(8):
	# 		for row_idx in range(10):
	# 			cell = digits_matrix[col_idx][row_idx]
	# 			cv2.imshow(f"C{col_idx+1}_R{row_idx}", cell)
	# 			cv2.waitKey(1000)  # Hiển thị 100ms mỗi ô

	# 	cv2.destroyAllWindows()


	return mssv_digits, ma_de_digits

def show_mouse_position(event, x, y, flags, param):
	if event == cv2.EVENT_MOUSEMOVE:
		logging.debug(f"Mouse at: ({x}, {y})")

if __name__ == '__main__':
	file_name = sys.argv[1]
	img = cv2.imread(file_name)

	# Resize ảnh để hiển thị (không ảnh hưởtng xử lý)
	scale_percent = 30
	width = int(img.shape[1] * scale_percent / 100)
	height = int(img.shape[0] * scale_percent / 100)
	dim = (width, height)
	resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
	
	# Xử lý ảnh
	list_ans_boxes, warped_img, scale, M, mssv_block = determine_answer_blocks(img)
	mssv_digits, ma_de_digits = process_mssv_block(mssv_block[0])
	list_answers, list_positions = process_ans_blocks(list_ans_boxes)
	list_ans, choice_positions = process_list_ans(list_answers, list_positions)
	answers = get_answers(list_ans)
	logging.debug(answers)

	for idx, (x1, y1, x2, y2) in enumerate(choice_positions):
		question_number = idx // 4 + 1
		answer_letter = map_answer(idx)

		if answer_letter in answers.get(question_number, []):
			cv2.rectangle(warped_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
			cv2.putText(warped_img, answer_letter, (x1 + 3, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
			
	# In lên ảnh MSSV và mã đề
	cv2.putText(warped_img, f"MSSV: {mssv_digits}", (30, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
	cv2.putText(warped_img, f"Ma de: {ma_de_digits}", (30, 60),
				cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
	
	# Tạo đối tượng JSON tổng
	warped_img_jpg = cv2.imencode('.jpg', warped_img)
	output = {
		"mssv": mssv_digits,
		"ma_de": ma_de_digits,
		"answers": answers,
		"result_img": base64.b64encode(warped_img_jpg[1]).decode()
	}

	# In ra JSON (pretty format)
	print(json.dumps(output, ensure_ascii=False, indent=2))

	# Hiển thị ảnh đã xử lý
	if DEBUG:
		cv2.imshow("Answer Choices (scaled 1000)", warped_img)   
		cv2.waitKey(0)
		cv2.destroyAllWindows()