#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import sys
import imutils
import numpy as np
import cv2
import logging
from math import ceil
from model import CNN_Model
from collections import defaultdict

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

def get_contour_corners(contour):
	peri = cv2.arcLength(contour, True)
	approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
	if len(approx) == 4:
		return approx.reshape(4, 2)
	return None

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

def detect_and_warp(image, low_thresh=20, high_thresh=80):
	orig_h, orig_w = image.shape[:2]
	target_h = 500
	ratio = orig_h / target_h
	new_w = int(orig_w / ratio)
	img = cv2.resize(image, (new_w, target_h))

	# Preprocessing
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, low_thresh, high_thresh)

	cnts, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(cnts, key=lambda c: cv2.contourArea(c), reverse=True)[:5]

	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		if len(approx) == 4 and cv2.contourArea(approx) > 10000:
			pts = approx.reshape(4, 2) * ratio
			return pts  # Trả về 4 điểm giấy A4 (trên ảnh gốc)

	return None

def is_well_aligned(corners, img_shape, tolerance=10):
	(h, w) = img_shape[:2]
	ordered = order_corners(corners)
	expected_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
	diffs = np.abs(ordered - expected_corners)
	return np.all(diffs < tolerance)

def scale_image(img, target_height=1100):
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

def is_scanned_image(img, verbose=DEBUG):
	h, w = img.shape[:2]
	ratio = w / h
	is_ratio_ok = abs(ratio - 0.707) < 0.06  # KHÔNG dùng như điều kiện chính nữa

	# Convert về gray nếu là ảnh màu
	if len(img.shape) == 3 and img.shape[2] == 3:
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		is_gray = False
	else:
		gray = img.copy()
		is_gray = True

	# Viền sáng
	border_pixels = np.concatenate([
		gray[:10, :].flatten(),
		gray[-10:, :].flatten(),
		gray[:, :10].flatten(),
		gray[:, -10:].flatten(),
	])
	border_brightness = np.mean(border_pixels)
	is_white_border = border_brightness > 200

	# Độ nhiễu thấp
	std_dev = np.std(gray)
	is_low_noise = std_dev < 70

	# ❗ Đổi cách tính: chỉ cần 2 trong 2 đặc trưng chính → SCANNED
	result = is_white_border and is_low_noise

	if verbose:
		logging.debug("🔍 [Scan Detection Debug]")
		logging.debug(f"- Ratio = {ratio:.4f} → {'OK' if is_ratio_ok else 'NO'}")
		logging.debug(f"- Gray image? → {is_gray}")
		logging.debug(f"- Border brightness = {border_brightness:.2f} → {'OK' if is_white_border else 'NO'}")
		logging.debug(f"- Gray std deviation = {std_dev:.2f} → {'OK' if is_low_noise else 'NO'}")
		logging.debug(f"=> ✅ Final result: {'SCANNED' if result else 'NOT SCANNED'}")
		logging.debug(f"==> RETURN VALUE: {result}")

	return result


def determine_answer_blocks(img):
	if DEBUG:
		logging.debug("=== Bắt đầu kiểm tra ảnh ===")

	if not is_scanned_image(img):  # CHẮC CHẮN PHẢI DÙNG not
		logging.debug("→ Ảnh chưa scan → gọi detect_and_warp")
		paper_pts = detect_and_warp(img)
		if paper_pts is not None:
			img, M = perspective_transform(img, paper_pts)
		else:
			raise ValueError("Không tìm thấy giấy A4.")
	else:
		logging.debug("→ Ảnh đã scan → bỏ qua bước detect")
		M = None

	img, scale = scale_image(img)

	warped_img = img.copy()
	# cv2.imshow("Warped Paper", warped_img)
	# cv2.waitKey(0)

	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
	img_canny = cv2.Canny(blurred, 50, 150)

	cnts = cv2.findContours(img_canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	# # 4. Tìm 5 contour lớn nhất
	# cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

	# img_contours = img.copy()

	# for i, c in enumerate(cnts):
	#     x, y, w, h = cv2.boundingRect(c)
	#     area = cv2.contourArea(c)
	#     cv2.rectangle(img_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)
	#     cv2.putText(img_contours, f"#{i+1} ({w}x{h}) {area:.0f}", (x, y - 10),
	#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

	#     logging.debug(f"Contour {i+1}: x={x}, y={y}, w={w}, h={h}, area={area:.0f}")

	# # Show hình có 5 contour lớn nhất
	# cv2.imshow("Top 5 Contours", img_contours)
	# cv2.imshow("Canny edges", img_canny)
	# cv2.waitKey(0)

	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)


	ans_blocks = []
	mssv_block = []

	# Đây là 3 block mẫu mới
	ref_blocks = [
	(543, 331, 214, 700),
	(293, 332, 215, 700),
	(51, 332, 212, 700)]

	ref_blocks2 = (440, 33, 241, 337)

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
		if is_similar(x, y, w, h, *ref_blocks2, tolerance=0.2, min_h=200):
			mssv_block.append((enhanced[y:y + h, x:x + w], [x, y, w, h]))

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
				

	return sorted_ans_blocks, warped_img, scale, M, mssv_block\

def process_ans_blocks(ans_blocks):
	list_answers = []
	list_positions = []

	for block_idx, ans_block in enumerate(ans_blocks):
		ans_block_img = np.array(ans_block[0])
		block_x, block_y, block_w, block_h = ans_block[1]

		vis_img = cv2.cvtColor(ans_block_img.copy(), cv2.COLOR_GRAY2BGR)

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
				# logging.debug(f'Block {block_idx + 1}, Group {i + 1}, Answer {j + 1}: Position = ({x1}, {y1}, {x2}, {y2})')
				
	#             cv2.rectangle(vis_img, (0, j * offset2), (ans_block_img.shape[1] - 1, (j + 1) * offset2), (255, 0, 0), 1)

	#     cv2.imshow(f'Block {block_idx + 1} - Split lines', vis_img)
	
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	# logging.debug(list_positions)
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
		if score[1] > 0.5:  # choiced confidence score > 0.9
			chosed_answer = map_answer(idx)
			results[question + 1].append(chosed_answer)

	return results

def process_mssv_block(mssv_block):
	mssv_img = mssv_block[0]
	h, w = mssv_img.shape[:2]

	row_h = h // 10  # vẫn chia 10 hàng

	# Tỉ lệ chiều ngang cho từng cột (dựa theo tỷ lệ bạn cung cấp)
	ratios = [0.9] + [0.88] * 5 + [0.83] * 3  # tổng 9 phần
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
			cell = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
			digit_col.append(cell)

		digits_matrix.append(digit_col)

	# Load model CNN
	model = CNN_Model('weight.h5').build_model(rt=True)

	mssv_digits = ""
	ma_de_digits = ""

	for col_idx, column in enumerate(digits_matrix):
		column_imgs = [cv2.resize(digit, (28, 28)).reshape(28, 28, 1) for digit in column]
		column_imgs = np.array(column_imgs) / 255.0
		preds = model.predict_on_batch(column_imgs)

		# Lấy số có xác suất tô cao
		selected = [i for i, p in enumerate(preds) if p[1] > 0.5]

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
	if DEBUG:
		fig, axes = plt.subplots(10, 8, figsize=(12, 15))  # 10 hàng, 8 cột
		fig.suptitle("Các ô trong block MSSV", fontsize=16)

		for col_idx in range(8):
			for row_idx in range(10):
				ax = axes[row_idx, col_idx]
				ax.imshow(digits_matrix[col_idx][row_idx], cmap='gray')
				ax.axis('off')
				ax.set_title(f"C{col_idx+1} R{row_idx}", fontsize=6)

		plt.tight_layout()
		plt.subplots_adjust(top=0.95)
		plt.show()

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
	# logging.debug(answers)

	for idx, (x1, y1, x2, y2) in enumerate(choice_positions):
		question_number = idx // 4 + 1
		answer_letter = map_answer(idx)

		if answer_letter in answers.get(question_number, []):
			cv2.rectangle(warped_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
			cv2.putText(warped_img, answer_letter, (x1 + 3, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
			
	# In lên ảnh MSSV và mã đề
	cv2.putText(warped_img, f"MSSV: {mssv_digits}", (30, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
	cv2.putText(warped_img, f"Ma de: {ma_de_digits}", (30, 60),
				cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
	
	# Tạo đối tượng JSON tổng
	output = {
		"mssv": mssv_digits,
		"ma_de": ma_de_digits,
		"answers": answers
	}

	# In ra JSON (pretty format)
	print(json.dumps(output, ensure_ascii=False, indent=2))

	# Hiển thị ảnh đã xử lý
	if DEBUG:
		cv2.imshow("Answer Choices (scaled 1000)", warped_img)   
		cv2.waitKey(0)
		cv2.destroyAllWindows()