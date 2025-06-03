import cv2

def get_binary_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
    return binary_image

def separate_parts(image):
    binary_image = get_binary_image(image)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < 500000:
            continue
        boxes.append((x, y, w, h))
    boxes = sorted(boxes, key = lambda box: box[1])
    
    parts = []
    for box in boxes:
        x, y, w, h = box
        parts.append(image[y:y+h, x:x+w])
    
    x = boxes[0][0]
    y = boxes[0][1] + boxes[0][3]
    w = boxes[0][2]
    h = boxes[1][1] - y
    parts.append(image[y:y+h, x:x+w])

    part_1 = parts[0]
    part_2 = parts[1]
    part_3 = parts[2]
    part_1_parts = separate_part_1(part_1)
    part_2_parts = separate_part_2(part_2)
    part_3_parts = separate_part_3(part_3)

    return part_1_parts, part_2_parts, part_3_parts

def separate_part_1(part_1):
    binary_image = get_binary_image(part_1)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < 1500:
            continue
        boxes.append((x, y, w, h))
    boxes = sorted(boxes, key = lambda box: (box[1], box[0]))
    
    needed_idx = [3, 8, 11, 13, 15, 17, 20, 22, 24, 26]
    boxes = [boxes[idx] for idx in needed_idx]
    
    parts = []
    for box in boxes:
        x, y, w, h = box
        parts.append(part_1[y:y+h, x:x+w])

    return parts

def separate_part_2(part_2):
    binary_image = get_binary_image(part_2)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < 1500:
            continue
        boxes.append((x, y, w, h))
    boxes = sorted(boxes, key = lambda box: (box[1], box[0]))
    
    needed_idx = [3, 6, 8, 10, 20, 23, 25, 27]
    boxes = [boxes[idx] for idx in needed_idx]

    parts = []
    for box in boxes:
        x, y, w, h = box
        parts.append(part_2[y:y+h, x:x+w])

    return parts

def separate_part_3(part_3):
    binary_image = get_binary_image(part_3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1500, 30))
    dilated_image = cv2.dilate(binary_image, kernel, iterations = 3)
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, w, h))
    boxes = sorted(boxes, key = lambda box: box[1])
    
    parts = []
    for box in boxes:
        x, y, w, h = box
        parts.append(part_3[y:y+h, x:x+w])

    return parts