import cv2
import os

def get_binary_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
    return binary_image

def separate_parts(image_path):
    image_name = os.path.basename(image_path).split('.')[0]
    save_path = f'./data/images/{image_name}'
    os.makedirs(save_path, exist_ok = True)
    
    image = cv2.imread(image_path)
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
    
    for idx, box in enumerate(boxes):
        x, y, w, h = box
        cv2.imwrite(f'{save_path}/part_{idx+1}.jpg', image[y:y+h, x:x+w])
    
    x = boxes[0][0]
    y = boxes[0][1] + boxes[0][3]
    w = boxes[0][2]
    h = boxes[1][1] - y
    cv2.imwrite(f'{save_path}/part_{idx+2}.jpg', image[y:y+h, x:x+w])

    part_1_path = f'{save_path}/part_1.jpg'
    part_2_path = f'{save_path}/part_2.jpg'
    part_3_path = f'{save_path}/part_3.jpg'
    separate_part_1(part_1_path)
    separate_part_2(part_2_path)
    separate_part_3(part_3_path)

def separate_part_1(part_1_path):
    save_path = os.path.dirname(part_1_path)

    part_1 = cv2.imread(part_1_path)
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
    
    for idx, box in enumerate(boxes):
        x, y, w, h = box
        cv2.imwrite(f'{save_path}/part_1_{idx+1}.jpg', part_1[y:y+h, x:x+w])

def separate_part_2(part_2_path):
    save_path = os.path.dirname(part_2_path)

    part_2 = cv2.imread(part_2_path)
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

    for idx, box in enumerate(boxes):
        x, y, w, h = box
        cv2.imwrite(f'{save_path}/part_2_{idx+1}.jpg', part_2[y:y+h, x:x+w])

def separate_part_3(part_3_path):
    save_path = os.path.dirname(part_3_path)
    
    part_3 = cv2.imread(part_3_path)
    binary_image = get_binary_image(part_3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1500, 20))
    dilated_image = cv2.dilate(binary_image, kernel, iterations = 3)
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, w, h))
    boxes = sorted(boxes, key = lambda box: box[1])
    '''
    x, y, w, h = boxes.pop(0)
    height = [0, 90, 172, 247, 360, 435, h]
    for idx in range(len(height)-1):
        boxes.append((x, y+height[idx], w, height[idx+1]-height[idx]))
    boxes = sorted(boxes, key = lambda box: box[1])    
    '''
    for idx, box in enumerate(boxes):
        x, y, w, h = box
        cv2.imwrite(f'{save_path}/part_3_{idx+1}.jpg', part_3[y:y+h, x:x+w])
