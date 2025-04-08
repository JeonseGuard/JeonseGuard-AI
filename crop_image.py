import cv2

def is_similar(rect1, rect2, threshold = 10):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    return (
        abs(x1 - x2) < threshold and
        abs(y1 - y2) < threshold and
        abs(w1 - w2) < threshold and
        abs(h1 - h2) < threshold
    )

file_path = './data/images/rentContract.jpg'
image = cv2.imread(file_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(image = blurred, threshold1 = 100, threshold2 = 200)

contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

label = 1
unique_boxes = []
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    if w * h < 10000:
        continue
    curr = (x, y, w, h)
    if any(is_similar(curr, other) for other in unique_boxes):
        continue
    unique_boxes.append(curr)
    print(curr)
    cv2.putText(image, str(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (0, 0, 255), 2)
    label += 1
    #crop_image = image[y:y+h, x:x+w]
    #cv2.imwrite(f'./data/images/crop_{i}.jpg', crop_image)
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
print(len(unique_boxes))
cv2.imwrite('./data/images/canny.jpg', image)