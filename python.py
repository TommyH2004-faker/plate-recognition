from PIL import ImageFont, ImageDraw, Image
import numpy as np
from easyocr import Reader
import cv2

# -----------------------------
# 1. Load ảnh + resize
# -----------------------------
img = cv2.imread('./images/image11.jpg')
img = cv2.resize(img, (800, 600))

# -----------------------------
# 2. Load font
# -----------------------------
fontpath = "./arial.ttf"
font = ImageFont.truetype(fontpath, 32)
text_color = (0, 255, 0, 0)  # màu chữ

# -----------------------------
# 3. Hiển thị ảnh gốc
# -----------------------------
cv2.imshow("1. Ảnh gốc", img)
cv2.waitKey(0)

# -----------------------------
# 4. Chuyển sang grayscale
# -----------------------------
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("2. Ảnh xám", grayscale)
cv2.waitKey(0)

# -----------------------------
# 5. Gaussian Blur
# -----------------------------
blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
cv2.imshow("3. Làm mờ Gaussian", blurred)
cv2.waitKey(0)

# -----------------------------
# 6. Canny Edge Detection
# -----------------------------
edged = cv2.Canny(blurred, 10, 200)
cv2.imshow("4. Phát hiện biên (Canny)", edged)
cv2.waitKey(0)

# -----------------------------
# 7. Tìm contour
# -----------------------------
contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv2.contourArea, reverse=True)[:5]

contour_img = img.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
cv2.imshow("5. Các contour lớn nhất", contour_img)
cv2.waitKey(0)

# -----------------------------
# 8. Tìm hình chữ nhật biển số
# -----------------------------
number_plate_shape = None
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        number_plate_shape = approx
        break

# -----------------------------
# 9. Cắt ROI biển số
# -----------------------------
(x, y, w, h) = cv2.boundingRect(number_plate_shape)
number_plate = grayscale[y:y+h, x:x+w]
cv2.imshow("6. Ảnh cắt biển số", number_plate)
cv2.waitKey(0)

# -----------------------------
# 10. OCR
# -----------------------------
reader = Reader(['en'])
detection = reader.readtext(number_plate)

# -----------------------------
# 11. Hiển thị kết quả cuối cùng
# -----------------------------
img_result = img.copy()
if len(detection) == 0:
    text = "Không thấy bảng số xe"
else:
    text = "Biển số: " + detection[0][1]

# Vẽ contour biển số
cv2.drawContours(img_result, [number_plate_shape], -1, (255, 0, 0), 3)

# Hiển thị text bằng PIL như code cũ
img_pil = Image.fromarray(img_result)
draw = ImageDraw.Draw(img_pil)
draw.text((200, 500), text, font=font, fill=text_color)
img_result = np.array(img_pil)

cv2.imshow("7. Kết quả dự đoán + khung biển số", img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
