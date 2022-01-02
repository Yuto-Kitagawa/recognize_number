import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 画像のパス指定
img = cv2.imread('./number.jpg')

# 画像の入りの調整
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 画像にガウス(ぼやけ)をかける
img_gaus = cv2.GaussianBlur(img_gray, (1, 1), 5)

# 輪郭の抽出
img_sobel = cv2.Sobel(img_gaus, cv2.CV_8U, 1, 0, ksize=1)

# 輪郭の抽出
img_canny = cv2.Canny(img_sobel, 250, 100)

# しきい値を設定して、表示されているかされていないかを判断
i, img_shold = cv2.threshold(img_canny, 0, 255, cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
img_dilate = cv2.dilate(img_shold, kernel)

i, j = cv2.findContours(img_shold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

result = None

for i1 in i:
    x, y, w, h = cv2.boundingRect(i1)
    if w > 2*h:
        result = img[y:y+h, x:x+w]

# 画像を読み込み
plt.imshow(result)
# 画像を表示
plt.show()
sys.exit()
