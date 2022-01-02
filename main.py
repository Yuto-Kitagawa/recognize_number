import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pyocr
import pyocr.builders
import argparse
from PIL import Image

###########################
# 文字認識するためにパスを通す必要がある
import os
# インストール済みのTesseractへパスを通す
path_tesseract = "C:\\Program Files\\Tesseract-OCR"
if path_tesseract not in os.environ["PATH"].split(os.pathsep):
    os.environ["PATH"] += os.pathsep + path_tesseract
##########################################

img = cv2.imread('./number.jpg')

# 画像を白黒にする
img_gray_cascade = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.cvtColor(img_gray_cascade, cv2.COLOR_BGR2RGB)

# ブラシ(ぼかし)をかける
img_blur = cv2.bilateralFilter(img_gray, 11, 17, 17)

# 輪郭を抽出
img_canny = cv2.Canny(img_blur, 30, 200)


# Numpyは多次元配列の数値計算は早いが各次元のサイズや方がそろっていることが前提。
# VisibleDeprecationWarning: Creating an ndarray from ragged nestedはその非推奨に対するエラー文

# これは全体の輪郭を実施
# contours, img_edge_cascade = cv2.findContours(
#     img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
# )

# これは画像の輪郭を抽出下部分を実施
contours, img_edge_cascade = cv2.findContours(
    img_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# なぜかエラーが出るので回避
# img_edge_cascade_second = imutils.grab_contours(img_edge_cascade)
# img_edge = sorted(img_edge_cascade_second,
#                   key=cv2.contourArea, reverse=True)[:10]

# ラムダ関数
# 小さい輪郭は誤検出として削除する
contours = list(filter(lambda x: cv2.contourArea(x) > 100, contours))

# 輪郭を描画する。
cv2.drawContours(img, contours, -1, color=(0, 0, 255), thickness=2)

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break


# mask = np.tile(img_gray.shape, np.uint8(0))
mask = np.zeros(img_gray.shape, np.uint8(0))
new_img = cv2.drawContours(mask, [location], 0, 255, -1)
# マスクのサイズと画像のサイズが同じじゃないとエラーを吐く
# new_img = cv2.bitwise_and(img, img, mask=mask)

# 三項だと引数が二個だけだが、maskを見てみると三つ格納されているので引数を三つにしている
(x, y, z) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_img = img_gray[x1:x2+1, y1:y2+1]

# WindowsのCuda(easyocr)が動かない(インストールとかなんとかめんどくさい)のでできない
# reader = easyocr.Reader(['ja', 'en'])
# result = reader.readtext(cropped_img)

# ocrで実行
parser = argparse.ArgumentParser(description='tesseract ocr test')
parser.add_argument(cv2.cvtColor(
    cropped_img, cv2.COLOR_BAYER_BG2BGR), help='image path')
args = parser.parse_args()
builder = pyocr.builders.TextBuilder(tesseract_layout=6)
# 実行結果を格納する配列を宣言
tools = pyocr.get_available_tools()
tool = tools[0]

# result = tool.image_to_string(cv2.cvtColor(
#     cropped_img, cv2.COLOR_BAYER_BG2BGR), lang="jpn", builder=builder)

# plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
# plt.show()
sys.exit()
