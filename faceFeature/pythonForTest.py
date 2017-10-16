import cv2
import math

img = cv2.imread("img/1.jpg",1)


sp = img.shape
print(sp)
ycrcb = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
print(img[0,0],ycrcb[0,0])
print(img[0,0,0],ycrcb[0,0,0])
for i in range(5,9):
    print(i)
print(i)


cv2.waitKey(0)
