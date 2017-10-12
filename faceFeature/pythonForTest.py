import cv2


img = cv2.imread("img/1.jpg",1)
cv2.imshow("=img",img)
ret1, th1 = cv2.threshold(img, (10,10,10), (255,255,255), cv2.THRESH_BINARY_INV)
cv2.imshow("th",th1)
cv2.waitKey(0)
