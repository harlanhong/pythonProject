import cv2
import numpy as np

if __name__ == '__main__':
    img = cv2.imread("img/1.png",1)
    sp = img.shape
    cv2.imshow("src",img)
    mask = np.zeros((img.shape[0]+2,img.shape[1]+2),np.uint8)
    cv2.floodFill(img, mask, (5, 5), (255, 255, 255), (5, 5, 5), (5, 5, 5), 8)
    cv2.floodFill(img, mask, (sp[1]-1, sp[0]-1), (255, 255, 255), (5, 5, 5), (5, 5, 5), 8)
    cv2.imshow("floodfill", img)
    cv2.waitKey(0)