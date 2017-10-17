import cv2
import math
import numpy as np

def delete_jut(src,uthreshold,vthreshold,type):
    threshold = 0
    dst = src.copy()
    sp = src.shape
    height = sp[0]
    width = sp[1]
    k = 0
    mode = (1-type)*255
    modeInver = 255-mode
    cont = 1;
    for i in range(height-1):
        for j in range(width-1):
            cont +=1
            #行消除
            if dst[i,j] == mode and dst[i,j+1] == modeInver:
                print(i, j,cont)
                if j+uthreshold >= width:
                    for k in range(j+1,width):
                        dst[i,k] = mode
                else:
                    for k in range(j+2,j+uthreshold+1):
                        if dst[i,k] == mode:
                            break
                    if dst[i,k]  == mode:
                        for h in range(j+1,k):
                            dst[i,h] = mode
            #列消除
            if dst[i,j] == mode and dst[i+1,j] == modeInver:

                if i+vthreshold >= height:
                    for k in range(i+1,height):
                        dst[k,j] = mode
                else:
                    for k in range(i+2,i+vthreshold+1):
                        if dst[k,j] == mode:
                            break
                    if dst[k,j] == mode:

                        for h in range(i+1,k):
                            dst[h,j] = mode
    return dst

if __name__ == '__main__':
    img = np.zeros((256,256),np.uint8)
    img[50,0:255] = 255
    img[0:255,100] = 255
    cv2.imshow("img",img)
    img = delete_jut(img,5,5,1)
    cv2.imshow("imgsrc",img)


    cv2.waitKey(0)
