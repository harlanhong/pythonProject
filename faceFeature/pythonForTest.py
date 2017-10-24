import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    import numpy as np
    import cv2
    from matplotlib import pyplot as plt

    img = cv2.imread('img/gpo6i.jpg',1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img.shape,img.shape[0],img.shape[1])
    x=0
    y=img.shape[1]-1
    sp= img.shape
    while x<img.shape[0] and y>=0:
        img[x,y] =(0,0,0)
        x+=1
        y-=1
    img[sp[1]-3,3] = (0,255,0)
    cv2.imshow("dfd",img)

    cv2.waitKey(0)