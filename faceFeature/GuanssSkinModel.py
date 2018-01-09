import cv2
import numpy as np

def P(img,u,s):
    result = np.zeros((img.shape[0],img.shape[1]))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
    sp = img.shape
    for i in range(sp[0]):
        for j in range(sp[1]):
            x = np.array([[img[i,j,2]],[img[i,j,1]]])
            x = np.matrix(x)
            result[i, j]=np.exp(-0.5*(x-u).T*s.I*(x-u))
    cv2.imshow("fdfd",result)

if __name__ == '__main__':
    u = np.array([[117.4361],[156.5599]])
    u = np.matrix(u)
    s = np.array([[160.1301,12.1430],[12.1430,299.4574]])
    s = np.matrix(s)
    img = cv2.imread("skeleton/41.jpg")
    P(img,u,s)
    print(type(u))
    cv2.waitKey(0)