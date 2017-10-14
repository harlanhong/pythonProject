import cv2
import numpy as np
def detectFaces(srcImg):
    img = srcImg.copy()
    #print 1
    face_cascade = cv2.CascadeClassifier("data/haarcascades/haarcascade_frontalface_default.xml")
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)#1.3and5  counts to the result of face recognation
    return faces

def detectEyes(srcImg):
    src = srcImg.copy()
    faces = detectFaces(src)
    eye_cascade = cv2.CascadeClassifier('data/haarcascades_cuda/haarcascade_eye_tree_eyeglasses.xml')
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    result = [[]]
    for j,(x,y,w,h) in enumerate(faces):
        cv2.rectangle(src, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = src[y:y + h,x:x + w]  # 检测视频中脸部的眼睛，并用vector保存眼睛的坐标、大小（用矩形表示）
        eyes = eye_cascade.detectMultiScale(roi_gray) #眼睛检测
        #选择最大的两个框当成眼睛
        max = 0
        temp = []
        index = 0
        for i,(ex,ey,ew,eh) in enumerate(eyes):
            if ew*eh>max and ex<w and ey<h/2:
                max = ew*eh
                index = i
                temp = (ex+x,ey+y,ew,eh)
        if index != 0:
            result[j].append(temp)
            cv2.rectangle(src, (temp[0], temp[1]), (temp[0] + temp[2], temp[1] + temp[3]), (0, 255, 0), 2)
        max = 0
        temp = []
        #用来确认是否找到眼睛
        flag = 0
        for i,(ex,ey,ew,eh) in enumerate(eyes):
            if ew*eh>max and i != index and ex<w and ey<h/2:
                print(ex,ey,x+w,y+h/2)
                max = ew*eh
                flag = 1
                temp = (ex+x,ey+y,ew,eh)
        if flag == 1:
            result[j].append(temp)
            cv2.rectangle(src, (temp[0], temp[1]), (temp[0] + temp[2], temp[1] + temp[3]), (0, 255, 0), 2)
    cv2.imshow("eye",src)
    return faces,result

def skinDetect(srcImg):
    img = srcImg.copy()
    img = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(img)
    result = np.zeros(img.shape, np.uint8)
    sp = img.shape
    #cr 77:127 cb 133:173
    for i in range(sp[0]):
        for j in range(sp[1]):
            if Cr[i,j]>=77 and Cr[i,j]<=127 and Cb[i,j]>=133 and Cb[i,j]<=173:
                result[i,j] = 255;
            else:
                result[i,j] = 0;
    return result

def skinModel(srcImg):
    img = srcImg.copy()
    rows, cols, channels = img.shape
    # light compensation
    gamma = 0.95
    for r in range(rows):
        for c in range(cols):
            # get values of blue, green, red
            B = img.item(r, c, 0)
            G = img.item(r, c, 1)
            R = img.item(r, c, 2)
            # gamma correction
            B = int(B ** gamma)
            G = int(G ** gamma)
            R = int(R ** gamma)
            # set values of blue, green, red
            img.itemset((r, c, 0), B)
            img.itemset((r, c, 1), G)
            img.itemset((r, c, 2), R)
            ###############################################################################
    # convert color space from rgb to ycbcr
    imgYcc = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    # convert color space from bgr to rgb
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # prepare an empty image space
    imgSkin = np.zeros(img.shape, np.uint8)
    # copy original image
    imgSkin = np.zeros(img.shape,np.uint8)
    ################################################################################
    # define variables for skin rules
    Wcb = 46.97
    Wcr = 38.76
    WHCb = 14
    WHCr = 10
    WLCb = 23
    WLCr = 20
    Ymin = 16
    Ymax = 235
    Kl = 125
    Kh = 188
    WCb = 0
    WCr = 0
    CbCenter = 0
    CrCenter = 0
    ################################################################################
    for r in range(rows):
        for c in range(cols):
            # non-skin area if skin equals 0, skin area otherwise
            skin = 0
            ########################################################################
            # color space transformation
            # get values from ycbcr color space
            Y = imgYcc.item(r, c, 0)
            Cr = imgYcc.item(r, c, 1)
            Cb = imgYcc.item(r, c, 2)
            if Y < Kl:
                WCr = WLCr + (Y - Ymin) * (Wcr - WLCr) / (Kl - Ymin)
                WCb = WLCb + (Y - Ymin) * (Wcb - WLCb) / (Kl - Ymin)
                CrCenter = 154 - (Kl - Y) * (154 - 144) / (Kl - Ymin)
                CbCenter = 108 + (Kl - Y) * (118 - 108) / (Kl - Ymin)
            elif Y > Kh:
                WCr = WHCr + (Y - Ymax) * (Wcr - WHCr) / (Ymax - Kh)
                WCb = WHCb + (Y - Ymax) * (Wcb - WHCb) / (Ymax - Kh)
                CrCenter = 154 + (Y - Kh) * (154 - 132) / (Ymax - Kh)
                CbCenter = 108 + (Y - Kh) * (118 - 108) / (Ymax - Kh)
            if Y < Kl or Y > Kh:
                Cr = (Cr - CrCenter) * Wcr / WCr + 154
                Cb = (Cb - CbCenter) * Wcb / WCb + 108
                ########################################################################
            # skin color detection
            if Cb > 77 and Cb < 127 and Cr > 133 and Cr < 173:
                skin = 1
                # print 'Skin detected!'
            if 0 == skin:
                imgSkin[r,c] = 0
            else:
                imgSkin[r,c] = 255

                # display original image and skin image
    cv2.imshow("imgSkin",imgSkin)
    return imgSkin
trackbar_name = "facethreshold"
win_name ="threshold"
def on_threshold(arg):
    value = cv2.getTrackbarPos(trackbar_name,win_name)
    imgsrc = cv2.threshold(img,value,255,cv2.THRESH_BINARY)
    cv2.imshow(win_name,imgsrc)
if __name__ == '__main__':
    img = cv2.imread("img/3.jpg",1)
    cv2.imshow("img",img)
    skin = skinModel(img)
    cv2.namedWindow(win_name)
    cv2.createTrackbar(trackbar_name,win_name,100,255,on_threshold)
    cv2.waitKey(0)