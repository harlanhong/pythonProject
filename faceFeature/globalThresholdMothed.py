import cv2
import math
import numpy as np
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
    imgSkin = np.zeros((img.shape[0],img.shape[1]),np.uint8)
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
    skinCounter=0
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
            #if Cb >= 80 and Cb <= 127 and Cr >= 141 and Cr <= 173:
            if Cb >= 80 and Cb <= 127 and Cr >= 133 and Cr <= 177:
                skin = 1
                # print 'Skin detected!'
            if 0 == skin:
                imgSkin[r,c] = 0
            else:
                imgSkin[r,c] = 255
                skinCounter+=1
                # display original image and skin image
    return skinCounter,imgSkin
def globalThreshod(inputImg):
    face_cascade = cv2.CascadeClassifier("data/haarcascades/haarcascade_frontalface_alt2.xml")
    Y=[0]
    #inputImg = cv2.resize(inputImg,(int(inputImg.shape[1]/2),int(inputImg.shape[0]/2)))
    ycrcb = cv2.cvtColor(inputImg,cv2.COLOR_RGB2YCrCb)
    #保存亮度通道
    gr = ycrcb[:,:,0]
    faces = face_cascade.detectMultiScale(gr, 1.3, 5)
    if len(faces)<1:
        return cv2.cvtColor(inputImg,cv2.COLOR_BGR2GRAY)
    facemax = -1
    for i, (x, y, w, h) in enumerate(faces):
        if w*h>facemax:
            facemax=w*h
            ansface = [x,y,w,h]
    avgback = 0
    s =1;ct =1
    for i in range(ansface[1]):
        for j in range(gr.shape[1]):
            s+=gr[i,j]
            ct+=1
    avgback = s/ct
    mask1 = np.zeros((inputImg.shape[0] + 2, inputImg.shape[1] + 2), np.uint8)
    cv2.floodFill(inputImg,mask1,(ansface[0],ansface[1]),(255, 255, 255), (3, 3, 3), (3, 3, 3))

    ycrcb = cv2.cvtColor(inputImg,cv2.COLOR_RGB2YCrCb)
    # for i in range(ycrcb.shape[0]):
    #     for j in range(ycrcb.shape[1]):
    #         cr = ycrcb[i,j,1]
    #         cb = ycrcb[i,j,2]
    #         if 77<=cr and cr<=127 and 133<=cb and cb<=173:
    #             inputImg[i,j]=(255,255,255)
    #         else:
    #             inputImg[i, j]=(0,0,0)
    skinCounter,skin = skinModel(inputImg)

    for i in range(skin.shape[0]):
        for j in range(skin.shape[1]):
            if skin[i,j]>100:
                inputImg[i, j] = (255, 255, 255)
            else:
                inputImg[i, j] = (0, 0, 0)
    resolution = 1.6
    ansface[0] = int(ansface[0]-ansface[2]*((resolution-1)/2))
    ansface[1] = int(ansface[1]-ansface[3]*((resolution-1)/2))
    ansface[2] = int(ansface[2]*resolution)
    ansface[3] = int(ansface[3]*resolution*0.85)
    if ansface[0]<0:
        ansface[2]=ansface[2]+ansface[0]
        ansface[0]=0
    if ansface[1]<0:
        ansface[3]=ansface[3]+ansface[1]
        ansface[1]=0
    if ansface[2]+ansface[0]>=inputImg.shape[1]:
        ansface[2]=inputImg.shape[1]-ansface[0]-1
    if ansface[3]+ansface[1]>=inputImg.shape[0]:
        ansface[3] = inputImg.shape[0]-ansface[1]-1
    print(ansface)
    inputROI = inputImg[ansface[1]:ansface[1]+ansface[3],ansface[0]:ansface[0]+ansface[2]]
    grROI = gr[ansface[1]:ansface[1]+ansface[3],ansface[0]:ansface[0]+ansface[2]]
    cv2.imshow("inputROI",inputROI)
    cv2.imshow("grROI",grROI)
    temp = cv2.cvtColor(inputROI,cv2.COLOR_RGB2GRAY)
    facebin = cv2.cvtColor(inputROI,cv2.COLOR_RGB2GRAY)
    binary, contours, hierarchy = cv2.findContours(temp, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    temp = grROI.copy()
    contour_ok = 0
    for c in range(len(contours)-1,-1,-1):
        result = cv2.convexHull(contours[c])
        if cv2.pointPolygonTest(result,(ansface[3]/2,ansface[2]/2),0)==1:
            contour_ok = 1
            py =0;s=0;ct =1;
            for i in range(temp.shape[0]):
                for j in range(temp.shape[1]):
                    if cv2.pointPolygonTest(result,(i,j),0)==1:
                        if facebin[i,j]:
                            s = s+temp[i,j]
                            ct+=1
                            Y[py] = temp[i,j]
                            Y.append(0)
                            py+=1
            Y.sort()
            out = [0]*300
            for i in range(Y.__len__()):
                out[Y[i]]+=1
            avg = s/ct
            mask = cv2.Canny(temp,avgback,avgback*2)
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.dilate(mask,element)
            cv2.imshow("cannyMask",mask)
            ret,temp = cv2.threshold(temp,avgback*avgback*0.9/1000+0.9*(avg+Y[int(py/2)])*(avg+Y[int(py/2)])/1000,255,cv2.THRESH_BINARY)

            for i in range(temp.shape[0]):
                for j in range(temp.shape[1]):
                    if cv2.pointPolygonTest(result,(i,j),0)!=1:
                        if temp[i,j]>avgback:
                            temp[i,j]=255
                        else:
                            temp[i,j]=0
                    if mask[i,j]:
                        temp[i,j]=0
            break
    if(contour_ok == 0):
        py=0
        s=0
        ct =1
        for i in range(temp.shape[0]):
            for j in range(temp.shape[1]):
                if facebin[i,j]:
                    s=s+temp[i,j]
                    ct+=1
                    Y[py] = temp[i, j]
                    Y.append(0)
                    py+=1
        Y.sort()
        out=[0]*300
        for i in range(Y.__len__()):
            out[Y[i]]+=1
        avg=s/ct
        mask = cv2.Canny(temp, avgback, avgback * 2)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.dilate(mask, element)
        cv2.imshow("cannyMask", mask)
        ret, temp = cv2.threshold(temp, avgback * avgback * 0.9 / 1000 + 0.9 * (avg + Y[int(py / 2)]) * (
        avg + Y[int(py / 2)]) / 1000, 255, cv2.THRESH_BINARY)
    temp = cv2.medianBlur(temp,3)
    return temp

if __name__ == '__main__':
    img = cv2.imread("imageTailor/1 (156).jpg")
    result = globalThreshod(img)
    cv2.imshow("result",result)
    cv2.waitKey(0)










