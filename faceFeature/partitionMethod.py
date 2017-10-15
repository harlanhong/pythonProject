# encoding:utf-8
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
    #cv2.imshow("eye",src)
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
    cv2.imshow("skin",imgSkin)
    return imgSkin

def removeBackground(srcImg):
    img = srcImg.copy()
    sp = img.shape
    mask = np.zeros((img.shape[0]+2,img.shape[1]+2),np.uint8)
    cv2.floodFill(img, mask, (5, 5), (255, 255, 255), (5, 5, 5), (5, 5, 5), 8)
    #mask = np.zeros((img.shape[0]+2,img.shape[1]+2),np.uint8)
    #cv2.floodFill(img, mask, (5,img.shape[0]-5), (255, 255, 255), (3, 3, 3), (3, 3, 3), 8)
    cv2.imshow("floodfill", img)
    return img

def tailorImg(img,faces):
    if len(faces) == 1:
        for i,(x,y,w,h) in enumerate(faces):
            #注意不要越界
            new_x = x - int(w/3) if x - int(w/3)>0 else 0
            new_y = y - int(h/3) if y - int(h/3)>0 else 0
            new_w = w +int(2/3*w) if new_x+w+int(2/3*w)<img.shape[1] else img.shape[1]-new_x
            new_h = h +int(2/3*h) if new_y+h +int(2/3*h)<img.shape[0] else img.shape[0]-new_y
            new_img = img[new_y:new_y+new_h,new_x:new_x+new_w]
            #cv2.imshow("new_img",new_img)
            return new_img

def meanBrightness(imgSKIN,imgGRAY):
    sp = imgSKIN.shape
    print(sp)
    brightness = 0
    count = 0
    for i in range(sp[0]):
        for j in range(sp[1]):
           if imgSKIN[i,j] >100:
               brightness += imgGRAY[i,j]
               count +=1
    return brightness/count

def skeletonComplete(imgResult,imgSKIN):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    imgSKIN = cv2.erode(imgSKIN, kernel)
    imgSKIN = cv2.GaussianBlur(imgSKIN, (7, 7), 0)  # 高斯平滑处理原图像降噪
    imgSKIN = cv2.dilate(imgSKIN, kernel)
    ret,new_skin = cv2.threshold(imgSKIN,100,255,cv2.THRESH_BINARY_INV)
    cv2.imshow("process_skin",new_skin)
    #binary, contours, hierarchy = cv2.findContours(new_skin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #cv2.drawContours(binary, contours, -1, (0, 0, 255), 3)
    #cv2.imshow("inver",binary)
    img = cv2.GaussianBlur(new_skin, (3, 3), 0)  # 高斯平滑处理原图像降噪
    canny = cv2.Canny(img, 50, 150)  # apertureSize默认为3
    ret,skeleton = cv2.threshold(canny,100,255,cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    skeleton = cv2.erode(skeleton, kernel)
    skeleton = cv2.erode(skeleton, kernel)
    result = cv2.bitwise_and(skeleton,imgResult)
    cv2.imshow("result",result)
    return result

def processImg(img):
    img = cv2.resize(img,(int(img.shape[1]/2) ,int(img.shape[0]/2)),interpolation=cv2.INTER_CUBIC)
    cv2.imshow("img",img)
    imgFront = removeBackground(img)
    #眼睛和人脸检测
    faces,eyes = detectEyes(img)
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #获取到的人物头像前景图
    imgFace = tailorImg(imgFront,faces)
    #获取imgFace的肤色图
    imgFace_Skin = skinModel(imgFace)
    imgFace_Gray = cv2.cvtColor(imgFace,cv2.COLOR_BGR2GRAY)
    #获取到作阈值化的阈值
    theta = meanBrightness(imgSKIN=imgFace_Skin,imgGRAY=imgFace_Gray)
    #进行阈值化
    cv2.imshow("imgFace_gray",imgFace_Gray)
    ret,imgFace_thresh = cv2.threshold(imgFace_Gray,0.7*int(theta),255,cv2.THRESH_BINARY)
    cv2.imshow("face_threshold",imgFace_thresh)
    #轮廓补充
    result = skeletonComplete(imgFace_thresh,imgSKIN=imgFace_Skin)
    return result

def RemoveSmallRegion(src,AreaLimit,CheckMode,NeiborMode):
    RemoveCount = 0;
    # 新建一幅标签图像初始化为0像素点，为了记录每个像素点检验状态的标签，0代表未检查，1代表正在检查, 2代表检查不合格（需要反转颜色），3代表检查合格或不需检查
    # 初始化的图像全部为0，未检查
    PointLabel = np.zeros(src.shape,np.uint8)
    sp = src.shape
    if(CheckMode == 1):#去除小连通区域的白色点
        print("去除小连通域")
        for i in range(sp[0]):
            for j in range(sp[1]):
                if src[i,j]<10:
                    PointLabel[i,j] = 3#将背景黑色点标记为合格，像素为3
    else:
        print("去除孔洞")
        for i in range(sp[0]):
            for j in range(sp[1]):
                if src[i,j]>10:
                    PointLabel[i,j] = 3#如果原图是白色区域，标记为合格，像素为3
    NeiborPos = [] #将邻域压进容器
    NeiborPos.append((-1,0))
    NeiborPos.append((1, 0))
    NeiborPos.append((0, -1))
    NeiborPos.append((0, 1))
    if NeiborMode ==1:
        print("Neighbor mode: 8邻域.")
        NeiborPos.append((-1, -1))
        NeiborPos.append((-1, 1))
        NeiborPos.append((1, -1))
        NeiborPos.append((1, 1))
    else:
        print("Neighbor mode: 4邻域.")
    NeihborCount = 4+4*NeiborMode;
    CurrX = 0
    CurrY =0
    for i in range(sp[0]):
        for j in range(sp[1]):
            if PointLabel[i,j] == 0:
                GrowBuffer = []
                GrowBuffer.append((j,i));
                PointLabel[i,j] = 1
                CheckResult = 0
                for z in range(len(GrowBuffer)):
                    for q in range(NeihborCount):
                        CurrX = GrowBuffer[z][0]+NeiborPos[q][0]
                        CurrY = GrowBuffer[z][1]+NeiborPos[q][1]
                        if CurrX>=0 and CurrX<sp[1] and CurrY >=0 and CurrY<sp[0]:
                            if PointLabel[CurrY,CurrX] == 0:
                                GrowBuffer.append((CurrX,CurrY))
                                PointLabel[CurrY,CurrX] = 1


if __name__ == '__main__':
    i = 1
    img = cv2.imread("img/5.jpg",1)

    processImg(img)
    cv2.waitKey(0)