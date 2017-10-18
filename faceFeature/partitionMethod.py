# encoding:utf-8
import cv2
import numpy as np
import math
#检测人脸
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
#在人脸的基础上检测眼睛
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
#皮肤检测。。。。。实践不行
def skinEllipse(srcImg):
    img = srcImg.copy()

    sp = img.shape
    model = np.zeros((256,256),np.uint8)
    result = np.zeros((sp[0],sp[1]),np.uint8)
    cv2.ellipse(model, (113, 155), (25, 17), 43, 0, 360, 255,-1)
    imgYcrcb = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
    for i in range(sp[0]):
        for j in range(sp[1]):
            #if model[imgYcrcb[i,j,1],imgYcrcb[i,j,2]] == 255:
            if imgYcrcb[i,j,1]>30 and imgYcrcb[i,j,1]<115 and imgYcrcb[i,j,2]>90 and imgYcrcb[i,j,2]<240:
                result[i,j]=255
            else:
                result[i,j]=0
    cv2.imshow("ellipseSkin",result)
    return result
#肤色检测
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
            #if Cb >= 80 and Cb <= 127 and Cr >= 141 and Cr <= 173:
            if Cb >= 77 and Cb <= 127 and Cr >= 140and Cr <= 175:
                skin = 1
                # print 'Skin detected!'
            if 0 == skin:
                imgSkin[r,c] = 0
            else:
                imgSkin[r,c] = 255
                # display original image and skin image
    cv2.imshow("skin",imgSkin)
    return imgSkin
#通过漫水填充去除背景
def removeBackground(srcImg):
    img = srcImg.copy()
    sp = img.shape
    mask = np.zeros((img.shape[0]+2,img.shape[1]+2),np.uint8)
    cv2.floodFill(img, mask, (5, 5), (255, 255, 255), (4.5, 4.5, 4.5), (3.5, 3.5, 3.5), 8)
    #mask = np.zeros((img.shape[0]+2,img.shape[1]+2),np.uint8)
    #cv2.floodFill(img, mask, (5,img.shape[0]-5), (255, 255, 255), (3, 3, 3), (3, 3, 3), 8)
    cv2.imshow("floodfill", img)
    return img
#通过人脸检测裁剪人脸
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
#计算肤色部分的平均亮度
def meanBrightness(imgSKIN,imgGRAY):
    sp = imgSKIN.shape
    brightness = 0
    count = 0
    for i in range(sp[0]):
        for j in range(sp[1]):
           if imgSKIN[i,j] >100:
               brightness += imgGRAY[i,j]
               count +=1
    if count !=0:
        return count,brightness/count
    else:
        return count,0
#轮廓完善
def skeletonComplete(imgResult,imgSKIN):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    new_skin = imgSKIN
    new_skin = cv2.erode(new_skin, kernel)
    new_skin = cv2.dilate(new_skin, kernel)
    new_skin = cv2.erode(new_skin, kernel)
    new_skin = cv2.dilate(new_skin, kernel)
    new_skin = RemoveSmallRegion(new_skin,3000,0,1)
    new_skin = RemoveSmallRegion(new_skin,3000,1,1)
    cv2.imshow("new_skinHAHA",new_skin)
    new_skin = delete_jut(new_skin, 10, 10, 0)
    new_skin = delete_jut(new_skin, 10, 10, 1)
    new_skin = RemoveSmallRegion(new_skin,2000,0,1)
    new_skin = RemoveSmallRegion(new_skin,2000,1,1)
    cv2.imshow("new_skin",new_skin)

    #binary, contours, hierarchy = cv2.findContours(new_skin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #cv2.drawContours(binary, contours, -1, (0, 0, 255), 3)
    #cv2.imshow("inver",binary)
    img = cv2.GaussianBlur(new_skin, (3, 3), 0)  # 高斯平滑处理原图像降噪
    canny = cv2.Canny(img, 50, 150)  # apertureSize默认为3
    ret,skeleton = cv2.threshold(canny,100,255,cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    skeleton = cv2.erode(skeleton, kernel)
    #skeleton = cv2.erode(skeleton, kernel)
    result = cv2.bitwise_and(skeleton,imgResult)
    cv2.imshow("result",result)
    return result
#去除二值图像边缘的突出部
def delete_jut(src,uthreshold,vthreshold,type):
    threshold = 0
    dst = src.copy()
    sp = src.shape
    height = sp[0]
    width = sp[1]
    k = 0
    mode = (1-type)*255
    modeInver = 255-mode
    for i in range(height-1):
        for j in range(width-1):
            #行消除
            if dst[i,j]== mode and dst[i,j+1] == modeInver:
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
#对图片进行处理

def RemoveSmallRegion(src,AreaLimit,CheckMode,NeiborMode):
    RemoveCount = 0
    # 新建一幅标签图像初始化为0像素点，为了记录每个像素点检验状态的标签，0代表未检查，1代表正在检查, 2代表检查不合格（需要反转颜色），3代表检查合格或不需检查
    # 初始化的图像全部为0，未检查
    PointLabel = np.zeros(src.shape,np.uint8)
    dst = np.zeros(src.shape, np.uint8)
    sp = src.shape
    if(CheckMode == 1):#去除小连通区域的白色点,除去白色的
        print("去除小连通域")
        for i in range(sp[0]):
            for j in range(sp[1]):
                if src[i,j]<10:
                    PointLabel[i,j] = 3#将背景黑色点标记为合格，像素为3
    else: #除去黑色的
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
                GrowBuffer.append((i,j))
                PointLabel[i,j] = 1
                CheckResult = 0
                #在这里说一下，python的for循环有点奇葩，就是范围是静态的，第一次获取到范围值后就不会改变了
                z=0
                while z<len(GrowBuffer):
                    for q in range(NeihborCount):
                        CurrX = GrowBuffer[z][0]+NeiborPos[q][0]
                        CurrY = GrowBuffer[z][1]+NeiborPos[q][1]
                        if CurrX >=0 and CurrX < sp[0] and CurrY >=0 and CurrY<sp[1]:
                            if PointLabel[CurrX,CurrY] == 0:
                                GrowBuffer.append((CurrX,CurrY))
                                PointLabel[CurrX,CurrY] = 1
                    z += 1
                #对整个连通域检查完
                if len(GrowBuffer)>AreaLimit:
                    CheckResult = 2
                else:
                    CheckResult = 1
                    RemoveCount +=1

                for z in range(len(GrowBuffer)):
                    CurrX = GrowBuffer[z][0]
                    CurrY = GrowBuffer[z][1]
                    PointLabel[CurrX,CurrY] += CheckResult
    CheckMode = 255*(1-CheckMode)
    for i in range(sp[0]):
        for j in range(sp[1]):
            if PointLabel[i,j] == 2:
                dst[i,j] = CheckMode
            if PointLabel[i,j] == 3:
                dst[i,j] = src[i,j]

    return dst

def RemoveSelectRegion(src,AreaHigh,AreaLow,CheckMode,NeiborMode):
    RemoveCount = 0
    # 新建一幅标签图像初始化为0像素点，为了记录每个像素点检验状态的标签，0代表未检查，1代表正在检查, 2代表检查不合格（需要反转颜色），3代表检查合格或不需检查
    # 初始化的图像全部为0，未检查
    PointLabel = np.zeros(src.shape,np.uint8)
    dst = np.zeros(src.shape, np.uint8)
    sp = src.shape
    if(CheckMode == 1):#去除小连通区域的白色点,除去白色的
        print("去除小连通域")
        for i in range(sp[0]):
            for j in range(sp[1]):
                if src[i,j]<10:
                    PointLabel[i,j] = 3#将背景黑色点标记为合格，像素为3
    else: #除去黑色的
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
                GrowBuffer.append((i,j))
                PointLabel[i,j] = 1
                CheckResult = 0
                #在这里说一下，python的for循环有点奇葩，就是范围是静态的，第一次获取到范围值后就不会改变了
                z=0
                while z<len(GrowBuffer):
                    for q in range(NeihborCount):
                        CurrX = GrowBuffer[z][0]+NeiborPos[q][0]
                        CurrY = GrowBuffer[z][1]+NeiborPos[q][1]
                        if CurrX >=0 and CurrX < sp[0] and CurrY >=0 and CurrY<sp[1]:
                            if PointLabel[CurrX,CurrY] == 0:
                                GrowBuffer.append((CurrX,CurrY))
                                PointLabel[CurrX,CurrY] = 1
                    z += 1
                #对整个连通域检查完
                if len(GrowBuffer)> AreaHigh or len(GrowBuffer) < AreaLow:
                    CheckResult = 2
                else:
                    CheckResult = 1
                    RemoveCount +=1

                for z in range(len(GrowBuffer)):
                    CurrX = GrowBuffer[z][0]
                    CurrY = GrowBuffer[z][1]
                    PointLabel[CurrX,CurrY] += CheckResult
    CheckMode = 255*(1-CheckMode)
    for i in range(sp[0]):
        for j in range(sp[1]):
            if PointLabel[i,j] == 2:
                dst[i,j] = CheckMode
            if PointLabel[i,j] == 3:
                dst[i,j] = src[i,j]

    return dst

def myThreshold(imgGray,imgSkin,skinPoint,thresh):
    sp =  imgGray.shape
    dst = imgGray.copy()
    for i in range(sp[0]):
        for j in range(sp[1]):
            if imgSkin[i,j] >100:
                if imgGray[i,j]>thresh:
                    dst[i,j] = 255
                else:
                    dst[i,j] = 0
    return dst

def divisionThreshold(imgSKIN,imgFace):
    useless,theta = meanBrightness(imgSKIN=imgSKIN, imgGRAY=imgFace)
    divisionCount = 25
    #初始化数组大小
    imgSegment = [None] * divisionCount
    imgSkinSegment = [None] * divisionCount
    for i in range(len(imgSegment)):
        imgSegment[i] = [0] * divisionCount
        imgSkinSegment[i] = [0] * divisionCount
    sp = imgSKIN.shape
    new_w = math.floor(sp[1]/divisionCount)
    new_h = math.floor(sp[0]/divisionCount)
    dst = imgSKIN.copy()
    dst = dst[0:new_h*divisionCount,0:new_w*divisionCount]
    newSkin = imgSKIN[0:new_h*divisionCount,0:new_w*divisionCount]
    for i in range(divisionCount):
        for j in range(divisionCount):
            imgSegment[i][j] = imgFace[i*new_h:(i+1)*new_h,j*new_w:(j+1)*new_w]
            imgSkinSegment[i][j] = imgSKIN[i*new_h:(i+1)*new_h,j*new_w:(j+1)*new_w]
            skinPoint,alpha = meanBrightness(imgSKIN=imgSkinSegment[i][j],imgGRAY=imgSegment[i][j])
            if alpha !=0:
                min_th = min(alpha,theta)
                max_th = max(alpha,theta)
                #imgSegment[i][j] = myThreshold(imgSegment[i][j],imgSkinSegment[i][j],skinPoint,0.71*thresh) gamma = 0.63 lamda = 0.855 bata = 0.145
                imgSegment[i][j] = myThreshold(imgSegment[i][j],imgSkinSegment[i][j],skinPoint,gamma*(lamda*max_th+bata*min_th))
    for i in range(divisionCount):
        for j in range(divisionCount):
            dst[i*new_h:(i+1)*new_h,j*new_w:(j+1)*new_w]=imgSegment[i][j]
    ret,dst = cv2.threshold(dst,100,255,cv2.THRESH_BINARY)
    return dst,newSkin

def processImg(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.resize(img,(int(img.shape[1]/2) ,int(img.shape[0]/2)),interpolation=cv2.INTER_CUBIC)
    cv2.imshow("img",img)
    #获取到前景图，但是这里因为外层头发比较稀疏，所以可能会被误识别成肤色，所以用形态学操作把这层头发给去掉
    imgFront = removeBackground(img)

    #眼睛和人脸检测
    faces,eyes = detectEyes(img)
    if len(faces)<1:
        print("cannot detect the face")
        exit(0)
    temp = img.copy()
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)

    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #获取到的人物头像前景图
    imgFace = tailorImg(imgFront,faces)
    #获取imgFace的肤色图
    imgFace_Skin = skinModel(imgFace)
    ####################################################
    #对肤色图进行修剪
    skinTemp = cv2.erode(imgFace_Skin, kernel)
    skinTemp = cv2.dilate(skinTemp, kernel)
    #harlan
    imgFace_Skin = RemoveSmallRegion(skinTemp, 2000, 1, 1)
    imgFace_Skin = RemoveSelectRegion(imgFace_Skin, 2000,200, 0, 1)
    imgFace_Skin = RemoveSelectRegion(imgFace_Skin, 50,1, 0, 1)
    cv2.imshow("skinTemp", imgFace_Skin)

    imgFace_Gray = cv2.cvtColor(imgFace,cv2.COLOR_BGR2GRAY)
    #获取到作阈值化的阈值
    #theta = meanBrightness(imgSKIN=imgFace_Skin,imgGRAY=imgFace_Gray)
    #进行阈值化
    cv2.imshow("imgFace_gray",imgFace_Gray)
    imgFace_thresh,newSkin = divisionThreshold(imgSKIN=imgFace_Skin,imgFace=imgFace_Gray)
    #ret,imgFace_thresh = cv2.threshold(imgFace_Gray,0.7*int(theta),255,cv2.THRESH_BINARY)
    cv2.imshow("face_threshold",imgFace_thresh)
    #轮廓补充
    result = skeletonComplete(imgFace_thresh,imgSKIN=newSkin)
    return result
def createResult():
    i = 1
    while i <= 22:
        print(i)
        img = cv2.imread("img/" + str(i) + ".jpg", 1)
        img = cv2.medianBlur(img, 3)
        dst = processImg(img)
        dst = cv2.medianBlur(dst, 3)
        dst = RemoveSmallRegion(dst, 10, 0, 1)
        dst = delete_jut(dst, 1, 1, 1)
        dst = delete_jut(dst, 1, 1, 0)
        cv2.imwrite("result/" + str(i) + ".jpg", dst)
        i = i + 1

def unitTest():
    img = cv2.imread("img/14.jpg",1)
    img = cv2.medianBlur(img,3)
    dst = processImg(img)
    dst = cv2.medianBlur(dst,3)
    dst = RemoveSmallRegion(dst,10,0,1)
    dst = delete_jut(dst,1,1,1)
    dst = delete_jut(dst,1,1,0)
    cv2.imshow("result",dst)
if __name__ == '__main__':
    gamma = 0.652
    lamda = 0.8250
    bata = 0.1750
    createResult()
    cv2.waitKey(0)

