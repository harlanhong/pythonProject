import cv2
import numpy as np
import math
#自定义直方图统计
def myCalHist(img):
    hist = np.zeros([256,1])
    sp = img.shape
    for i in range(sp[0]):
        for j in range(sp[1]):
            hist[img[i,j]]+=1
    hist[255] = 0
    return hist
def DrawHist(hist, color,thresh = -1):
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
    histImg = np.zeros([512,256,3], np.uint8)

    hpt = int(0.9* 256);
    #cv2.line(histImg, (0, 256), (256, 256), [0,0,255])
    for h in range(256):
            intensity = int(hist[h]*hpt/maxVal)
            if intensity!=0:
                cv2.line(histImg,(h,256), (h,256-intensity), color)
    if thresh != -1:
        cv2.line(histImg, (thresh, 512), (thresh,0), [0,0,255])
    return histImg;
#直方图平滑处理
#插值平滑
def interpolation(hist,step):
    start = int(step - step/2)
    end = int(256 - step/2)
    temp =0
    res = hist.copy()
    for i in range(start,end):
        temp =0
        for j in range(int(0-step/2),int(step/2)):
            temp += hist[i+j]
        temp /=step
        res[i] = int(temp)
    return res
#频率滤波
def fourier(hist):
    ft = hist.copy()
    N = 256
    Cu = 1
    for u in range(N):
        if u == 0:
            Cu= 1.0/math.sqrt(2)
        sum = 0
        for x in range(N):
            sum += (hist[x]*math.cos(((2*x)+1)*u*math.pi/(2*N)))
        temp = Cu*math.sqrt(2.0/N)*sum
        ft[u] = int(temp)
    return ft
# def fourier(hist,alterComponent):
#     ft = hist.copy()
#     N = 256
#     Cu = 1
#     for u in range(N):
#         if u == 0:
#             Cu= 1.0/math.sqrt(2)
#         sum = 0
#         for x in range(N):
#             theta = ((2*x)+1)*u*math.pi/(2*N)
#             if theta == 0 or u<alterComponent:
#                 sum += (hist[x]*math.cos(theta))
#         temp = Cu*math.sqrt(2.0/N)*sum
#         ft[u] = int(temp)
#     return ft
def inverFourier(hist):
    inverseft = hist.copy()
    N = 256
    for x in range(N):
        sum =(1/math.sqrt(2.0))*hist[0]
        for u in range(1,N):
            sum+=hist[u]*math.cos((2*x+1)*u*math.pi/(2*N))
        temp = math.sqrt(2.0/N)*sum
        inverseft[x] = math.fabs(temp)
    return inverseft

#求直方图的导数图
def coefficientHist(hist):
    res = hist.copy()
    N = 256
    res[0]=0
    res[255]=0
    for x in range(1,255):
        res[x]=hist[x+1]-hist[x]
    return res

#迭代最佳阈值
def GetIterativeBestThreshold(hist):
    X, Iter = 0,0;
    MinValue, MaxValue=0,0;
    for MinValue in range(256):
        if hist[MinValue]!=0:
            break;
    for MaxValue in range(255,-1,-1):
        if hist[MaxValue]!=0:
            break;
    if MaxValue == MinValue: return MaxValue; # 图像中只有一个颜色
    if MinValue + 1 == MaxValue: return MinValue; # 图像中只有二个颜色

    Threshold = MinValue;
    NewThreshold = (MaxValue + MinValue)/2;
    while Threshold != NewThreshold: # 当前后两次迭代的获得阈值相同时，结束迭代
        SumOne = 0;
        SumIntegralOne = 0;
        SumTwo = 0;
        SumIntegralTwo = 0;
        Threshold = NewThreshold;
        for X in range(math.ceil(MinValue),math.ceil(Threshold)):# 根据阈值将图像分割成目标和背景两部分，求出两部分的平均灰度值
            SumIntegralOne += hist[X] * X;
            SumOne += hist[X];
        MeanValueOne = SumIntegralOne / SumOne;
        for X in range(math.ceil(Threshold+1),math.ceil(MaxValue+1)):
            SumIntegralTwo += hist[X] * X;
            SumTwo += hist[X];
        MeanValueTwo = SumIntegralTwo / SumTwo;
        NewThreshold = (MeanValueOne + MeanValueTwo)/2; # 求出新的阈值
        Iter+=1;
        if Iter >= 1000:
            return -1;
    return Threshold;
#OSTU大津法
def GetOSTUThreshold(histGram):
    X, Y, Amount = 0,0,0;
    MinValue, MaxValue=0,0;
    Threshold = 0;
    for MinValue in range(256):
        if histGram[MinValue]!=0:
            break;
    for MaxValue in range(255,-1,-1):
        if histGram[MaxValue]!=0:
            break;
    if MaxValue == MinValue: return MaxValue; # 图像中只有一个颜色
    if MinValue + 1 == MaxValue: return MinValue; # 图像中只有二个颜色
    for Y in range(MinValue,MaxValue+1):Amount += histGram[Y];#像素个数
    PixelIntegral = 0;
    for Y in range(MinValue, MaxValue + 1): PixelIntegral += histGram[Y] * Y;#像素总值
    SigmaB = -1;
    PixelBack = 0
    PixelIntegralBack=0
    for Y in range(MinValue, MaxValue + 1):
        PixelBack = PixelBack + histGram[Y];
        PixelFore = Amount - PixelBack;
        OmegaBack = float(PixelBack / Amount);
        OmegaFore = float(PixelFore / Amount);
        PixelIntegralBack += histGram[Y] * Y;
        PixelIntegralFore = PixelIntegral - PixelIntegralBack;
        MicroBack = float(PixelIntegralBack / PixelBack);
        MicroFore = float(PixelIntegralFore / PixelFore);
        Sigma = OmegaBack * OmegaFore * (MicroBack - MicroFore) * (MicroBack - MicroFore);
        if (Sigma > SigmaB):
            SigmaB = Sigma;
            Threshold = Y;
    return Threshold;
#力矩保持法
def GetMomentPreservingThreshold(HistGram):
    X, Y, Index = 0,0,0; Amount = 0;
    Avec = [0]*256
    X2, X1, X0, Min=0,0,0,0;
    for Y in range(256):Amount+=HistGram[Y]
    for Y in range(256):
        Avec[Y]=A(HistGram,Y)/Amount
    X2 = (B(HistGram, 255) * C(HistGram, 255) - A(HistGram, 255) * D(HistGram, 255)) / (
        A(HistGram, 255) * C(HistGram, 255) - B(HistGram, 255) * B(HistGram, 255));
    X1 = (B(HistGram, 255) * D(HistGram, 255) - C(HistGram, 255) * C(HistGram, 255)) / (
        A(HistGram, 255) * C(HistGram, 255) - B(HistGram, 255) * B(HistGram, 255));
    X0 = 0.5 - (B(HistGram, 255) / A(HistGram, 255) + X2 / 2) / math.sqrt(X2 * X2 - 4 * X1);

    Min = 9999999999
    for Y in range(256):
        if (math.fabs(Avec[Y] - X0) < Min):
            Min = math.fabs(Avec[Y] - X0);
            Index = Y;
    return Index
def A(hist,index):
    sum = 0
    for Y in range(index+1):
        sum+=hist[Y]
    return sum;
def B(hist,index):
    sum = 0
    for Y in range(index + 1):
        sum += hist[Y]*Y
    return sum;
def C(hist,index):
    sum = 0
    for Y in range(index + 1):
        sum += Y*Y*hist[Y]
    return sum;
def D(hist,index):
    sum = 0
    for Y in range(index + 1):
        sum += Y*Y*Y*hist[Y]
    return sum;
#谷底最小值
def GetMinimumThreshold(HistGram):
    Y, Iter = 0,0;
    HistGramC = HistGram.copy();
    HistGramCC = HistGram.copy();
    # 通过三点求均值来平滑直方图
    while IsDimodal(HistGramCC) == False: # 判断是否已经是双峰的图像了
        HistGramCC[0] = (HistGramC[0] + HistGramC[0] + HistGramC[1]) / 3; # 第一点
        for Y in range(1,255):
            HistGramCC[Y] = (HistGramC[Y - 1] + HistGramC[Y] + HistGramC[Y + 1]) / 3; # 中间的点
        HistGramCC[255] = (HistGramC[254] + HistGramC[255] + HistGramC[255]) / 3; # 最后一点
        HistGramC = HistGramCC.copy();
        Iter+=1;
        if (Iter >= 1000):
            return -1; # 直方图无法平滑为双峰的，返回错误代码
    # 阈值极为两峰之间的最小值
    Peakfound = False;
    for Y in range(1, 255):
        if (HistGramCC[Y - 1] < HistGramCC[Y] and HistGramCC[Y + 1] < HistGramCC[Y]): Peakfound = True;
        if (Peakfound == True and HistGramCC[Y - 1] >= HistGramCC[Y] and HistGramCC[Y + 1] >= HistGramCC[Y]):
            histImg = DrawHist(HistGramCC, [255, 255, 255],Y-1)
            cv2.imshow("newHist", histImg)
            return Y - 1;
    return -1;
def IsDimodal(HistGram):
    # 对直方图的峰进行计数，只有峰数位2才为双峰
    Count = 0;
    for Y in range(1, 255):
        if (HistGram[Y - 1] < HistGram[Y] and HistGram[Y + 1] < HistGram[Y]):
            Count +=1;
            if (Count > 2):
                return False;
    if (Count == 2):
        return True;
    else:
        return False;
def adaptiveThreshold(img):
        hist = myCalHist(img)
        histImg = DrawHist(hist, [255, 255, 255])
        cv2.imshow("histInit", histImg)
        min, max, hist = removeNoiseOfHist(hist)
        histImg = DrawHist(hist, [255, 255, 255])
        cv2.imshow("histIMG", histImg)
        thresh = GetMinimumThreshold(hist)
        return min, max, thresh + 15
def removeNoiseOfHist(hist):
    result = hist.copy()
    counter = 0
    for i in range(255,-1,-1):
        if result[i] != 0:
            counter += 1
        result[i] = 0
        if counter == 10:
            break
    counter = 0
    for i in range(255):
        if result[i] !=0:
            counter+=1
        result[i] = 0
        if counter == 10:
            break

    for  i in range(255):
        if result[i]<50:
            result[i] = 0
        else:
            break
    for  i in range(255,-1,-1):
        if result[i]<50:
            result[i] = 0
        else:
            break
    for i in range(70):
        if result[i]>500:
            break
        result[i]=0
    min ,max =0,0
    for i in range(255):
        if result[i]!=0:
            min = i;
            break
    for i in range(255,-1,-1):
        if result[i]!=0:
            max = i;
            break
    return min,max,result
def subtractFun(src,erode_ouput,threshold):
    result = np.zeros(src.shape,np.int64)
    sp = src.shape
    for i in range(sp[0]):
        for j in range(sp[1]):
            if src[i,j,0]-erode_ouput[i,j,0]>threshold or src[i, j, 1] - erode_ouput[i, j, 1] > threshold or src[i, j, 2] - erode_ouput[i, j,2] > threshold:
                result[i,j,0] =255
                result[i,j,1] =255
                result[i,j,2] =255

            else:
                result[i,j,0]=0
                result[i,j,1]=0
                result[i,j,2]=0

    return result
def removeNoiseOfHist(hist):
    result = hist.copy()
    counter = 0
    min=0;max =0
    for i in range(256):
        counter+=hist[i]
        if counter>200:
            break
        else:
            result[i]=0;
            min = i
    counter = 0
    for i in range(255,-1,-1):
        counter+=hist[i]
        if counter>200:
            break
        else:
            result[i]=0;
            max = i
    return min,max,result

def getSkeleton(img,removedetails_times,erode_times):#输入为去除背景后的bgr图

    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    img = cv2.erode(img,element,iterations=2)
    erode_output = img.copy()
    outLineResult = img.copy()
    openResult = img.copy()
    closeResult = img.copy()
    #噪声滤除
    if removedetails_times ==1:
        closeResult=cv2.morphologyEx(img,cv2.MORPH_CLOSE,element,iterations=1)
        openResult = cv2.morphologyEx(closeResult,cv2.MORPH_OPEN,element,iterations=1)
    elif removedetails_times == 2:
        openResult = cv2.morphologyEx(img,cv2.MORPH_OPEN,element,iterations=1)
        closeResult=cv2.morphologyEx(openResult,cv2.MORPH_CLOSE,element,iterations=1)
        openResult = cv2.morphologyEx(closeResult,cv2.MORPH_OPEN,element,iterations=1)
        closeResult=cv2.morphologyEx(openResult,cv2.MORPH_CLOSE,element,iterations=1)
    #腐蚀操作
    erode_output = cv2.erode(closeResult,element,erode_times)
    ret,closeResult = cv2.threshold(closeResult,5,255,cv2.THRESH_BINARY)
    ret,erode_output = cv2.threshold(erode_output,5,255,cv2.THRESH_BINARY)
    outLineResult = cv2.subtract(closeResult,erode_output)
    result = cv2.medianBlur(outLineResult,3)
    if result.ndim >1:
        result = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
    ret,result = cv2.threshold(result,100,255,cv2.THRESH_BINARY_INV)
    result = cv2.erode(result,element,iterations=1)
    return result
#去除指定大小的区域
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


