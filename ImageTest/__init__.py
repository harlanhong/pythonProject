import cv2

def fillRunVectors(bwImage):
    rows = bwImage.shape[0]
    cols = bwImage.shape[1]
    NumberOfRuns = 0;
    stRun = []
    enRun = []
    rowRun = []
    for i in range(rows):
        if bwImage[rows,0]==255:
            NumberOfRuns+=1
            stRun.append(0)
            rowRun.append(i)
        for j in range(cols):
            if bwImage[i,j-1] == 0 and bwImage[i,j]==255:
                NumberOfRuns+=1
                stRun.append(j)
                rowRun.append(i)
            elif bwImage[i,j-1]==255 and bwImage[i,j]==0:
                enRun.append(j-1)
        if bwImage[i,cols-1]:
            enRun.append(cols-1)
    return NumberOfRuns,stRun,enRun,rowRun

def firstPass(NumberOfRuns,stRun,enRun,rowRun,offset):
    runLabels = [0]*NumberOfRuns;
    idxLabel = 1
    curRowIdx = 0
    firstRunOnCur = 0
    firstRunOnPre = 0
    lastRunOnPre = -1
    equivalence =[(None,None)]
    for i in range(NumberOfRuns):
        if rowRun[i] != curRowIdx:
            curRowIdx = rowRun[i]  #换一行
            firstRunOnPre = firstRunOnCur
            lastRunOnPre = i-1
            firstRunOnCur = i
        for j in range(firstRunOnPre,lastRunOnPre+1):
            if stRun[i]<=enRun[j]+offset and enRun[i]>=stRun[j]-offset and rowRun == rowRun[j]+1:
                if runLabels[i]==0:
                    runLabels[i] = runLabels[j]
                elif runLabels[i]!=runLabels[j]:
                    equivalence.append((runLabels[i],runLabels.append(j)))
        if runLabels[i] == 0:
            runLabels[i] = idxLabel
            idxLabel+=1


if __name__ == '__main__':
    t = [0]*5
    print(t)
