import numpy as np
from os import listdir

def kNN(inx, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diff = np.tile(inx, (dataSetSize, 1)) - dataSet                     #各变量求差
    distance = ((diff**2).sum(axis = 1))**0.5                           #欧式距离计算
    sortedDistance = distance.argsort()
    classCount = {}
    for i in range(k):
        classLabel = labels[sortedDistance[i]]
        classCount[classLabel] = classCount.get(classLabel, 0) + 1      #对应标签的统计数加一
    sortedClassCount = sorted(classCount.items(), key=lambda item:item[1], reverse=True)
    return sortedClassCount[0][0]

def vectorization(filename):
    returnVect = np.zeros((1,1024))
    fp = open(filename)
    for i in range(32):
        line = fp.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(line[j])
    return returnVect

def handwritingTest():
    hwLabels = []  
    trainingFileList = listdir('./data/train')                                                               
    m = len(trainingFileList)
    training = np.zeros((m, 1024))
    for i in range(m):
        fileName = trainingFileList[i]
        fileStr = fileName.split('.')[0]
        classNum = fileStr.split('_')[0]
        hwLabels.append(classNum)
        training[i,:] = vectorization('./data/train/%s' % fileName)
    testFileList = listdir('./data/test')
    errorCount = 0
    mTest = len(testFileList)
    for i in range(mTest):
        fileName = testFileList[i]
        fileStr = fileName.split('.')[0]
        classNum = int(fileStr.split('_')[0])
        vectorTest = vectorization('./data/test/%s' % fileName)
        kNNResult = int(kNN(vectorTest, training, hwLabels, 3))
        print("the kNN result is: %d, the real answer is: %d" % (kNNResult, classNum))
        if(kNNResult != classNum):
            errorCount += 1
    print("the errorCount is %d" % errorCount)
    print("the error rate is %f%%" % (errorCount/float(mTest)*100))

if __name__ == '__main__':
    handwritingTest()