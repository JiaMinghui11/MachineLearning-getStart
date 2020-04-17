# -*- encoding: UTF-8 -*-
"""
@file       kNN_sklearn.py
@brief      调用sklearn库kNN算法实现手写数字识别
@author     Jia Minghui
@modify     2020-4-7
@note       仅提供参考例程，详情请参考官方文档：
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
"""
import numpy as np 
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as kNN

"""
@name       img_to_vector
@brief      将图像矩阵转换为一维向量
@param      filename        图像文件名
@return     returnVect      转换后的向量
"""
def img_to_vector(filename):
    returnVect = np.zeros((1, 1024))
    with open(filename, 'r') as fr:
        for i in range(32):
            line = fr.readline()
            for j in range(32):
                returnVect[0, 32*i+j] = int(line[j])
    return returnVect

"""
@name       handwriting_classfier
@brief      手写数字识别分类器
@param      void
@return     void
"""
def handwriting_classfier():

    #训练
    hwLabels = []                                   #训练集标签
    trainFileList = listdir('./data/train')         #训练集文件名列表
    trainNum = len(trainFileList)                   #训练样本数
    trainMat = np.zeros((trainNum, 1024))           #训练样本矩阵，kNN的输入参数
    for i in range(trainNum):
        filenameStr = trainFileList[i]
        classNum = int(filenameStr.split('_')[0])
        hwLabels.append(classNum)
        trainMat[i, :] = img_to_vector('./data/train/%s' % filenameStr)
    neigh = kNN(n_neighbors=3)                      #构造kNN分类器，k = 3
    neigh.fit(trainMat, hwLabels)                   #输入训练数据

    #测试
    testFileList = listdir('./data/test')           #测试集文件列表
    errorCount = 0.0                                #测试错误计数
    testNum = len(testFileList)                     #测试样本数
    for i in range(testNum):
        filenameStr = testFileList[i]
        classNum = int(filenameStr.split('_')[0])
        testVect = img_to_vector('./data/test/%s' % filenameStr)
        classfierResult = neigh.predict(testVect)   #测试
        print("The predict result is %d\t\tThe real result is %d" % (classfierResult, classNum))
        if classfierResult != classNum:
            errorCount += 1.0
    print("The total error count is %d\nThe error rate is %f%%" % (errorCount, errorCount/testNum*100))

if __name__ == '__main__':
    handwriting_classfier() 