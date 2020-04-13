# -*- coding: UTF-8 -*-
"""
@file       test.py
@brief      读取保存的模型，测试数据
@author     Jia Minghui
@modify     2020-4-13
"""

import pickle

"""
@name       tree_load
@brief      读取保存的树模型
@param      filename        文件名
@return     pickle.load(f)  树模型
"""
def tree_load(filename):
    f = open(filename, 'rb')
    return pickle.load(f)

"""
@name       classify
@brief      调用树模型进行分类
@param      inputTree       树模型
            featLabels      决策依据的属性标签，在构建树时产生
            testVec         待测样本属性向量
@return     classLabel      样本所属类别
"""
def classify(inputTree, featLabels, testVec):
    firstStr = next(iter(inputTree))              #获取决策树结点
    secondDict = inputTree[firstStr]              #下一个字典
    featIndex = featLabels.index(firstStr)   
    for key in secondDict.keys():
        #print(type(key))
        #print(type(secondDict[key]).__name__)
        if str(testVec[featIndex]) == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else: classLabel = secondDict[key]
    return classLabel


if __name__ == '__main__':
    featLabels = ['纹理', '触感', '根蒂', '色泽', '触感']   #构建树的时候得到的
    myTree = tree_load('deciTreeClassfier.txt')
    print(myTree)
    #testVec = [2, 3, 1, 2, 2, 2]                         #西瓜集2.0的最后一个样本用作测试
                                                          #这种选择测试样本的方法并不正确因为训练时用到了这个样本，这里只做演示使用
    testVec = [2, 2, 3, 2]                                #只需输入featLabels对应属性即可，无需输入冗余属性
    test_label = classify(myTree, featLabels, testVec)
    print('是否好瓜:', test_label)