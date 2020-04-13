# -*- coding: UTF-8 -*-
"""
@file       decisionTree.py
@brief      参照教材决策树构建基本流程构建决策树（以信息熵为依据划分）
@author     Jia Minghui
@modify     2020-4-13
"""


from math import log
import pickle


"""
@name       dataset_creat
@brief      创建数据集
@param      void
@return     dataSet     数据集
            labels      分类标签
"""
def dataset_creat():
    dataSet = []
    with open('data.txt', 'r') as f:
        dataSetList = f.readlines()
    for i in range(len(dataSetList)):                           
        dataSet.append(dataSetList[i].split('\t'))
        dataSet[i][-1] = dataSet[i][-1].split('\n')[0]
    labels = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']        #特征标签
    return dataSet, labels


"""
@name       Ent_calc
@brief      依据公式4.1计算信息熵
@param      dataSet     需要计算的样本集合
@return     End         样本信息熵
"""
def Ent_calc(dataSet):
    num = len(dataSet)              #统计样本数
    labelCounts = {}                #统计各个类别的个数
    for each in dataSet:
        currentLabel = each[-1]     #类别标签在最后一列
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    Ent = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / num
        Ent -= prob * log(prob, 2)  #公式4.1
    return Ent


"""
@name       dataSet_split
@brief      按照给定的特征和值划分数据集
@param      dataSet     待划分数据集
            axis        划分数据集所用的特征
            value       划分依据的特征值
@return     resetDara   满足条件的类别集合
"""
def dataSet_split(dataSet, axis, value):
    resetData = []
    for exam in dataSet:
        if exam[axis] == value:
            examReduce = exam[:axis]
            examReduce.extend(exam[axis+1:])    #axis对应的特征已经用过，删除其对应的样本数据
            resetData.append(examReduce)
    return resetData


"""
@name       bestFeature_toSplit
@brief      选择最优划分属性
@param      dataSet     准备划分的集合
@return     bestFeature 最优划分属性
"""
def bestFeature_toSplit(dataSet):
    numFeature = len(dataSet[0]) - 1    #统计属性总数
    baseEnt = Ent_calc(dataSet)         #计算原始信息熵
    bestInfoGain = 0.0                  #最大信息增益
    bestFeature = -1                    #最优划分属性
    for i in range(numFeature):
        featList = [exam[i] for exam in dataSet]
        uniqueFeat = set(featList)      #消重
        newEnt = 0.0                    #划分后新的信息熵
        for value in uniqueFeat:
            subDataSet = dataSet_split(dataSet, i, value)
            prob = float(len(subDataSet)) / len(dataSet)
            newEnt += prob * Ent_calc(subDataSet)
        infoGain = baseEnt - newEnt     #教材式4.2
        #print(infoGain)
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
    

"""
@name       majority_count
@brief      统计样本集合中最多的类别
@param      classList               样本集合
@return     sortedClassCount[0][0]  个数最多的类别
"""
def majority_count(classList):
    classCount = {}                             #统计各类别样本数量
    for eachClass in classList:
        if eachClass not in classCount.keys():
            classCount[eachClass] = 0
        classCount[eachClass] += 1
    sortedClassCount = sorted(classCount.items(), 
                              key=lambda item: item[1],
                              reverse=True)     #排序
    return sortedClassCount[0][0]


"""
@name       tree_generate
@brief      参考教材图4.2使用递归构建决策树
@param      dataSet     样本集合
            labels      属性标签
            featLabels  记录每次划分用到的属性
@return     myTree      用字典方式存储的决策树
"""
def tree_generate(dataSet, labels, featLabels):
    classList = [exam[-1] for exam in dataSet]
    #print(classList, '\n')
    if classList.count(classList[0]) == len(classList): #样本全属于同一类别
        return classList[0]
    if len(dataSet[0]) == 1 or len(labels) == 0:        #所有属性已经用完
        return majority_count(classList)
    bestFeat = bestFeature_toSplit(dataSet)             #选择最优特征
    bestFeatLabel = labels[bestFeat]                    #最优特征的标签
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel:{}}                         #根据最优特征的标签生成树
    del(labels[bestFeat])                               #删除已经使用特征标签
    #print(labels)
    featValues = [exam[bestFeat] for exam in dataSet]   #得到训练集中所有最优特征的属性值
    uniqueVals = set(featValues)                        #去掉重复的属性值
    for value in uniqueVals:                            #遍历特征，创建决策树。        
        subLabels = labels[:]               
        myTree[bestFeatLabel][value] = tree_generate(dataSet_split(dataSet, bestFeat, value),
                                                     subLabels, featLabels)
    return myTree


"""
@name       tree_save
@brief      以二进制形式存储决策树模型（直接打开文件看到的会是乱码）
@param      inputTree       需要存储的树
            filename        文件名
@return     void
"""
def tree_save(inputTree, filename):
    with open(filename, 'wb') as f:
        pickle.dump(inputTree, f)


if __name__ == '__main__':
    dataSet, labels = dataset_creat()
    #print(dataSet, '\n', labels)
    featLabels = []
    myTree = tree_generate(dataSet, labels, featLabels)
    print(myTree, '\n', featLabels)
    tree_save(myTree, 'deciTreeClassfier.txt')