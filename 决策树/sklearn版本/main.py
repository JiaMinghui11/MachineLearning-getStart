# -*- coding: UTF-8 -*-
"""
@file       main.py
@brief      基于lenses数据集使用sklearn库构建决策树并实现可视化
@author     Jia Minghui
@modify     2020-4-12
@version    1.0.0
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import graphviz

if __name__ == '__main__':

    #数据读取
    with open('lenses.txt', 'r') as f:
        lenses = [inst.strip().split('\t') for inst in f.readlines()]
    #print(lenses)
    lenses_target = []      #提取数据的类别标签
    for each in lenses:
        lenses_target.append(each[-1])
    
    #数据转化
    #因为fit方法不支持字符串类型输入，因此需要将字符类属性序列化
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']   #属性标签
    lenses_list = []                                                #临时列表
    lenses_dict = {}                                                #用于生成pandas的字典
    for each_label in lensesLabels:
        for each in lenses:
            lenses_list.append(each[lensesLabels.index(each_label)])
        lenses_dict[each_label] = lenses_list
        lenses_list = []
    #print(lenses_dict)
    lenses_pd = pd.DataFrame(lenses_dict)   #生成pandas.DataFrame       
    le = LabelEncoder()                     #创建LabelEncoder对象
    for col in lenses_pd.columns:           #为每一列序列化
        lenses_pd[col] = le.fit_transform(lenses_pd[col])   
    print(lenses_pd)
    
    #决策树构建
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(lenses_pd.values.tolist(), lenses_target)

    #数据可视化
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=lenses_pd.keys(),
                                    class_names=clf.classes_,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("tree")                                #在文件相同目录中会出现可视化的图形tree.pdf文件 
    
    #数据预测
    print(clf.predict([[2, 1, 0, 1], [1, 0, 1, 0]]))    #输入应为序列化后的属性值