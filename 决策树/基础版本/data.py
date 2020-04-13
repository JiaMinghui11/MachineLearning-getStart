#属性转化字典
colors = {'浅白':1, '青绿':2, '乌黑':3}
roots = {'硬挺':1, '稍蜷':2, '蜷缩':3}
voices = {'沉闷':1, '浊响':2, '清脆':3}
textures = {'模糊':1, '稍糊':2, '清晰':3}
umbilicals = {'凹陷':1, '稍凹':2, '平坦':3}
touchs = {'软粘':1, '硬滑':2}
results = {'是':'yes', '否':'no'}

class Watermelon():
    """
    定义西瓜样本类
    """
    def __init__(self, feature):
        self.color = str(colors[feature[0]])                     #色泽 
        self.root = str(roots[feature[1]])                       #根蒂
        self.voice = str(voices[feature[2]])                     #敲声
        self.texture = str(textures[feature[3]])                 #纹理
        self.umbilical = str(umbilicals[feature[4]])             #脐部
        self.touch = str(touchs[feature[5]])                     #触感
        self.result = str(results[feature[6].split('\n')[0]])    #是否好瓜

"""
读取数据文件，量化样本集
"""
with open('./西瓜数据集2.0.txt', 'r', encoding='GBK') as f:
    lines = f.readlines()
watermelons = []
for i in range(1, len(lines)):
    temp = lines[i].split(',')[1:]
    #print(temp)
    watermelon = Watermelon(temp)
    watermelons.append(watermelon)
    #print(watermelons)
with open ('data.txt', 'w') as f:
    for i in range(len(watermelons)):
        f.writelines([watermelons[i].color,'\t',watermelons[i].root,'\t',watermelons[i].voice,'\t',
                      watermelons[i].texture,'\t',watermelons[i].umbilical,'\t',watermelons[i].touch,'\t',watermelons[i].result,'\n'])