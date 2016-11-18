#coding:utf-8

import math

class DesicionTree():
    def __init__(self):
        pass

    def _calcShannonEnt(self, classList): ## 计算数据集的熵
        classCounts = {}
        for currentLabel in classList:
            if currentLabel not in classCounts:
                classCounts[currentLabel] = 1
            else:
                classCounts[currentLabel] += 1
            # if currentLabel not in classCounts:
            #     classCounts[currentLabel] = 0
            # classCounts[currentLabel] += 1
        '''
        信息  -log2(pi)
        熵：信息的期望 sigma(-pi*log2(pi))
        '''
        shannonEnt = 0.0
        for key in classCounts:
            prob = classCounts[key]/float(len(classList))
            shannonEnt -= prob*math.log(prob, 2) # log base 2
        return shannonEnt

    def _splitDataSet(self, dataArr, classList, axis, value):
        retFeatData = []
        retLabelData = []
        for data, label in zip(dataArr, classList):
            # print data[axis]
            if data[axis] == value:
                reducedFeat = data[:axis]
                reducedFeat.extend(data[axis+1:])
                retFeatData.append(reducedFeat)
                retLabelData.append(label)
        return retFeatData, retLabelData

    def _chooseBestFeatureToSplit(self, dataArr, classList):
        baseEntropy = self._calcShannonEnt(classList)
        bestInfoGain = 0
        bestFeature = -1
        numFeatures = len(dataArr[0])
        for i in range(numFeatures): # 依次迭代所有的特征
            featList = [data[i] for data in dataArr]
            values = set(featList)
            '''
            条件熵：sigma(pj*子数据集的熵)
            '''
            ## 计算每个特征对数据集的条件熵
            newEntropy = 0.0
            for value in values:
                subDataArr, subClassList = self._splitDataSet(dataArr, classList, i, value)
                prob = len(subClassList)/float(len(classList))
                newEntropy += prob*self._calcShannonEnt(subClassList)
            '''
            信息增益 = 熵-条件熵
            '''
            infoGain = baseEntropy-newEntropy
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeature = i
        return bestFeature

    def _majorityCnt(self, classList):
        classCount = {}
        for currentLabel in classList:
            if currentLabel not in classCount:
                classCount[currentLabel] = 1
            else:
                classCount[currentLabel] += 1
            # if currentLabel not in classCount:
            #     classCount[currentLabel] = 0
            # classCount[currentLabel] += 1
        sortedClassCount = sorted(classCount.items(), key=lambda xx:xx[1], reverse=True)
        return sortedClassCount[0][0]

    def fit(self, dataArr, classList, featLabels):
        if classList.count(classList[0]) == len(classList):
            return classList[0] # 所有的类标签都相同，则返回类标签
        if len(dataArr[0]) == 0: # 所有的类标签不完全相同，但用完所有特征，则返回次数最多的类标签
            return self._majorityCnt(classList)
        bestFeat = self._chooseBestFeatureToSplit(dataArr, classList)
        bestFeatLabel = featLabels[bestFeat]
        tree = {bestFeatLabel:{}}
        featLabels_copy = featLabels[:] # 这样不会改变输入的featLabels
        featLabels_copy.remove(bestFeatLabel)
        featList = [data[bestFeat] for data in dataArr]
        values = set(featList)
        for value in values:
            subFeatLabels_copy = featLabels_copy[:] # 列表复制，非列表引用
            subDataArr = self._splitDataSet(dataArr, classList, bestFeat, value)[0]
            subClassList = self._splitDataSet(dataArr, classList, bestFeat, value)[1]
            tree[bestFeatLabel][value] = self.fit(subDataArr, subClassList, subFeatLabels_copy)
        return tree

    def predict(self, tree, featLabels, testVec):
        firstStr = tree.keys()[0]
        secondDict = tree[firstStr]
        featIndex = featLabels.index(firstStr)
        key = testVec[featIndex]
        valueOfFeat = secondDict[key]
        if isinstance(valueOfFeat, dict):
            classLabel = self.predict(valueOfFeat, featLabels, testVec)
        else:
            classLabel = valueOfFeat
        return classLabel

def loadDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    featData = []
    labelData = []
    for data in dataSet:
        featData.append(data[:-1])
        labelData.append(data[-1])
    featLabels = ['no surfacing', 'flippers'] # 特征标签
    return featData, labelData, featLabels

if __name__ == '__main__':
    myFeatData, myLabelData, myFeatLabels = loadDataSet()
    print myFeatData, myLabelData, myFeatLabels
    dt = DesicionTree()
    myTree = dt.fit(myFeatData, myLabelData, myFeatLabels)
    print myTree
    results = dt.predict(myTree, myFeatLabels, [1, 1])
    print results
    results = dt.predict(myTree, myFeatLabels, [0, 1])
    print results
