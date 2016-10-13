#coding:utf-8

import math

class DesicionTree():
    def __init__(self):
        pass

    def _calcShannonEnt(self, dataSet): ## 计算数据集的熵
        numEntries = len(dataSet)
        classCounts = {}
        for data in dataSet:
            currentLabel = data[-1]
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
            prob = classCounts[key]/float(numEntries)
            shannonEnt -= prob*math.log(prob, 2) # log base 2
        return shannonEnt

    def _splitDataSet(self, dataSet, axis, value):
        retDataSet = []
        for data in dataSet:
            # print data[axis]
            if data[axis] == value:
                reduceddata = data[:axis]
                reduceddata.extend(data[axis+1:])
                retDataSet.append(reduceddata)
        return retDataSet

    def _chooseBestFeatureToSplit(self, dataSet):
        numFeatures = len(dataSet[0])-1 # 最后一列是类标签
        baseEntropy = self._calcShannonEnt(dataSet)
        bestInfoGain = 0
        bestFeature = -1
        for i in range(numFeatures): # 依次迭代所有的特征
            featList = [data[i] for data in dataSet]
            values = set(featList)
            '''
            条件熵：sigma(pj*子数据集的熵)
            '''
            ## 计算每个特征对数据集的条件熵
            newEntropy = 0.0
            for value in values:
                subDataSet = self._splitDataSet(dataSet, i, value)
                prob = len(subDataSet)/float(len(dataSet))
                newEntropy += prob*self._calcShannonEnt(subDataSet)
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
        for vote in classList:
            if vote not in classCount:
                classCount[vote] = 1
            else:
                classCount[vote] += 1
            # if vote not in classCount:
            #     classCount[vote] = 0
            # classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(), key=lambda xx:xx[1], reverse=True)
        return sortedClassCount[0][0]

    def fit(self, dataSet, featLabels):
        classList = [data[-1] for data in dataSet]
        if classList.count(classList[0]) == len(classList):
            return classList[0] # 所有的类标签都相同，则返回类标签
        if len(dataSet[0]) == 1: # 所有的类标签不完全相同，但用完所有特征，则返回次数最多的类标签
            return self._majorityCnt(classList)
        bestFeat = self._chooseBestFeatureToSplit(dataSet)
        bestFeatLabel = featLabels[bestFeat]
        tree = {bestFeatLabel:{}}
        featLabels_copy = featLabels[:] # 这样不会改变输入的featLabels
        featLabels_copy.remove(bestFeatLabel)
        featList = [data[bestFeat] for data in dataSet]
        values = set(featList)
        for value in values:
            subfeatLabels_copy = featLabels_copy[:] # 列表复制，非列表引用
            tree[bestFeatLabel][value] = self.fit(self._splitDataSet(dataSet, bestFeat, value), subfeatLabels_copy)
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
    featLabels = ['no surfacing', 'flippers'] # 特征标签
    return dataSet, featLabels

if __name__ == '__main__':
    myDataSet, myFeatLabels = loadDataSet()
    print myDataSet, myFeatLabels
    dt = DesicionTree()
    myTree = dt.fit(myDataSet, myFeatLabels)
    print myTree
    results = dt.predict(myTree, myFeatLabels, [1, 1])
    print results
    results = dt.predict(myTree, myFeatLabels, [0, 1])
    print results
