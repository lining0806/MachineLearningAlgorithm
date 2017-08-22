#coding:utf-8
import numpy as np

class NaiveBayes():
    def __init__(self):
        pass

    def createVocabList(self, train_x):
        vocabSet = set([])
        for wordList in train_x:
            vocabSet = vocabSet | set(wordList)
        return list(vocabSet)

    def listOfWords2Vec(self, vocabList, wordList):
        wordsVec = [0] * len(vocabList)
        for word in wordList:
            if word in vocabList:
                wordsVec[vocabList.index(word)] = 1 # 词集模型
                # wordsVec[vocabList.index(word)] += 1 # 词袋模型
            # else:
            #     print "the word:%s is not in my vocabulary!" % word
        return wordsVec

    def fit(self, train_x, train_y):
        vocabList = self.createVocabList(train_x)
        trainMat = []
        for wordList in train_x:
            trainMat.append(self.listOfWords2Vec(vocabList, wordList))
        trainMatrix = np.array(trainMat) ## array
        trainLabel = np.array(train_y) ## array
        numTrainDocs = len(trainMatrix) # 统计样本个数
        numWords = len(trainMatrix[0])  # 统计特征个数，理论上是词库的长度
        ## 计算p(c0)，p(c1) ## 对应0/1分类
        p0 = sum(trainLabel==0)/float(numTrainDocs) # 对应p(c0)
        p1 = sum(trainLabel==1)/float(numTrainDocs) # 对应p(c1)
        ## 计算p(wi|c0)，p(wi|c1)
        p0Num = np.ones(numWords) # 初始样本个数为1，防止条件概率为0，影响结果
        p1Num = np.ones(numWords)
        p0InAll = 2.0 # 词库中只有两类，所以此处初始化为2
        p1InAll = 2.0
        for i in range(numTrainDocs):
            if trainLabel[i] == 1:
                p1Num += trainMatrix[i]
                p1InAll += sum(trainMatrix[i])
            else:
                p0Num += trainMatrix[i]
                p0InAll += sum(trainMatrix[i])
        p0Vec = np.log(p0Num/p0InAll) # 对应p(wi|c0)
        p1Vec = np.log(p1Num/p1InAll) # 对应p(wi|c1)
        ## 整合参数
        param = p0, p1, p0Vec, p1Vec
        return vocabList, param

    def predict(self, test_X, vocabList, param):
        p0, p1, p0Vec, p1Vec = param
        testMat = []
        for wordList in test_X:
            testMat.append(self.listOfWords2Vec(vocabList, wordList))
        testMatrix = np.array(testMat) ## array
        predict_y = []
        for vec in testMatrix:
            prob_y0 = sum(vec*p0Vec)+np.log(p0) # 对应p(w1|c0)*p(w2|c0)*...*p(c0)，log(a*b) = log(a)+log(b)
            prob_y1 = sum(vec*p1Vec)+np.log(p1) # 对应p(w1|c1)*p(w2|c1)*...*p(c1)，log(a*b) = log(a)+log(b)
            if prob_y0 < prob_y1: ## 对应0/1分类
                predict_y.append(1)
            else:
                predict_y.append(0)
        predictLabel = np.array(predict_y) ## array
        return predictLabel

    def predict1(self, test_X, test_y, vocabList, param):
        p0, p1, p0Vec, p1Vec = param
        testMat = []
        for wordList in test_X:
            testMat.append(self.listOfWords2Vec(vocabList, wordList))
        testMatrix = np.array(testMat) ## array
        m = testMatrix.shape[0]
        predict_y = []
        for vec in testMatrix:
            prob_y0 = sum(vec*p0Vec)+np.log(p0) # 对应p(w1|c0)*p(w2|c0)*...*p(c0)，log(a*b) = log(a)+log(b)
            prob_y1 = sum(vec*p1Vec)+np.log(p1) # 对应p(w1|c1)*p(w2|c1)*...*p(c1)，log(a*b) = log(a)+log(b)
            if prob_y0 < prob_y1: ## 对应0/1分类
                predict_y.append(1)
            else:
                predict_y.append(0)
        testLabel = np.array(test_y) ## array
        predictLabel = np.array(predict_y) ## array
        print 'accuracy:', sum(testLabel==predictLabel)/float(m)
        return predictLabel

class LogisticRegression(): # 二分类，0/1分类
    def __init__(self):
        pass

    def createVocabList(self, train_x):
        vocabSet = set([])
        for wordList in train_x:
            vocabSet = vocabSet | set(wordList)
        return list(vocabSet)

    def listOfWords2Vec(self, vocabList, wordList):
        wordsVec = [0] * len(vocabList)
        for word in wordList:
            if word in vocabList:
                wordsVec[vocabList.index(word)] = 1 # 词集模型
                # wordsVec[vocabList.index(word)] += 1 # 词袋模型
            # else:
            #     print "the word:%s is not in my vocabulary!" % word
        return wordsVec

    def sigmoid(self, inX):
        return 1.0/(1 + np.exp(-inX))

    # 使用梯度下降方法训练模型，alpha为步长（学习率），maxCycles最大迭代次数
    def fit(self, train_x, train_y, alpha=0.01, maxCycles=100):
        vocabList = self.createVocabList(train_x)
        trainMat = []
        for wordList in train_x:
            trainMat.append(self.listOfWords2Vec(vocabList, wordList))
        trainMatrix = np.matrix(trainMat) ## matrix是二维的 # size: m*n
        trainLabel = np.matrix(train_y).T ## matrix是二维的 # size: m*1
        m, n = trainMatrix.shape
        weigh = np.matrix(np.ones((n, 1))) # size: n*1
        for i in range(maxCycles):
            hx = self.sigmoid(trainMatrix*weigh) # size: m*1 sigmoid把线性回归转换到[0,1]之间，对应概率
            error = np.abs(trainLabel-hx) # size: m*1
            weigh += alpha*trainMatrix.T*error # size: n*1
        return vocabList, weigh

    # 使用学习得到的参数进行分类
    def predict(self, test_X, vocabList, weigh):
        testMat = []
        for wordList in test_X:
            testMat.append(self.listOfWords2Vec(vocabList, wordList))
        testMatrix = np.matrix(testMat) ## matrix是二维的
        m = testMatrix.shape[0]
        hx = self.sigmoid(testMatrix*weigh) # size: m*1 sigmoid把线性回归转换到[0,1]之间，对应概率
        predict_y = []
        for i in range(m): ## 对应0/1分类
            if hx[i][0] > 0.5:
                predict_y.append(1)
            else:
                predict_y.append(0)
        predictLabel = np.array(predict_y) ## array
        return predictLabel

    # 使用学习得到的参数进行分类
    def predict1(self, test_X, test_y, vocabList, weigh):
        testMat = []
        for wordList in test_X:
            testMat.append(self.listOfWords2Vec(vocabList, wordList))
        testMatrix = np.matrix(testMat) ## matrix是二维的
        m = testMatrix.shape[0]
        hx = self.sigmoid(testMatrix*weigh) # size: m*1 sigmoid把线性回归转换到[0,1]之间，对应概率
        predict_y = []
        for i in range(m): ## 对应0/1分类
            if hx[i][0] > 0.5:
                predict_y.append(1)
            else:
                predict_y.append(0)
        predictLabel = np.array(predict_y) ## array
        testLabel = np.array(test_y) ## array
        print 'accuracy:', sum(testLabel==predictLabel)/float(m)
        return predictLabel

def loadTrainDataSet():
    train_x = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
               ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
               ['my', 'dalmation', 'is', 'so', 'cute', ' and', 'I', 'love', 'him'],
               ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
               ['mr', 'licks','ate','my', 'steak', 'how', 'to', 'stop', 'him'],
               ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    train_y = [0,1,0,1,0,1]
    return train_x, train_y

def loadTestDataSet():
    test_X = [['love', 'my', 'girl', 'friend'],
              ['stupid', 'garbage'],
              ['Haha', 'I', 'really', "Love", "You"],
              ['This', 'is', "my", "dog"]]
    test_y = [0,1,0,0]
    return test_X, test_y


class NaiveBayes_neg1_pos1():
    def __init__(self):
        pass

    def createVocabList(self, train_x):
        vocabSet = set([])
        for wordList in train_x:
            vocabSet = vocabSet | set(wordList)
        return list(vocabSet)

    def listOfWords2Vec(self, vocabList, wordList):
        wordsVec = [0] * len(vocabList)
        for word in wordList:
            if word in vocabList:
                wordsVec[vocabList.index(word)] = 1 # 词集模型
                # wordsVec[vocabList.index(word)] += 1 # 词袋模型
            # else:
            #     print "the word:%s is not in my vocabulary!" % word
        return wordsVec

    def fit(self, train_x, train_y):
        vocabList = self.createVocabList(train_x)
        trainMat = []
        for wordList in train_x:
            trainMat.append(self.listOfWords2Vec(vocabList, wordList))
        trainMatrix = np.array(trainMat) ## array
        trainLabel = np.array(train_y) ## array
        numTrainDocs = len(trainMatrix) # 统计样本个数
        numWords = len(trainMatrix[0])  # 统计特征个数，理论上是词库的长度
        ## 计算p(c0)，p(c1) ## 对应-1/1分类
        p0 = sum(trainLabel==-1)/float(numTrainDocs) # 对应p(c0)
        p1 = sum(trainLabel==1)/float(numTrainDocs) # 对应p(c1)
        ## 计算p(wi|c0)，p(wi|c1)
        p0Num = np.ones(numWords) # 初始样本个数为1，防止条件概率为0，影响结果
        p1Num = np.ones(numWords)
        p0InAll = 2.0 # 词库中只有两类，所以此处初始化为2
        p1InAll = 2.0
        for i in range(numTrainDocs):
            if trainLabel[i] == 1:
                p1Num += trainMatrix[i]
                p1InAll += sum(trainMatrix[i])
            else:
                p0Num += trainMatrix[i]
                p0InAll += sum(trainMatrix[i])
        p0Vec = np.log(p0Num/p0InAll) # 对应p(wi|c0)
        p1Vec = np.log(p1Num/p1InAll) # 对应p(wi|c1)
        ## 整合参数
        param = p0, p1, p0Vec, p1Vec
        return vocabList, param

    def predict(self, test_X, vocabList, param):
        p0, p1, p0Vec, p1Vec = param
        testMat = []
        for wordList in test_X:
            testMat.append(self.listOfWords2Vec(vocabList, wordList))
        testMatrix = np.array(testMat) ## array
        predict_y = []
        for vec in testMatrix:
            prob_y0 = sum(vec*p0Vec)+np.log(p0) # 对应p(w1|c0)*p(w2|c0)*...*p(c0)，log(a*b) = log(a)+log(b)
            prob_y1 = sum(vec*p1Vec)+np.log(p1) # 对应p(w1|c1)*p(w2|c1)*...*p(c1)，log(a*b) = log(a)+log(b)
            if prob_y0 < prob_y1: ## 对应-1/1分类，但是NaiveBayes可以修改成多分类
                predict_y.append(1)
            else:
                predict_y.append(-1)
        predictLabel = np.array(predict_y) ## array
        return predictLabel

    def predict1(self, test_X, test_y, vocabList, param):
        p0, p1, p0Vec, p1Vec = param
        testMat = []
        for wordList in test_X:
            testMat.append(self.listOfWords2Vec(vocabList, wordList))
        testMatrix = np.array(testMat) ## array
        m = testMatrix.shape[0]
        predict_y = []
        for vec in testMatrix:
            prob_y0 = sum(vec*p0Vec)+np.log(p0) # 对应p(w1|c0)*p(w2|c0)*...*p(c0)，log(a*b) = log(a)+log(b)
            prob_y1 = sum(vec*p1Vec)+np.log(p1) # 对应p(w1|c1)*p(w2|c1)*...*p(c1)，log(a*b) = log(a)+log(b)
            if prob_y0 < prob_y1: ## 对应-1/1分类，但是NaiveBayes可以修改成多分类
                predict_y.append(1)
            else:
                predict_y.append(-1)
        testLabel = np.array(test_y) ## array
        predictLabel = np.array(predict_y) ## array
        print 'accuracy:', sum(testLabel==predictLabel)/float(m)
        return predictLabel

class LogisticRegression_neg1_pos1(): # 二分类，-1/1分类
    def __init__(self):
        pass

    def createVocabList(self, train_x):
        vocabSet = set([])
        for wordList in train_x:
            vocabSet = vocabSet | set(wordList)
        return list(vocabSet)

    def listOfWords2Vec(self, vocabList, wordList):
        wordsVec = [0] * len(vocabList)
        for word in wordList:
            if word in vocabList:
                wordsVec[vocabList.index(word)] = 1 # 词集模型
                # wordsVec[vocabList.index(word)] += 1 # 词袋模型
            # else:
            #     print "the word:%s is not in my vocabulary!" % word
        return wordsVec

    def sigmoid(self, inX):
        return 1.0/(1 + np.exp(-inX))

    # 使用梯度下降方法训练模型，alpha为步长（学习率），maxCycles最大迭代次数
    def fit(self, train_x, train_y, alpha=0.01, maxCycles=100):
        vocabList = self.createVocabList(train_x)
        trainMat = []
        for wordList in train_x:
            trainMat.append(self.listOfWords2Vec(vocabList, wordList))
        trainMatrix = np.matrix(trainMat) ## matrix是二维的 # size: m*n
        trainLabel = np.matrix(train_y).T ## matrix是二维的 # size: m*1
        m, n = trainMatrix.shape
        weigh = np.matrix(np.ones((n, 1))) # size: n*1
        for i in range(maxCycles):
            # hx = self.sigmoid(trainMatrix*weigh) # size: m*1 sigmoid把线性回归转换到[0,1]之间，对应概率
            hx = self.sigmoid(trainMatrix*weigh)*2-1
            error = np.abs(trainLabel-hx) # size: m*1
            weigh += alpha*trainMatrix.T*error # size: n*1
        return vocabList, weigh

    # 使用学习得到的参数进行分类
    def predict(self, test_X, vocabList, weigh):
        testMat = []
        for wordList in test_X:
            testMat.append(self.listOfWords2Vec(vocabList, wordList))
        testMatrix = np.matrix(testMat) ## matrix是二维的
        m = testMatrix.shape[0]
        # hx = self.sigmoid(testMatrix*weigh) # size: m*1 sigmoid把线性回归转换到[0,1]之间，对应概率
        hx = self.sigmoid(testMatrix*weigh)*2-1
        predict_y = []
        for i in range(m): ## 对应-1/1分类
            if hx[i][0] > 0.0:
                predict_y.append(1)
            else:
                predict_y.append(-1)
        predictLabel = np.array(predict_y) ## array
        return predictLabel

    # 使用学习得到的参数进行分类
    def predict1(self, test_X, test_y, vocabList, weigh):
        testMat = []
        for wordList in test_X:
            testMat.append(self.listOfWords2Vec(vocabList, wordList))
        testMatrix = np.matrix(testMat) ## matrix是二维的
        m = testMatrix.shape[0]
        # hx = self.sigmoid(testMatrix*weigh) # size: m*1 sigmoid把线性回归转换到[0,1]之间，对应概率
        hx = self.sigmoid(testMatrix*weigh)*2-1
        predict_y = []
        for i in range(m): ## 对应-1/1分类
            if hx[i][0] > 0.0:
                predict_y.append(1)
            else:
                predict_y.append(-1)
        predictLabel = np.array(predict_y) ## array
        testLabel = np.array(test_y) ## array
        print 'accuracy:', sum(testLabel==predictLabel)/float(m)
        return predictLabel

def loadTrainDataSet_neg1_pos1():
    train_x = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
               ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
               ['my', 'dalmation', 'is', 'so', 'cute', ' and', 'I', 'love', 'him'],
               ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
               ['mr', 'licks','ate','my', 'steak', 'how', 'to', 'stop', 'him'],
               ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    train_y = [-1,1,-1,1,-1,1]
    return train_x, train_y

def loadTestDataSet_neg1_pos1():
    test_X = [['love', 'my', 'girl', 'friend'],
              ['stupid', 'garbage'],
              ['Haha', 'I', 'really', "Love", "You"],
              ['This', 'is', "my", "dog"]]
    test_y = [-1,1,-1,-1]
    return test_X, test_y


if __name__ == '__main__':

    train_X, train_y = loadTrainDataSet()
    test_X, test_y = loadTestDataSet()
    clf = NaiveBayes()
    vocabList, param = clf.fit(train_X, train_y)
    results = clf.predict(test_X, vocabList, param)
    print results
    results1 = clf.predict1(test_X, test_y, vocabList, param)
    print results1
    clf = LogisticRegression()
    vocabList, weigh = clf.fit(train_X, train_y)
    results = clf.predict(test_X, vocabList, weigh)
    print results
    results1 = clf.predict1(test_X, test_y, vocabList, weigh)
    print results1

    train_X, train_y = loadTrainDataSet_neg1_pos1()
    test_X, test_y = loadTestDataSet_neg1_pos1()
    clf = NaiveBayes_neg1_pos1()
    vocabList, param = clf.fit(train_X, train_y)
    results = clf.predict(test_X, vocabList, param)
    print results
    results1 = clf.predict1(test_X, test_y, vocabList, param)
    print results1
    clf = LogisticRegression_neg1_pos1()
    vocabList, weigh = clf.fit(train_X, train_y)
    results = clf.predict(test_X, vocabList, weigh)
    print results
    results1 = clf.predict1(test_X, test_y, vocabList, weigh)
    print results1