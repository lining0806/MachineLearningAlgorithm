#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

'''
np.linalg   Core Linear Algebra Tools
xx.T    矩阵的转置
xx.I    矩阵的逆
m   样本点数
n   特征维数
'''

def loadDataSet(datafile):
    featData = []
    labelData = []
    with open(datafile, 'r') as fr_file:
        for eachLine in fr_file:
            oneLine = eachLine.split('\t')
            tempArr = []
            for i in range(len(oneLine)-1):
                tempArr.append(float(oneLine[i]))
            featData.append(tempArr)
            labelData.append(float(oneLine[-1].strip())) # float型连续变量
    featData = np.array(featData) # 转换为array
    labelData = np.array(labelData) # 转换为array
    return featData, labelData

def rssError(yArr, yHat):
    return np.sum((yArr-yHat)**2)

def showRegres(xArr, yArr, yHat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xArr[:, 1], yArr)
    '''
    因为数据假定了x0=1，因此yHat=ws[0]+ws[1]*x1，看yHat与x1之间的线性关系
    '''
    srtInd = xArr[:, 1].argsort(0)
    # print srtInd
    ax.plot(xArr[srtInd, 1], yHat[srtInd]) # 拟合前需要将点升序排列
    plt.show()

'''标准的线性回归：最小二乘法（平方误差最小），适用于m>=n情况'''
def standRegres(xMat, yMat):
    xTx = xMat.T*xMat # n*n
    if np.linalg.det(xTx) == 0.0:
        print 'This matrix is singular, cannot do inverse'
        return
    ## 方法1
    ws = xTx.I*(xMat.T*yMat) # n*1
    ## 方法2
    # ws = np.linalg.solve(xTx, xMat.T*yMat) # n*1
    # yHat = xMat*ws # m*1
    return ws

def standRegresTest(xArr, yArr):
    xMat = np.matrix(xArr) # m*n
    yMat = np.matrix(yArr).T # m*1
    ws = standRegres(xMat, yMat) # n*1
    # print ws
    yHat = xMat*ws # m*1
    yHat = np.array(yHat).reshape(1, -1)[0] ## [[xx1][xx2]]二维matrix为[xx1, xx2]一维array
    return yHat

'''局部加权线性回归，适用于m>=n情况'''
def lwlr(testPoint, xMat, yMat, k=1.0):
    m = np.shape(xMat)[0]
    weights = np.matrix(np.eye(m)) # 创建对角矩阵
    for j in range(m):
        diffMat = testPoint-xMat[j, :]
        weights[j, j] = np.exp(diffMat*diffMat.T/(-2.0*k**2)) # 高斯核
    print weights
    xTx = xMat.T*(weights*xMat)
    if np.linalg.det(xTx) == 0.0:
        print 'This matrix is singular, cannot do inverse'
        return
    ws = xTx.I*(xMat.T*(weights*yMat))
    return testPoint*ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
    xMat = np.matrix(xArr) # m*n
    yMat = np.matrix(yArr).T # m*1
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xMat, yMat, k)
    return yHat

'''岭回归，适用于m>=n及m<n情况'''
def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx+np.eye(np.shape(xMat)[1])*lam
    if np.linalg.det(denom) == 0.0:
        print 'This matrix is singular, cannot do inverse'
        return
    ws = denom.I*(xMat.T*yMat)
    return ws

def ridgeTest(xArr, yArr):
    xMat = np.matrix(xArr)
    yMat = np.matrix(yArr).T
    '''标准化XY'''
    ## regularize Y's
    yMean = np.mean(yMat, 0)
    yMat = yMat-yMean # to eliminate X0 take mean off of Y
    ## regularize X's
    xMeans = np.mean(xMat, 0) # calc mean then subtract it off
    xVar = np.var(xMat, 0) # calc variance of Xi then divide by it
    xMat = (xMat-xMeans)/xVar
    '''计算wMat'''
    numTestPts = 30
    wMat = np.matrix(np.zeros((numTestPts, np.shape(xMat)[1])))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, np.exp(i-10))
        wMat[i, :] = ws.T
    return wMat

if __name__ == '__main__':
    ####################################################################################
    ## 标准的线性回归
    xArr, yArr = loadDataSet('ex.txt')
    yHat = standRegresTest(xArr, yArr)
    print yHat
    showRegres(xArr, yArr, yHat)
    coef = np.corrcoef(yArr, yHat)
    print coef
    print (coef[0, 1]+coef[1, 0])/2.0
    print rssError(yArr, yHat)
    ####################################################################################
    ## 局部加权线性回归
    xArr, yArr = loadDataSet('ex.txt')
    yHat = lwlrTest(xArr, xArr, yArr, k=0.01)
    print yHat
    showRegres(xArr, yArr, yHat)
    coef = np.corrcoef(yArr, yHat)
    print coef
    print (coef[0, 1]+coef[1, 0])/2.0
    print rssError(yArr, yHat)
    # '''寻找使相关系数最大的k'''
    # max_k = 0
    # max_coef = 0
    # for k in range(1, 100):
    #     k /= 1000.0
    #     yHat = lwlrTest(xArr, xArr, yArr, k)
    #     coef = np.corrcoef(yArr, yHat)
    #     temp_coef = (coef[0, 1]+coef[1, 0])/2.0
    #     if temp_coef > max_coef:
    #         max_coef = temp_coef
    #         max_k = k
    # print max_k, max_coef
    # '''寻找使平方误差最小的k'''
    # min_k = 0
    # min_error = np.inf
    # for k in range(1, 100):
    #     k /= 1000.0
    #     yHat = lwlrTest(xArr, xArr, yArr, k)
    #     temp_error = rssError(yArr, yHat)
    #     if temp_error < min_error:
    #         min_error = temp_error
    #         min_k = k
    # print min_k, min_error
    ####################################################################################
    ## 岭回归
    xArr, yArr = loadDataSet('abalone.txt')
    wMat = ridgeTest(xArr, yArr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(wMat) # 描述回归系数与log(lam)的关系
    plt.show()
    '''
    在最左边时，lam为np.exp(0-10)=0，回归系数为原始值(即不缩减)，跟标准的线性回归一致
    在最右边时，lam为np.exp(20-10)=e^10，回归系数全部缩减为0
    因此，在中间的某部分取值，lam能得到最好的预测效果，去掉不重要回归参数，参数的大小表示其重要性
    '''
