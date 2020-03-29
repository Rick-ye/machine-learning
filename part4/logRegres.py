import random

import numpy as np


def loadDataSet():
    """
    读取文件准备书
    :return: 返回训练数据和目标值
    """
    trainMat, labelMat = [], []
    fr = open('data/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        # 1.0有何用
        trainMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[-1]))
    return trainMat, labelMat


def sigmoid(inX):
    """
    sigmoid函数
    :param inX:
    :return:
    """
    # return 1.0/(1+np.exp(-inX))
    # 优化
    if inX >= 0:
        return 1.0/(1+np.exp(-inX))
    else:
        return np.exp(inX)/(1+np.exp(inX))


def gradAscent(dataMatIn, classLabels):
    """
    梯度上升优化算法找到最佳参数
    :param dataMatIn:
    :param classLabels:
    :return:
    """
    # 转换为Numpy矩阵数据类型
    dataMat = np.mat(dataMatIn)
    # transpose()将行向量转换为列向量
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMat)
    print(m, n)
    alpha = 0.001
    maxCycle = 500
    weights = np.ones((n, 1))
    for k in range(maxCycle):
        # h: 100 X 1
        h = sigmoid(dataMat * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMat.transpose() * error
    return weights


def stocGradAscent0(dataMatrix, classLabels, numIter=150):
    """
    随机梯度上升算法
    :param dataMatrix:
    :param classLabels:
    :param numIter: 迭代次数
    :return:
    """
    m, n = np.shape(dataMatrix)
    dataMatrix = np.array(dataMatrix)
    # alpha每次迭代时都需要调整
    # alpha = 0.01
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0 + j + i) + 0.01
            # 随机选取样本更新
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


def plotBestFit(weights):
    """
    画出数据集和Logistic回归最佳拟合直线
    :param weights:
    :return:
    """
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1, ycord1 = [], []
    xcord2, ycord2 = [], []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    frTrain = open('data/horseColicTraining.txt')
    frTest = open('data/horseColicTest.txt')
    trainSet = []
    trainLabel = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineSet = []
        for i in range(21):
            lineSet.append(float(currLine[i]))
        trainSet.append(lineSet)
        trainLabel.append(float(currLine[-1]))
    weights = stocGradAscent0(trainSet, trainLabel)
    errorCount, numTestVec = 0, 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(lineArr, weights)) != int(currLine[-1]):
            errorCount += 1
    errorRate = float(errorCount)/numTestVec
    print("the error rate of this test is: ", errorRate)
    return errorRate


def multiTest():
    count, rateSum = 10, 0.0
    for i in range(count):
        rate = colicTest()
        rateSum += rate
    rateAvg = rateSum/float(count)
    print("rateAvg: ", rateAvg)
    return rateAvg



if __name__ == '__main__':
    # trainMat, labelMat = loadDataSet()
    # weights = gradAscent(trainMat, labelMat)
    # weights = stocGradAscent0(trainMat, labelMat)
    # plotBestFit(weights)
    # print(weights)

    multiTest()