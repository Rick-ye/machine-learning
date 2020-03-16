from array import array

from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

from os import listdir

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    """
    k-近邻算法
    :param inX: 输入向量
    :param dataSet: 训练集样本
    :param labels: 标签向量
    :param k: 选在最近邻居的数目
    :return:
    """
    # 1. 计算已知类别数据集中的点与当前点之间的距离
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5

    # 2.按照距离递增次序排序
    sortedDistIndicies = distances.argsort()

    # 3.选取与当前距离最小的k个点
    classCount = {}
    count = 0
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 4.确定前k个点所在类别的出现频率
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    # 5.返回前k个点出现频率最高的类别作为当前点的预测分类
    return sortedClassCount[0][0]


def file2matrix(filename):
    """
    1. 准备数据：将文本记录转换为NumPy的解析程序
    :param filename:
    :return:
    """
    love_dictionary = {'largeDoses': 3, 'smallDoses': 2, 'didntLike': 1}
    fr = open(filename)  # 返回文件对象
    # 读取文件
    arrayOLines = fr.readlines()  # 返回列表
    numberOfLines = len(arrayOLines)
    # 创建返回的Numpy矩阵
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        # 解析文件数据列表
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        if listFromLine[-1].isdigit():
            # -1表示列表中的最后一列元素
            classLabelVector.append(int(listFromLine[-1]))
        else:
            classLabelVector.append(love_dictionary.get(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    """
    第三步，归一化数据
    :param dataSet:
    :return:
    """
    # 从列表中取出最小值，而不是选取当前行最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    # shape(dataSet)
    m = dataSet.shape[0]
    # tile函数改变矩阵大小。
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    """
    分类器针对约会网站的测试代码
    :return:
    """
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('./data/datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        # [i, :]取出第i行所有数据
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("%d: the classifier came back with: %d, the real answer is: %d" % (i, classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))


def img2vector(filename):
    """
    将图像转换成向量  32*32-->1*1024
    :param filename:
    :return:
    """
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handWritingClassTest():
    """
    识别手写数字
    :return:
    """
    hwLabels = []
    trainingFileList = listdir('./data/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNum = int(fileStr.split('_')[0])
        hwLabels.append(classNum)
        trainingMat[i, :] = img2vector('./data/trainingDigits/%s' % fileNameStr)
    testFileList = listdir('./data/testDigits')
    n = len(testFileList)
    errorCount = 0.0
    for j in range(n):
        fileNameStr = testFileList[j]
        fileStr = fileNameStr.split('.')[0]
        testNum = int(fileStr.split('_')[0])
        testMat = img2vector('./data/testDigits/%s' % fileNameStr)
        result = classify0(testMat, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (result, testNum))
        if testNum != result:
            errorCount += 1
    print("the error count: %d, the error rate is: %f" % (errorCount, errorCount/float(n)))


if __name__ == '__main__':
    # datingClassTest()
    handWritingClassTest()

    # group, labels = createDataSet()
    # sort = classify0([0, 2], group, labels, 3)
    # datingDataMat, datingLabels = file2matrix('./data/datingTestSet.txt')
    # normMat, ranges, minVals = autoNorm(datingDataMat)
    # # 第二部分析数据
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # # [:, n]取所有集合的第n个数据
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0*array(datingLabels), 15.0*array(datingLabels))
    # plt.show()
    # print(datingDataMat, datingLabels[0:20])
    #
    # print("normMat: (%s)" %(normMat))
    # print(ranges, minVals)
