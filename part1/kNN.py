from array import array

from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

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
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 4.确定前k个点所在类别的出现频率
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    print(sortedClassCount)
    # 5.返回前k个点出现频率最高的类别作为当前点的预测分类
    return sortedClassCount[0][0]


def file2matrix(filename):
    """
    分析数据：将文本记录转换为NumPy的解析程序
    :param filename:
    :return:
    """
    love_dictionary = {'largeDoses': 3, 'smallDoses': 2, 'didntLike': 1}
    fr = open(filename)
    # 读取文件
    arrayOLines = fr.readlines()
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


if __name__ == '__main__':
    # group, labels = createDataSet()
    # sort = classify0([0, 2], group, labels, 3)
    datingDataMat, datingLabels = file2matrix('./data/datingTestSet.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # [:, n]取所有集合的第n个数据
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0*array(datingLabels), 15.0*array(datingLabels))
    plt.show()
    print(datingDataMat, datingLabels[0:20])
