from numpy import *

"""
训练数据集中的每个样板都赋予一个权重值，初始化每个权重值都相等，
第一次在数据集上训练出一个弱分类器，再次在相同数据集上训练弱分类器。
在第二次训练中，第一次预测分类的结果与真实的类别比较，将分类正确的
数据权重降低，反之，分类错误的数据权重提高。

为了从每个弱分类器中得到最终的结果，AdaBoost算法会对每个弱分类器分配一个
权重值alpha，这些alpha是通过每个弱分类器的错误率计算来的
"""


def loadSimpData():
    dataMat = matrix([[1.0, 2.1],
                      [2.0, 1.1],
                      [1.3, 1.0],
                      [1.0, 1.0],
                      [2.0, 1.0]
                      ])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'It':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    """
    单层决策树生成函数
    :param dataArr:
    :param classLabels:
    :param D:
    :return:
    """
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = mat(zeros((m, 1)))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin)/numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f"
                       % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst


if __name__ == '__main__':
    dataMat, labealMat = loadSimpData()
    D = mat(ones((5, 1))/5)
    bestStump, minError, bestClassEst = buildStump(dataMat, labealMat, D)
    print(bestStump, minError, bestClassEst)