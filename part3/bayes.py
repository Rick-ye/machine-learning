from numpy.ma import zeros, log, array


def loadDataSet():
    """
    创建实验样本
    :return:
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 1代表侮辱性词汇，0代表正常言论
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


def createVocabList(dataSet):
    """
    创建一个包含所有文档中出现的不重复词的列表，
    :param dataSet: 输入数据集
    :return:
    """
    vocabSet = set([])
    for document in dataSet:
        # 创建两个集合的并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """
    识别一个文档是在在指定词汇表中是否出现
    :param vocabList: 词汇表
    :param inputSet: 某个文档
    :return: 文档向量，向量的每一元素为1或0，表示词汇表中的单词在输入文档中是否出现
    """
    # 创建一个其中所含元素都为0的向量
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my vocabulary!" % word)
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    """

    :param trainMatrix: 训练集输入向量
    :param trainCategory: 每篇文档类别标签所构成的向量
    :return: pAbusive：侮辱性文档的概率；
    给定文档类别条件下p0:词汇表中正常单词出现的概率，p1:词汇表中侮辱性单词出现的概率
    """
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    # sum函数求和
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = zeros(numWords)
    p1Num = zeros(numWords)
    p0Denom, p1Denom = 0.0, 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            print("trainMatrix[i]:", trainMatrix[i])
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num/p1Denom
    p0Vect = p0Num/p0Denom
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listoPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listoPosts)
    trainMat = []
    for postinDoc in listoPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


if __name__ == '__main__':
    listoPosts, listClasses = loadDataSet()
    vocaList = createVocabList(listoPosts)
    print(vocaList)
    trainMat = []
    for postinDoc in listoPosts:
        trainMat.append(setOfWords2Vec(vocaList, postinDoc))
    print(trainMat)
    p0, p1, pa = trainNB0(trainMat, listClasses)
    print(p0, p1, pa)
    # vec = setOfWords2Vec(vocaList, listoPosts[1])
    # print(vec)

    # testingNB()

