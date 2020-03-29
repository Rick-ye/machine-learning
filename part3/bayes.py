import random

from numpy.ma import zeros, log, array, ones


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


def bagOfWords2Vec(vocabList, inputSet):
    """
    识别一个文档是在指定词汇表中是否出现
    :param vocabList: 词汇表
    :param inputSet: 某个文档
    :return: 文档向量，向量的每一元素为1或0，表示词汇表中的单词在输入文档中是否出现
    """
    # 创建一个其中所含元素都为0的向量
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word: %s is not in my vocabulary!" % word)
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    """
    朴素贝叶斯分类器训练函数
    :param trainMatrix: 训练集输入向量
    :param trainCategory: 每篇文档类别标签所构成的向量
    :return: pAbusive：侮辱性文档的概率；
    给定文档类别条件下p0:词汇表中正常单词出现的概率，p1:词汇表中侮辱性单词出现的概率
    """
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    # sum函数求和
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # p0Num, p1Num = zeros(numWords), zeros(numWords)
    # p0Denom, p1Denom = 0.0, 0.0
    # 为防止出现概率为0的情况，将所有词的出现数初始化为1，分母初始化为2
    p0Num, p1Num = ones(numWords), ones(numWords)
    p0Denom, p1Denom = 2.0, 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # p1Vect = p1Num/p1Denom
    # p0Vect = p0Num/p0Denom
    # 为防止下溢出，太多很小的数相乘，在python里会出现下溢出变为0
    # 对每个元素除以该类别中的总词数
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    朴素贝叶斯分类函数
    :param vec2Classify:
    :param p0Vec:
    :param p1Vec:
    :param pClass1:
    :return:
    """
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
        trainMat.append(bagOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(bagOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(bagOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


def textParse(bigString):
    """
    将大字符串解析为字符串列表
    :param bigString: 大字符串
    :return: 字符串列表
    """
    import re
    # re.split()方法可以使用正则表达式匹配
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        # 导入并解析文本文件
        wordList = textParse(open('data/email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('data/email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    # 创建不重复词列表
    vocabList = createVocabList(docList)
    # range()返回range对象，不返回数组对象，创建一个0到50的列表
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount), len(testSet), float(errorCount)/len(testSet))


def localWords(feed1, feed0):
    import feedparser
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    print(ny)
    print(len(ny['feed']))

if __name__ == '__main__':
    # listoPosts, listClasses = loadDataSet()
    # vocaList = createVocabList(listoPosts)
    # print(vocaList)
    # trainMat = []
    # for postinDoc in listoPosts:
    #     trainMat.append(setOfWords2Vec(vocaList, postinDoc))
    # print(trainMat)
    # p0, p1, pa = trainNB0(trainMat, listClasses)
    # print(p0, p1, pa)
    # vec = setOfWords2Vec(vocaList, listoPosts[1])
    # print(vec)


    # testingNB()

    localWords(1, 1)
