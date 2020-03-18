from numpy import *
import numpy as numpy

if __name__ == '__main__':
    # data = numpy.array([[1, 2, 5], [2, 3, 5], [3, 4, 5], [2, 3, 6]])
    # shape = zeros(shape(data))
    # m = data.shape[0]
    # print(shape)
    # print(m)
    #
    # data = [0, 0, 0.001156]
    # tile = tile(data, [1000, 1])
    # print(tile)

    data = numpy.array([[1, 2, 5], [2, 3, 5], [3, 4, 5], [2, 3, 6]])
    data2 = numpy.array([[1, 3, 2], [1, 4, 6], [4, 1, 9], [4, 6, 3]])
    # print(data[:, 1])
    # print(data[1, :])
    # size = data.shape[0]
    # for i in range(size):
    #     print(data[i, :])
    #     #print(data[1:3, :])

    # 计算距离
    # data1 = data - data2
    # print(data1)
    # diff = data1**2
    # print(diff)
    # print(diff.sum(axis=1))

    print(data.min(0))
    print(data.max(0))

    print(len(data), len(data[0]), sum(data))

    list = ['soft', 'hard', 'soft', 'hard', 'soft', 'hard', 'soft', 'no lenses', 'no lenses', 'hard', 'soft',
            'no lenses']
    list = set(list)
    print(list)


