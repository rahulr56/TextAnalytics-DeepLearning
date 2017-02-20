from datetime import datetime as dt
import copy
import math


class Utility:
    def sigmoid(x):
        from math import exp
        return (1/(1+exp(-1*x)))

    def dSigmoid(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def dotProd(l1, l2):
        from numpy import dot
        return round(dot(l1, l2), 3)

    def getSeed(self):
        i=dt.now()
        return int(i.strftime("%H%M%S"))

    def generateWeights(size, a=0 ,b=1):
        from random import uniform
        randArr=[]
        for x in range(size):
            randArr.append(round(uniform(a,b), 3))
        return randArr


class DeepLearning:
    def __init__(self):
        layers=100
        function="logarithamic"

    def intializeNN(self, ilayers,ifunction):
        layers=ilayers
        function=ifunction

    def createNN(self, input, result):
        inputRows=input.shape[0]
        inputCols=input.shape[0]

myUtils = Utility
neuronValue = [[0,	0,	8,	15,	16,	13,	0,	0,
                0,	1,	11,	9,	11,	16,	1,	0,
                0,	0,	0,	0,	7,	14,	0,	0,
                0,	0,	3,	4,	14,	12,	2,	0,
                0,	1,	16,	16,	16,	16,	10,	0,
                0,	2,	12,	16,	10,	0,	0,	0,
                0,	0,	2,	16,	4,	0,	0,	0,
                0,	0,	9,	14,	0,	0,	0,	0]]
expectedResult = 7
weights = [Utility.generateWeights(len(neuronValue[0]))]

neurons = []
counter = 0


N = math.ceil(40*len(neuronValue[0])/100)
prev_num_neurons = len(neuronValue[0])
neurons_in_each_layer = []

#  NN1
for i in range(N):
    neuronValue.append([])
    weights.append([])
    if i > 0:
        prev_num_neurons = math.ceil(math.log(N - i + 1, 2) * 2 + 1)
    num_of_neurons = math.ceil(math.log(N - i, 2) * 2 + 1)
    neurons_in_each_layer.append(num_of_neurons)
    print('\nLevel %d with %d neuron[s]; previous layer %d with %d neuron[s]\n' % (i + 1, num_of_neurons, i, prev_num_neurons))
    for j in range(num_of_neurons):
        counter += 1
        print('\tNeuron %d, id %d' % (j, counter))
        neuronValue[i+1].append(myUtils.sigmoid(myUtils.dotProd(neuronValue[i], weights[i])))
        weights[i + 1].append(myUtils.generateWeights(1)[0])

prediction = myUtils.sigmoid(myUtils.dotProd(neuronValue[-1], weights[-1]))
error = ((prediction - expectedResult)**2)/2
print('Error %.2f' % error)

# while error < 10:
#       neuronValue[3] = myUtil.dSigmoid(neuronValue[4])
#       neuronValue[2] = myUtil.dSigmoid(neuronValue[3])
#       neuronValue[1] = myUtil.dSigmoid(neuronValue[2])
#       call NN1
#
    # pass
