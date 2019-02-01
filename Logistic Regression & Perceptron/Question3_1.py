# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 17:17:39 2018

@author: Kunal
CS 520 - Question 3
Preceptron Model for classification
"""

import numpy as np
import random as rnd

def generateWeightMatrix():
    weightsMatrix = np.random.random((25,1))
    return weightsMatrix

def activationFunction(inputValue):
    if inputValue >= 0:
        return 1
    else:
        return 0

def forward(inputData, weights):
    inputToFunction = np.dot(inputData, weights)
    output = activationFunction(inputToFunction)
    return output

def test(testImage, weights, label):
    output = forward(testImage, weights)
    if output == 0:
        label = 'A'
    else:
        label = 'B'
    print("predicted output label - ", output, ": Class Label - ", label)
    return output, label

if __name__ == '__main__':
    
    learningRate = 1
    trainingData = []
    # A label images
    trainingImg1 = np.array([1,0,0,1,1,1,1,1,0,0,1,1,1,0,0,1,0,0,1,1,0,0,0,0,1])
    trainingImg2 = np.array([0,0,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,0,0,0,0])
    trainingImg3 = np.array([1,1,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,1])
    trainingImg4 = np.array([0,0,1,1,1,1,0,0,0,0,1,1,1,0,0,1,1,0,0,1,1,0,0,0,0])
    trainingImg5 = np.array([0,0,0,1,1,0,0,0,0,0,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1])
    
    # B label images
    trainingImg6 = np.array([1,1,0,0,1,1,1,0,0,1,0,1,0,1,1,0,0,0,1,1,1,0,0,1,1])
    trainingImg7 = np.array([1,0,0,1,1,0,0,0,0,1,0,0,0,0,1,1,0,0,1,0,1,1,1,1,0])
    trainingImg8 = np.array([1,0,1,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,1,1,1,0,1])
    trainingImg9 = np.array([1,1,1,1,0,1,0,0,1,0,0,0,0,1,1,0,0,1,1,0,0,0,1,1,0])
    trainingImg10 = np.array([1,0,1,1,0,0,0,0,1,0,0,1,0,1,0,1,1,0,1,0,1,1,0,1,0])
    
    actual_label = np.array([0,0,0,0,0,1,1,1,1,1])
    
    trainingData.append(trainingImg1)
    trainingData.append(trainingImg2)
    trainingData.append(trainingImg3)
    trainingData.append(trainingImg4)
    trainingData.append(trainingImg5)
    trainingData.append(trainingImg6)
    trainingData.append(trainingImg7)
    trainingData.append(trainingImg8)
    trainingData.append(trainingImg9)
    trainingData.append(trainingImg10)
    
    # test data:
    testData1 = np.array([1,0,0,1,0,1,0,0,0,0,0,0,1,1,1,0,0,1,0,1,1,0,1,0,1])
    testData2 = np.array([1,1,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,1,0,1,0,1,1,1,1])
    testData3 = np.array([0,0,1,1,1,1,1,1,0,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1])
    testData4 = np.array([1,0,0,0,1,1,0,0,0,1,0,0,1,1,1,1,1,0,0,1,1,1,1,0,0])
    testData5 = np.array([0,1,1,1,1,0,1,0,0,1,1,1,1,1,0,1,0,0,1,0,1,1,1,1,1])
    weights = generateWeightMatrix()
    for i in range(1000):
        # pick a random sample.
        index = rnd.randint(0,9)
        trainingImg = trainingData[index]
        trainingImg = np.reshape(trainingImg, (1,25))
        actualOutput = actual_label[index]
        output = forward(trainingImg, weights)
        difference = output - actualOutput
        weights = weights - learningRate * difference * np.transpose(trainingImg)
    test(trainingImg1, weights, 'A')
    test(trainingImg2, weights, 'A')
    test(trainingImg3, weights, 'A')
    test(trainingImg4, weights, 'A')
    test(trainingImg5, weights, 'A')
    
    test(trainingImg6, weights, 'B')
    test(trainingImg7, weights, 'B')
    test(trainingImg8, weights, 'B')
    test(trainingImg9, weights, 'B')
    test(trainingImg10, weights, 'B')
    
    test(testData1, weights, '')
    test(testData2, weights, '')
    test(testData3, weights, '')
    test(testData4, weights, '')
    test(testData5, weights, '')
