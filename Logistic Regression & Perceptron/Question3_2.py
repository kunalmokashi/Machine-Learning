# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 17:17:39 2018

@author: Kunal
CS 520 - Question 3
Logistic regression model for classification
"""

import numpy as np
import random as rnd
import math as mth

# generate weight matrix randomly.
def generateWeightMatrix():
    weightsMatrix = np.random.random((25,1))
    return weightsMatrix

# uses sigmoid activation function.
def activationFunction(inputValue):
    return 1 / (1 + np.exp(-inputValue))

# return the output based on the output of the sigmoid function.
def forward(inputData, weights):
    inputToFunction = np.dot(inputData, weights) 
    output = activationFunction(inputToFunction)
    return output

# Used to test images after the model has been trained.
def test(testImage, weights, label):
    output = forward(testImage, weights)
    if label == '':
        if output <= 0.5:
            # classify as A
            label = 'A'
        else:
            # classify as B
            label = 'B'
    print("predicted output label - ", output, ":predicted class label - ", label)
    return output, label

if __name__ == '__main__':
    
    learningRate = 0.01
    lambdaRate = 0.0001 # For regularizer
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
    
    # true labels for the training data
    labels = np.array([0,0,0,0,0,1,1,1,1,1])
    
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
    for i in range(50000):
        # pick a random sample.
        index = rnd.randint(0,9)
        trainingImg = trainingData[index]
        trainingImg = np.reshape(trainingImg,(1,25))
        actualLabel = labels[index]
        predictedLabel = forward(trainingImg, weights)
        loss = (-actualLabel * mth.log(predictedLabel)) - ((1 - actualLabel) * mth.log(1-predictedLabel))
        if loss > 0 and loss < 0.0001:
            break
        difference = predictedLabel - actualLabel
        weights = weights - learningRate * (difference * np.transpose(trainingImg) + lambdaRate * weights)
    # test the data now.
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
    
    # These are the mystery images.
    output1, label1 = test(testData1, weights, '')
    output2, label2 = test(testData2, weights, '')
    output3, label3 = test(testData3, weights, '')
    output4, label4 = test(testData4, weights, '')
    output5, label5 = test(testData5, weights, '')
    
    
