#Implement an image colorizer using linear regression model with stochastic gradient descent optimizer

from PIL import Image
import matplotlib.pyplot as plt
from os import listdir
import numpy as np
import random as rnd

# Training Data handling


def loadImages(path):
    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList:
        img = Image.open(path + image)
        loadedImages.append(img)
    return loadedImages

# Color To Gray Conversion Function


def colortogray(imgarr):
    grayImages = []
    i = 0
    for img in imgarr:
        graytemp = 0.21*img[:, :, 0] + 0.72*img[:, :, 1] + 0.07*img[:, :, 2]
        grayImages.append(graytemp.astype(np.float))
        # print(i)
        i += 1
    return grayImages

# 3-by-3 image slice


def input_slice(gray_training_img):
    i = rnd.randint(1, 253)
    j = rnd.randint(1, 253)
    centre_pixel_i = i+1
    centre_pixel_j = j+1
    test_slice = gray_training_img[i:i+3, j:j+3]
    return test_slice, centre_pixel_i, centre_pixel_j

# RGB pixel for loss calculation


def true_rgb_val(training_img_true, pixel_i, pixel_j):
    return training_img_true[pixel_i, pixel_j, :]

# Sigmoid Function


def sgm(x):
    return 1 / (1 + np.exp(-x))

# Derivative of Sigmoid by passing sgm(x) values


def der_sgm(x):
    return x * (1 - x)

# Random Weight Matrix Initialization


def gen_weightmatrix1():
    weights_layer1 = np.ones((9, 6))
    for i in range(9):
        for j in range(6):
            weights_layer1[i, j] = 0.2
    return weights_layer1


def gen_weightmatrix2():
    weights_layer2 = np.ones((6, 3))
    for i in range(6):
        for j in range(3):
            weights_layer2[i, j] = 0.1
    return weights_layer2

# Layer1_output


def out_layer1(input_slice, weights_layer1):
    temp_out = np.zeros((1, 3))
    temp_in = np.resize(input_slice, (1, 9))
    temp_out = sgm(np.dot(temp_in, weights_layer1))
    return temp_out

# Layer2_output


def out_layer2(Layer1_output, weights_layer2):
    temp_out2 = np.zeros((1, 3))
    temp_out2 = sgm(np.dot(Layer1_output, weights_layer2))
    return temp_out2


def forward_propagate(inputData, weights_layer1, weights_layer2):
    inputToLayer1 = np.dot(inputData, weights_layer1)
    outputLayer1 = sgm(inputToLayer1)
    
    inputToLayer2 = np.dot(outputLayer1, weights_layer2)
    outputLayer2 = sgm(inputToLayer2)
    return outputLayer2, outputLayer1

def back_propagate(inputData, true_output, predicted_output, weights_layer2, outputLayer1):
    error = predicted_output - true_output
    delta_error = error * der_sgm(predicted_output)
    #delta_error_final = np.transpose(np.dot(np.transpose(delta_error), outputLayer1))
    
    error_hidden = np.dot(delta_error, np.transpose(weights_layer2))
    delta_error_hidden = error_hidden * der_sgm(outputLayer1)
    #delta_error_hidden_final = np.dot(np.transpose(inputData), delta_error_hidden)
    return delta_error, delta_error_hidden
    
def delta_loss_output_hidden(layer2_predicted_output, true_output, layer1_output, weights_layer2):
    difference = layer2_predicted_output - true_output
    derivative = der_sgm(layer2_predicted_output)
    difference = np.squeeze(np.asarray(difference))
    intermediateResult = difference * derivative
    intermediateResult = np.reshape(intermediateResult, (1, 3))
    result = np.dot(np.transpose(intermediateResult), layer1_output)
    return result, intermediateResult

def delta_loss_hidden_input(layer2_predicted_output, true_output, output_layer_1, weights_layer2, input_layer, weights_layer1, result):
    result2 = np.dot(result, np.transpose(weights_layer2))
    result2 = np.squeeze(np.asarray(result2))
    
    derivative2 = der_sgm(output_layer_1)
    derivative2 = np.squeeze(np.asarray(derivative2))
    
    deltaHidden = result2 * derivative2
    deltaHidden = np.reshape(deltaHidden, (3, 1))
    
    deltaWeights = np.transpose(np.dot(deltaHidden, input_layer))
    print(deltaWeights)
    return deltaWeights

def test_trained_weights(weights_layer1, weights_layer2, training_img_true, gray_training_img, recon):
    for i in range(253):
        for j in range(253):
            input_slice_test = gray_training_img[i:i+3, j:j+3]
            inputData = np.reshape(input_slice_test, (1,9))
            output, output_layer1 = forward_propagate(inputData, weights_layer1, weights_layer2)
            recon[i+1, j+1, :] = output
    print("shape:", np.shape(recon))
    recon = recon * 255
    print(recon)
    recon_img = Image.fromarray(recon.astype(np.uint8))
    recon_img.save("recon1.png")
    training_img_true = training_img_true*255

    training_img_true = Image.fromarray(training_img_true.astype(np.uint8))
    training_img_true.save("true_img.png")

if __name__ == '__main__':
    path = "./img_db/"
    print("working")
    input_slice_test = np.zeros((3,3))
    recon = np.zeros((256, 256, 3))
    color_imgs_true = loadImages(path)
    print("color image returned:", color_imgs_true)
    test_img = []
    color_img_arr = []
    for img in color_imgs_true:
        img = img.resize((256, 256), Image.BOX)
        color_img_arr.append(np.array(img))

    training_img_true = color_img_arr[0]
    test_img = color_img_arr[0]

    training_img_true = training_img_true/255

    gray_imgs = colortogray(color_img_arr)
    ##
    gray_training_img = []
    gray_training_img = gray_imgs[0]
    img1 = Image.fromarray(gray_training_img.astype(np.uint8))
    img1.save("beforeTrain.png")
    gray_training_img = gray_training_img/255

    weights_layer1 = gen_weightmatrix1()
    weights_layer2 = gen_weightmatrix2()

    # for loop for iterations.
    print("gray shape:", np.shape(gray_imgs))
    print("color shape:", np.shape(color_img_arr[0]))
    j = 0
    for img in gray_imgs:
        training_img_true = color_img_arr[j]
        training_img_true = training_img_true/255
        print("getting next image:", training_img_true)
        for i in range(100000):
            temp_input_slice, pix_i, pix_j = input_slice(img)
            true_inp_slice = true_rgb_val(training_img_true, pix_i, pix_j)
            temp_input_slice = np.reshape(temp_input_slice, (1,9))
            # forward propagate.
            output, outputLayer1 = forward_propagate(temp_input_slice, weights_layer1, weights_layer2)
    
            meanSquareError = np.mean(np.square(output - true_inp_slice))
            
            delta_error, delta_error_hidden = back_propagate(temp_input_slice, true_inp_slice, output, weights_layer2, outputLayer1)
            
            # update weights
            diffLayer1 = np.dot(np.transpose(temp_input_slice), delta_error_hidden)
            diffLayer2 = np.dot(np.transpose(outputLayer1), delta_error)
            weights_layer1 = weights_layer1 - 0.01 * diffLayer1
            weights_layer2 = weights_layer2 - 0.01 * diffLayer2
        print("weights1:", weights_layer1)
        print("weights2:", weights_layer2)
        print("Mean square error:", meanSquareError)
        j+=1
    test_trained_weights(weights_layer1, weights_layer2, training_img_true, gray_training_img, recon)
