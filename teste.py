import numpy as np
import math
import random

def NN(value1, value2, weight1, weight2, bias):
    result = value1 * weight1 + value2 * weight2 + bias
    return sigmoid(result)


def sigmoid(value):
    return 1/(1 + np.exp(-value))


def cost(number):
    return (number -4) ** 2

def num_slope(number):
    h = 0.0001
    return (cost(number+h) - cost(number))/h

def slope(number):
    return 2 * (number - 4)




weight1 = np.random.randn()
weight2 = np.random.randn()
bias = np.random.randn()

b = 6







arr = np.zeros(shape=(2,3))
arr[0] = [1,2,3]

rand = np.random.uniform(-1,1,(5,1))

print(rand)