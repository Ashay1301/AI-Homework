import numpy as np
from Perceptron1 import Perceptron

trainin_inputs = []
trainin_inputs.append(np.array([1,1,1]))
trainin_inputs.append(np.array([0,1,0]))
trainin_inputs.append(np.array([1,0,1]))
trainin_inputs.append(np.array([0,0,0]))
trainin_inputs.append(np.array([0,0,1]))
trainin_inputs.append(np.array([0,1,1]))

labels = np.array([1,0,0,0,0,0])

perceptron = Perceptron(3)
perceptron.train(trainin_inputs,labels)

inputs = np.array([1,1,0])
print(" [1,1,0] ",perceptron.predict(inputs))

inputs = np.array([1,0,0])
print(" [1,0,0] ",perceptron.predict(inputs))