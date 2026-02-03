import numpy as np

def sigmoid(x):
    #activation function: f(x) = 1 / (1 + e^(-x))
    return 1/(1+np.exp(-x))

class Neuron:
    def __init__(self, weights, bais):
        self.weights = weights
        self.bais = bais
    def feedfroward(self, inputs):
        #weight inputs, add bais, then use the activation function
        total = np.dot(self.weights, inputs) + self.bais
        return sigmoid(total)
    
weights = np.array([0, 1]) # w1 = 0, w2 = 1
bais = 4                   #b = 4

n = Neuron(weights, bais)

x = np.array([2, 3])      # w1 = 2, w2 = 3
print(n.feedfroward(x))   #0.9990889488055994

class OurNeuralNetwork:
    '''
    A neural network with:
        - 2 inputs
        - a hidden layer with 2 neurons (h1, h2)
    Each neuron has the same weights and bais:
        - w = [0, 1]
        - b = 0
    '''
    def __init__(self):
        weights = np.array([0, 1])
        bais = 0

        self.h1 = Neuron(weights, bais)
        self.h2 = Neuron(weights, bais)
        self.o1 = Neuron(weights, bais)
    def feedforward(self, x):
        out_h1 = self.h1.feedfroward(x)
        out_h2 = self.h2.feedfroward(x)

        #the inputs for o1 are tge outputs form h1 and h2
        out_o1 = self.o1.feedfroward(np.array([out_h1, out_h2]))
        return out_o1

network = OurNeuralNetwork()
x = np.array([2, 3])
print(network.feedforward(x))