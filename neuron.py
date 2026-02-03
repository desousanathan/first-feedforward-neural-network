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