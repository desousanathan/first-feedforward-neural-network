# First Feedforward Neural Network

A simple feedforward neural network built from scratch using only NumPy.

## Overview

This project implements a basic neural network to understand the fundamentals of how neural networks work without relying on machine learning frameworks.

## Network Architecture

- **Input layer**: 2 inputs
- **Hidden layer**: 2 neurons (h1, h2)
- **Output layer**: 1 neuron (o1)
- **Activation function**: Sigmoid

## Implementation Details

Each neuron in the network uses:
- **Weights**: `[0, 1]`
- **Bias**: `0`

The network performs forward propagation by:
1. Computing weighted sum of inputs plus bias for each neuron
2. Applying sigmoid activation function
3. Passing hidden layer outputs to the output neuron

## Dependencies

- NumPy


## What I Learned

- How neurons process inputs through weighted sums and activation functions
- How to connect layers in a feedforward network
- The basics of forward propagation
