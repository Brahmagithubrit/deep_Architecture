# Perceptron Implementation in Python

This repository contains a simple implementation of the Perceptron algorithm using NumPy. The perceptron is a fundamental building block of neural networks and is used here to learn the AND logic gate.

---

## What is a Perceptron?

A perceptron is a type of linear classifier, i.e., a classification algorithm that makes its predictions based on a linear predictor function combining a set of weights with the feature vector. It is the simplest type of artificial neural network and can only solve linearly separable problems.

---

## How It Works

1. **Initialization:**  
   - Weights and bias are initialized to zero.
2. **Training:**  
   - For each input sample, the perceptron computes a weighted sum.
   - The output is determined by a step function (if sum ≥ 0, output 1; else 0).
   - If the prediction is incorrect, weights and bias are updated using the perceptron learning rule.
   - This process repeats for a fixed number of epochs.
3. **Testing:**  
   - After training, the perceptron is tested on the input data to check if it has learned the correct mapping.

---

## Code Overview

```python
import numpy as np

# Input data for AND gate
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])

# Initialize weights and bias
weight = np.zeros(X.shape[1])
bias = 0
learningRate = 0.1
epoch = 20

# Training loop
for i in range(epoch):
    for j in range(len(X)):
        val = np.dot(X[j], weight) + bias
        prediction = 1 if val >= 0 else 0
        error = y[j] - prediction
        weight += learningRate * error * X[j]
        bias += learningRate * error
        print(f"weight = {weight} , bias = {bias}")

# Testing
for i in range(len(X)):
    val = np.dot(weight, X[i]) + bias
    prediction = 1 if val >= 0 else 0
    print(f"x[{i}] = {X[i]} => prediction: {prediction}")
```

---

## Output Example

```
weight = [0. 0.] , bias = 0.0
weight = [0. 0.] , bias = 0.0
weight = [0. 0.] , bias = 0.0
weight = [0.1 0.1] , bias = 0.1
...
x[0] = [0 0] => prediction: 0
x[1] = [0 1] => prediction: 0
x[2] = [1 0] => prediction: 0
x[3] = [1 1] => prediction: 1
```

---

## Applications

- Basic introduction to neural networks and machine learning.
- Demonstrates how a perceptron can learn simple logic gates (AND, OR, etc.).
- Foundation for more complex neural network architectures.

---

## References

- [Wikipedia: Perceptron](https://en.wikipedia.org/wiki/Perceptron)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)

---

