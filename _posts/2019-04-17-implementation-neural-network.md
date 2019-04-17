---
title: Implementation of a Neural Network from scratch
description: Have you ever tried to implement a neural network from scratch, using Logistic Regression? This post reveals the fundamentals of building a neural network and training it with multiple training samples with the help of vectorization.
---

### 1. What is a Neural Network?

![](https://i.ibb.co/d79YnGD/Screenshot-from-2019-04-17-15-58-08.png)

- As shown in the above image, basically a neural network is a neuron (evaluated using Logistic Regression) repeated multiple times
- In neural network notation, we don't count the input layer. So, the above shown neural network is a **"2 layer NN"**

### 2. Neural Network Representation

![](https://i.ibb.co/T0VYmkT/Screenshot-from-2019-04-17-16-06-35.png)

### 3. Computing a Neural Network's Output

- The hidden layer has four neurons, so the output or activation from the hidden layer is a column vector with four single valued rows and is denoted by **a[1]** as it is the first layer of the network
- In logistic regression, when input is X, a neuron computes the following two equations to get **z & a**:
    - z = wx + b
    - a = sigmoid(x)
    - yhat = a (output of the neuron/layer)
    - **Size and dimension of the layers depends on the number of neurons contained by the layer**

![](https://i.ibb.co/0BQqjXc/Screenshot-from-2019-04-17-16-50-35.png)

- Now, for a neural network with hidden layers and multiple neurons, this is how the weights and biases are calculated
- First the input layer (input values = X), is multiplied with the transpose of the first layer weights W[1].T, and passed through a sigmoid function to get the output from the first hidden layer (this implements Logistic Regression on the first layer)
    - Same thing is repeated for the second layer where the output from the first acts as input to the second
- From the output of the output layer, the cost function L is evaluated

![](https://i.ibb.co/q96kxQX/Screenshot-from-2019-04-17-16-53-42.png)

- Final algorithm to implement the neural network for one example at a time

![](https://i.ibb.co/ygS5JcW/Screenshot-from-2019-04-17-17-00-41.png)

### 4. Vectorizing across multiple examples

![](https://i.ibb.co/BCykzTw/Screenshot-from-2019-04-17-17-07-46.png)

- As shown in the above image, to train all the samples from a training set, we just stack each data point horizontally to create a matrix X
    - X = [X1 X2 ... Xm]
- Similarly, the Z vector is evaluated using the formula **z = w.T + b** for each column of the vector X to finally form matrix Z
    - Z = [Z1 Z2 ... Zm]
    - For each layer, there is a separate Z matrix
    - e.g. Z[1], Z[2] represent the first and second layers of the network
- The final output matrix of a layer A is the sigmoid of the matrix Z
    - A = sigmoid(Z)
    - This is the output of a layer and acts an input to the second layer
    - If we go down vertically in a column of matrix A, it represents the activations from nodes of that hidden/output layer
