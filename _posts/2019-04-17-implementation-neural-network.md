---
title: Implementation of a Neural Network from scratch
description: Have you ever tried to implement a neural network from scratch, using Logistic Regression? This post reveals the fundamentals of building a neural network and training it with multiple training samples with the help of vectorization.
categories: [Tutorial]
tags: [machine_learning]
---

Are you someone who is always fond of using machine learning and deep learning libraries such as Scikit-learn, TensorFlow,
PyTorch, etc, but don't really understand how a neural network works underneath the libraries? If so, this post tries
its best to explain what a neural network is, how it works, and how anyone can implement it with some **NumPy** basics.

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

#### Explanation about the dimensions of the vectors W, X, Z, and A

- Vector X is formed by stacking all the data points horizontally.
    - X = [x1 x2 ... xm], where "m" is the number of samples
    - Dimension of X is **(nx, m)**
        - nx: the number of features in a data point
        - m: the number of training samples

- Vector W is formed by stacking the number of neurons/nodes in the layer for each data point of X
    - W = [w1 w2 ... wm], where the number of rows is the number of nodes in the layer
    - So, W.T is the transpose of W to make it compatible for multiplication with X
    - Dimension of W.T is **(k, nx)**
        - k: the number of nodes in the layer
        - nx: the number of features in a data point
        
- Vector Z = W.T * X
    - Its dimension is (k, nx) * (nx, m) = **(k, m)**
    - k: the number of nodes in the layer
    - m: the number of training samples

- Vector A = sigmoid(Z)
    - A is the result of using an activation function over Z to make the output in a range 0-1 (which is what the sigmoid does)
    - Dimension is the same as that of Z, i.e. **(k, m)**

### 5. Activation Functions

- Hyperbolic tangent function almost always works better than a sigmoid function
    - Sigmoid has an output range (0, 1) and tanh function has an output range (-1, 1)
    - Only place where a sigmoid function can be useful is at the output layer of a binary classification, where you want the output to be between (0, 1)
    
- One of the downsides of both the sigmoid and tanh functions is that if the value of **z** is very large or very small, the slope of the function approximates to nearly zero. This can drastically slow down the gradient descent and can hinder convergence in those cases.


- **RULE OF THUMB**
    - Just use **Relu (Rectified Linear Unit)** function for all hidden layers and only use sigmoid at the output layer if you are trying to implement a binary classifier

![](https://i.ibb.co/XLNFmTX/Screenshot-from-2019-04-18-17-06-27.png)

- Sometimes, leaky relu performs better than relu, but relu is the ultimate choice in most cases.

#### Why do we need to use non-linear activation functions?

- The purpose of the activation function is to introduce non-linearity into the network

- In turn, this allows you to model a response variable (aka target variable, class label, or score) that varies non-linearly with its explanatory variables

- Non-linear means that the output cannot be reproduced from a linear combination of the inputs (which is not the same as output that renders to a straight line--the word for this is affine).

- Another way to think of it: without a non-linear activation function in the network, a NN, no matter how many layers it had, would behave just like a single-layer perceptron, because summing these layers would give you just another linear function

### 6. Derivatives of Activation Functions for Backpropagation

![](https://i.ibb.co/NxXTGb2/Screenshot-from-2019-04-18-17-17-53.png)

![](https://i.ibb.co/WyXmCCG/Screenshot-from-2019-04-18-17-19-02.png)

![](https://i.ibb.co/1md51kM/Screenshot-from-2019-04-18-17-20-01.png)

### 7. Gradient Descent for Neural Networks

- Formula for computing derivatives for backpropagation

![](https://i.ibb.co/kQD8wNk/Screenshot-from-2019-04-18-17-36-05.png)

- Deriving the derivative equations for gradient descent from scratch is quite complicated and requires the knowledge of linear algebra and matrix calculus

![](https://i.ibb.co/F8VzRhv/Screenshot-from-2019-04-18-17-46-44.png)