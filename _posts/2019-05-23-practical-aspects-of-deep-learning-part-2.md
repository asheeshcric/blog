---
title: Practical Aspects of Deep Learning - 2
description: In order to improve a deep neural network that you built, you need to understand the practical aspects of Deep Learning and Neural Networks.
categories: [Illustration]
tags: [machine_learning]
---

Welcome to the second part of "Practical Aspects of Deep Learning". If you haven't already gone through the first part,
then you can read the post [here.]({{ site.url }}/practical-aspects-of-deep-learning-part-1){:target="_blank"} In this
post, we will be discussing on how to prevent the model from diverging from a good solution and also efficient
processes to check the gradients of the network.


### 9. Normalizing Inputs

- Before training a neural network, it is essential to normalize your input training set
- It make the elongation between the feature axes of the dataset uniform, resulting in zero mean and unit standard deviation

![](https://i.ibb.co/0qQt5jH/Screenshot-from-2019-05-24-09-40-00.png)

- **Why do we need normalization?**
    - It shrinks the cost function of the network as shown in the image below
    - This helps the model to learn faster even at a low learning rate
    - It also helps in easy and faster optimization

![](https://i.ibb.co/TqqDZxR/Screenshot-from-2019-05-24-09-43-38.png)

### 10. Vanishing and Exploding Gradients

- For very deep neural networks, the activations can exponentially rise or diminish resulting in either exploding or vanishing gradients when backpropagating through the layers
- The **vanishing gradient** problem decays information as it goes deep into the network, making the network to never converge on a good solution
    - Neurons in the earlier layers learn more slowly than the ones in the latter layers
- The **exploding gradient** problem on the other hand makes the gradient bigger and bigger, and as a result forces the network to diverge
    - In this case, the earlier layers explode with very large gradients, making the model useless

![](https://i.ibb.co/MNqQW13/Screenshot-from-2019-05-24-09-59-36.png)

### 11. Weight Initialization for Deep Neural Networks

- Initializing weights W to zero
    - We know that if we initialize all the weights to zero, our network acts like a linear model as all the layers basically learn the same thing. This makes the model just a linear combination of layers
    - So, the most important thing is to not initialize all the weights to zero and use a random initialization approach

- Initializing weights randomly
    - Although this may sound an appropriate approach to initialize the weights of a network, in some conditions (when proper activations are not used), it may lead to vanishing or exploding gradients
    - So, this method cannot be considered bullet proof although it works most of the time with **RELU** activations

- Using some heuristic to initialize weights
    - This is considered as the proper way when it comes to weight initialization of deep neural networks
    - We can use some heuristics from the model to assign the weights of the layers according to the activation function used in a layer
    - The images below show how we should actually initialize weights in case of **RELU**, **tanh**, and other activations

![](https://i.ibb.co/nCsYSMB/Screenshot-from-2019-05-25-09-07-09.png)


- **NOTE:** Here, *"size_l"* refers to the number of nodes in the lth layer, i.e. *n[l]*

![](https://i.ibb.co/TRj2GyG/Screenshot-from-2019-05-25-09-25-36.png)



- This concludes the key points that you should be aware of when building a practical deep neural network. The image below demonstrates the workflow of a deep neural network along with forward and backward propagation. This might generate an intuition for you to build practical neural networks from scratch in Python.

![](https://i.ibb.co/Qdy8W0D/Screenshot-from-2019-05-07-20-19-13.png)

- It is very important to understand how forward propagation and backward propagation (with gradient descent) work in a deep neural network. The following formulae will always keep you in track when building a neural network from scratch.

![](https://i.ibb.co/85rXKYB/Screenshot-from-2019-05-06-20-41-00.png)


Please add your suggestions about the post in the comments section below. I would love to have your insights on
ways to make this post better. Thank you for reading.



