---
title: Practical Aspects of Deep Learning - 2
description: In order to improve a deep neural network that you built, you need to understand the practical aspects of Deep Learning and Neural Networks.
---

Welcome to the second part of "Practical Aspects of Deep Learning". If you haven't already gone through the first part,
then you can read the post [here.]({{ site.url }}/practical-aspects-of-deep-learning-part-1){:target="_blank"} In this
post, we will be discussing on how to prevent the model from diverging from a good solution and also efficient
processes to check the gradients of the network.


### 9. Normalizing Inputs

- Before training a neural network, it is very important to normalize you input training set
- It uniforms the elongation between the feature axes of the dataset resulting in zero mean and zero standard deviation

![](https://i.ibb.co/0qQt5jH/Screenshot-from-2019-05-24-09-40-00.png)

- **Why do we need normalization?**
    - It shrinks the cost function of the network as shown in the image below
    - This helps the model to learn faster at a lower learning rate
    - It also helps in easy and faster optimization

![](https://i.ibb.co/TqqDZxR/Screenshot-from-2019-05-24-09-43-38.png)

### 10. Vanishing and Exploding Gradients

- For very deep neural networks, the activations can exponentially rise or diminish resulting in either exploding or vanishing gradients throughout the consecutive layers
- The **vanishing gradient** problem decays information as it goes deep into the network, making the network to never converge on a good solution
    - Neurons in the earlier layers learn more slowly than the ones in the latter layers
- The **exploding gradient** problem on the other hand makes the gradient bigger and bigger, and as a result forces the network to diverge
    - In this case, the earlier layers explode with very large gradients, making the model useless

![](https://i.ibb.co/MNqQW13/Screenshot-from-2019-05-24-09-59-36.png)