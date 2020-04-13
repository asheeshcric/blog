---
title: Practical Aspects of Deep Learning - 1
description: In order to improve a deep neural network that you built, you need to understand the practical aspects of Deep Learning and Neural Networks.
categories: [Tutorial]
tags: [machine_learning]
---

Deep Learning is a subset of Machine Learning which has come to evolve highly in the past few years. It involves
neural networks with the number of hidden layers greater than one, hence the term **"deep"**. The basis of a neural
network in deep learning is **Logistic Regression** and one should understand it well before moving ahead constructing
neural networks. Following are a few basic terminologies and their explanations that are vastly used in deep learning.
This should give you a basic understanding of how a deep neural network is trained and the strategies to make it
efficient.

### 1. Train/dev/test sets

- Generally, when training a neural network, we divide the entire dataset into three parts:
    - Training set
    - Dev (or Validation) set
    - Testing set
- When the size of the dataset is large (~ 1 million or above), the ratio of the sets is usually taken as 98:1:1
- Dev or validation set is used to tune the network and hence it can get overfitted with the network that we are training on
- Dev set and Test set should come from the same distribution for proper validation of the neural network


### 2. Bias and Variance

- Bias in machine learning tends to denote how well our trained network did well on the training set
    - High bias refers to higher training error rate in a neural network
    - Generally, high bias is a result of **underfitting**
        - This means that the model is too weak to understand the complicated relation of your training set and the target set
- Variance on the other hand refers to how well our trained model is able to generalize on new data (i.e. data it has never seen before)
    - If a model has overfit its training set, it does not do well with the test set and new data that it encounters
    - High variance is a sign for **overfitting**

### 3. Basic Recipe of Machine Learning

**a. High Bias**

- When there is a high training error rate on the model we trained, then it is said to have **high bias**
    - It reflects the inability of the model to learn precisely from its data
- High bias generally occurs due to the use of a simpler model for a dataset with complicated relationship
- Tips for preventing High Bias:
    - Making the network larger
        - Generally, when a smaller (or shallow) neural network is used (which covers only linear relationships), it is not able to account for the complicated relationship that our dataset has
        - This leads to **underfitting**, where all the features of the dataset are not learned by the model
        - So, increasing the size of the model used or going deeper into the network can make the model perform better on the given dataset

    - Manipulating the Neural Network structure
        - Sometimes, interchanging the layers used in the network with one another can also do the trick to learn more essential features of the dataset


**b. High Variance**

- Variance occurs when the model performs well on the training set but does not generalize well for the test set or new data
- Greater the difference between the training error rate and the test error rate, greater is the variance for the model
- High variance for a model indicates that the model has simply crammed the entire training set and is unlikely to perform well on the data that it has not seen before
- This condition is called **overfitting** where the model performs well only on the training set and not on the testing set

- Tips for preventing High Variance
    - Include more training data
        - Simply introducing more data in the training set can remove overfitting of a complicated model as it previously may not have sufficient data to generalize well
        - New data can mean that the model needs to learn newer features and can somehow not overfit the entire training set

    - Smaller Network
        - Sometimes a complicated network is large enough to remember all the feature relationships the dataset we are training on
        - This generally results in overfitting as it learns the entire dataset with ease
        - So reducing the number of layers in the model or altering the layer size can help the model to not overfit on the training dataset

    - Regularization
        - The best technique to avoid overfitting the data and prevent high variance is to use a **regularization** technique
        - Regularization techniques tend to decrease the quantitative increase in the weight matrix of a neural network. This helps in not allowing the neural network to learn all the data in the training set
        - Some of the regularization techniques are L1/L2 Regularization, Dropout Regularization, Data Augmentation, Early Stopping, etc

### 4. Regularization

- When overfitting occurs during training a model, it is best to use a suitable **regularization** technique that can prevent it
- There are generally two types of **regularization** techniques used widely (except in CNNs) are:
    - L1 Regularization -- Lasso Regression
    - L2 Regularization -- Ridge Regression

- **Lasso (Least Absolute Shrinkage and Selection Operator) Regression / L1 Regularization**
    - This technique adds **absolute value of magnitude** (mag(W)) of the weight matrix as penalty term to the loss function (J)
    - It shrinks the less important feature's coefficient to zero thus, removing some features altogether
        - Hence, it is generally used for **feature selection** in case we have a huge number of features
- **Ridge Regression / L2 Regularization**
    - It adds **squared magnitude** (W^2) as penalty term to the loss function
    - If the value of **lambda** is too large, it will shrink the Weight matrix of the neural network to nearly zero which can make the model underfit the dataset
    - This technique works very well to avoid the overfitting issue
    - For neural networks, the L2 norm is often called **Frobenius Norm**


![](https://i.ibb.co/jgC2rT1/Screenshot-from-2019-05-15-20-08-36.png)

### 5. Why regularization prevents overfitting?

- The regularization terms added to the loss function (J) encourages the weight matrix to diminish quantitavely
- As the regularization hyperparameter **lambda** increases, the magnitude of the weight matrix **W** decreases
    - This means more nodes are discarded (or made of less significance in the network) which kind of compells the model to generalize the training dataset
- Also, making **lambda** very large can make the model to learn linear relationships only. This can make the model useless as it cannot learn complex (quadratic) relationships in the dataset
    - e.g. For **tanh** activation, if the value of lambda is **very large**, it transforms the function to linearity inciting only linear relationships in the model

### 6. Dropout Regularization

- Most commonly used for Convolutional Neural Networks (CNNs), **dropout regularization** is a technique to randomly remove nodes for each training data and iteration
- To select if a node should be removed or not is chosen by using a probability randomness given by **keep_prob** hyperparameter
    - e.g. If **keep_prob = 0.6**, then the chance for removing the node from that layer is 40%
- The probability for node removal may vary for each layer

```ruby
d = np.random.randn(activation.shape[0], activation.shape[1])
activation = np.multiply(activation, d)
activation /= keep_prob   # inverted dropout (to neutralize the changes of dropout in the test set)
```

### 7. Understanding dropout regularization technique

- In a neural network, dropout is generally applied to the hidden layers and not to the input or output layers
- This is because the model should not be ignoring input features from the first layer as it may not perform well on the dataset
- Dropout is a famous technique mostly used in **Computer Vision** applications

### 8. Other regularization techniques

- Apart from L1 regularization, L2 regularization, and Dropout regularization, we can also use some other techniques to avoid overfitting suitable for our applications such as:
    1. Data Augmentation
        -  Generally applied for image datasets, where an image is rotated, flipped or cropped to introduce new input to the dataset
        - This technique helps in increasing the size of the dataset, thus helping the model to avoid overfitting

    2. Early Stopping
        - When training a neural network, we can track the training error and the dev/validation error. There is always a point to which the training error continuously decreases and then starts rising down again
        - Such points are regarded to be the boundary between a generalized model and a overfitted model
        - So, if we can apply early stopping (based on the number of epochs), we can stop the model from overfitting

        ![](https://i.ibb.co/y5NYXjC/Screenshot-from-2019-05-23-20-31-23.png)


This concludes the first part of "Practical Aspects of Deep Learning". For more details, please go on to the next
post from [here.]({{ site.url }}/practical-aspects-of-deep-learning-part-2)