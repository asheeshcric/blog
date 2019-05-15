---
title: Practical Aspects of Deep Learning
description: In order to improve a deep neural network that you built, you need to understand the practical aspects of Deep Learning and Neural Networks.
---


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