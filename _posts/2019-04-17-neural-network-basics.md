---
title: Neural Network Basics with Logistic Regression
description: This post explains about the fundamentals of neural networks by explaining how a logistic regression model works for each neuron.
---

These are some of my notes taken from the first course of Deep Learning Specialization
(Neural Networks and Deep Learning) by **deeplearning.ai**. This might help you gain
insights on how a neural network is formed and how Logistic Regression is used to form
one single neuron to stack up and form a neural network. Further down, there is a
discussion on how vectorization helps us improve the efficiency of code and algorithms
in Python using Numpy.

## 1. Binary Classification

![](https://i.ibb.co/LzfPgPC/Screenshot-from-2019-04-16-06-58-34.png)

## 2. Logistic Regression

![](https://i.ibb.co/JKzpNgV/Screenshot-from-2019-04-16-06-52-21.png)

- In the following figure, the loss function (L) computes the error for a single training example; the cost function (J) is the average of the loss functions of the entire training set.

![](https://i.ibb.co/QPJdHby/Screenshot-from-2019-04-16-07-03-26.png)

## 3. Gradient Descent
- Gradient descent is a method to determine the optimum values of the parameters using slopes, derivatives and a learning rate

![](https://i.ibb.co/yBZfDrg/Screenshot-from-2019-04-16-07-10-12.png)


- For logistic regression, the parameters **w & b** are optimized as follows, where **alpha** is the learning rate

![](https://i.ibb.co/P9S0CWJ/Screenshot-from-2019-04-16-07-12-56.png)

## 4. Computing Derivatives for Gradient Descent

- The following computation graph is made by breaking down the formula for evaluating the cost function of the model
    - Forward pass: Evaluation of the cost function
    - Backward pass: Evaluation of the authenticity of the model and also the best fit parameters by finding the derivative of final output w.r.t different model variables
- One backward pass in a computation graph gives us the derivative of the current variable with respect to the one that we are passing to.
- e.g. Moving from block *J to block v*, we find the derivative of J w.r.t. v
- We apply chain rule which propagating back through the graph

![](https://i.ibb.co/WzMs0Qm/Screenshot-from-2019-04-16-07-34-44.png)

## 5. Logistic Regression Derivatives

- The following image shows the gradient descent for Logistic regression method using only one data point
    - That is why, we are taking loss function (L) instead of the cost function (J) as we only have one data point for now
- After we calculate "da=dL/da" & "dz=dL/dz", we can use the following formula to get the change in w1 and w2:
    - w1 = w1 - alpha * dw1
    - w2 = w2 - alpha * dw2

![](https://i.ibb.co/DYKX1tZ/Screenshot-from-2019-04-16-14-16-04.png)

## 6. Logistic Regression on "m" examples

- Algorithm to implement logistic regression for a dataset (m input samples) with 2 features only
- Although there is a **for loop** being used in the algorithm below, we will need **vectorization** techniques to simplify and optimize the code to work on large datasets.


![](https://i.ibb.co/R49bBth/Screenshot-from-2019-04-16-14-27-23.png)

## 7. Python and Vectorization

- Here, `np.dot(W, X)` is simply evaluating `W transpose . X`

![](https://i.ibb.co/dP67djg/Screenshot-from-2019-04-16-14-35-13.png)

![](https://i.ibb.co/4JVSkjW/Screenshot-from-2019-04-16-14-39-26.png)

## 8. Vectorizing Logistic Regression

- In the last equation, although "b" is a (1, 1) real number, python automatically converts it to a (1, m) row vector. This is called **Broadcasting**

![](https://i.ibb.co/5sqQJc3/Screenshot-from-2019-04-16-14-44-44.png)

## 9. Implementing Logistic Regression

![](https://i.ibb.co/sPFHQwt/Screenshot-from-2019-04-16-14-51-50.png)

- Finally, the code to implement logistic regression in python is given below (right part)

![](https://i.ibb.co/L1sC3ZV/Screenshot-from-2019-04-16-14-53-40.png)

## 10. Broadcasting in Python

- When **axis=0**, the numpy operations are carried out on the vertical axis

![](https://i.ibb.co/W0NR5sQ/Screenshot-from-2019-04-16-15-00-30.png)

## 11. Python and Numpy Tips

- Do not use **Rank 1** vectors as shown in the image below
- Use **assert()** statements in your code to validate the dimensions of the arrays/vectors you are working with

![](https://i.ibb.co/q1nVv3Y/Screenshot-from-2019-04-16-15-05-16.png)

## 12. Explanation (derivation) of Loss function for Logistic Regression

- **Loss Function**

![](https://i.ibb.co/z6BBv1x/Screenshot-from-2019-04-16-15-12-45.png)

- **Cost Function**

![](https://i.ibb.co/K0G14fD/Screenshot-from-2019-04-16-15-16-09.png)