---
title: Machine Learning Fundamentals
---

In this post, I have pointed out a few basic key notes about some popular
machine learning terminologies that you are going to observe as a beginner
in this field.

## 1. Cross Validation
- **k-fold cross validation** approach: (k-1 training blocks with 1 testing block)
- **Leave One Out Cross Validation** approach: (each data point is considered as a block)
- Cross validation is also useful to determine the best hyperparameters for the model being trained.

## 2. Confusion Matrix
- Helps in summarizing the performance of the testing data on our trained model
- **Rows** in the confusion matrix correspond to the predicted data from the model and **Columns** correspond to the actual output of the data
- ![Confusion Matrix](https://raw.githubusercontent.com/asheeshcric/learning-ml-crashcourse/master/Machine%20Learning%20-%20StatQuest/images/confusion-matrix.png?token=ATcLOps5SjzuYDAzsp4aaSB8EWaQ4FEuks5cd8cqwA==)


## 3. Sensitivity & Specificity
- **Accuracy**
	- ACC is the ratio of correct predictions to the total number of data points
	- `Accuracy = (True Positive + True Negative) / Total`
	- `ACC = 1 - ERR`, where **ERR** is the Error Rate

- **Sensitivity (Recall or True Positive Rate)**
	- The number of correct positive predictions divided by the total number of positives
	- The best sensitivity is 1.0, whereas the worst is 0.0
	- `Sensitivity = True Positive / (True Positive + False Negatives)` 

- **Specificity (True Negative Rate)**
	- The number of correct negative predictions divided by the total number of negatives
	- The best specificity is 1.0, whereas the worst is 0.0
	- `Specificity = True Negative / (True Negative + False Postive)`

- **Precision (Positive Predictive Value)**
	- Precision (PREC) is the ratio of correct positive predictions to the total number of positive predictions
	- `Precision = True Positive / (True Positive + False Postive)`



## 4. Bias & Variance
**Bias**
- High bias can lead models to underfit the data
- For example: A straight line generally underfits a practically complicated training dataset (or model) as it has high bias and cannot curve according to the data available. This is the problem faced by **Linear Regression** while fitting complicated models

**Variance**
- On the other hand, a model can **overfit** the training data and can lead to very **high variance**
- Generally, complicated high dimension models lead to overfitting a relatively small training dataset
- Adding a small amount of **bias** to the model while training on the data can significantly reduce its resulting **variance**. Such technique is also called **Regularization**

**NOTE**: A model that has **low bias & low variance** generally performs well in real datasets.

- Three commonly used methods for finding the sweet spot between simple and complicated models are:
	- **Regularization**
	- **Boosting**
	- **Bagging**
	

## 5. AUC - ROC

- A performance visualization for classification problems at various threshold settings.
- **ROC** is a probability curve and **AUC** represents degree or measure of separability
	- Higher the AUC, better the model is at predicting classes as it tells how the model is capable of distinguishing between classes

- **ROC** is plotted with TPR (y-axis) against FPR (x-axis); where `FPR = 1- Specificity` 
###  
- **ROC** with AUC ~ 1, i.e. a model having ideal classification ability
- ![AUC=1](https://i.ibb.co/7WBPC1V/roc-1.png)
- **ROC** with AUC = 0.5, i.e. a model similar to randomly guessed classification
- ![AUC=0.5](https://i.ibb.co/x872KDk/roc-2.png)
- **ROC** with AUC = 0, i.e. a model that perfectly predicts the outcome oppositely, i.e. 1s for 0s and 0s for 1s
- ![AUC=0](https://i.ibb.co/Gs36NZT/roc-3.png)
