---
title: RandomForest with scikit-learn (Part 1)
date: 2019-03-16 11:23:00 +0515
categories: [Tutorial]
tags: [machine_learning]
description: Learn how to apply a RandomForest classifier for any arbitrary dataset and generate surprisingly accurate results.
---

Among many machine learning techniques, RandomForest is one of the most widely used
**ensemble** supervised algorithm for both classification and regression tasks. Yes, it
works smoothly for both categorical and continuous prediction labels.

Before diving into the code directly, let's first discuss the theoretical aspects of 
RandomForest. It is an ensemble method (i.e. a collection or group of various other
models) which is highly suitable for almost any kind of dataset.

RandomForest uses a collection of **Decision Trees** to predict an outcome. A Decision
Tree Classifier contains a root node, followed by branch nodes, and end up with the leaf
nodes as shown in the diagram below.
![Decision Tree](https://i.ibb.co/jTXTWmw/Screenshot-from-2019-03-16-20-24-03.png)

Intuitively, a single decision tree on a large dataset would perform really poorly
as there are many different ways an outcome can be reached. This is where RandomForest
comes in. The principle behind RandomForest is that it takes down a part of the dataset
(also called **bootstrapping**) and forms a decision tree by further taking a random 
subset of the features (or variables) only. Such decision trees are formed in large
numbers, each time with a bootstrapped dataset and a random subset of features. Once
all the trees are tuned and ready, prediction is made by taking out the average of
outcomes from each tree.

This method surprisingly works for almost all kinds of dataset. The reason behind this
is that every tree overfits on its own bootstrapped dataset (though learns something about
that portion of the data), and when outputs from the trees are averaged, the final result
does not come from an overfitted model. Hence, this method builds a highly generalizable
machine learning model.

For evaluating the model, a technique called **out-of-bag** is used. Each time a
decision tree is trained on a bootstrapped dataset, it does not see all the data in
that set. So, the data that are not seen by the tree can be used to evaluate the performance
for the tree. These performances can be added for all the trees and a final accuracy
metric can be evaluated.

This concludes a really quick overview of how **RandomForest** works under-the-hood.
Now, let's get into implementing our own model using a famous machine learning library
in Python called **scikit-learn**. The code for implementing a RandomForest model is
discussed in the second part of this post: [RandomForest - Part 2]({{ site.url }}/randomforest-part-2)