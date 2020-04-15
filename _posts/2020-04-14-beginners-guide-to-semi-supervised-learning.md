---
title: "A Beginner's Guide to Semi-Supervised Learning"
date: 2020-04-14 21:19:32 -0500
categories: [Illustration]
tags: [machine_learning]
description: This post discusses about different methods, surveys, and metrics that have been introduced in the field of video description. Video description is one of the popular fields in today's research that involves understanding and detection of occurrences of many entities in a video.
---

When we try to divide machine learning methods, we generally tend to separate them into two fundamentally different groups:
*supervised* and *unsupervised*. So, where do semi-supervised methods fit in? As it is clear by the name, there is some element
of supervised principles involved in this category, but what is it that makes it different from supervised learning. If I
have to explain it in layman words, semi-supervised learning is a mixture of both supervised and unsupervised techniques
so that we enable ourselves to make use of abundant unlabeled data to train our supervised algorithms.

By definition, **semi-supervised learning** is a branch of machine learning that combines a small amount of labeled data
with a large amount of unlabeled data during training to produce considerable improvement in the learning accuracy. It
falls between unsupervised learning and supervised learning. The concept of semi-supervised learning is important because
it allows us to prevent the cost of manually labeling the training data and make use of huge amount of unlabeled data that
we can find in today's digital world. Although, it may sound simple when we say mixing supervised and unsupervised methods
is the basis of formation of semi-supervised methods, there are different ways we can approach this problem. The diagram
below broadly represents the taxonomy of semi-supervised learning methods that are present to date.

![Semi-supervised Learning]({{ site.assets_url }}/img/posts/taxonomy_semi_supervised_learning.png "Taxonomy of Semi-supervised Learning") 
 


## Basic Concept and Assumptions

We can find unlabeled data everywhere in the form of text, images, audio signals, etc but it is very difficult to utilize
them to good effect. Most of the machine learning methods (including deep learning) can only perform well only when we have
properly labeled data to train the model and labeling data comes with great economic and labor cost. Semi-supervised
learning methods try to make use of the unlabeled data with the help of small amount of available labeled data. But for
the unlabeled data to be of any use for our algorithm, there are three main assumptions about the distribution of the
unlabeled data.

Smoothness Assumption
- We assume that if two data points in the sample space are close to each other, then their labels in the output space
should also be close to each other.
- Basically, if x and x' are close, then y and y' should also be close enough, where **x** is the input data and **y** is
its label.

Low-density Assumption
- This assumption ensures that the decision boundary between classes does not pass through high-density areas, i.e.
through areas where there are a lot of data points grouped together.
- It is quite intuitive that regions with higher density in sample space generally belong to a particular class and the
decision boundary that separates two classes should be somewhere where almost no data are present.

Manifold Assumption
- According to manifold assumption, data points on the same low-dimensional manifold should have the same label.
- By same low-dimensional manifold, we mean the projection of data points to a lower-dimensional space using methods like
Principal Component Analysis (PCA). So, if data points projected at lower dimensions are closer, then they should most
probably belong to the same label or class.


![Assumptions in Semi-supervised Learning]({{ site.assets_url }}/img/posts/semi-supervised_assumptions.png "Semi-supervised assumptions of the underlying data")

The image above clarifies the different assumptions that are undertaken in almost all semi-supervised methods. In addition
to these assumptions, there is one more called **cluster assumption** which somewhat is a generalization of all three
assumptions. It states that if data points (both labeled and unlabeled) cannot be meaningfully clustered, it is impossible
for a semi-supervised learning method to improve on a supervised learning method. Once these assumptions about our
unlabeled data are satisfied, we can move forward to applying semi-supervised methods on the unlabeled data that we have.