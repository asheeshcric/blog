---
title: Dimensionality Reduction - Machine Learning
---

In Machine Learning or Statistics, **dimesionality reduction** is the process
of reducing the number of random variables under consideration by obtaining a 
set of principal variables. This can be achieved using two processes: **Feature
Selection** and **Feature Projection**


### Feature Selection

- **Feature selection** methods try to find a subset of the original variables by
the following strategies:
    - the Filter strategy (e.g. information gain)
    - the wrapper strategy (e.g. search guided accuracy)
    - the embedded strategy (features are selected to add or be removed while building
    the model based on the prediction scores)


### Feature Projection

- In this post, we are more focused on approaches that help with **feature extraction**
for high-dimensional data. **Feature projection** transforms data in the high-dimensional
space to a space of fewer dimensions. The transformation may be linear or nonlinear
depending on the approach we take and the type of data we have on hand.


### Feature Projection Techniques


- **Principal Component Analysis (PCA)**

    - PCA takes a dataset with a lot of dimensions and flattens it to 2 or 3 dimensions
    so that we can take a look at it
    - It tries to find a meaningful way to flatten the data by focusing on the things
    that are different among the variables (or features)
    - PCA looks at the genes with the most variation


- **Linear Discriminant Analysis (LDA)**
    - LDA is like PCA, but it focuses on maximizing the separability among known
    categories
    - LDA tries to maximize the separation of known categories