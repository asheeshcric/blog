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
    - the *Filter* strategy (e.g. information gain)
    - the *wrapper* strategy (e.g. search guided accuracy)
    - the *embedded* strategy (features are selected to add or be removed while building
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
    - PCA looks at the variables/features with the most variation
    - The first principal component (or axis) lies in the direction where the 
    variation in the dataset is the maximum.
        - The second **PC** is in the direction of the second most variation axis
        - and so on...
    - The number of dimensions of the dataset is equal to the number of PCs
    after the dataset has been projected from higher dimension to lower dimension
    
    - **Formulation of Principal Component**
        1. A random axis (passing through origin) is drawn at first in the sample space and each point in
        the dataset is then projected to the axis.
        2. The distance of each projected point from the origin is calculated and
        their sum of squares is calculated
        3. This is done for every possible axis passing through origin
        4. The one with the highest square sum gives the 1st PC (as it is the one
        along the direction with maximum variation in the dataset)
        5. Similarly, the second largest sum for the axis is taken as the 2nd PC and
        so on...
        - *NOTE*: The PCs are orthogonal to one another.
        
    - **Implementing PCA on a 2-D dataset**
        1. ___Normalize the data___
            - Done by subtracting the respective means from the numbers for each
            feature
            - This produces a dataset whose mean is zero
            
        2. ___Covariance Matrix___
            - Compute the covariance matrix for the dataset
            > Matrix (Covariance) = $$ \begin{bmatrix}Var[X_2] & Cov[X_1, X_2]\\Cov[X_2, X_1] & Var[X_2]\end{bmatrix} $$

        3. ___Eigenvalues and Eigenvectors calculation___
            - Calculate the eigen values and vectors for the above calculated
            covariance matrix
            - **$$\lambda$$** can be defined as the eigen value of a matrix **A** if
            if satisfies the following characteristic equation
                > det($$\lambda$$I - A) = 0
            - Also, for each eigen value $$\lambda$$, there exists a corresponding eigen
            vector **v** such that
                > ($$\lambda$$I - A)v = 0
                
        4. ___Forming a feature vector___
            - Order the obtained eigenvalues from largest to smallest so that it
            sorts in the order of its significance
            - If we have a dataset with **n** variables (or features), then we will
            have **n** number of eigenvalues and eigenvectors
            - To reduce the dimensions of the dataset, just select the first **p** eigenvalues
            and ignore the rest.
            - Now, we form a feature vector which is a matrix of the **eigenvectors** as shown
            below
                > Feature Vector = ($$eig_1, eig_2, eig_3,  ... $$)
                
        5. ___Forming Principal Components___
            - We now form our principal components using the above calculated figures
                > NewData = $$ FeatureVector^T * ScaledData^T$$
                
            - So, 
                - *NewData* is the matrix consisting of the principal components
                - *FeatureVector* is the matrix containing the eigenvectors
                - 
            
- **Linear Discriminant Analysis (LDA)**
    - LDA is like PCA, but it focuses on maximizing the separability among known
    categories
    - LDA tries to maximize the separation of known categories