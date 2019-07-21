---
title: Hyperparameter Tuning in Deep Neural Networks
description: One way to make your deep learning model more accurate and generate better results is to tune your model's hyperparameters. By doing so, you can speed up your training process and optimize the outputs provided by the model.
---


One way to make your deep learning model more accurate and generate better results is to tune your model's hyperparameters.
By doing so, you can speed up your training process and optimize the outputs provided by the model. In this post, we try
to figure out some ways to make sure that we choose the right hyperparameters every time we train a deep neural network.


### 1. Tuning Process

- The most common hyperparameters one needs to choose for training their neural networks are as follows:
    - Learning rate (alpha: "$$\alpha$$")
    - Mini-batch size
    - Number of hidden units for each layer
    - Momentum term (beta: "$$\beta$$"); generally $$\beta$$ = 0.9 is taken
    - Number of layers
    - Learning rate decay (Changing $$\alpha$$ as the learning progresses)
    - For Adam Optimizers: ($$\beta_{1}$$, $$\beta_{2}$$, and $$\epsilon$$)
        - Generally, $$\beta_{1}$$ = 0.9, $$\beta_{2}$$ = 0.999, and $$\epsilon$$ = $$10^{-8}$$
        
- The hyperparameters are listed in the order of their significance (in tuning) while training a deep neural network, but the order may vary according to the requirements

- When tuning hyperparameters, try to sample the values of the parameters in random so that we can find the ones that perform the best for our model


### 2. Using an appropriate scale to pick hyperparameters

- In cases when we try to sample values for a hyperparameter like **learning_rate ($$\alpha$$)**, we need to be smart while taking random values at different scales
- For example, the acceptable value for the **learning_rate** can be anything in between 0 and 1, but we know that the values that are less than **0.1** are more plausible that the higher values
    - In such case, we can divide the scale from 0 to 1 logarithmically and then take random values from each scale
        - e.g. r = -4 * np.random.rand();   r --> [-4, 0]
        - $$\alpha$$ = $$10^{r}$$; $$\alpha$$ --> [$$10^{-4}$$, 1]

