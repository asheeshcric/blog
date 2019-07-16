---
title: Optimization Algorithms in Deep Learning
description: Building deep neural networks is one thing, but optimizing it to train faster with better accuracy is a completely different set of domain. It is very important to focus on optimizing our training algorithms and expedite the process.
---

Building deep neural networks is one thing, but optimizing it to train faster with better accuracy is a completely different
set of domain. So, it is very important that we focus on optimizing our algorithms to converge faster with
desirable accuracy and details. In this post, we discuss about a few optimization algorithms that are generally used
to expedite the training process even with indefinitely larger training datasets. The usage of the algorithms solely depend
upon our application and the type of dataset that we use.


### 1. Mini-batch Gradient Descent

- Gradient descent is an algorithm in machine learning that is used to evaluate the parameters that are used in the model
- The main downside of Gradient Descent is that it has to go through the entire training set on each descent (or iteration)
    - So, if the training dataset used is very large, then the algorithm takes huge amount of time
- To mitigate this problem, we use **mini-batch gradient descent**, taking batches from training data for each descent
    - This helps gradient descent to progress smoothly by not requiring the entire training dataset on each step (or descent)
- A mini-batch from the training set is represented as $$X^{\{t\}}$$, $$Y^{\{t\}}$$ (we use curly braces to represent the $$t^{th}$$ mini-batch)
![](https://i.ibb.co/ZVnG6dD/Screenshot-from-2019-07-15-11-52-11.png)

- Each step of the descent is on a mini-batch instead of the whole training set
- Size of each mini-batch is the size of the dataset in each loop
- 1 epoch = A single pass through the entire training set (going through all mini-batches of the set for once)
    - The only difference is that the gradient descent gets updated after each mini-batch is processed within a running epoch unlike the full gradient descent

#### Size of Mini-batch

- Size = "m": Batch Gradient Descent (too slow although has better accuracy)
- Size = "1": Stochastic Gradient Descent
    - Too prone to noise and outliers
    - We lose the advantage of speed of vectorization as each mini-batch is only a single example
- Size = "between 1 and m": Generally taken sizes in practice are among 64, 128, 256, 512, 1024, etc

- Hence, size of mini-batch is also another **hyperparameter** to consider

### 2. Exponentially Weighted Averages

- Used on basically any data that is in sequence
- It is also referred as **smoothing** of the data (or timeseries)
    > $$v_{t}$$ = $$\beta$$$$v_{t-1}$$ + (1 - $$\beta$$)$$\theta_{t}$$
- Generally, we take $$\beta$$ = 0.9 for practical consideration
![](https://i.ibb.co/7VmTvRh/Screenshot-from-2019-07-16-10-15-20.png)

#### Bias Correction

- In exponentially weighted averages, when the initial value $$v_{0}$$ = 0, then it can create an unwanted bias making 
the initial averages to be much lower than the actual. So, we use the following formula to update the value of $$v_{t}$$
    > $$v_{t}$$ = $$v_{t}$$ / (1 - $$\beta^{t}$$)
    - This is required for bias correction and not letting the initial values be affected by a fixed bias towards 
    **zero** or **origin**
    
    
### 3. Gradient Descent with Momentum

- While using **mini-batch gradient descent**, the parameters get updated after each mini-batch cycle (having some variance
in each update). This make the gradient descent to oscillate a lot while moving towards the convergence
- So, gradient descent with momentum computes an exponentially weighted averages of the gradients and then use that gradient
to update the weights instead
- This helps in reducing the oscillations during G.D. and makes the convergence faster
    > $$v_{dw}$$ = $$\beta$$ $$v_{dw}$$ + (1 - $$\beta$$) dw  
    > $$v_{db}$$ = $$\beta$$ $$v_{db}$$ + (1 - $$\beta$$) db  
    
    > w = w - $$\alpha$$ $$v_{dw}$$  
    > b = b - $$\alpha$$ $$v_{db}$$  
    
- So, here "$$\beta$$" is a new hyperparameter involved which basically carries out exponentially weighted averages on each update making the convergence faster
    - Generally, the practically considered value of $$\beta$$ is ~ 0.9
- Hence, this method is basically taking *"exponentially weighted moving averages"* method and merging
it to the *"mini-batch gradient descent"* algorithm


### 4. RMSprop

- RMSprop is quite similar to G.D with Momentum except for the fact that it restricts the oscillations in the vertical direction
    - This allows the descent to take greater leaps in the horizontal direction with greater **learning rate** as the vertical movement is restricted
    
- In this case, the exponentially weighted moving averages are calculated differently as shown below
    > $$s_{dw}$$ = $$\beta$$ $$s_{dw}$$ + (1 - $$\beta$$) $$dw^{2}$$  
    > $$s_{db}$$ = $$\beta$$ $$s_{db}$$ + (1 - $$\beta$$) $$db^{2}$$  
    
    > w = w - $$\alpha$$ * dw / $$\sqrt{s_{dw} + \epsilon}$$  
    > b = b - $$\alpha$$ * db / $$\sqrt{s_{db} + \epsilon}$$  
    
- RMSprop and Momentum algorithms both decrease the vertical oscillations and increase horizontal speed, making the descent converge faster for a given cost function
