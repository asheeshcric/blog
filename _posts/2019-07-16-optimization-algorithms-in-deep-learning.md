---
title: Optimization Algorithms in Deep Learning
description: Building deep neural networks is one thing, but optimizing it to train faster with better accuracy is a completely different set of domain. It is very important to focus on optimizing our training algorithms and expedite the process.
categories: [Tutorial]
tags: [machine_learning]
---

Building deep neural networks is one thing, but optimizing it to train faster with better accuracy is a completely different
set of domain. So, it is very important that we focus on optimizing our algorithms to converge faster with
desirable accuracy and details. In this post, we discuss about a few optimization algorithms that are generally used
to expedite the training process even with indefinitely larger training datasets. The usage of the algorithms solely depend
upon our application and the type of dataset that we use.

If you want to check out the implementation of optimization algorithms in deep neural networks, kindly visit this link here:
**[Optimization Algorithms - DNN](https://github.com/asheeshcric/deeplearning.ai/blob/master/Assignments/2.%20Improving%20Deep%20Neural%20Networks:%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/4.%20Optimization%2Bmethods.ipynb){:target="_blank"}**


Here are the main differences among different types of gradient descents.

![](https://i.ibb.co/F3b2rKb/SGDvBGD.png)
---
![](https://i.ibb.co/VgJZHHW/SGDvMBGD.png)

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
    - $$v_{t}$$ = $$\beta$$$$v_{t-1}$$ + (1 - $$\beta$$)$$\theta_{t}$$


- Generally, we take $$\beta$$ = 0.9 for practical consideration
![](https://i.ibb.co/7VmTvRh/Screenshot-from-2019-07-16-10-15-20.png)

#### Bias Correction

- In exponentially weighted averages, when the initial value $$v_{0}$$ = 0, then it can create an unwanted bias making 
the initial averages to be much lower than the actual. So, we use the following formula to update the value of $$v_{t}$$
    - $$v_{t}$$ = $$v_{t}$$ / (1 - $$\beta^{t}$$)



- This is required for bias correction and not letting the initial values be affected by a fixed bias towards 
**zero** or **origin**
    
    
### 3. Gradient Descent with Momentum

- While using **mini-batch gradient descent**, the parameters get updated after each mini-batch cycle (having some variance
in each update). This make the gradient descent to oscillate a lot while moving towards the convergence
- So, gradient descent with momentum computes an exponentially weighted averages of the gradients and then use that gradient
to update the weights instead
- This helps in reducing the oscillations during G.D. and makes the convergence faster

    - $$v_{dw}$$ = $$\beta$$ $$v_{dw}$$ + (1 - $$\beta$$) dw
  
    - $$v_{db}$$ = $$\beta$$ $$v_{db}$$ + (1 - $$\beta$$) db  


    - w = w - $$\alpha$$ $$v_{dw}$$  

    - b = b - $$\alpha$$ $$v_{db}$$  


- So, here "$$\beta$$" is a new hyperparameter involved which basically carries out exponentially weighted averages on each update making the convergence faster
- Generally, the practically considered value of $$\beta$$ is ~ 0.9
- Hence, this method is basically taking *"exponentially weighted moving averages"* method and merging
it to the *"mini-batch gradient descent"* algorithm


### 4. RMSprop Optimizer

- RMSprop is quite similar to G.D with Momentum except for the fact that it restricts the oscillations in the vertical direction
    - This allows the descent to take greater leaps in the horizontal direction with greater **learning rate** as the vertical movement is restricted
    
- In this case, the exponentially weighted moving averages are calculated differently as shown below

    - $$s_{dw}$$ = $$\beta$$ $$s_{dw}$$ + (1 - $$\beta$$) $$dw^{2}$$  

    - $$s_{db}$$ = $$\beta$$ $$s_{db}$$ + (1 - $$\beta$$) $$db^{2}$$  


    - w = w - $$\alpha$$ * dw / $$\sqrt{s_{dw} + \epsilon}$$  

    - b = b - $$\alpha$$ * db / $$\sqrt{s_{db} + \epsilon}$$  


- RMSprop and Momentum algorithms both decrease the vertical oscillations and increase horizontal speed, making the descent converge faster for a given cost function


### 5. Adam Optimization Algorithm

- It combines the techniques from both RMSprop and Momentum algorithms to calculate the gradients
- The term **Adam** is derived from **Adaptive Moment Estimation**
- First, it calculates gradients using the momentum method:

    - $$v_{dw}$$ = $$\beta_{1}$$ $$v_{dw}$$ + (1 - $$\beta_{1}$$)dw  

    - $$v_{db}$$ = $$\beta_{1}$$ $$v_{db}$$ + (1 - $$\beta_{1}$$)db


    - $$v_{dw_{c}}$$ = $$v_{dw}$$ / (1 - $$\beta_{1}^{t}$$),  -- *where  $$v_{dw_{c}}$$ is the corrected form of $$v_{dw}$$*

    - $$v_{db_{c}}$$ = $$v_{db}$$ / (1 - $$\beta_{1}^{t}$$),  -- *where  $$v_{db_{c}}$$ is the corrected form of $$v_{db}$$*

    
- Then we have the gradients using the RMSprop method:

    - $$s_{dw}$$ = $$\beta_{2}$$ $$s_{dw}$$ + (1 - $$\beta_{2}$$) $$dw^{2}$$  

    - $$s_{db}$$ = $$\beta_{2}$$ $$s_{db}$$ + (1 - $$\beta_{2}$$) $$db^{2}$$


    - $$s_{dw_{c}}$$ = $$s_{dw}$$ / (1 - $$\beta_{2}^{t}$$),   -- *where  $$s_{dw_{c}}$$ is the corrected form of $$s_{dw}$$*

    - $$s_{db_{c}}$$ = $$s_{db}$$ / (1 - $$\beta_{2}^{t}$$),   -- *where  $$s_{db_{c}}$$ is the corrected form of $$s_{db}$$*

    
    
- Finally the weights are updated as follows:

    - w = w - $$\alpha$$ *  $$v_{dw_{c}}$$ / $$\sqrt{s_{dw_{c}} + \epsilon}$$  

    - b = b - $$\alpha$$ *  $$v_{db_{c}}$$ / $$\sqrt{s_{db_{c}} + \epsilon}$$


- Here, the hyperparameters are $$\alpha$$, $$\beta_{1}$$ = 0.9, $$\beta_{2}$$ = 0.999, and $$\epsilon$$ = $$10^{-8}$$ with practical use-case values


### 6. Learning Rate Decay

- During gradient descent, the pathway may oscillate around the minimum if the learning rate is sufficiently large to avoid convergence.
- So, there is a technique to lower down the learning rate as it approaches the minimum so that it converges faster
- The formula for a **decaying learning rate** is given below:

    - $$\alpha$$ = $$\frac{\alpha_{o}}{1\; + \;decay\_rate\; *  \;epoch\_num}$$


- Here, $$\alpha_{0}$$ is the initial learning rate

#### Other learning rate formulae that can be used

- $$\alpha$$ = $$\alpha_{o}$$ * $$0.95^{\:epoch\_num}$$


- Discrete Staircase

    - $$\alpha$$ = $$\frac{k\: *\: \alpha_{o}}{\sqrt{epoch\_num}}\;\;$$ or $$\;\frac{k * \:\alpha_{o}}{\sqrt{t}}$$


- One option is to manually decay the learning rate during the training process which is not feasible most of the times

#### NOTE

- Generally there are **local optimum** during gradient descent which are also called **saddle points** where the pathway of the descent may get stuck not resulting in a convergence
    - So, we should be aware of such points in the descent
    
- Also, plateaus in the learning curve may make the learning slow


To conclude, these were some of the popular optimization algorithms that are used to speed up
the convergence process in a deep neural networks. Incorporating these algorithms can speed up
the training process from days to hours or even sometimes to minutes. Don't forget to check
out other posts related to machine learning and deep learning on my blog. Thank you for reading
and cheers for your next machine learning model.


If you want to check out the implementation of optimization algorithms in deep neural networks, kindly visit this link here:
**[Optimization Algorithms - DNN](https://github.com/asheeshcric/deeplearning.ai/blob/master/Assignments/2.%20Improving%20Deep%20Neural%20Networks:%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/4.%20Optimization%2Bmethods.ipynb){:target="_blank"}**