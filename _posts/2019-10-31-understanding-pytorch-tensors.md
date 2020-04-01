---
title: Understanding PyTorch Tensors
date: 2019-10-31 11:23:00 +0515
categories: [Tutorial]
tags: [pytorch python deep_learning]
description: Understanding how tensor operations are carried out in PyTorch with examples.
published: false
---

This post is just intended to present example codes to see how tensors are operated and manipulated in
PyTorch. There are no explanations or descriptions for any of the code. This post is just for reference purposes and will soon be removed.

```python
import torch
```

###### Creating a tensor in PyTorch


```python
a = torch.tensor([1, 2, 3, 4])
```


```python
a
```




    tensor([1, 2, 3, 4])



###### Checking the tensor type


```python
a.type()
```




    'torch.LongTensor'



###### Type of data stored in the tensor


```python
a.dtype
```




    torch.int64



###### Create a tensor of specific type


```python
b = torch.FloatTensor([1, 2, 3, 4])
b.type()
```




    'torch.FloatTensor'



###### Size and dimension of a tensor


```python
print(a.size())
print(a.ndimension())
```

    torch.Size([4])
    1


###### Adding a new dimension to a tensor


```python
a_col = a.view(4, 1) # 4 rows and 1 column
# OR
a_col = a.view(-1, 1) # In case you don't know the number of rows, both do the same thing

a_col.size()
```




    torch.Size([4, 1])



###### Numpy array and tensors


```python
# Numpy array and tensors

import numpy as np

np_array = np.array([1, 2, 3, 4])
# From numpy array to a torch tensor
torch_tensor = torch.from_numpy(np_array)

# Back to torch tensor from a numpy array
back_to_np_array = torch_tensor.numpy()

"""
Here, the tensors and np_arrays carry reference of the ones they were assigned from
torch_tensor points back to np_array, and
back_to_np_array points back to torch_tensor

So, if you change the variable np_array, both torch_tensor and back_to_np_array change
"""

print(torch_tensor)
print(back_to_np_array)
```

    tensor([1, 2, 3, 4])
    [1 2 3 4]



```python
# Tensor to list
this_tensor = torch.tensor([1, 2, 3, 4])
this_tensor.tolist()
```




    [1, 2, 3, 4]



### Vector Addition and Subtraction


```python
u = torch.tensor([1.0, 2.1, 4.2])
v = torch.tensor([3.0, 1.9, -0.2])

z = u + v
z
```




    tensor([4.0000, 4.0000, 4.0000])



###### Multiplication and dot product of two tensors


```python
# Multiplying two tensors
m = u * v

# Dot product of two tensors
d = torch.dot(u, v)

print(m)
print(d)
```

    tensor([ 3.0000,  3.9900, -0.8400])
    tensor(6.1500)


###### Adding scalar to a tensor (Broadcasting)


```python
# Adding a scalar to a tensor is similar to broadcasting in numpy
m+1
```




    tensor([4.0000, 4.9900, 0.1600])



### Universal Function in PyTorch


```python
m.mean()
```




    tensor(2.0500)




```python
m.max()
```




    tensor(3.9900)



### Getting evenly spaced numbers from one point (lower) to another (upper)


```python
torch.linspace(-2, 2, steps=9).type(torch.IntTensor)
```




    tensor([-2, -1, -1,  0,  0,  0,  1,  1,  2], dtype=torch.int32)



###### Plotting a sin(x) function


```python
x = torch.linspace(0, 2*np.pi, 100)
y = torch.sin(x)

import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(x.numpy(), y.numpy())
```




    [<matplotlib.lines.Line2D at 0x7f9855de8898>]




![png]({{ site.assets_url }}/img/posts/pytorch_tensors_plot.png)


### Derivatives in PyTorch


```python
x = torch.tensor(2.0, requires_grad=True)
y = x**2
y.backward()
x.grad
```




    tensor(4.)



![](https://i.ibb.co/n3k5z7k/derivative-in-pytorch.png)


```python
x = torch.linspace(-10, 10, 10, requires_grad=True)
Y = x**2
y = torch.sum(x**2)

y.backward()

plt.plot(x.detach().numpy(), Y.detach().numpy(), label='function')
plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label='derivative')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f9852d27780>




![png]({{ site.assets_url }}/img/posts/pytorch_tensors_plot_2.png)


![](https://i.ibb.co/9wsW4wk/Screenshot-from-2019-11-01-11-58-09.png)

## Dataset Class in PyTorch


```python
from torch.utils.data import Dataset
```

###### The class below inherits the Dataset class and is used to represent data in a table (x, y)


```python
class ToySet(Dataset):
    def __init__(self, length=100, transform=None):
        self.x = 2*torch.ones(length, 2)
        self.y = torch.ones(length, 1)
        self.len = length
        self.transform = transform
        
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    def __len__(self):
        return self.len
```


```python
dataset = ToySet()
for i in range(3):
    x, y = dataset[i]
    print(i, 'x: {}, y: {}'.format(x, y))
```

    0 x: tensor([2., 2.]), y: tensor([1.])
    1 x: tensor([2., 2.]), y: tensor([1.])
    2 x: tensor([2., 2.]), y: tensor([1.])


###### Making a Transform class for our dataset that transforms our samples


```python
class add_mult(object):
    def __init__(self, addx=1, muly=1):
        self.addx = addx
        self.muly = muly
        
    def __call__(self, sample):
        x, y = sample
        x += self.addx
        y *= self.muly
        return x, y
```


```python
a_m = add_mult()
dataset = ToySet(transform=a_m)
dataset[0]
```




    (tensor([3., 3.]), tensor([1.]))




```python
# Another transform class to modify our data samples
class mult(object):
    def __init__(self, mul=100):
        self.mul = mul
        
    def __call__(self, sample):
        x, y = sample
        x *= self.mul
        y *= self.mul
        return x, y
```

###### Now, if we need to use multiple transforms to our data samples


```python
from torchvision import transforms
```


```python
# This applies the two transforms one by one to the data samples
data_transform = transforms.Compose([add_mult(), mult()])

data_set = ToySet(transform=data_transform)
data_set[0]
```




    (tensor([300., 300.]), tensor([100.]))


