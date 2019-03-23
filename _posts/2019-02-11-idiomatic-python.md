---
title: Tricks for writing idiomatic Python code
description: This post reveals some cool Python3 tricks that can make your code look elegant and much efficient. The principle behind this is to write idiomatic python code so that it is more human-readable.
---

This post reveals some cool **Python3** tricks that can make your code look elegant and
much efficient. The principle behind this is to write idiomatic python code so that
it is more human-readable.


**1. Indexing and looping through a list**
- Using the **enumerate()** function you can easily get the `index` of an item and the
`item` itself of a list (or any iterable) in pairs.

{% highlight python linenos %}
    >>> some_list = ['a', 'b', 'c', 'd']
    >>> for i, alphabet in enumerate(some_list):
    ...     print('{0}: {1}'.format(i, alphabet))
    ... 
    0: a
    1: b
    2: c
    3: d
{% endhighlight %}

**2. Using the "zip" function**
- If you ever need to pair the corresponding elements of two list, then you can use the
`zip` function to easily do so.
- Just make sure that both the lists are of same length to retrieve all the pair items
from both the lists.
    - You can try yourself to see the result if you zip two lists with different lengths.

{% highlight python linenos %}
    >>> first_list = ['a', 'b', 'c']
    >>> second_list = [1, 2, 3]
    >>> list(zip(first_list, second_list))
    [('a', 1), ('b', 2), ('c', 3)]
{% endhighlight %}

**3. Swapping variables**
- Although this might seem trivial in Python, it can be beneficial specifically when
you are trying to shorten your code.
- You can swap two variables in Python in just one line.
{% highlight python linenos %}
    >>> a, b = 1, 10
    >>> a, b = b, a
    >>> print(a, b)
    10 1
{% endhighlight %}

**4. Dictionary Lookups**
- If you are ever retrieving a value for a key from a dictionary, and you want to avoid
a `KeyError`, then this might come in handy
- This helps you retrieve the value for a `key` if it is present in the dictionary.
Otherwise, you can assign a default value in case the key is not present.
{% highlight python linenos %}
    >>> x = {'a': 1, 'b': 2}
    >>> x.get('a', 'Key Not Present')
    1
    >>> x.get('c', 'Key Not Present')
    'Key Not Present'
{% endhighlight %}
    
**5. Merging Dictionaries**
- Merging two dictionaries is very simple in Python3. You can accomplish this in just
a single command.
- This is a very handy trick to make your code look idiomatic and beautiful.
{% highlight python linenos %}
    >>> x = {'a': 1}
    >>> y = {'b': 2}
    
    >>> {**x, **y}
    {'a': 1, 'b': 2}
{% endhighlight %}

**6. Check if a string contains an integer**
- Most of the times, you receive integer ids from form-data in the form of strings, and
you may need to check if the data you received is actually an integer id or not.
- In such cases, you can implement the following two techniques.
{% highlight python linenos %}
    # First technique (Only works for positive integers)
    >>> string_int = '32'
    >>> string_int.isdigit()
    True
    
    # Second method (Works for all integers)
    >>> def is_integer(some_integer_in_string):
    ...     try:
    ...             int(some_integer_in_string)
    ...             return True
    ...     except ValueError:
    ...             return False
    ... 
    >>> is_integer('-32')
    True
    >>> is_integer('49')
    True
    >>> is_integer('a')
    False
{% endhighlight %}

**7. Reverse a list or string**
- You can easily reverse a list or string or any iterable that supports slicing in 
one line.
{% highlight python linenos %}
    # Reversing a list
    >>> original_list = [1, 2, 3, 4]
    >>> reversed_list = original_list[::-1]
    >>> reversed_list
    [4, 3, 2, 1]
    
    # Reversing a string
    >>> some_string = 'This is a string'
    >>> reversed_string = some_string[::-1]
    >>> reversed_string
    'gnirts a si sihT'
{% endhighlight %}

**8. Function argument unpacking**
- In python you can assign dynamic arguments while making a function call
- There are ways through which you can send any number of argument values to a user-defined
function
{% highlight python linenos %}
    >>> def some_function(x, y, z):
    ...     print(x, y, z)
    ... 
    >>> some_function(*[1, 2, 3])
    1 2 3
    >>> some_function(**{'x': 1, 'y': 2, 'z': 3})
    1 2 3
    
    # Another way
    >>> def another_function(*args, **kwargs):
    ...     print(args)
    ...     print(kwargs)
    ...
    >>> another_function(1, 2, 3, {'a': 1, 'b': 2}, (1, 2), **{'x': 1, 'y': 2})
    (1, 2, 3, {'a': 1, 'b': 2}, (1, 2))
    {'x': 1, 'y': 2}
{% endhighlight %}



- Comment below if you know some additional tips and tricks for Python3 that I missed 
out.