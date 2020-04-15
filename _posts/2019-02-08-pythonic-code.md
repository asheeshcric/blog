---
title: How to write pythonic code?
date: 2019-02-08 11:23:00 +0515
categories: [Demo]
tags: [best_practice]
description: Who decides if a code written in Python is actually pythonic or not?
  These are some questions which I will try to discuss in this post so that you can
  have some insights.
seo:
  date_modified: 2020-04-13 14:54:59 -0500
---

![Pythonic Code]({{ site.assets_url }}/img/posts/pythonic-code.jpg "Pythonic Code")
Ever since I shifted from C to Python, my life has been terribly easy as Python
is such a neat and powerful language. But I have had my problems in the beginning.
Although I started writing code in Python, the structure was still vastly influenced
by my experience with C.

Working with people experienced in Python, I slowly began to get a hang of this
language. Not only was I starting to reduce chunks of code into a few lines, I
was also beginning to understand its ecosystem  .

The major agenda for today is: "What exactly is **pythonic**?"
Who decides if a code written in Python is actually pythonic or not? These are
some questions which I will try to discuss in this post so that you can have some
insights.

### The Zen of Python, by Tim Peters
Open your python console and type in the following command. This will show you a
list of succinct guiding principles for Python's design. **Keep them in mind**

```ruby
>> import this
```

```
- Beautiful is better than ugly.
- Explicit is better than implicit.
- Simple is better than complex.
- Complex is better than complicated.
- Flat is better than nested.
- Sparse is better than dense.
- Readability counts.
- Special cases aren't special enough to break the rules.
- Although practicality beats purity.
- Errors should never pass silently.
- Unless explicitly silenced.
- In the face of ambiguity, refuse the temptation to guess.
- There should be one-- and preferably only one --obvious way to do it.
- Although that way may not be obvious at first unless you're Dutch.
- Now is better than never.
- Although never is often better than *right* now.
- If the implementation is hard to explain, it's a bad idea.
- If the implementation is easy to explain, it may be a good idea.
- Namespaces are one honking great idea -- let's do more of those!
```

After understanding the abstract rules and paradigms of Python, let's get into
ways which can help you write actual standard python code.

### PEP8 -- Style Guide for Python Code

**PEP** is an acronym for Python Enhancement Proposal and PEP8 is the most widely
followed coding conventions for Python. It standardizes:
- Code Layout
- Naming Conventions
- Comments Styling
- Programming Recommendations & Annotations
- and much more...

For further information on PEP8, follow the official documentation: [PEP8](https://www.python.org/dev/peps/pep-0008/){:target="_blank"}

### Linters
Although PEP8 documentation is a great place to start writing structured code, it
is very difficult to track layout errors in your code with the naked eye. In my
personal experience, linters play a vital role in analysing code for potential errors.
They check for both Programmatic and Stylistic errors. Detecting such errors while writing
code and not when running the app can save you a lot of time.

My personal favorite is **Code Inspection** which is provided as default by the PyCharm IDE.
You can obviously choose your favorite as per your requirements. This article explains it
nicely on improving [code quality with linters](https://realpython.com/python-code-quality/#linters){:target="_blank"}.


### Checkout some standard open-source Python Libraries
It is a difficult task to read others' code (especially when you are a beginner in programming), but this
is a good habit to develop. If one has to drive in the path of becoming a great Python programmer, they should have
the ability to read, understand, and comprehend excellent code. One way of making 
yourself comfortable is to read code from popular open-source libraries out there. Some
examples include [Requests - An HTTP library written for humans](https://github.com/requests/requests){:target="_blank"}
and [Flask - A microframework for Python](https://github.com/pallets/flask){:target="_blank"}. 
You can find plenty of such examples on platforms like Github.


### Welcome feedback on your code
As a programmer, I have always believed that feedback on your code from experienced
people weigh pretty high on improving your coding habits. If you can put your ego
aside and think on the comments that others make about your code, this can help you
learn faster than anything else. Obviously, not every feedback will be of high standards,
but you will slowly learn to distinguish between what is bad and what is not.


### Ensuring clean code with parameterized python
Parameterization in Python is one of the most elegant ways to produce clean and pythonic code. This technique mainly
helps us prevent using copy-paste programming with the introduction of functions and classes. If you want to have a
deeper understanding on writing parameterized python code, then checkout this cool and insightful post from
[Toptal.](https://www.toptal.com/python/python-parameterized-design-patterns){:target="_blank"}


### Writing Idiomatic Python code
Keeping the above points in your mind will set you up with a mindset to write better
python code. Sometimes, it can be a hassle to go through all the documentations to
find out which code is better than the other. So, I have written a post that makes an
effort to reveal the most useful python tricks that you can implement in your code on
the go. Checkout this post [Tricks for writing Idiomatic Python Code]({{ site.url }}/idiomatic-python).