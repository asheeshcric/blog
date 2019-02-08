---
title: How to write pythonic code?
---

![Pythonic Code](img/pythonic-code.jpg "Pythonic Code")
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
{% highlight python %}
>> import this
{% endhighlight %}
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
code and when running the app can save you a lot of time.

My personal favorite is **Code Inspection** which is provided as default by the PyCharm IDE.
You can obviously choose your favorite as per your requirements. This article explains it
nicely on improving [code quality with linters](https://realpython.com/python-code-quality/#linters){:target="_blank"}.




