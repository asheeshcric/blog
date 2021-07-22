---
title: "Progress of Transformers in Computer Vision"
date: 2021-07-17 12:58:32 -0500
categories: [Illustration]
tags: [machine_learning]
description: This post discusses how Transformers have been adapted from NLP and implemented in several computer vision tasks with better efficiency 
---

The breakthrough in transformer networks in NLP came after the paper "Attention is all you need" was published.

Transformers enable modeling long dependencies between input sequence units and support parallel processing of sequence as compared to reccurrent networks.

The main characteristics of transformers are self-attention, large-scale unsupervised pre-training, and bidirectional feature encoding. Transformers assume minimal prior knowledge about the structure of the problem as compared to their convolutional counterparts in deep learning which is why they are typically pre-trained using pretext tasks on large-scale (unlabelled) datasets. Such pre-training not only avoids costly manual annotations, but also allows the model to learn generalization representations with rich relationships between the entities present in a given dataset.

## Self-attention

Given a sequence of items, self-attention mechanism estimates the relevance of one item to other items (e.g., which words are likely related to each other in a sentence). A self-attention layer updates each component of a sequence by adding more context to it using aggregated global information from the complete input sentence.