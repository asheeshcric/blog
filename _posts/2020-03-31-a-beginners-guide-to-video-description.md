---
title: A Beginner's Guide to Understanding Video Description
date: 2020-03-31 16:51:32 -0500
categories: [Tutorial]
tags: [deep_learning vision nlp]
description: This post discusses about different methods, surveys, and metrics that have been introduced in the field of video description. Video description is one of the popular fields in today's research that involves understanding and detection of occurrences of many entities in a video.
---

I am writing this at the time when it has been more than two weeks since the United States (including most parts of the
world) have gone into lockdown and all the colleges and work places have been moved to be operating remotely. This is the
reason I've had some spare time to read a few research papers covering topics like video analysis, self-supervised learning,
and human-action recognition. I would like to grab this opportunity to write a comprehensive article on things that I learned
about video description and its current state of the art methodologies, datasets, and results.

Video description is one of the popular fields in today's research that involves understanding and detection of occurrences
of many entities in a video. By "entities", we mean things like *background scene*, *humans*, *objects*, *human actions*,
*human-object interactions*, *other events*, and the order in which these events occur in the video. To handle such problem
in hand, over the past few years, Computer Vision (CV) and Natural Language Processing (NLP) techniques have joined forces
to address the upsurge of research interests in understanding and describing images and videos. Automatic video description
has many applications in human-robot interaction, automatic video subtitling and video surveillance. Moreover, it can be
used to help the visually impaired with the help of verbal description of surroundings through speech synthesis generated
from such models. Currently, these are generally achieved through very costly and time-consuming manual processes which is
the reason why any advancement in automatic video description opens up enormous opportunities in many application domains.

Before starting with the methodologies, we would want to understand what the following terminologies refer to and what are
the key differences among them.
- **Visual Description**: Automatic generation of single or multiple natural language sentences that convey the information
in still images or video clips.

- **Video Captioning**: A single automatically generated natural language sentence for a single video clip based on the
premise that the video clips usually contain only one main event.

- **Video Description**: Automatically generating multiple natural language sentences that provide a narrative of a relatively
longer video clip. It is sometimes also referred to as *story telling* or *paragraph generation*.

- **Dense Video Captioning**: Detecting and conveying information of all, possibly overlapping events of different lengths
in a videos using a natural language sentence per event.

![Visual Description]({{ site.assets_url }}/img/posts/visual_description.png "Visual Description")


Video description research has mainly been through three phases. On the basis of these three phases, the video description
literature can be divided into three main categories (phases):

- Classical Methods Phase
- Statistical Methods Phase
- Deep Learning Phase


## **Classical Methods Phase**

The SVO (Subject, Object, Verb) tuples based methods are among the first successful methods used specifically for video
description. However, the first attempt goes back to [Koller et al.](https://ieeexplore.ieee.org/document/139667) in 1991,
who developed a system that was able to characterize motion of vehicles in real traffic scenes using natural language verbs.
In 1997, [Brand et al.](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.52.5341) extended this method and described
it as "Inverse Hollywood Problem" (since in movies, scripts/descriptions get converted into video, here the problem is opposite
). His team described a series of actions into semantic tag summaries in order to develop a storyboard from instructional
videos, developing a system called "video gister", that was able to heuristically parse the video into a series of key actions
and generate a script that describes actions detected in the video. However, video gister was limited to only one human arm
(actor) interacting with non liquid objects and was able to understand only five actions (touch, put, get, add, remove).

As discussed earlier, the only successful method during that phase was SVO tuple based methods which dealt with generation
of video description in two stages: *content identification* which focuses on visual recognition and classification of the
main objects in the video clip and *sentence generation* which maps the objects identified in the first stage to Subject,
Verb, and Object and filling in handcrafted templates for grammatically sound sentences. These templates are created using
rule-based systems and hence are only effective in very constrained environments. Several recognition techniques have been
used for the first stage of SVO tuples based approach which are described below.
- Object Recognition:
    - Object recognition in SVO approaches was performed using conventional methods such as [model-based shape matching
    through edge detection or color matching](http://www.cs.ait.ac.th/~mdailey/cvreadings/Kojima-ActionRecognition.pdf),
    [HAAR](https://ieeexplore.ieee.org/document/990517) features matching, [context-based object recognition](https://www.cs.ubc.ca/~murphyk/Papers/iccv03.pdf),
    Scale Invariant Feature Transform [(SIFT)](https://www.cs.ubc.ca/~lowe/papers/iccv99.pdf), discriminatively trained
    part-based [models](http://cs.brown.edu/people/pfelzens/papers/lsvm-pami.pdf), and Deformable Parts Model [(DPM)](http://rogerioferis.com/VisualRecognitionAndSearch2013/material/Class4DPM2.pdf).
    These methods were particularly applied using hand-crafted features and extra-domain knowledge to make the model perform
    better (but in a constrained environment).
    
- Human and Activity Detection:
    - For human detection in a video clip, popular methods employed features such as Histograms of Oriented Gradients 
    [(HOG)](https://ieeexplore.ieee.org/document/1467360) followed by SVM classifier. For activity detection, features like
    Spatiotemporal Interest Points such as Histogram of Oriented Optical Flow [(HOOF)](http://www.cis.jhu.edu/~rizwanch/papers/ChaudhryCVPR09.pdf),
    Bayesian Networks [(BN)](https://ieeexplore.ieee.org/abstract/document/905296/similar#similar), Dynamic Bayesian Networks
    [(DBNs)](https://www.eecs.qmul.ac.uk/~sgg/papers/GongXiangICCV03.pdf), Hidden Markov Models [(HMM)](https://ieeexplore.ieee.org/abstract/document/643892?section=abstract),
    state machines, and [PNF](https://ieeexplore.ieee.org/document/698711) Networks have been used by SVO approaches.
    

- Integrated Approaches:
    - Instead of detecting the description-relevant entities separately, Stochastic Attribute Image Grammar [(SAIG)](http://www.stat.ucla.edu/~sczhu/papers/Reprint_Grammar.pdf)
    and Stochastic Context Free Grammars [(SCFG)](https://www.aaai.org/Papers/AAAI/2002/AAAI02-116.pdf) allow for compositional
    representation of visual entities present in a video, an image or a scene based on their spatial and functional relations.
    In these methods, the content of an images is first extracted as a parse graph and then a parsing algorithm is then used
    to find the best scoring entities that describe the video.
    
    
For the second stage of SVO approach, *sentence generation*, a variety of methods have been proposed including [(HALogen)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.499.8680&rep=rep1&type=pdf)
representation, Head-driven Phrase Structure Grammar [(HPSG)](https://www.press.uchicago.edu/ucp/books/book/chicago/H/bo3618318.html),
planner and surface realizer. The major task of these methods is to define templates. By template, we mean a user-defined
language structure containing placeholders and comprises of three parts named *lexicons*, *grammar*, and *template rules*.
We shall not go into the details as this post is more focused on the overview of the methods been applied so far. The figure
below shows the basic workflow of a SVO tuple based approach. For more details, please use the reference links embedded with
the text above.

![SVO Tuple Approach]({{ site.assets_url }}/img/posts/svop_method.png "SVO Tuple Approach") 


    
## **Statistical Methods Phase**

The SVO tuple rule-based methods being popular in the classical phase only work in a very constrained dataset and are naive and
inadequate to describe open domain videos and large datasets, such as YouTubeClips and other open domain datasets. The
important differences between these open domain and previous datasets are that these open videos contain unforeseeable
diverse set of subjects, objects, activities, and places. Moreover, due to the sophisticated nature of human languages and
lengthy videos, they are annotated with multiple viable meaningful descriptions. But description of such lengthy videos
with multiple sentences or paragraphs is more desirable in today's world as it can reduce the cost of labor in many fields.

To incorporate such longer open domain videos, [Rohrbach et al](http://ivan-titov.org/papers/iccv13.pdf) in 2013 proposed
a machine learning method to convert visual content into natural language. They used parallel corpora of videos and associated
annotations where they first made the network to learn intermediate semantic labels to represent the video using Maximum
Posterior Estimate (MAP). Then, it translates the semantic labels into natural language sentences using Statistical Machine
Translation [(SMT)](https://www.aclweb.org/anthology/P07-2045/) techniques. The methods for object and activity recognition
improved from earlier threshold-based detection to manual feature engineering and traditional classifiers. Machine learning
methods were used for sentence generation to address the issue of large vocabulary. The models used are generally learned
in a weakly supervised or fully supervised fashion.

Although, these methods showed some improvement on the classical methods, the separation of the two stages (*learning visual
features* and *sentence generation*) made these methods incapable of capturing the interplay of visual features and linguistic
patterns, let alone learning a transferable space between visual artifacts and linguistic representations. This approach was
soon overtaken by the rise of modern deep learning approaches.


## **Deep Learning Models Phase**

The soar of deep learning's success in almost all subfields of Computer Vision (CV) and Natural Language Processing (NLP)
has revolutionized video description approaches. Convolutional Neural Networks [(CNNs)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
in particular, are the state-of-the-art for modeling visual data and extracting meaningful features from the provided
visual content. These networks excel at tasks such as [Object Recognition](https://arxiv.org/abs/1409.1556) where as the
more general deep Recurrent Neural Networks (RNNs) like the Long Short-Term Memory [(LSTMs)](),
on the other hand, are now dominating the area of sequence modeling and perform well for tasks like [speech recognition](http://proceedings.mlr.press/v32/graves14.pdf),
and [image captioning](https://arxiv.org/abs/1411.4389)