---
title: An Overview of research on Video Description
date: 2020-04-01 22:35:32 -0500
categories: [Illustration]
tags: [machine_learning]
description: This post discusses about different methods, surveys, and metrics that have been introduced in the field of video description. Video description is one of the popular fields in today's research that involves understanding and detection of occurrences of many entities in a video.
---

The following post is inspired by the paper [Video Description: A Survey of Methods, Datasets and Evaluation Metrics](https://arxiv.org/abs/1806.00186).
I have tried to summarize the research that has been undergone in the field of visual recognition and video description
in this writing. This post discusses the methods that have been used since the past by categorizing them into three
special groups so that it is easier to understand the progress of classical methods in comparison to modern deep learning
methods.


Video description is one of the popular fields in today's research that involves understanding and detection of occurrences
of many entities in a video. By "entities", we mean things like *background scene*, *humans*, *objects*, *human actions*,
*human-object interactions*, *other events*, and the order in which these events occur in the video. To handle such problems
in hand, over the past few years, Computer Vision (CV) and Natural Language Processing (NLP) techniques have joined forces
to address the upsurge of research interests in understanding and describing images and videos. Automatic video description
has many applications in human-robot interaction, automatic video subtitling, and video surveillance. Moreover, it can be
used to help the visually impaired with the help of a verbal description of surroundings through speech synthesis generated
from such models. Currently, these are generally achieved through very costly and time-consuming manual processes which is
the reason why any advancement in automatic video description opens up enormous opportunities in many application domains.

Before starting with the methodologies, we would want to understand what the following terminologies refer to and what are
the key differences among them.
- **Visual Description**: Automatic generation of single or multiple natural language sentences that convey the information
in still images or video clips.

- **Video Captioning**: A single automatically generated natural language sentence for a single video clip based on the
premise that the video clips usually contain only one main event.

- **Video Description**: Automatically generating multiple natural language sentences that provide a narrative of a relatively
longer video clip. It is sometimes also referred to as *storytelling* or *paragraph generation*.

- **Dense Video Captioning**: Detecting and conveying information of all, possibly overlapping events of different lengths
in a video using a natural language sentence per event.

![Visual Description]({{ site.assets_url }}/img/posts/visual_description.png "Visual Description")


Video description research has mainly been through three phases. On the basis of these three phases, the video description
literature can be divided into three main categories (phases):

- Classical Methods Phase
- Statistical Methods Phase
- Deep Learning Phase


## **Classical Methods Phase**

The SVO (Subject, Object, Verb) tuples based methods are among the first successful methods used specifically for video
description. However, the first attempt goes back to [Koller et al.](https://ieeexplore.ieee.org/document/139667) in 1991,
who developed a system that was able to characterize the motion of vehicles in real traffic scenes using natural language verbs.
In 1997, [Brand et al.](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.52.5341) extended this method and described
it as "Inverse Hollywood Problem" (since in movies, scripts/descriptions get converted into a video, here the problem is opposite
). His team described a series of actions into semantic tag summaries in order to develop a storyboard from instructional
videos, developing a system called "video gister", that was able to heuristically parse the video into a series of key actions
and generate a script that describes actions detected in the video. However, video gister was limited to only one human arm
(actor) interacting with non-liquid objects and was able to understand only five actions (touch, put, get, add, remove).

As discussed earlier, the only successful method during that phase was SVO tuple based methods which dealt with the generation
of video description in two stages: *content identification* which focuses on visual recognition and classification of the
main objects in the video clip and *sentence generation* which maps the objects identified in the first stage to Subject,
Verb, and Object and filling in handcrafted templates for grammatically sound sentences. These templates are created using
rule-based systems and hence are only effective in very constrained environments. Several recognition techniques have been
used for the first stage of the SVO tuples based approach, which are described below.
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
    In these methods, the content of an image is first extracted as a parse graph and then a parsing algorithm is then used
    to find the best scoring entities that describe the video.
    
    
For the second stage of the SVO approach, *sentence generation*, a variety of methods have been proposed including [(HALogen)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.499.8680&rep=rep1&type=pdf)
representation, Head-driven Phrase Structure Grammar [(HPSG)](https://www.press.uchicago.edu/ucp/books/book/chicago/H/bo3618318.html),
planner and surface realizer. The major task of these methods is to define templates. By template, we mean a user-defined
language structure containing placeholders and comprises of three parts named *lexicons*, *grammar*, and *template rules*.
We shall not go into the details as this post is more focused on the overview of the methods been applied so far. The figure
below shows the basic workflow of an SVO tuple based approach. For more details, please use the reference links embedded with
the text above.

![SVO Tuple Approach]({{ site.assets_url }}/img/posts/svop_method.png "SVO Tuple Approach") 


    
## **Statistical Methods Phase**

The SVO tuple rule-based methods being popular in the classical phase, only work in a very constrained dataset and are naive and
inadequate to describe open domain videos and large datasets, such as YouTubeClips and other open domain datasets. The
important differences between these open-domain and previous datasets are that these open videos contain an unforeseeable
diverse set of subjects, objects, activities, and places. Moreover, due to the sophisticated nature of human languages and
lengthy videos, they are annotated with multiple viable meaningful descriptions. But the description of such lengthy videos
with multiple sentences or paragraphs is more desirable in today's world as it can reduce the cost of labor in many fields.

To incorporate such longer open domain videos, [Rohrbach et al](http://ivan-titov.org/papers/iccv13.pdf) in 2013 proposed
a machine learning method to convert visual content into natural language. They used parallel corpora of videos and associated
annotations where they first made the network to learn intermediate semantic labels to represent the video using Maximum
Posterior Estimate (MAP). Then, it translates the semantic labels into natural language sentences using Statistical Machine
Translation [(SMT)](https://www.aclweb.org/anthology/P07-2045/) techniques. The methods for object and activity recognition
improved from earlier threshold-based detection to manual feature engineering and traditional classifiers. Machine learning
methods were used for sentence generation to address the issue of a large vocabulary. The models used are generally learned
in a weakly supervised or fully supervised fashion.

Although, these methods showed some improvement on the classical methods, the separation of the two stages (*learning visual
features* and *sentence generation*) made these methods incapable of capturing the interplay of visual features and linguistic
patterns, let alone learning a transferable space between visual artifacts and linguistic representations. This approach was
soon overtaken by the rise of modern deep learning approaches.


## **Deep Learning Models Phase**

The soar of deep learning's success in almost all subfields of Computer Vision (CV) and Natural Language Processing (NLP)
has revolutionized video description approaches. Convolutional Neural Networks [(CNNs)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
in particular, are state-of-the-art for modeling visual data and extracting meaningful features from the provided
visual content. These networks excel at tasks such as [Object Recognition](https://arxiv.org/abs/1409.1556) whereas the
more general deep Recurrent Neural Networks (RNNs) like the Long Short-Term Memory [(LSTMs)](),
on the other hand, are now dominating the area of sequence modeling and perform well for tasks like [speech recognition](http://proceedings.mlr.press/v32/graves14.pdf),
and [image captioning](https://arxiv.org/abs/1411.4389). The figure below shows the basic model of modern video description
systems.

![Video Description Model]({{ site.assets_url }}/img/posts/conv_lstm.png "Basic Convolutional LSTM/GRU Model")

The block representing *visual model* in the above figure is generally a CNN-based network that is responsible for extracting
important visual features from the input image or video frames. The extracted features output from the visual model is
passed to one or multiple LSTM/GRU layers in sequential order which try to learn the correlation between the input frames.
The exaggeration of the above figure is shown below with detailed flow in a scenario where the input is a video frame sequence
 and the output is a sequence of words or sentences.

![Video Description Methods Summary]({{ site.assets_url }}/img/posts/summary_deep_learning_methods.png "Summary of deep learning based video description methods")


In video description, there are three main architectures that people tend to follow when it comes to deep learning approaches.
These architectures are divided on the basis of their combination of the encoding and decoding stages. Here, in the encoding
stage, the model tries to capture relevant and useful information from the input video sequence whereas in the decoding stage,
the extracted relevant features that represent the input are used in predicting the description of the input video, i.e
in generating sequences of words and sentences. These architectures are individually described below.

### CNN-RNN Video Description
As it is already clear by the name, in this architecture, convolutional networks are used for visual encoding. CNN
is still by far the most popular network structure used for visual encoding. The encoding process can be broadly
categorized into *fixed-size* and *variable-size* video encoding.

- [Donahue et al.](https://arxiv.org/abs/1411.4389) in 2014 were the first to use deep neural networks to solve the problem
of video captioning. They proposed three different architectures to solve this problem. The first one, LSTM
encoder-decoder with CRF max where they replaced the [SMT](http://ivan-titov.org/papers/iccv13.pdf) module with
a stacked LSTM comprising two layers for encoding and decoding. Other variants of this architecture, LSTM decoder with
CRF max and LSTM decoder with CRF probabilities also performed better than the SMT based approach. But they were still
not trainable in an end-to-end fashion.

- [Venugopalan et al.](https://arxiv.org/abs/1412.4729) presented the first end-to-end trainable network architecture
for generating a natural language description of videos. Their model is able to simultaneously learn the semantic as well
as grammatical structure of the associated language and also reported results on open domain YouTube Clips. The model was
built by directly connecting an LSTM to the output of the CNN. This model has been the base of many recent models in
sequence learning. As usual, the CNN layers extract feature vectors which are input to the first LSTM layer, where the
hidden state of the first LSTM becomes the input to the second LSTM unit for caption generation. This end-to-end model
performed better than the previous video description systems at the time and was effectively generating sentences
without any templates. However, as a result of simple averaging, valuable temporal information of the video such as
the order of appearances of any two objects are lost. Therefore, this approach is only capable of generating captions
for short clips with a single main event in the clip.

- With the success of [C3D](https://arxiv.org/abs/1412.0767v1) in capturing spatio-temporal action dynamics in videos,
[Li et al.](https://arxiv.org/abs/1502.08029) proposed a novel 3D-CNN to model the spatio-temporal information in
videos. The 3D-CNN part of their model is based on [GoogLeNet](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43022.pdf)
and pre-trained on an activity recognition dataset. The specialty of their model is that it was able to capture local
fine motion information between consecutive frames. This local information is subsequently summarized and preserved
through higher-level representations by modeling a video as a 3D spatio-temporal cuboid with the concatenation of [HoG,
HoF](https://link.springer.com/chapter/10.1007/11744047_33), [MbH](https://www.irisa.fr/vista/Papers/2009_bmvc_wang.pdf).
These transformations not only help capture local motion features but also reduce the computation of the subsequent
3D CNN. They also introduced temporal attention mechanisms in RNN which improved the results.

- Recently, [GRU-EVE](https://arxiv.org/pdf/1902.10322.pdf) was proposed for video captioning which effectively uses
a standard GRU (Gated Recurrent Unit) for language modeling but enriched with Short Fourier Transform (SFT) on
2D/3D-CNN features in a hierarchical manner to encapsulate the spatio-temporal video dynamics. They further enrich
the visual features with high-level semantics of the detected objects and actions in the video as an extra signal to the
network. Apparently, the enriched features obtained by applying SFT on 2D-CNN features alone outperformed C3D features.

- When it comes to *variable-size visual representation* (unlike above methods), the models are able to directly map
input videos comprising a different number of frames to variable-length words or sentences (outputs) by successfully
modeling various complex temporal dynamics. [Venugopalan et al.](https://arxiv.org/abs/1505.00487) in 2015, again proposed
architecture using a sequence-to-sequence approach for video captioning with a two-layered LSTM framework.

- [Yu et al.](https://arxiv.org/abs/1510.07712) introduced a hierarchical recurrent neural network (h-RNN) that
applies [attention mechanisms](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) on both the temporal
and spatial aspects. They focused on the sentence decoder and introduced a hierarchical framework that comprises of
a sentence generator and on top of that a paragraph generator.

![CNN-RNN Basic Architecture]({{ site.assets_url }}/img/posts/video_description_deep_learning.png "CNN-RNN Basic Architecture")
The figure above represents the basic blocks of commonly-used CNN-RNN architecture. The CNN block is responsible for
extracting enriched visual features that represent the incoming video frames whereas on the other hand, the RNN block acts
as a decoder for text/sentence generation.

- **Popular Video Features Extraction Methods and Action Recognition**

    The following are some methods that employ extraction of enriched features from videos and can be used as base or
    pre-trained models for video description encoders.
    - Long-term Recurrent Convolutional Networks for Visual Recogntion and Description [(LRCN)](https://arxiv.org/abs/1411.4389)
    - Learning Spatiotemporal Features with 3D Convolutional Networks [(C3D)](https://arxiv.org/pdf/1412.0767.pdf)
    - Describing Videos by Exploiting Temporal Structure [(Conv3D & Attention)](https://arxiv.org/abs/1502.08029)
    - [TwoStreamFusion](https://arxiv.org/abs/1604.06573)
    - Temporal Segment Networks [(TSN)](https://arxiv.org/abs/1608.00859)
    - [ActionVlad](https://arxiv.org/pdf/1704.02895.pdf): Learning spatio-temporal aggregation for action classification
    - [Hidden Two-Stream](https://arxiv.org/abs/1704.00389) Convolutional Networks for Action Recognition
    - Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset [(I3D)](https://arxiv.org/abs/1705.07750)
    - Temporal 3D ConvNets [(T3D)](https://arxiv.org/abs/1711.08200)
    
    
### RNN-RNN Video Description
Unlike CNN-RNN architecture, this approach is not quite popular, but people still use it in some cases. Here, RNNs are
used to encode the visual information.

- [Srivastava et al.](https://arxiv.org/abs/1502.04681) used one LSTM layer to extract features from video frames
(i.e. encoding) and then passed the encoded feature vector through another LSTM for decoding. They also extended their
model to predict future sentences from past frames. They adopted a machine translation model for visual recognition but
could not achieve significant improvement in classification accuracy.

- [Yu et al.](https://arxiv.org/abs/1510.07712) also proposed a similar approach using two RNN structures for the video
description task. They used GRU units for sentence generation with a hierarchical decoder structure. Another paragraph
generator unit is then fed input (from the decoder) to model the time dependencies between the output sentences while
focusing on linguistic aspects. However, their model is inefficient for videos involving fine-grained activities and small
interactive objects.


### Deep Reinforcement Learning Models
Deep Reinforcement Learning (DRL) has outperformed humans in many real-world games like Chess and Go. In DRL, artificial
intelligent agents learn from the environment through trial and error and adjust learning policies purely from environmental
rewards or punishments. These approaches have been popularized by [Google Deep Mind](https://deepmind.com/) since 2013.
Still, due to the absence of a straight forward cost function, learning mechanisms in reinforcement learning are
considerably difficult to devise as compared to traditional supervised methods. There are two main challenges with this
approach unlike other problems:
1. The model does not have full access to the function being optimized. It has to query the function through interaction.
2. The interaction with the environment is state-based where the present input depends on previous actions. Hence, the
choice of reinforcement learning algorithms depends on the scope of the problem at hand.

- [Xwang et al.](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Video_Captioning_via_CVPR_2018_paper.pdf) proposed a fully-differentiable neural network architecture using reinforcement learning which follows
a general encoder-decoder framework. The encoder uses ResNet-152 as the base model to capture the video frame features.
The video features are processed through two-stage encoder i.e. low-level LSTM followed by a high-level LSTM. The decoder
stage employed Hierarchical Reinforcement Learning [(HRL)](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Video_Captioning_via_CVPR_2018_paper.pdf)
to generate the word by word natural language descriptions.

- Similarly, in 2018, [Chen et al.](https://arxiv.org/abs/1803.01457) introduced an RL based model selecting *key
informative frames* to represent a complete video, in an attempt to minimize computational complexity and unnecessary
noise. The key frames (a compact subset of 6-8 frames) are selected such that they maximize visual diversity and minimize
the textual discrepancy. The method also did not use motion features for encoding, a design trade-off between speed and
accuracy.

DRL based methods are gaining popularity and have shown comparable results in video description. Since their learning
methodology is unlike other supervised methods, they are less likely to suffer from the scarcity of labeled training data,
computational constraints, and overfitting problems. We can expect these methods to flourish in the near future.