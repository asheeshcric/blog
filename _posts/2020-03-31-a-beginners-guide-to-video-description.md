---
title: A Beginner's Guide to Understanding Video Description
date: 2020-03-31 16:51:32 -0500
categories: [Tutorial]
tags: [deep_learning vision nlp]
description: This post discusses about different methods, surveys, and metrics that have been introduced in the field of video description. Video description is one of the popular fields in today's research that involves understanding and detection of occurrences of many entities in a video.
---

I am writing this at the time when it has been more than two weeks since the United States (including
most parts of the world) have gone into lockdown and all the colleges and work places have been moved
to be operating remotely. This is the reason I've had some decent opportunities to read a few research
papers covering topics like video analysis, self-supervised learning, and human-action recognition. I
would like to grab this opportunity to write a comprehensive article on things that I learned about 
video description and its current state of the art methodologies.

Video description is one of the popular fields in today's research that involves understanding and
detection of occurrences of many entities in a video. By "entities", we mean things like *background scene*,
*humans*, *objects*, *human actions*, *human-object interactions*, *other events*, and the order in which
these events occur in the video. To handle such problem in hand, over the past few years, Computer Vision (CV)
and Natural Language Processing (NLP) techniques have joined forces to address the upsurge of research interests in 
understanding and describing images and videos. Automatic video description has many applications in human-robot
interaction, automatic video subtitling and video surveillance. Moreover, it can be used to help the visually impaired
with the help of verbal description of surroundings through speech synthesis generated from such models. Currently,
these are generally achieved through very costly and time-consuming manual processes which is the reason why any
advancement of video description opens up enormous opportunities in many application domains.