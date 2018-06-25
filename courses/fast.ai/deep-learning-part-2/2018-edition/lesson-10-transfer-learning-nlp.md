# Lesson 10 - NLP Classification, Transfer Learning for NLP and Translation

_These are my personal notes from fast.ai course and will continue to be updated and improved if I find anything useful and relevant while I continue to review the course to study much more in-depth. Thanks for reading and happy learning!_

## Topics

* Jump into NLP.
  * Start with an introduction to the new fastai.text library.
  * Cover a lot of the same ground as lesson 4.
* How to get much more accurate results, by using transfer learning for NLP.
* How pre-training a full language model can greatly surpass previous approaches based on simple word vectors.
  * Use language model to show a new state of the art result in text classification.
* How to complete and understand ablation studies.

## Lesson Resources

* [Website](http://course.fast.ai/lessons/lesson10.html)
* [Video](https://youtu.be/h5Tz7gZT9Fo)
* [Wiki](http://forums.fast.ai/t/part-2-lesson-10-wiki)
* Jupyter Notebook and code
  * [imdb.ipynb](https://nbviewer.jupyter.org/github/fastai/fastai/blob/master/courses/dl2/imdb.ipynb)
* Dataset
  * [IMDB large movie reviews](http://ai.stanford.edu/~amaas/data/sentiment/) / [direct download link](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)
* [Slides](http://files.fast.ai/part2_2/lesson10_2.pptx)

## Assignments

### Papers

* Must read
  * [Regularizing and Optimizing LSTM Language Models](https://arxiv.org/abs/1708.02182) on understanding dropout in LSTM models by Stephen Merity et al.
  * [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146) (ULMFiT or FitLam) - transfer learning for NLP by Jeremy Howard and Sebastian Ruder
  * [A disciplined approach to neural network hyper-parameters](https://arxiv.org/abs/1803.09820) by Leslie N. Smith
  * [Learning non-maximum suppression](https://arxiv.org/abs/1705.02950) (NMS) - end-to-end convolutional network to replace manual NMS by Jan Hosang et al.

### Other Resources

#### Other Useful Information

* [List of mathematical symbols](https://en.wikipedia.org/wiki/List_of_mathematical_symbols)
* [Stanford CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/index.html)

## My Notes

### ([0:00:14](https://youtu.be/h5Tz7gZT9Fo?t=14s)) Review lesson 9 - SSD

- Many students are struggling with last week's material, so if you are finding it difficult, that's fine. The reason Jeremy put it up there up front is so that we have something to cogitate about, think about, and gradually work towards, so by lesson 14, you'll get a second crack at it.
- To understand the pieces, you'll need to understand the shapes of convolutional layer outputs, receptive fields, and loss functions — which are all the things you'll need to understand for all of your deep learning studies anyway.
- One key thing is that we started out with something simple — a single object classifier, single object bounding box without a classifier, and then single object classifier and bounding box. The bit where we go to multiple objects is actually almost identical to that except we first have to solve the matching problem. We ended up creating far more activations than we need for our our number of ground truth bounding boxes, so we match each ground truth object to a subset of those activations. Once we've done that, the loss function that we then do to each matched pair is almost identical to this loss function (i.e. the one for single object classifier and bounding box).
- If you are feeling stuck, go back to lesson 8 and make sure you understand Dataset, DataLoader, and most importantly the loss function.
- So once we have something which can predict the class and bounding box for one object, we went to multiple objects by just creating more activations [[0:02:40](https://youtu.be/h5Tz7gZT9Fo?t=2m40s)]. We had to then deal with the matching problem, having dealt with a matching problem, we then moved each of those anchor boxes in and out a little bit and around a little bit, so they tried to line up with particular ground truth objects.
- We talked about how we took advantage of the convolutional nature of the network to try to have activations that had a receptive field that was similar to the ground truth object we were predicting. Chloe provided the following fantastic picture to talk about what `SSD_MultiHead.forward` does line by line:

![Visualizing the `SSD_MultiHead.forward` line-by-line by [Chloe Sultan](http://forums.fast.ai/u/chloews)](/images/ssd_multihead_linebyline.png)

What Chloe's done here is she's focused particularly on the dimensions of the tensor at each point in the path as we gradually downsampled using stride 2 convolutions, making sure she understands why those grid sizes happen then understanding how the outputs come out of those.

- This is where you've got to remember this `pbd.set_trace()`. I just went in just before the class and went into `SSD_MultiHead.forward` and entered `pdb.set_trace()` and then I ran a single batch. Then I could just print out the size of all these. We make mistakes and that's why we have debuggers and know how to check things and do things in small little bits along the way.
- We then talked about increasing k [00:05:49] which is the number of anchor boxes for each convolutional grid cell which we can do with different zooms, aspect ratios, and that gives us a plethora of activations and therefore predicted bounding boxes.
- Then we went down to a small number using Non Maximum Suppression.
- Non Maximum Suppression is kind of hacky, ugly, and totally heuristic, and we did not even talk about the code because it seems hideous. Somebody actually came up with a [paper](https://arxiv.org/abs/1705.02950) recently which attempts to do an end-to-end conv net to replace that NMS piece.
- Have you been reading the papers?
  - Not enough people are reading papers! What we are doing in class now is implementing papers, the papers are the real ground truth. And I think you know from talking to people a lot of the reason people aren't reading paper is because a lot of people don't think they are capable of reading papers. They don't think they are the kind of people that read papers, but you are. You are here. We started looking at a paper last week and we read the words that were in English and we largely understood them. If you look at the picture above carefully, you’ll realize `SSD_MultiHead.forward` is not doing the same. You might then wonder if this is better. My answer is probably. Because `SSD_MultiHead.forward` was the first thing I tried just to get something out there. Between this and YOLO v3 paper, they are probably much better ways.
  - One thing you'll notice in particular they use a smaller `k` but they have a lot more sets of grids 1x1, 3x3, 5x5, 10x10, 19x19, 38x38 — 8732 per class. A lot more than we had, so that'll be an interesting thing to experiment with.
  - Another thing I noticed is that we had 4x4, 2x2, 1c1 which means there are a lot of overlap — every set fits within every other set. In this case where you've got 1, 3, 5, you don't have that overlap. So it might actually make it easier to learn.
  
    :memo: There's lots of interesting you can play with.
- Match the equations to the code.
  - :bookmark: Perhaps most important thing I would recommend is to put the code and the equations next to each other. You are either math person or code person. By having them side by side, you will learn a little bit of the other.
  - Learning the math is hard because of the notation might seem hard to look up but there are good resources such as [wikipedia](https://en.wikipedia.org/wiki/List_of_mathematical_symbols).
  - Another thing you should try doing is to re-create things that you see in the papers. Here was the key most important figure 1 from the focal loss paper.
  - Last lesson's code had a minor bug—after fixing it the predicted boxes look a lot better! The way Jeremy was flattening out the convolutional activations did not line up with how he was using them in the loss function, and fixing that made it quite a bit better.

### ([0:16:40](https://youtu.be/h5Tz7gZT9Fo?t=16m40s)) Introducing fastai.text

_WIP_

### IMDB

#### ([0:20:30](https://youtu.be/h5Tz7gZT9Fo?t=16m40s)) IMDB with fastai.text

_WIP_

#### ([0:23:10](https://youtu.be/h5Tz7gZT9Fo?t=23m10s)) The standard format of text classification dataset

_WIP_

#### ([0:28:08](https://youtu.be/h5Tz7gZT9Fo?t=28m08s)) Difference between tokens and words 1 - spaCy

_WIP_

#### ([0:29:59](https://youtu.be/h5Tz7gZT9Fo?t=29m59s)) Pandas 'chunksize' to deal with a large corpus

_WIP_

#### ([0:32:38](https://youtu.be/h5Tz7gZT9Fo?t=32m38s)) {BOS} (beginning of words) and {FLD} (field) tokens

_WIP_

#### ([0:33:57](https://youtu.be/h5Tz7gZT9Fo?t=33m57s)) Run spaCy on multi-cores with proc_all_mp()

_WIP_

#### ([0:35:40](https://youtu.be/h5Tz7gZT9Fo?t=35m40s)) Diffrence between tokens and word 2 - capture semantic of letter case and others

_WIP_

#### ([0:38:05](https://youtu.be/h5Tz7gZT9Fo?t=38m05s)) Numericalise tokens - Python Counter() class

_WIP_

### Pre-trained Language Model - PreTraining

#### ([0:42:16](https://youtu.be/h5Tz7gZT9Fo?t=42m16s)) Pre-trained language model

_WIP_

#### ([0:47:13](https://youtu.be/h5Tz7gZT9Fo?t=47m13s)) Map imdb index to wiki text index

_WIP_

#### ([0:53:09](https://youtu.be/h5Tz7gZT9Fo?t=53m09s)) fastai documentation project

_WIP_

#### ([0:58:24](https://youtu.be/h5Tz7gZT9Fo?t=58m24s)) Difference between pre-trained LM and embeddings 1 - word2vec

_WIP_

#### ([1:01:25](https://youtu.be/h5Tz7gZT9Fo?t=1h1m25s)) The idea behind using average of embeddings for non-equivalent tokens

_WIP_

### Pre-trained Language Model - Training

#### ([1:02:34](https://youtu.be/h5Tz7gZT9Fo?t=1h2m34s)) Dive into source code of LanguageModelLoader()

_WIP_

#### ([1:09:55](https://youtu.be/h5Tz7gZT9Fo?t=1h9m55s)) Create a custom Learner and ModelData class

_WIP_

#### ([1:20:35](https://youtu.be/h5Tz7gZT9Fo?t=1h20m35s)) Guidance to tune dropout in LM

_WIP_

#### ([1:21:43](https://youtu.be/h5Tz7gZT9Fo?t=1h21m43s)) The reason to measure accuracy than cross entropy loss in LM

_WIP_

#### ([1:25:23](https://youtu.be/h5Tz7gZT9Fo?t=1h25m23s)) Guidance of reading paper vs coding

_WIP_

#### ([1:28:10](https://youtu.be/h5Tz7gZT9Fo?t=1h28m10s)) Tips to vary dropout for eash layer

_WIP_

#### ([1:28:44](https://youtu.be/h5Tz7gZT9Fo?t=1h28m44s)) Difference between pre-trained LM and embeddings 2 - Comparison of NLP and CV

_WIP_

#### ([1:31:21](https://youtu.be/h5Tz7gZT9Fo?t=1h31m21s)) Accuracy vs cross entropy as a loss function

_WIP_

#### ([1:33:37](https://youtu.be/h5Tz7gZT9Fo?t=1h33m37s)) Shuffle documents; Sort-ish to save computation

_WIP_

### Paper ULMFiT (FiTLaM)

#### ([1:44:00](https://youtu.be/h5Tz7gZT9Fo?t=1h44m00s)) Paper: ULMFiT - pre-trained LM

_WIP_

#### ([1:49:09](https://youtu.be/h5Tz7gZT9Fo?t=1h49m09s)) New version of Cyclical Learning Rate

_WIP_

#### ([1:51:34](https://youtu.be/h5Tz7gZT9Fo?t=1h51m34s)) Concat Pooling

_WIP_

#### ([1:52:44](https://youtu.be/h5Tz7gZT9Fo?t=1h52m44s)) RNN encoder and MultiBatchRNN encoder - BPTT for text classification ( BPT3C )

_WIP_

### Tricks to conduct ablation studies

#### ([1:58:35](https://youtu.be/h5Tz7gZT9Fo?t=1h58m35s)) VNC and Google Fire Library

_WIP_

#### ([2:05:10](https://youtu.be/h5Tz7gZT9Fo?t=2h05m10s)) SentenPice; Tokenise Sub-Word units

_WIP_
