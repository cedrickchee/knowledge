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
****
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

![Visualizing the `SSD_MultiHead.forward` line-by-line](/images/ssd_multihead_linebyline.png)

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
