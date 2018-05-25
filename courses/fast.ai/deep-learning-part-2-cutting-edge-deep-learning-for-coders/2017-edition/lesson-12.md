# Lesson 12 - Attentional Models

Topics:

* Clustering in TensorFlow
* Attentional models

Lesson

* [Website](http://course17.fast.ai/lessons/lesson12.html)
* [Video](https://youtu.be/jy1w0mPCHb0)
* [Wiki](http://forums.fast.ai/t/lesson-12-wiki)
* [Forum discussion](http://forums.fast.ai/t/lesson-12-discussion)

## Coursework

### Jupyter Notebook Used

- [time: 00:00:00] [kmeans_test.ipynb](https://github.com/fastai/courses/blob/master/deeplearning2/kmeans_test.ipynb)
- [time: 00:40:00] [spelling\_bee\_RNN.ipynb](https://github.com/fastai/courses/blob/master/deeplearning2/spelling_bee_RNN.ipynb)

### Reading: Paper \[TODO\]

- [time: 01:37:28] [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) by Dzmitry Bahdanau, et. al
- [time: 01:44:08] [Grammar as a Foreign Language](https://arxiv.org/abs/1412.7449) by Oriol Vinyals et. al
- [Massive Exploration of Neural Machine Translation Architectures](https://arxiv.org/abs/1703.03906)
    - This is a very recent research paper on the subject (released last week 21 Mar 2017)
    - Open source code: https://github.com/google/seq2seq/

## My Notes

- [time: 00:00:00] Clustering again. Kmeans.
- [time: 00:40:00] Intro to next step: NLP and translation deep-dive, with CMU pronouncing dictionary via `spelling_bee_RNN.ipynb`
- [time: 01:32:00] Attention models.  I actually really like these, I think they're great.  And really the paper that introduced these, quite an extraordinary paper, introduced both GRUs and attention models at the same time.  I think it might even be before the guy had his PhD, if I remember correctly.  It was just a wonderful paper, very successful.
- [time: 01:28:38] Even though I try to teach things which I think are going to stand the test of time, I'm not at all convinced that any technique for reinforcement learning is going to stand the test of time.  So I don't think we're going to touch that.
- [time: 01:44:08] This whole thing is summarized in another paper, actually a very cool paper, Grammar as a Foreign Language.  Lots of names you probably recognize here, Geoffrey Hinton, who's kind of the father of deep learning; Illya Stuskever, who's Director of Science at OpenAI; Oriel Vinyals, who's done lots of cool stuff.  This paper is kind of neat and fun anyway.  It basically says, What if you didn't know anything about grammar and you attempted to build a neural net which assigned grammar to sentences.  It turns out you actually end up with something more accurate than any rule-based grammar system that's been built.
- [time: 01:49:50] AMA - Question:  Any advice on imbalanced datasets? Seems to be a recurring issue with real world data.
