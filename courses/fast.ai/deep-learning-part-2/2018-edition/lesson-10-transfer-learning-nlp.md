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

### Review of Last Week

- Many students are struggling with last week's material, so if you are finding it difficult, that's fine. The reason Jeremy put it up there up front is so that we have something to cogitate about, think about, and gradually work towards, so by lesson 14, you'll get a second crack at it.
