# Lesson 11 - Neural Translation

_These are my personal notes from fast.ai course and will continue to be updated and improved if I find anything useful and relevant while I continue to review the course to study much more in-depth. Thanks for reading and happy learning!_

## Topics

* Learn to translate French into English.
  * How to add attention to an LSTM in order to build a sequence to sequence (seq2seq) model.
  * Review of some key RNN foundations, since a solid understanding of those will be critical to understanding the rest of this lesson.
* A seq2seq model is one where both the input and the output are sequences, and can be of difference lengths.
* Learn an attention mechanism to figure out which words to focus on at each time step.
* Tricks to improve seq2seq results, including teacher forcing and bidirectional models.
* Discuss the DeViSE paper, which shows how we can bridge the divide between text and images, using them both in the same model.

## Lesson Resources

* [Website](http://course.fast.ai/lessons/lesson11.html)
* [Video](https://youtu.be/tY0n9OT5_nA)
* [Wiki](http://forums.fast.ai/t/part-2-lesson-11-wiki)
* Jupyter Notebook and code
  * [translate.ipynb](https://nbviewer.jupyter.org/github/fastai/fastai/blob/master/courses/dl2/translate.ipynb)
  * [devise.ipynb](https://nbviewer.jupyter.org/github/fastai/fastai/blob/master/courses/dl2/devise.ipynb)
* Dataset
  * [Parallel Giga French-English corpus](http://www.statmt.org/wmt15/translation-task.html) / [direct download link](http://www.statmt.org/wmt10/training-giga-fren.tar) (2.3 GB)

## Assignments

### Papers

* Must read
  * [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/abs/1609.08144v2)
  * [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) by Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio - original paper explaining the attention approach explained in class
  * [Grammar as a Foreign Language](https://arxiv.org/abs/1412.7449) by Oriol Vinyals, Geoffrey Hinton, et. al - concise summary of attention in this paper
  * [DeViSE: A Deep Visual-Semantic Embedding Model](http://papers.nips.cc/paper/5204-devise-a-deep-visual-semantic-embedding-model.pdf)
* Additional papers \(optional\)
  * [Papers and SoTA results](https://github.com/RedditSota/state-of-the-art-result-for-machine-learning-problems)
  * [BLEU: a Method for Automatic Evaluation of Machine Translation](https://www.aclweb.org/anthology/P02-1040.pdf) - understanding BiLingual Evaluation Understudy (BLEU) score

### Other Resources

#### Other Useful Information

* [Stephen Merity's talk on Attention and Memory in Deep Learning Networks](https://www.youtube.com/watch?v=uuPZFWJ-4bE&t=1261s9)

### Useful Tools and Libraries

* [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) - a Google DL mini-library with many datasets and tutorials for various seq2seq tasks

### Code Snippets

* Useful function to transform PyTorch `nn.module` class to fastai `Learner` class
  ```python
  rnn = Seq2SeqRNN(fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_90)
  learn = RNN_Learner(md, SingleModel(to_gpu(rnn)), opt_fn=opt_fn)
  ```

## My Notes

### Before getting started:

- [The 1cycle policy](https://sgugger.github.io/the-1cycle-policy.html#the-1cycle-policy) by Sylvain Gugger. Based on Leslie Smith's new paper which takes the previous two key papers (cyclical learning rate and super convergence) and built on them with a number of experiments to show how you can achieve super convergence. Super convergence lets you train models five times faster than the previous stepwise approach (and faster than CLR, although it is less than five times). Super convergence lets you get up to massively high learning rates by somewhere between 1 and 3. The interesting thing about super convergence is that you train at those very high learning rates for quite a large percentage of your epochs and during that time, the loss doesn't really improve very much. But the trick is it's doing a lot of searching through the space to find really generalizable areas it seems. Sylvain implemented it in fastai by flushing out the pieces that were missing then confirmed that he actually achieved super convergence on training on CIFAR10. It is currently called `use_clr_beta` but will be renamed in future. He also added cyclical momentum to fastai library.
- [How To Create Data Products That Are Magical Using Sequence-to-Sequence Models](https://towardsdatascience.com/how-to-create-data-products-that-are-magical-using-sequence-to-sequence-models-703f86a231f8) by Hamel Husain. He blogged about training a model to summarize GitHub issues. Here is the [demo](http://gh-demo.kubeflow.org/) Kubeflow team created based on his blog.

### Neural Machine Translation [[00:05:36](https://youtu.be/tY0n9OT5_nA?t=5m36s)]

Let's build a sequence-to-sequence model! We are going to be working on machine translation. Machine translation is something that's been around for a long time, but we are going to look at an approach called neural translation which uses neural networks for translation. Neural machine translation appeared a couple years ago and it was not as good as the statistical machine translation approaches that use classic feature engineering and standard NLP approaches like stemming, fiddling around with word frequencies, n-grams, etc. By a year later, it was better than everything else. It is based on a metric called BLEU — we are not going to discuss the metric because it is not a very good metric and it is not interesting, but everybody uses it.

![Progress in Machine Translation](/images/translate_notebook_001.png)

We are seeing machine translation starting down the path that we saw starting computer vision object classification in 2012 which just surpassed the state-of-the-art and now zipping past it at great rate. It is unlikely that anybody watching this is actually going to build a machine translation model because https://translate.google.com/ works quite well. So **why are we learning about machine translation**? The reason we are learning about machine translation is that the general idea of taking some kind of input like a sentence in French and transforming it into some other kind of output with arbitrary length such as a sentence in English is **a really useful thing to do**. For example, as we just saw, Hamel took GitHub issues and turn them into summaries. Another example is taking videos and turning them into descriptions, or basically anything where you are spitting out an arbitrary sized output which is very often a sentence. Maybe taking a CT scan and spitting out a radiology report — this is where you can use sequence to sequence learning.

#### Four big wins of Neural Machine Translation [[00:08:36](https://youtu.be/tY0n9OT5_nA?t=8m36s)]

![Four big wins of Neural MT](/images/translate_notebook_002.png)

- End-to-end training: No fussing around with heuristics and hacky feature engineering.
- We are able to build these distributed representations which are shared by lots of concepts within a single network.
- We are able to use long term state in the RNN so it uses a lot more context than n-gram type approaches.
- In the end, text we are generating uses RNN as well so we can build something that is more fluid.

#### BiLSTMs(+Attention) not just for neural MT [[00:09:20](https://youtu.be/tY0n9OT5_nA?t=9m20s)]

![](/images/translate_notebook_003.png)

We are going to use bi-directional GRU (basically the same as LSTM) with attention — these general ideas can also be used for lots of other things as you see above.

#### Let's jump into the code [[00:09:47](https://youtu.be/tY0n9OT5_nA?t=9m47s)]

[translate.ipynb](https://nbviewer.jupyter.org/github/fastai/fastai/blob/master/courses/dl2/translate.ipynb)

We are going to try to translate French into English by following the standard neural network approach:

1. Data
2. Architecture
3. Loss Function

#### 1. Data

As usual, we need `(x, y)` pair. In this case, x: French sentence, y: English sentence which you will compare your prediction against. We need lots of these tuples of French sentences with their equivalent English sentence — that is called "parallel corpus" and harder to find than a corpus for a language model. For a language model, we just need text in some language. For any living language, there will be a few gigabytes at least of text floating around the Internet for you to grab. For translation, there are some pretty good parallel corpus available for European languages. The European Parliament has every sentence in every European language. Anything that goes to the UN is translated to lots of languages. For French to English, we have particularly nice thing which is pretty much any semi official Canadian website will have a French version and an English version[[00:12:13](https://youtu.be/tY0n9OT5_nA?t=12m13s)].

