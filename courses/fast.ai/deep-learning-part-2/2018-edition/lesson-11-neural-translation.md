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

#### Translation files

```python
from fastai.text import *
```

French-English parallel texts from http://www.statmt.org/wmt15/translation-task.html . It was created by Chris Callison-Burch, who crawled millions of web pages and then used a set of simple heuristics to transform French URLs onto English URLs (i.e. replacing "fr" with "en" and about 40 other hand-written rules), and assume that these documents are translations of each other.

```python
PATH = Path('data/translate')
TMP_PATH = PATH / 'tmp'
TMP_PATH.mkdir(exist_ok=True)
fname = 'giga-fren.release2.fixed'
en_fname = PATH / f'{fname}.en'
fr_fname = PATH / f'{fname}.fr'
```

For bounding boxes, all of the interesting stuff was in the loss function, but for neural translation, all of the interesting stuff is going to be in the architecture [00:13:01]. Let's zip through this pretty quickly and one of the things Jeremy wants you to think about particularly is what are the relationships or the similarities in terms of the tasks we are doing and how we do it between language modeling vs. neural translation.

![We're going to look at Neural Translation, using a form of seq2seq learning](/images/translate_notebook_004.png)

The first step is to do the exact same thing we do in a language model which is to take a sentence and chuck it through an RNN[00:13:35].

![We already saw the idea of concat pooling in part 1](/images/translate_notebook_005.png)

Now with the classification model, we had a decoder which took the RNN output and grabbed three things: `maxpool` and `meanpool` over all of the time steps, and the value of the RNN at the last time step, stack all those together and put it through a linear layer [00:14:24]. Most people do not do that and just use the last time step, so all the things we will be talking about today uses the last time step.

We start out by chucking the input sentence through an RNN and out of it comes some "hidden state" (i.e. some vector that represents the output of an RNN that has encoded the sentence).

#### Encoder ≈ Backbone [[00:15:18](https://youtu.be/tY0n9OT5_nA?t=15m18s)]

Stephen used the word "encoder", but we tend to use the word "backbone". Like when we talked about adding a custom head to an existing model, the existing pre-trained ImageNet model, for example, we say that is our backbone and then we stick on top of it some head that does the task we want. In sequence to sequence learning, they use the word encoder, but it basically is the same thing — it is some piece of a neural network architecture that takes the input and turns it into some representation which we can then stick a few more layers on top to grab something out of it such as we did for the classifier where we stack a linear layer on top of it to turn int into a sentiment. This time though, we have something that's a little bit harder than just creating sentiment [00:16:12]. Instead of turning the hidden state into a positive or negative sentiment, we want to turn it into a sequence of tokens where that sequence of token is the German sentence in Stephen's example.

This is sounding more like the language model than the classifier because the language had multiple tokens (for every input word, there was an output word). But the language model was also much easier because the number of tokens in the language model output was the same length as the number of tokens in the language model input. Not only they were the same length, but they exactly matched up (e.g. after word one comes word two, after word two comes word three, and so forth). For translating language, you don't necessarily know that the word "he" will be translated as the first word in the output (unfortunately, it is in this particular case). Very often, the subject object order will be different or there will be some extra words inserted, or some pronouns we will need to add some gendered article, etc. This is the key issue we are going to have to deal with is the fact that we have an arbitrary length output where the tokens in the output do not correspond to the same order or the specific tokens in the input [00:17:31]. But the general idea is the same. Here is an RNN to encode the input, turns it into some hidden state, then this is the new thing we are going to learn is generating a sequence output.

#### Sequence output [[00:17:47](https://youtu.be/tY0n9OT5_nA?t=17m47s)]

We already know:

- Sequence to class (IMDB classifier)
- Sequence to equal length sequence (Language model)

But we do not know yet how to do a general purpose sequence to sequence, so that's the new thing today. Very little of this will make sense unless you really understand lesson 6 how an RNN works.

#### Quick review of [Lesson 6](https://medium.com/@hiromi_suenaga/deep-learning-2-part-1-lesson-6-de70d626976c) [[00:18:20](https://youtu.be/tY0n9OT5_nA?t=18m20s)]

We learnt that an RNN at its heart is a standard fully connected network. Below is one with 4 layers — takes an input and puts it through four layers, but at the second layer, it concatenates in the second input, third layer concatenated in the third input, but we actually wrote this in Python as just a four layer neural network. There was nothing else we used other than linear layers and ReLUs. We used the same weight matrix every time when an input came in, we used the same matrix every time when we went from one of the hidden states to the next — that is why these arrows are the same color.

![](/images/translate_notebook_006.png)

We can redraw the above diagram like the below [00:19:29].

![](/images/translate_notebook_007.png)

Not only did we redraw it but we took the four lines of linear code in PyTorch and we replaced it with a for loop. Remember, we had something that did exactly the same thing as below, but it just had four lines of code saying `self.l_in(input)` and we replaced it with a for loop because that's nice to refactor. The refactoring which does not change any of the math, any of the ideas, or any of the outputs is an RNN. It's turning a bunch of separate lines in the code into a Python for loop.

![](/images/translate_notebook_008.png)

We could take the output so that it is not outside the loop and put it inside the loop [00:20:25]. If we do that, we are now going to generate a separate output for every input. The code above, the hidden state gets replaced each time and we end up just spitting out the final hidden state. But if instead, we had something that said `hs.append(h)` and returned `hs` at the end, that would be the picture below.

![](/images/translate_notebook_009.png)

The main thing to remember is when we say hidden state, we are referring to a vector — technically a vector for each thing in the mini-batch so it's a matrix, but generally when Jeremy speaks about these things, he ignores the mini-batch piece and treat it for just a single item.

![](/images/translate_notebook_010.png)

We also learned that you can stack these layers on top of each other [00:21:41]. So rather than the left RNN (in the diagram above) spitting out output, they could just spit out inputs into a second RNN. If you are thinking at this point "I think I understand this but I am not quite sure" that means you don't understand this. The only way you know that you actually understand it is to go and write this from scratch in PyTorch or Numpy. If you can't do that, then you know you don't understand it and you can go back and re-watch lesson 6 and check out the notebook and copy some of the ideas until you can. It is really important that you can write that from scratch — it's less than a screen of code. So you want to make sure you can create a 2 layer RNN. Below is what it looks like if you unroll it.

![](/images/translate_notebook_011.png)

To get to a point that we have (x, y) pairs of sentences, we will start by downloading the dataset [00:22:39]. Training a translation model takes a long time. Google's translation model has eight layers of RNN stacked on top of each other. There is no conceptual difference between eight layers and two layers. If you are Google and you have more GPUs or TPUs than you know what to do with, then you are fine doing that. Where else, in our case, it's pretty likely that the kind of sequence to sequence models we are building are not going to require that level of computation. So to keep things simple [00:23:22], let's do a cut-down thing where rather than learning how to translate French into English for any sentence, let's learn to translate French questions into English questions — specifically questions that start with what/where/which/when. So here is a regex which looks for things that start with "wh" and end with a question mark.

```python
# Question regex search filters
re_eq = re.compile('^(Wh[^?.!]+\?)')
re_fq = re.compile('^([^?.!]+\?)')

# grabbing lines from the English and French source texts
lines = ( (re_eq.search(eq), re_fq.search(fq))
         for eq, fq in zip(open(en_fname, encoding='utf-8'), open(fr_fname, encoding='utf-8')))

# isolate the questions
qs = [(e.group(), f.group()) for e, f in lines if e and f]
```

We go through the corpus [00:23:43], open up each of the two files, each line is one parallel text, zip them together, grab the English question and the French question, and check whether they match the regular expressions.

```python
# save the questions for later
pickle.dump(qs, (PATH / 'fr-en-qs.pkl').open('wb'))

# load in pickled questions
qs = pickle.load((PATH / 'fr-en-qs.pkl').open('rb'))
```

Dump that out as a pickle so we don't have to do it again and so now we have 52,000 sentence pairs and here are some examples:

```python
print(len(qs))
print(qs[:5])

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
52331
[('What is light ?', 'Qu’est-ce que la lumière?'), ('Who are we?', 'Où sommes-nous?'), ('Where did we come from?', "D'où venons-nous?"), ('What would we do without it?', 'Que ferions-nous sans elle ?'), ('What is the absolute location (latitude and longitude) of Badger, Newfoundland and Labrador?', 'Quelle sont les coordonnées (latitude et longitude) de Badger, à Terre-Neuve-etLabrador?')]
```

One nice thing about this is that what/who/where type questions tend to be fairly short [00:24:08]. But the idea that we could learn from scratch with no previous understanding of the idea of language let alone of English or French that we could create something that can translate one to the other for any arbitrary question with only 50k sentences sounds like a ludicrously difficult thing to ask this to do. So it would be impressive if we can make any progress what so ever. This is very little data to do a very complex exercise.

`qs` contains the tuples of French and English [00:24:48]. You can use this handy idiom to split them apart into a list of English questions and a list of French questions.

```python
en_qs, fr_qs = zip(*qs)
```

Then we tokenize the English questions and tokenize the French questions. So remember that just means splitting them up into separate words or word-like things. By default [00:25:11], the tokenizer that we have here (remember this is a wrapper around the spaCy tokenizer which is a fantastic tokenizer) assumes English. So to ask for French, you just add an extra parameter `'fr'`. The first time you do this, you will get an error saying you don’t have the spaCy French model installed so you can run `python -m spacy download fr` to grab the French model.

```python
en_tok = Tokenizer.proc_all_mp(partition_by_cores(en_qs))
fr_tok = Tokenizer.proc_all_mp(partition_by_cores(fr_qs), 'fr')
```

It is unlikely that any of you are going to have RAM problems here because this is not particularly big corpus but some of the students were trying to train a new language models during the week and were having RAM problems. If you do, it’s worth knowing what these functions (`proc_all_mp`) are actually doing. `proc_all_mp` is processing every sentence across multiple processes [00:25:59]:

![](/images/translate_notebook_012.png)

The function above finds out how many CPUs you have, divide it by two (because normally with hyper-threading they don’t actually all work in parallel), then in parallel run this `proc_all` function. So that is going to spit out a whole separate Python processes for every CPU you have. If you have a lot of cores, that is a lot of Python processes — everyone is going to load all this data in and that can potentially use up all your RAM. So you could replace that with just `proc_all` rather than `proc_all_mp` to use less RAM. Or you could just use less cores. At the moment, we are calling `partition_by_cores` which calls `partition` on a list and asks to split it into a number of equal length things according to how many CPUs you have. So you could replace that to split into a smaller list and run it on less things.

![](/images/translate_notebook_013.png)

Having tokenized the English and French, you can see how it gets split up [00:28:04]:

```python
en_tok[:3], fr_tok[:3]

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
([['what', 'is', 'light', '?'],
  ['who', 'are', 'we', '?'],
  ['where', 'did', 'we', 'come', 'from', '?']],
 [['qu’', 'est', '-ce', 'que', 'la', 'lumière', '?'],
  ['où', 'sommes', '-', 'nous', '?'],
  ["d'", 'où', 'venons', '-', 'nous', '?']])
```

You can see the tokenization for French is quite different looking because French loves their apostrophes and their hyphens. So if you try to use an English tokenizer for a French sentence, you’re going to get a pretty crappy outcome. You don’t need to know heaps of NLP ideas to use deep learning for NLP, but just some basic stuff like use the right tokenizer for your language is important [00:28:23]. Some of the students this week in our study group have been trying to build language models for Chinese instance which of course doesn’t really have the concept of a tokenizer in the same way, so we’ve been starting to look at [SentencePiece](https://github.com/google/sentencepiece) which splits things into arbitrary sub-word units and so when Jeremy says tokenize, if you are using a language that doesn’t have spaces in, you should probably be checking out SentencePiece or some other similar sub-word unit thing instead. Hopefully in the next week or two, we will be able to report back with some early results of these experiments with Chinese.

```python
# 90th percentile of English and French sentences length.
np.percentile([len(o) for o in en_tok], 90), np.percentile([len(o) for o in fr_tok], 90)

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
(23.0, 28.0)

# We are keeping tokens that are less than 30 chars. The filter is applied on the English words, and the same tokens are kept for French.
keep = np.array([len(o) < 30 for o in en_tok])

en_tok = np.array(en_tok)[keep]
fr_tok = np.array(fr_tok)[keep]

# save our work
pickle.dump(en_tok, (PATH / 'en_tok.pkl').open('wb'))
pickle.dump(fr_tok, (PATH / 'fr_tok.pkl').open('wb'))

en_tok = pickle.load((PATH / 'en_tok.pkl').open('rb'))
fr_tok = pickle.load((PATH / 'fr_tok.pkl').open('rb'))
```

So having tokenized it [00:29:25], we will save that to disk. Then remember, the next step after we create tokens is to turn them into numbers. To do that, we have two steps — the first is to get a list of all of the words that appear and then we turn every word into the index. If there are more than 40,000 words that appear, then let’s cut it off there so it doesn’t go too crazy. We insert a few extra tokens for beginning of stream (`_bos_`), padding (`_pad_`), end of stream (`_eos_`), and unknown (`_unk`). So if we try to look up something that wasn’t in the 40,000 most common, then we use a `defaultdict` to return 3 which is unknown.

```python
def toks2ids(tok, pre):
    """
    Numericalize words to integers.

    Arguments:
        tok: token
        pre: prefix
    """
    freq = Counter(p for o in tok for p in o)
    itos = [o for o, c in freq.most_common(40000)] # 40k most common words
    itos.insert(0, '_bos_')
    itos.insert(1, '_pad_')
    itos.insert(2, '_eos_')
    itos.insert(3, '_unk')
    stoi = collections.defaultdict(lambda: 3, { v: k for k, v in enumerate(itos) }) #reverse
    ids = np.array([ ([stoi[o] for o in p] + [2]) for p in tok ])
    np.save(TMP_PATH / f'{pre}_ids.npy', ids)
    pickle.dump(itos, open(TMP_PATH / f'{pre}_itos.pkl', 'wb'))
    return ids, itos, stoi
```

Now we can go ahead and turn every token into an ID by putting it through the string to integer dictionary (`stoi`) we just created and then at the end of that let’s add the number 2 which is the end of stream. The code you see here is the code Jeremy writes when he is iterating and experimenting [00:30:25]. Because 99% of the code he writes while iterating and experimenting turns out to be totally wrong or stupid or embarrassing and you don’t get to see it. But there is not point refactoring that and making it beautiful when he’s writing it so he wanted you to see all the little shortcuts he has. Rather than having some constant for `_eos_` marker and using that, when he is prototyping he just does the easy stuff. Not so much that he ends up with broken code but he tries to find some middle ground between beautiful code and code that works.

:question: Just heard him mention that we divide the number of CPUs by 2 because with hyper-threading, we don’t get a speed-up using all the hyper threaded cores. Is this based on practical experience or is there some underlying reason why we wouldn’t get additional speedup [00:31:18]?

Yes, it’s just practical experience and it’s not all things seemed like this, but I definitely noticed with tokenization — hyper-threading seemed to slow things down a little bit. Also if I use all the cores, often I want to do something else at the same time (like running some interactive notebook) and I don’t have any spare room to do that.

Now for our English and French, we can grab a list of IDs `en_ids` [00:32:01]. When we do that, of course, we need to make sure that we also store the vocabulary. There is no point having IDs if we don’t know what a number 5 represents, there is no point having a number 5. So that’s our vocabulary `en_itos` and reverse mapping `en_stoi` that we can use to convert more corpuses in the future.

```python
en_ids, en_itos, en_stoi = toks2ids(en_tok, 'en')
fr_ids, fr_itos, fr_stoi = toks2ids(fr_tok, 'fr')
```

Just to confirm it’s working, we can go through each ID, convert the int to a string, and spit that out — there we have our sentence back now with an end of stream marker at the end. Our English vocab is 17,000 and our French vocab is 25,000, so that’s not too big and not too complex vocab that we are dealing with.

```python
def load_ids(pre):
    ids = np.load(TMP_PATH / f'{pre}_ids.npy')
    itos = pickle.load(open(TMP_PATH / f'{pre}_itos.pkl', 'rb'))
    stoi = collections.defaultdict(lambda: 3, { v: k for k, v in enumerate(itos) })
    return ids, itos, stoi

en_ids, en_itos, en_stoi = load_ids('en')
fr_ids, fr_itos, fr_stoi = load_ids('fr')

# Sanity check
[fr_itos[o] for o in fr_ids[0]], len(en_itos), len(fr_itos)

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
(['qu’', 'est', '-ce', 'que', 'la', 'lumière', '?', '_eos_'], 17573, 24793)
```

#### Word vectors [[00:32:53](https://youtu.be/tY0n9OT5_nA?t=32m53s)]

We spent a lot of time on the forum during the week discussing how pointless word vectors are and how you should stop getting so excited about them — and now we are going to use them. Why? All the stuff we’ve been learning about using language models and pre-trained proper models rather than pre-trained linear single layers which is what word vectors are, applies equally well to sequence to sequence. But Jeremy and Sebastian are starting to look at that. There is a whole thing for anybody interested in creating some genuinely new highly publishable results, the entire area of sequence to sequence with pre-trained language models has not been touched yet. Jeremy believes it is going to be just as good as classifications. If you work on this and you get to the point where you have something that is looking exciting and you want help publishing it, Jeremy is very happy to help co-author papers. So feel free to reach out when you have some interesting results.

At this stage, we do not have any of that, so we are going to use very little fastai [00:34:14]. All we have is word vectors — so let’s at least use decent word vectors. Word2vec is very old word word vectors. There are better word vectors now and fastText is a pretty good source of word vectors. There is hundreds of languages available for them, and your language is likely to be represented.

fastText word vectors available from https://fasttext.cc/docs/en/english-vectors.html

fasttext Python library is not available in PyPI but here is a handy trick [00:35:03]. **If there is a GitHub repo that has a `setup.py` and `requirements.txt` in it, you can just chuck `git+` at the start then stick that in your `pip install` and it works.** Hardly anybody seems to know this and if you go to the fastText repo, they won’t tell you this — they’ll say you have to download it and cd into it and blah but you don’t. You can just run this:

```python
!pip install git+https://github.com/facebookresearch/fastText.git
```

**Download word vectors:**

To use the fastText library, you’ll need to download [fastText word vectors](https://fasttext.cc/docs/en/pretrained-vectors.html) for your language (download the ‘bin plus text’ ones).

We are using the pre-trained word vectors for English and French language, trained on Wikipedia using fastText. These vectors in dimension 300 were obtained using the skip-gram model.

```bash
!aria2c --file-allocation=none -c -x 5 -s 5 -d data/translate https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip

!aria2c --file-allocation=none -c -x 5 -s 5 -d data/translate https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.fr.zip
```

```python
import fastText as ft

# use the fastText library
en_vecs = ft.load_model(str((PATH / 'wiki.en.bin')))
fr_vecs = ft.load_model(str((PATH / 'wiki.fr.bin')))
```

Above are our English and French models. There are a text version and a binary version. The binary version is faster, so we will use that. The text version is also a bit buggy. We are going to convert it into a standard Python dictionary to make it a bit easier to work with [00:35:55]. This is just going through each word with a dictionary comprehension and save it as a pickle dictionary:

```python
def get_vecs(lang, ft_vecs):
    """
    Convert fastText word vectors into a standard Python dictionary to make it a bit easier to work with.
    This is just going through each word with a dictionary comprehension and save it as a pickle dictionary.

    get_word_vector:
        [method] get the vector representation of word.
    get_words:
        [method] get the entire list of words of the dictionary optionally
        including the frequency of the individual words. This
        does not include any subwords.
    """
    vecd = { w: ft_vecs.get_word_vector(w) for w in ft_vecs.get_words() }
    pickle.dump(vecd, open(PATH / f'wiki.{lang}.pkl', 'wb'))
    return vecd

en_vecd = get_vecs('en', en_vecs)
fr_vecd = get_vecs('fr', fr_vecs)

en_vecd = pickle.load(open(PATH / 'wiki.en.pkl', 'rb'))
fr_vecd = pickle.load(open(PATH / 'wiki.fr.pkl', 'rb'))

# Test
ft_vecs = en_vecs
ft_words = ft_vecs.get_words(include_freq=True)
ft_word_dict = { k: v for k, v in zip(*ft_words) }
ft_words = sorted(ft_word_dict.keys(), key=lambda x: ft_word_dict[x])
len(ft_words)

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
2519370
```

Now we have our pickle dictionary, we can go ahead and look up a word, for example, a comma [00:36:07]. That will return a vector. The length of the vector is the dimensionality of this set of word vectors. In this case, we have 300 dimensional English and French word vectors.

```python
dim_en_vec = len(en_vecd[','])
dim_fr_vec = len(fr_vecd[','])
dim_en_vec, dim_fr_vec

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
(300, 300)
```

For reasons you will see in a moment, we also want to find out what the mean and standard deviation of our vectors are. So the mean is about zero and standard deviation is about 0.3.

```python
# en_vecd type is dict
en_vecs = np.stack(list(en_vecd.values())) # convert dict_values to list and then stack it
en_vecs.mean(), en_vecs.std()

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
(0.0075652334, 0.29283327)
```

#### Model data [[00:36:48](https://youtu.be/tY0n9OT5_nA?t=36m48s)]

Often corpuses have a pretty long tailed distribution of sequence length and it’s the longest sequences that tend to overwhelm how long things take, how much memory is used, etc. So in this case, we are going to grab 99th to 97th percentile of the English and French and truncate them to that amount. Originally Jeremy was using 90 percentiles (hence the variable name):

```python
enlen_90 = int(np.percentile([len(o) for o in en_ids], 99))
frlen_90 = int(np.percentile([len(o) for o in fr_ids], 99))
enlen_90, frlen_90

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
(29, 38)
```

**Create our Dataset, DataLoaders**

We are nearly there [00:37:24]. We’ve got our tokenized, numericalized English and French dataset. We’ve got some word vectors. So now we need to get it ready for PyTorch. PyTorch expects a `Dataset` object and hopefully by now you can say that a Dataset object requires two things — a length (`__len__`)and an indexer (`__getitem__`). Jeremy started out writing `Seq2SeqDataset` which turned out to be just a generic `Dataset` [00:37:52].

```python
en_ids_tr = np.array([o[:enlen_90] for o in en_ids])
fr_ids_tr = np.array([o[:frlen_90] for o in fr_ids])

class Seq2SeqDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __getitem__(self, idx):
        return A(self.x[idx], self.y[idx]) # A for Arrays

    def __len__(self):
        return len(self.x)
```

- `A` : Arrays. It will go through each of the thing you pass it, if it is not already a numpy array, it converts into a numpy array and returns back a tuple of all of the things you passed it which are now guaranteed to be numpy arrays [00:38:32].
- `V` : Variables
- `T` : Tensors

#### Training set and validation set [[00:39:03](https://youtu.be/tY0n9OT5_nA?t=39m3s)]

Now we need to grab our English and French IDs and get a training set and a validation set.

:warning: *One of the things which is pretty disappointing about a lot of code out there on the Internet is that they don’t follow some simple best practices.*

For example, if you go to PyTorch website, they have an example section for sequence to sequence translation. Their example does not have a separate validation set. Jeremy tried training according to their settings and tested it with a validation set and it turned out that it overfit massively. So this is not just a theoretical problem — the actual PyTorch repo has the actual official sequence to sequence translation example which does not check for overfitting and overfits horribly [00:39:41]. Also it fails to use mini-batches so it actually fails to utilize any of the efficiency of PyTorch whatsoever. Even if you find code in the official PyTorch repo, don’t assume it’s any good at all. The other thing you’ll notice is that pretty much every other sequence to sequence model Jeremy found in PyTorch anywhere on the Internet has clearly copied from that crappy PyTorch repo because all has the same variable names, it has the same problems, it has the same mistakes.

Another example is that nearly every PyTorch convolutional neural network Jeremy found does not use an adaptive pooling layer [00:40:27]. So in other words, the final layer is always average pool (7, 7). They assume that the previous layer is 7 by 7 and if you use any other size input, you get an exception, and therefore nearly everybody Jeremy has spoken who uses PyTorch thinks that there is a fundamental limitation of CNNs that they are tied to the input size and that has not been true since VGG. So every time Jeremy grabs a new model and stick it in the fastai repo, he has to go and search for "pool" and add "adaptive" to the start and replace the 7 with a 1 and now it works on any sized object. So just be careful. It’s still early days and believe it or not, even though most of you have only started in the last year your deep learning journey, you know quite a lot more about a lot of the more important practical aspects than the vast majority of people that have publishing and writing stuff in official repos. So you need to have a little more self-confidence than you might expect when it comes to reading other people’s code. If you find yourself thinking "that looks odd", it’s not necessarily you.

If the repo you are looking at doesn’t have a section on it saying here is the test we did where we got the same results as the paper that’s supposed to be implementing, that almost certainly means they haven’t got the same results of the paper they’re implementing, but probably haven’t even checked [00:42:13]. If you run it, definitely won’t get those results because it’s hard to get things right the first time — it takes Jeremy 12 goes. If they haven’t tested it once, it’s almost certainly won’t work.

Here is an easy way to get training and validation sets [00:42:45]. Grab a bunch of random numbers — one for each row of your data, and see if they are bigger than 0.1 or not. That gets you a list of booleans. Index into your array with that list of booleans to grab a training set, index into that array with the opposite of that list of booleans to get your validation set.

```python
np.random.seed(42)
trn_keep = np.random.rand(len(en_ids_tr)) > 0.1
en_trn, fr_trn = en_ids_tr[trn_keep], fr_ids_tr[trn_keep] # training set
en_val, fr_val = en_ids_tr[~trn_keep], fr_ids_tr[~trn_keep] # validation set
len(en_trn), len(en_val)

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
(45219, 5041)
```

Now we can create our dataset with our X’s and Y’s (i.e. French and English)[00:43:12]. If you want to translate instead English to French, switch these two around and you’re done.

```python
trn_ds = Seq2SeqDataset(fr_trn, en_trn)
val_ds = Seq2SeqDataset(fr_val, en_val)
```

Now we need to create DataLoaders [00:43:22]. We can just grab our data loader and pass in our dataset and batch size. We actually have to transpose the arrays — we won’t go into the details about why, but we can talk about it during the week if you’re interested but have a think about why we might need to transpose their orientation. Since we’ve already done all the pre-processing, there is no point spawning off multiple workers to do augmentation, etc because there is no work to do. So making `num_workers = 1` will save you some time. We have to tell it what our padding index is — that is pretty important because what’s going to happen is that we’ve got different length sentences and fastai will automatically stick them together and pad the shorter ones so that they are all equal length. Remember a tensor has to be rectangular.

```python
# Set batch size
bs = 125

# arranges sentences so that similar lengths are close to each other
trn_samp = SortishSampler(en_trn, key=lambda x: len(en_trn[x]), bs=bs)
val_samp = SortSampler(en_val, key=lambda x: len(en_val[x]))

# Create DataLoaders
trn_dl = DataLoader(trn_ds, bs, transpose=True, transpose_y=True, num_workers=1,
                    pad_idx=1, pre_pad=False, sampler=trn_samp)
val_dl = DataLoader(val_ds, int(bs * 1.6), transpose=True, transpose_y=True, num_workers=1,
                    pad_idx=1, pre_pad=False, sampler=val_samp)
md = ModelData(PATH, trn_dl, val_dl)
```

In the decoder in particular, we want our padding to be at the end, not at the start [00:44:29]:

- Classifier → padding in the beginning. Because we want that final token to represent the last word of the movie review.
- Decoder → padding at the end. As you will see, it actually is going to work out a bit better to have the padding at the end.

**Sampler** [00:44:54] Finally, since we’ve got sentences of different lengths coming in and they all have to be put together in a mini-batch to be the same size by padding, we would much prefer that the sentences in a mini-batch are of similar sizes already. Otherwise it is going to be as long as the longest sentence and that is going to end up wasting time and memory. Therefore, we are going to use the sampler tricks that we learnt last time which is the validation set, we are going to ask it to sort everything by length first. Then for the training set, we are going to randomize the order of things but to roughly make it so that things of similar length are about in the same spot.

**Model Data** [00:45:40] At this point, we can create a model data object — remember a model data object really does one thing which is it says "I have a training set and a validation set, and an optional test set" and sticks them into a single object. We also has a path so that it has somewhere to store temporary files, models, stuff like that.

We are not using fastai for very much at all in this example. We used PyTorch compatible Dataset and and DataLoader — behind the scene it is actually using the fastai version because we need it to do the automatic padding for convenience, so there is a few tweaks in fastai version that are a bit faster and a bit more convenient. We are also using fastai’s Samplers, but there is not too much going on here.

#### Architecture [[00:46:59](https://youtu.be/tY0n9OT5_nA?t=46m59s)]

![](/images/translate_notebook_014.png)

- The architecture is going to take our sequence of tokens.
- It is going to spit them into an encoder (a.k.a. backbone).
- That is going to spit out the final hidden state which for each sentence, it’s just a single vector.

None of this is going to be new [00:47:41]. That is all going to be using very direct simple techniques that we’ve already learned.

- Then we are going to take that, and we will spit it into a different RNN which is a decoder. That’s going to have some new stuff because we need something that can go through one word at a time. And it keeps going until it thinks it’s finished the sentence. It doesn’t know how long the sentence is going to be ahead of time. It keeps going until it thinks it’s finished the sentence and then it stops and returns a sentence.

```python
def create_emb(vecs, itos, em_sz):
    """
    Creates embedding:
    1. rows = number of vocab
    2. cols = embedding size dimension

    Will randomly initialize the embedding
    """
    emb = nn.Embedding(len(itos), em_sz, padding_idx=1)
    wgts = emb.weight.data
    miss = []

    # goes through the embedding and replace
    # the initialized weights with existing word vectors
    # multiply x3 to compensate for the stdev 0.3
    for i, w in enumerate(itos):
        try:
            wgts[i] = torch.from_numpy(vecs[w] * 3)
        except:
            miss.append(w)
    print(len(miss), miss[5:10])
    return emb

nh, nl = 256, 2
```

Let’s start with the encoder [00:48:15]. In terms of the variable naming here, there is identical attributes for encoder and decoder. The encoder version has `enc` the decoder version has `dec`.

- `emb_enc`: Embeddings for the encoder
- `gru` : RNN. GRU and LSTM are nearly the same thing.

We need to create an embedding layer because remember — what we are being passed is the index of the words into a vocabulary. And we want to grab their fastText embedding. Then over time, we might want to also fine tune to train that embedding end-to-end.

`create_emb` [00:49:37]: It is important that you know now how to set the rows and columns for your embedding so the number of rows has to be equal to your vocabulary size — so each vocabulary has a word vector. The size of the embedding is determined by fastText and fastText embeddings are size 300. So we have to use size 300 as well otherwise we can’t start out by using their embeddings.

`nn.Embedding` will initially going to give us a random set of embeddings [00:50:12]. So we will go through each one of these and if we find it in fastText, we will replace it with the fastText embedding. Again, something you should already know is that (`emb.weight.data`):

- A PyTorch module that is learnable has `weight` attribute
- `weight` attribute is a `Variable` that has `data` attribute
- The `data` attribute is a tensor

Now that we’ve got our weight tensor, we can just go through our vocabulary and we can look up the word in our pre-trained vectors and if we find it, we will replace the random weights with that pre-trained vector [00:52:35]. The random weights have a standard deviation of 1. Our pre-trained vectors has a standard deviation of about 0.3. So again, this is the kind of hacky thing Jeremy does when he is prototyping stuff, he just multiplied it by 3. By the time you see the video of this, we may able to put all this sequence to sequence stuff into the fastai library, you won’t find horrible hacks like that in there (sure hope). But hack away when you are prototyping. Some things won’t be in fastText in which case, we’ll just keep track of it [00:53:22]. The print statement is there so that we can see what’s going on (i.e. why are we missing stuff?). Remember we had about 30,000 so we are not missing too many.

```python
3097 ['l’', "d'", 't_up', 'd’', "qu'"]
1285 ["'s", '’s', "n't", 'n’t', ':']
```

Jeremy has started doing some stuff around incorporating large vocabulary handling into fastai — it’s not finished yet but hopefully by the time we get here, this kind of stuff will be possible [00:56:50].

```python
class Seq2SeqRNN(nn.Module):
    def __init__(self, vecs_enc, itos_enc, em_sz_enc, vecs_dec, itos_dec, em_sz_dec, nh, out_sl, nl=2):
        super().__init__()

        # encoder (enc)
        self.nl, self.nh, self.out_sl = nl, nh, out_sl

        # for each word, pull up the 300M vector and create an embedding
        self.emb_enc = create_emb(vecs_enc, itos_enc, em_sz_enc)
        self.emb_enc_drop = nn.Dropout(0.15)

        # GRU - similiar to LSTM
        self.gru_enc = nn.GRU(em_sz_enc, nh, num_layers=nl, dropout=0.25)
        self.out_enc = nn.Linear(nh, em_sz_dec, bias=False)

        # decoder (dec)
        self.emb_dec = create_emb(vecs_dec, itos_dec, em_sz_dec)
        self.gru_dec = nn.GRU(em_sz_dec, em_sz_dec, num_layers=nl, dropout=0.1)
        self.out_drop = nn.Dropout(0.35)
        self.out = nn.Linear(em_sz_dec, len(itos_dec))
        self.out.weight.data = self.emb_dec.weight.data

    def forward(self, inp):
        sl, bs = inp.size()

        # ==================================================
        # Encoder version
        # ==================================================

        # initialize the hidden layer
        h = self.initHidden(bs)

        # run the input through our embeddings + apply dropout
        emb = self.emb_enc_drop(self.emb_enc(inp))

        # run it through the RNN layer
        enc_out, h = self.gru_enc(emb, h)

        # run the hidden state through our linear layer
        h = self.out_enc(h)

        # ==================================================
        # Decoder version
        # ==================================================

        # starting with a 0 (or beginning of string _BOS_)
        dec_inp = V(torch.zeros(bs).long())
        res = []

        # will loop as long as the longest english sentence
        for i in range(self.out_sl):

            # embedding - we are only looking at a section at time
            # which is why the .unsqueeze is required
            emb = self.emb_dec(dec_inp).unsqueeze(0)

            # rnn - typically works with whole phrases, but we passing
            # only 1 unit at a time in a loop
            outp, h = self.gru_dec(emb, h)

            # dropout
            outp = self.out(self.out_drop(outp[0]))

            res.append(outp)

            # highest probability word
            dec_inp = V(outp.data.max(1)[1])

            # if its padding ,we are at the end of the sentence
            if (dec_inp == 1).all():
                break

        # stack the output into a single tensor
        return torch.stack(res)

    def initHidden(self, bs):
        return V(torch.zeros(self.nl, bs, self.nh))
```

The key thing to know is that encoder takes our inputs and spits out a hidden vector that hopefully will learn to contain all of the information about what that sentence says and how it sets it [00:58:49]. If it can’t do that, we can’t feed it into a decoder and hope it to spit our our sentence in a different language. So that’s what we want it to learn to do. We are not going to do anything special to make it learn to do that — we are just going to do the three things (data, architecture, loss function) and cross our fingers.

**Decoder** [00:59:58]: How do we now do the new bit? The basic idea of the new bit is the same. We are going to do exactly the same thing, but we are going to write our own for loop. The for loop is going to do exactly what the for loop inside PyTorch does for encoder, but we are going to do it manually. How big is the for loop? It’s an output sequence length (`out_sl`) which was something passed to the constructor which is equal to the length of the largest English sentence. Since we are translating into English, so it can’t possibly be longer than that at least in this corpus. If we then used it on some different corpus that was longer, this is going to fail — you could always pass in a different parameter, of course. So the basic idea is the same [01:01:06].

- We are going to go through and put it through the embedding.
- We are going to stick it through the RNN, dropout, and a linear layer.
- We will then append the output to a list which will be stacked into a single tensor and get returned.

Normally, a recurrent neural network works on a whole sequence at a time, but we have a for loop to go through each part of the sequence separately [01:01:37]. We have to add a leading unit axis to the start (`.unsqueeze(0)`) to basicaly say this is a sequence of length one. We are not really taking advantage of the recurrent net much at all — we could easily re-write this with a linear layer.

One thing to be aware of is `dec_inp` [01:02:34]: What is the input to the embedding? The answer is it is the previous word that we translated. The basic idea is if you are trying to translate the 4th word of the new sentence but you don’t know what the third word you just said was, that is going to be really hard. So we are going to feed that in at each time step. What was the previous word at the start? There was none. Specifically, we are going to start out with a beginning of stream token (`_bos_`) which is zero.

`outp` [01:05:24]: it is a tensor whose length is equal to the number of words in our English vocabulary and it contains the probability for every one of those words that it is that word.

`outp.data.max` : it looks in its tensor to find out which word has the highest probability. `max` in PyTorch returns two things: the first thing is what is that max probability and the second is what is the index into the array of that max probability. So we want that second item which is the word index with the largest thing.

`dec_inp` : it contains the word index into the vocabulary of the word. If it’s one (i.e. padding), that means we are done — we reached the end because we finished with a bunch of padding. If it’s not one, let’s go back and continue.

Each time, we appended our outputs (not the word but the probabilities) to the list [01:06:48] which we stack up into a tensor and we can now go ahead and feed that to a loss function.

#### Loss function [[01:07:13](https://youtu.be/tY0n9OT5_nA?t=1h7m13s)]

The loss function is categorical cross entropy loss. We have a list of probabilities for each of our classes where the classes are all the words in our English vocab and we have a target which is the correct class (i.e. which is the correct word at this location). There are two tweaks which is why we need to write our own loss function but you can see basically it is going to be cross entropy loss.

```python
def seq2seq_loss(input, target):
    """
    Loss function - modified version of cross entropy
    """
    sl, bs = target.size()
    sl_in, bs_in, nc = input.size()

    # sequence length could be shorter than the original
    # need to add padding to even out the size
    if sl > sl_in:
        input = F.pad(input, (0, 0, 0, 0, 0, sl - sl_in))
    input = input[:sl]
    return F.cross_entropy(input.view(-1, nc), target.view(-1))#, ignore_index=1)
```

Tweaks [01:07:40]:

1. If the generated sequence length is shorter than the sequence length of the target, we need to add some padding. PyTorch padding function requires a tuple of 6 to pad a rank 3 tensor (sequence length, batch size, by number of words in the vocab). Each pair represents padding before and after that dimension.
2. `F.cross_entropy` expects a rank 2 tensor, but we have sequence length by batch size, so let’s just flatten out. That is what `view(-1, ...)` does.

```python
opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
```

The difference between `.cuda()` and `to_gpu()` : `to_gpu` will not put to the GPU if you do not have one. You can also set `fastai.core.USE_GPU` to `false` to force it to not use GPU that can be handy for debugging.

```python
rnn = Seq2SeqRNN(fr_vecd, fr_itos, dim_fr_vec, en_vecd, en_itos, dim_en_vec, nh, enlen_90)
learn = RNN_Learner(md, SingleModel(to_gpu(rnn)), opt_fn=opt_fn)
learn.crit = seq2seq_loss

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
3097 ['l’', "d'", 't_up', 'd’', "qu'"]
1285 ["'s", '’s', "n't", 'n’t', ':']
```

We then need something that tells it how to handle learning rate groups so there is a thing called `SingleModel` that you can pass it to which treats the whole thing as a single learning rate group [01:09:40]. So this is the easiest way to turn a PyTorch module into a fastai model.

![](/images/translate_notebook_015.png)

We could just call `Learner` to turn that into a learner, but if we call `RNN_Learner`, it does add in `save_encoder` and `load_encoder` that can be handy sometimes. In this case, we really could have said `Leaner` but `RNN_Learner` also works.

```python
# Find the learning rate
learn.lr_find()
learn.sched.plot()
```

![](/images/translate_notebook_016.png)

```python
# Fit the model (15-20 mins to train)
lr = 3e-3
learn.fit(lr, 1, cycle_len=12, use_clr=(20, 10))

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
epoch      trn_loss   val_loss
    0      5.209042   5.980303
    1      4.513244   4.566818
    2      4.056711   4.515142
    3      3.775803   4.026515
    4      3.595237   3.857968
    5      3.519258   3.773164
    6      3.160189   3.705156
    7      3.108818   3.66531
    8      3.142783   3.613333
    9      3.192778   3.680305
    10     2.844773   3.637095
    11     2.857365   3.5963
[array([3.5963])]

learn.save('initial')
learn.load('initial')
```

![Training loss plot](/images/translate_notebook_017.png)

It took me ~2 minutes (109.80s) to train 1 epoch on K80, roughly 3.35 iteration/s.
The full training took me ~20 minutes.

#### Test [[01:11:01](https://youtu.be/tY0n9OT5_nA?t=1h11m1s)]

Remember the model attribute of a learner is a standard PyTorch model so we can pass some x which we can grab out of our validation set or you could `learn.predict_array` or whatever you like to get some predictions. Then we convert those predictions into words by going `.max()[1]` to grab the index of the highest probability words to get some predictions. Then we can go through a few examples and print out the French, the correct English, and the predicted English for things that are not padding.

```python
x, y = next(iter(val_dl))
probs = learn.model(V(x))
preds = to_np(probs.max(2)[1])

for i in range(180, 190):
    print(' '.join([ fr_itos[o] for o in x[:, i] if o != 1 ]))
    print(' '.join([ en_itos[o] for o in y[:, i] if o != 1 ]))
    print(' '.join([ en_itos[o] for o in preds[:, i] if o != 1 ]))
    print()

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
quelles composantes des différents aspects de la performance devraient être mesurées , quelles données pertinentes recueillir et comment ? _eos_
which components within various performance areas should be measured , whatkinds of data are appropriate to collect , and how should this be done ? _eos_
what aspects of the and and be be be be be be be be be ? ? _eos_

le premier ministre doit - il nommer un ministre d’ état à la santé mentale , à la maladie mentale et à la toxicomanie ? _eos_
what role can the federal government play to ensure that individuals with mental illness and addiction have access to the drug therapy they need ? _eos_
what minister the minister minister minister minister minister , , , , health health and health ? ? ? ? _eos_

quelles sont les conséquences de la hausse des formes d’ emploi non conformes aux normes chez les travailleurs hautement qualifiés et chez ceux qui occupent des emplois plus marginaux ? _eos_
what is the impact of growing forms of non - standard employment for highly skilled workers and for those employed in more marginal occupations ? _eos_
what are the consequences of workers workers workers workers workers workers and and workers and workers workers workers workers workers workers ? ? ? _eos_ _eos_

que se produit - il si le gestionnaire n’ est pas en mesure de donner à l’ employé nommé pour une période déterminée un préavis de cessation d’ emploi d’ un mois ou s’ il néglige de le
what happens if the manager is unable to or neglects to give a term employee the one - month notice of non - renewal ? _eos_
what happens the the employee employee employee employee employee the the the the the or or the the the ? ? _eos_

quelles personnes , communautés ou entités sont considérées comme potentiels i ) bénéficiaires de la protection et ii ) titulaires de droits ? _eos_
which persons , communities or entities are identified as potential ( i ) beneficiaries of protection and / or ( ii ) rights holders ? _eos_
who , , , , , or or or or or or or or protection ? ? ? ? _eos_

quelles conditions particulières doivent être remplies pendant l’ examen préliminaire international en ce qui concerne les listages des séquences de nucléotides ou d’ acides aminés ou les tableaux y relatifs ? _eos_
what special requirements apply during the international preliminary examination to nucleotide and / or amino acid sequence listings and / or tables related thereto ? _eos_
what specific must be be be be sequence sequence or or or or sequence or or sequence or sequence or sequence in in ? ? ? ? _eos_ _eos_

pourquoi cette soudaine réticence à promouvoir l’ égalité des genres et à protéger les femmes de ce que , dans la plupart des cas , on peut qualifier de violations grossières des droits humains ? _eos_
why this sudden reluctance to effectively promote gender equality and protect women from what are – in many cases – egregious human rights violations ? _eos_
why is the so for such of of of of of and rights and rights rights of rights rights ? ? ? ? _eos_ _eos_

pouvez - vous dire comment votre bagage culturel vous a aidée à aborder votre nouvelle vie au canada ( à vous adapter au mode de vie canadien ) ? _eos_
what are some things from your cultural background that have helped you navigate canadian life ( helped you adjust to life in canada ) ? _eos_
what are you new to to to to to to to to life life life life ? ? ? ? _eos_ _eos_

selon vous , quels seront , dans les dix prochaines années , les cinq enjeux les plus urgents en matière d' environnement et d' avenir viable pour vous et votre région ? _eos_
which do you think will be the five most pressing environmental and sustainability issues for you and your region in the next ten years ? _eos_
what do you see the next priorities priorities next the next the and and in in in in in in ? ? ? ? ? _eos_ _eos_

dans quelle mesure l’ expert est-il motivé et capable de partager ses connaissances , et dans quelle mesure son successeur est-il motivé et capable de recevoir ce savoir ? _eos_
what is the expert ’s level of motivation and capability for sharing knowledge , and the successor ’s motivation and capability of acquiring it ? _eos_
what is the nature and and and and and and and and and and and and and to to to ? ? ? ? _eos_ _eos_ _eos_
```

Amazingly enough, this kind of simplest possible written largely from scratch PyTorch module on only fifty thousand sentences is sometimes capable, on validation set, of giving you exactly the right answer. Sometimes the right answer is in slightly different wording, and sometimes sentences that really aren’t grammatically sensible or even have too many question marks. So we are well on the right track. We think you would agree even the simplest possible seq-to-seq trained for a very small number of epochs without any pre-training other than the use of word embeddings is surprisingly good. We are going to improve this later but the message here is even sequence to sequence models you think is simpler than they could possibly work even with less data than you think you could learn from can be surprisingly effective and in certain situations this may be enough for your needs.

:question: Would it help to normalize punctuation (e.g. `’` vs. `'`)? [01:13:10]

The answer to this particular case is probably yes — the difference between curly quotes and straight quotes is really semantic. You do have to be very careful though because it may turn out that people using beautiful curly quotes like using more formal language and they are writing in a different way. So if you are going to do some kind of pre-processing like punctuation normalization, you should definitely check your results with and without because nearly always that kind of pre-processing make things worse even when you’re sure it won’t.

:question: What might be some ways of regularizing these seq2seq models besides dropout and weight decay? [01:14:17]

:bookmark: Let me think about that during the week. AWD-LSTM which we have been relying a lot has dropouts of many different kinds and there is also a kind of a regularization based on activations and on changes. Jeremy has not seen anybody put anything like that amount of work into regularizing sequence to sequence model and there is a huge opportunity for somebody to do like the AWD-LSTM of seq-to-seq which might be as simple as stealing all the ideas from AWD-LSTM and using them directly in seq-to-seq that would be pretty easy to try. There’s been an interesting paper that Stephen Merity added in the last couple weeks where he used an idea which take all of these different AWD-LSTM hyper parameters and train a bunch of different models and then use a random forest to find out the feature importance — which ones actually matter the most and then figure out how to set them. You could totally use this approach to figure out for sequence to sequence regularization approaches which one is the best and optimize them and that would be amazing. But at the moment, we don’t know if there are additional ideas to sequence to sequence regularization beyond what is in that paper for regular language model.

#### Tricks [[01:16:28](https://youtu.be/tY0n9OT5_nA?t=1h16m28s)]

##### Trick #1 : Go bi-directional

For classification, the approach to bi-directional Jeremy suggested to use is take all of your token sequences, spin them around, train a new language model, and train a new classifier. He also mentioned that WikiText pre-trained model if you replace `fwd` with `bwd` in the name, you will get the pre-trained backward model he created for you. Get a set of predictions and then average the predictions just like a normal ensemble. That is how we do bi-dir for that kind of classification. There may be ways to do it end-to-end, but Jeremy hasn’t quite figured them out yet and they are not in fastai yet. So if you figure it out, that’s an interesting line of research. But because we are not doing massive documents where we have to chunk it into separate bits and then pool over them, we can do bi-dir very easily in this case. It is literally as simple as adding `bidirectional = True` to our encoder. People tend not to do bi-directional for the decoder partly because it is kind of considered cheating but maybe it can work in some situations although it might need to be more of an ensemble approach in the decoder because it’s a bit less obvious. But encoder it’s very simple — `bidirectional = True` and we now have a second RNN that is going the opposite direction. The second RNN is visiting each token in the opposing order so when we get to the final hidden state, it is the first (i.e. left most) token . But the hidden state is the same size, so the final result is that we end up with a tensor with an extra axis of length 2. Depending on what library you use, often that will be then combined with the number of layers, so if you have 2 layers and bi-directional — that tensor dimension is now length 4. With PyTorch it depends which bit of the process you are looking at as to whether you get a separate result for each layer and/or for each bidirectional bit. You have to look up the documentation and it will tell you input’s output’s tensor sizes appropriate for the number of layers and whether you have `bidirectional = True`.

In this particular case, you will see all the changes that had to be made [01:19:38]. For example ,when we added `bidirectional = True`, the `Linear` layer now needs number of hidden times 2 (i.e. `nh * 2`) to reflect the fact that we have that second direction in our hidden state. Also in `initHidden` it’s now `self.nl * 2`.

