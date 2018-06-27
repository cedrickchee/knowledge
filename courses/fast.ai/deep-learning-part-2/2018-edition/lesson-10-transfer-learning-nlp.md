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

### Useful Tools and Libraries

* [spaCy](https://spacy.io/) - tokenization

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
  - Not enough people are reading papers! What we are doing in class now is implementing papers, the papers are the real ground truth. And I think you know from talking to people a lot of the reason people aren't reading paper is because a lot of people don't think they are capable of reading papers. They don't think they are the kind of people that read papers, but you are. You are here. We started looking at a paper last week and we read the words that were in English and we largely understood them. If you look at the picture above carefully, you'll realize `SSD_MultiHead.forward` is not doing the same. You might then wonder if this is better. My answer is probably. Because `SSD_MultiHead.forward` was the first thing I tried just to get something out there. Between this and YOLO v3 paper, they are probably much better ways.
  - One thing you'll notice in particular they use a smaller `k` but they have a lot more sets of grids 1x1, 3x3, 5x5, 10x10, 19x19, 38x38 — 8732 per class. A lot more than we had, so that'll be an interesting thing to experiment with.
  - Another thing I noticed is that we had 4x4, 2x2, 1c1 which means there are a lot of overlap — every set fits within every other set. In this case where you've got 1, 3, 5, you don't have that overlap. So it might actually make it easier to learn.

    :memo: There's lots of interesting you can play with.
- Match the equations to the code.
  - :bookmark: Perhaps most important thing I would recommend is to put the code and the equations next to each other. You are either math person or code person. By having them side by side, you will learn a little bit of the other.
  - Learning the math is hard because of the notation might seem hard to look up but there are good resources such as [wikipedia](https://en.wikipedia.org/wiki/List_of_mathematical_symbols).
  - Another thing you should try doing is to re-create things that you see in the papers. Here was the key most important figure 1 from the focal loss paper.
  - Last lesson's code had a minor bug—after fixing it the predicted boxes look a lot better! The way Jeremy was flattening out the convolutional activations did not line up with how he was using them in the loss function, and fixing that made it quite a bit better.

### ([0:14:10](https://youtu.be/h5Tz7gZT9Fo?t=14m10s)) Natural Language Processing (NLP)

#### Where we are going

We have seen in every lesson this idea of taking a pre-trained model, whip off some some stuff on the top, replace it with something new, and get it to do something similar. We've kind of dived in a little bit deeper to that to say with `ConvLearner.pretrained` it had a standard way of sticking stuff on the top which does a particular thing (i.e. classification). Then we learned actually we can stick any PyTorch module we like on the end and have it do anything we like with a `custom_head` and so suddenly you discover there's some really interesting things we can do.

In fact, Yang Lu said "what if we did a different kind of custom head?" and the different custom head was let's take the original pictures, rotate them, and the make our dependent variable the opposite of that rotation and see if it can learn to un-rotate it. This is **a super useful thing**, in fact, I think Google Photos nowadays has this option that it'll actually automatically rotate your photos for you. But the cool thing is, as he showed here, you can build that network right now by doing exactly the same as our previous lesson. But your custom head is one that spits out a single number which is how much to rotate by, and your dataset has a dependent variable which is how much you rotated by.
[(Fun with lesson8) Rotation adjustment – things you can do without annotated dataset!](http://forums.fast.ai/t/fun-with-lesson8-rotation-adjustment-things-you-can-do-without-annotated-dataset/14261/1)

So you suddenly realize with this idea of a backbone plus a custom head, you can do almost anything you can think about [[00:16:30](https://youtu.be/h5Tz7gZT9Fo?t=16m30s)].

- Today, we are going to look at the same idea and see how that applies to NLP.
- In the next lesson, we are going to go further and say if NLP and computer vision lets you do the same basic ideas, how do we combine the two. We are going to learn about a model that can actually learn to find word structures from images, images from word structures, or images from images. That will form the basis if you wanted to go further of doing things like going from an image to a sentence (i.e. image captioning) or going from a sentence to an image which we kind of started to do, a phrase to image.
- From there, we've got to go deeper then into computer vision to think what other kinds of things we can do with this idea of pre-trained network plus a custom head. So we will look at various kinds of image enhancement like increasing the resolution of a low-res photo to guess what was missing or adding artistic filters on top of photos, or changing photos of horses into photos of zebras, etc.
- Then finally that's going to bring us all the way back to bounding boxes again. To get there, we're going to first of all learn about segmentation which is not just figuring out where a bounding box is, but figuring out what every single pixel in an image is a part of — so this pixel is a part of a person, this pixel is a part of a car. Then we are going to use that idea, particularly an idea called UNet, which turns out that this idea of UNet, we can apply to bounding boxes — where it's called feature pyramids. We'll use that to get really good results with bounding boxes. That's kind of our path from here. It's all going to build on each other but take us into lots of different areas.

### ([0:16:40](https://youtu.be/h5Tz7gZT9Fo?t=16m40s)) Introducing fastai.text

We've moved from torchtext to fastai.text.

For NLP, last part, we relied on a library called torchtext but as good as it was, Jeremy have since then found the limitation of it too problematic to keep using it.

- Very slow. No parrallel processing.
- Doesn't remember what you did last time and it does it all over again from scratch.
- Hard to do fairly simple things (like multi-label problem).
- Somewhat convoluted API.

To fix all these problems, we've created a new library called fastai.text. Fastai.text is a replacement for the combination of torchtext and fastai.nlp. So don't use fastai.nlp anymore — that's obsolete. It's slower, it's more confusing, it's less good in every way, but there's a lot of overlaps. Intentionally, a lot of the classes and functions have the same names, but this is the non-torchtext version.

```Python
from fastai.text import *
import html
```

### IMDB

#### ([0:20:30](https://youtu.be/h5Tz7gZT9Fo?t=16m40s)) IMDB with fastai.text

[imdb.ipynb](https://nbviewer.jupyter.org/github/fastai/fastai/blob/master/courses/dl2/imdb.ipynb)

We will work with IMDb again. For those of you who have forgotten, go back and checkout [lesson 4](https://medium.com/@hiromi_suenaga/deep-learning-2-part-1-lesson-4-2048a26d58aa). This is a dataset of movie reviews.

> Quick review of lesson 4 NLP `bptt` concept
>
> ```Python
> bptt = 2
>
> Example of text data: 'a b c d e f g h i j k l m n o p q r s t u v w x'
>
> Tokenized: [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
>
> Split into 8:
> [1 2 3], [4 5 6],  [7 8 9], [10 11 12], [13 14 15], [16 17 18], [19 20 21], [22 23 24]
>
> Stack and transpose:
>
> a d g j m p s v
> b e h k n q t v
> c f i l o r u x
>
> Transpose vectors:
> [1  [4  [7  [10  [13  [16  [19  [22
>  2   5   8   11   14   17   20   23
>  3]  6]  9]  12]  15]  18]  21]  24]
>
> Stack vectors:
> [1  4  7  10  13  16  19  22
>  2  5  8  11  14  17  20  23
>  3  6  9  12  15  18  21  24]
> Result
> matrix with dimension (3 x 8)
>
> # We then grab a little chunk at time and those chunk lengths are approximately equal to `BPTT`. Here, we grab a little `2` long section and that is the first thing we chuck into our GPU (i.e. the batch).
>
> # The first 3 chunks looks like: [1 2], [3 4], [5 6], ...
> ```

##### Dataset

We need to download the IMDB large movie reviews from this site: http://ai.stanford.edu/~amaas/data/sentiment/
Direct link : [link](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz) and `untar` it into the `PATH` location. We use `pathlib` which makes directory traveral a breeze.

```bash
%cd data
!aria2c --file-allocation=none -c -x 5 -s 5 http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

[#6acf06 79MiB/80MiB(99%) CN:1 DL:14MiB]
06/26 15:59:49 [NOTICE] Download complete: /home/ubuntu/data/aclImdb_v1.tar.gz

Download Results:
gid   |stat|avg speed  |path/URI
======+====+===========+=======================================================
6acf06|OK  |    14MiB/s|/home/ubuntu/data/aclImdb_v1.tar.gz

Status Legend:
(OK):download completed.

!tar -zxf aclImdb_v1.tar.gz
%cd ..
```

Initial setup:

```Python
PATH = Path('data/aclImdb/')

BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag

!ls -lah {PATH}

total 1.7M
drwxr-xr-x 4 ubuntu ubuntu 4.0K Jun 26  2011 .
drwxrwxr-x 8 ubuntu ubuntu 4.0K Jun 26 16:17 ..
-rw-r--r-- 1 ubuntu ubuntu 882K Jun 11  2011 imdbEr.txt
-rw-r--r-- 1 ubuntu ubuntu 827K Apr 12  2011 imdb.vocab
-rw-r--r-- 1 ubuntu ubuntu 4.0K Jun 26  2011 README
drwxr-xr-x 4 ubuntu ubuntu 4.0K Jun 26 16:02 test
drwxr-xr-x 5 ubuntu ubuntu 4.0K Jun 26 16:02 train
```

#### ([0:23:10](https://youtu.be/h5Tz7gZT9Fo?t=23m10s)) The standard format of text classification dataset

The basic paths for NLP is that we have to take sentences and turn them into numbers, and there is a couple to get there. At the moment, somewhat intentionally, fastai.text does not provide that many helper functions. It's really designed more to let you handle things in a fairly flexible way.

```python
CLAS_PATH = Path('data/imdb_clas/')
CLAS_PATH.mkdir(exist_ok=True)

LM_PATH = Path('data/imdb_lm/')
LM_PATH.mkdir(exist_ok=True)
```
As you can see here, Jeremy wrote something called `get_texts` which goes through each thing in `CLASSES`. There are three classes in IMDb: negative, positive, and then there's another folder "unsupervised" which contains the ones they haven't gotten around to labeling yet — so we will just call that a class for now. So we just go through each one of those classes, then find every file in that folder, and open it up, read it, and chuck it into the end of the array. As you can see, with `pathlib`, it's super easy to grab stuff and pull it in, and then the label is just whatever class we are up to so far. We will do that for both training set and test set.

```python
CLASSES = ['neg', 'pos', 'unsup']

def get_texts(path):
    texts, labels = [], []

    for idx, label in enumerate(CLASSES):
        for fname in (path/label).glob('*.*'):
            texts.append(fname.open('r').read())
            labels.append(idx)
    return np.array(texts), np.array(labels)

trn_texts, trn_labels = get_texts(PATH / 'train')
val_texts, val_labels = get_texts(PATH / 'test')

len(trn_texts), len(val_texts)

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
(75000, 25000)
```

There are 75,000 in train, 25,000 in test. 50,000 in the train set are unsupervised, and we won't actually be able to use them when we get to the classification. Jeremy found this much easier than torch.text approach of having lots of layers and wrappers because in the end, reading text files is not that hard.

```python
col_names = ['labels', 'text']
```
One thing that's always good idea is to sort things randomly [00:23:19]. It is useful to know this simple trick for sorting things randomly particularly when you've got multiple things you have to sort the same way. In this case, you have labels and texts. `np.random.permutation`, if you give it an integer, it gives you a random list from 0 up to and not including the number you give it in some random order.

```python
# We use a random permutation numpy array to shuffle the text reviews.
np.random.seed(42)
trn_idx = np.random.permutation(len(trn_texts))
val_idx = np.random.permutation(len(val_texts))
```

You can them pass that in as an indexer to give you a list that's sorted in that random order. So in this case, it is going to sort `trn_texts` and `trn_labels` in the same random way. So that's a **useful little idiom to use**.

```python
trn_texts = trn_texts[trn_idx]
val_texts = val_texts[val_idx]

trn_labels = trn_labels[trn_idx]
val_labels = val_labels[val_idx]
```
Now we have our texts and labels sorted, we can create a dataframe from them [00:24:07].

```python
df_trn = pd.DataFrame({ 'text': trn_texts, 'labels': trn_labels }, columns=col_names)
df_val = pd.DataFrame({ 'text': val_texts, 'labels': val_labels }, columns=col_names)
```

Why are we doing this? The reason is because there is a somewhat standard approach starting to appear for text classification datasets which is to have your training set as a CSV file with the labels first, and the text of the NLP documents second. So it basically looks like this:

```python
df_trn.head()
```

![Training set dataframe](/images/imdb_notebook_001.png)

```python
# we remove everything that has a label of 2 because label of 2 is "unsupervised" and we can't use it.
df_trn[df_trn['labels'] != 2].to_csv(CLAS_PATH / 'train.csv', header=False, index=False)

df_val.to_csv(CLAS_PATH / 'test.csv', header=False, index=False)

(CLAS_PATH / 'classes.txt').open('w').writelines(f'{o}\n' for o in CLASSES)
```

So you have your labels and texts, and then a file called `classes.txt` which just lists the classes. I say somewhat standard because in a reasonably recent academic paper Yann LeCun and a team of researcher looked at quite a few datasets and they use this format for all of them. So that's what I started using as well for my recent paper. You'll find that this notebook, if you put your data into this format, the whole notebook will work every time [00:25:17]. So rather than having a thousand different formats, I just said let's just pick a standard format and your job is to put your data in that format which is the CSV file. The CSV files have no header by default.

You'll notice at the start, we have two different paths [00:25:51]. One was the classification path, and the other was the language model path.

`CLAS_PATH = Path('data/imdb_clas/')`

`LM_PATH = Path('data/imdb_lm/')`

In NLP, you'll see LM all the time. LM means language model. The classification path is going to contain the information that we are going to use to create a sentiment analysis model. The language model path is going to contain the information we need to create a language model. So they are a little bit different.

**Difference between classification model and language model**

One thing that is different is that when we create the `train.csv` in the classification path, we remove everything that has a label of 2 `df_trn['labels'] != 2` because label of 2 is "unsupervised" and we can't use it.

The second difference is the labels [00:26:51]. For the classification path, the labels are the actual labels, but for the language model, there are no labels so we just use a bunch of zeros and that just makes it a little easier because we can use a consistent dataframe/CSV format.

**Language model goal**

The LM's goal is to learn the structure of the English language. It learns language by trying to predict the next word given a set of previous words(ngrams). Since the LM does not classify reviews, the labels can be ignored.

The LM can benefit from all the textual data and there is no need to exclude the unsup/unclassified movie reviews.

We start by creating the data for the LM. We can create our own validation set, so you've probably come across by now, `sklearn.model_selection.train_test_split` which is a really simple function that grabs a dataset and randomly splits it into a training set and a validation set according to whatever proportion you specify. In this case, we concatenate our classification training and validation together, split it by 10%, now we have 90,000 training, 10,000 validation for our language model. So that's getting the data in a standard format for our language model and our classifier.

```python
# Concat all the train(pos/neg/unsup = **75k**) and test(pos/neg=**25k**) reviews into a big chunk of **100k** reviews.
#
# And then we use sklearn splitter to divide up the 100k texts into 90% training and 10% validation sets.
trn_texts, val_texts = sklearn.model_selection.train_test_split(
    np.concatenate([trn_texts, val_texts]), test_size=0.1)
```

Save Pandas Dataframes to CSV files:

```python
df_trn = pd.DataFrame({ 'text': trn_texts, 'labels': [0] * len(trn_texts) }, columns=col_names)
df_val = pd.DataFrame({ 'text': val_texts, 'labels': [0] * len(val_texts) }, columns=col_names)

df_trn.to_csv(LM_PATH / 'train.csv', header=False, index=False)
df_val.to_csv(LM_PATH / 'test.csv', header=False, index=False)
```

#### ([0:28:08](https://youtu.be/h5Tz7gZT9Fo?t=28m08s)) Language model tokens

The next thing we need to do is tokenization. Tokenization means at this stage, for a document (i.e. a movie review), we have a big long string and we want to turn it into a list of tokens which is similar to a list of words but not quite. For example, `don't`, we want it to be `do` and `n't`, we probably want full stop to be a token, and so forth. **Tokenization is something that we passed off to a terrific library called [spaCy](https://spacy.io/)** . We put a bit of stuff on top of spaCy but the vast majority of the work's been done by spaCy.

**Pre-processing**

We start cleaning up the messy text. There are 2 main activities we need to perform:

1. Clean up extra spaces, tab chars, new line chars and other characters and replace them with standard ones.
2. Use the spaCy library to tokenize the data. Since **spaCy does not provide a parallel/multicore version of the tokenizer**, the fastai library adds this functionality. This parallel version uses all the cores of your CPUs and runs much faster than the serial version of the spacy tokenizer.

Tokenization is the process of splitting the text into separate tokens so that each token can be assigned a unique index. This means we can convert the text into integer indexes our models can use.

#### ([0:29:59](https://youtu.be/h5Tz7gZT9Fo?t=29m59s)) Pandas 'chunksize' to deal with a large corpus

We use an appropriate `chunksize` as the tokenization process is memory intensive.

```python
chunksize = 24000
```

Before we pass it to spaCy, Jeremy wrote this simple `fixup` function which is each time he's looked at different datasets (about a dozen in building this), every one had different weird things that needed to be replaced. So here are all the ones he's come up with so far, and hopefully this will help you out as well. All the entities are HTML unescaped and there are bunch more things we replace. Have a look at the result of running this on text that you put in and make sure there's no more weird tokens in there.

```python
re1 = re.compile(r'  +')

def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))

def get_texts(df, n_lbls=1):
    labels = df.iloc[:, range(n_lbls)].values.astype(np.int64)
    texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
    for i in range(n_lbls + 1, len(df.columns)):
        texts += f' {FLD} {i - n_lbls} ' + df[i].astype(str)
    texts = texts.apply(fixup).values.astype(str)

    tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
    return tok, list(labels)
```

`get_all function` calls `get_texts` and `get_texts` is going to do a few things [00:29:40]. One of which is to apply that `fixup` that we just mentioned.

```python
def get_all(df, n_lbls):
    tok, labels = [], []

    for i, r in enumerate(df):
        print(i)
        tok_, labels_ = get_texts(r, n_lbls)
        tok += tok_
        labels += labels_
    return tok, labels
```

Let's look through this because there is some interesting things to point out [00:29:57]. We are going to use Pandas to open our `train.csv` from the language model path, but we are passing in an extra parameter you may not have seen before called `chunksize`. Python and Pandas can both be pretty inefficient when it comes to storing and using text data. So you'll see that very few people in NLP are working with large corpuses. And Jeremy thinks the part of the reason is that traditional tools made it really difficult — you run out of memory all the time. So this process he is showing us today, he has used on corpuses of over a billion words successfully using this exact code. One of the simple trick is this thing called `chunksize` with Pandas. That that means is that Pandas does not return a data frame, but it returns an iterator that we can iterate through chunks of a data frame. That is why we don't say `tok_trn = get_text(df_trn)` but instead we call `get_all` which loops through the data frame but actually what it's really doing is it's looping through chunks of the data frame so each of those chunks is basically a data frame representing a subset of the data [00:31:05].

:question: When I'm working with NLP data, many times I come across data with foreign texts/characters. Is it better to discard them or keep them [31:31]? No no, definitely keep them. This whole process is unicode and I've actually used this on Chinese text. This is designed to work on pretty much anything. In general, most of the time, it's not a good idea to remove anything. Old-fashioned NLP approaches tended to do all this like lemmatization and all these normalization steps to get rid things, lower case everything, etc. But that's throwing away information which you don't know ahead of time whether it's useful or not. So don't throw away information.

So we go through each chunk each of which is a data frame and we call `get_texts` [00:32:19]. `get_texts` will grab the labels and makes them into integers, and it's going to grab the texts.

#### ([0:32:38](https://youtu.be/h5Tz7gZT9Fo?t=32m38s)) {BOS} (beginning of stream) and {FLD} (field) tokens

A couple things to point out:

- Before we include the text, we have "beginning of stream" (`BOS`) token which we defined in the beginning. There's nothing special about these particular strings of letters — they are just ones I figured don't appear in normal texts very often. So every text is going to start with 'xbos' — why is that? Because it's often useful for your model to know when a new text is starting. For example, if it's a language model, we are going to concatenate all the texts together. So it would be really helpful for it to know all this articles finished and a new one started so I should probably forget some of their context now.
- Ditto is quite often texts have multiple fields like a title and abstract, and then a main document. So by the same token, we've got this thing here which lets us actually have multiple fields in our CSV. So this process is designed to be very flexible. Again at the start of each one, we put a special "field starts here" token followed by the number of the field that's starting here for as many fields as we have. Then we apply `fixup` to it.

#### ([0:33:57](https://youtu.be/h5Tz7gZT9Fo?t=33m57s)) Run spaCy on multi-cores with `proc_all_mp()`

Then most importantly [00:33:54], we tokenize it — we tokenize it by doing a "process all multiprocessing" (`proc_all_mp`). Tokenizing tends to be pretty slow but we've all got multiple cores in our machines now, and some of the better machines on AWS can have dozens of cores. spaCy is not very amenable to multi processing but Jeremy finally figured out how to get it to work. The good news is that it's all wrapped up in this one function now. So all you need to pass to that function is a list of things to tokenize which each part of that list will be tokenized on a different core. There is also a function called `partition_by_cores` which takes a list and splits it into sublists. The number of sublists is the number of cores that you have in your computer. On Jeremy's machine without multiprocessing, this takes about an hour and a half, and with multiprocessing, it takes about 2 minutes. So it's a really hand thing to have. Feel free to look inside it and take advantage of it for your own stuff. Remember, we all have multiple cores even in our laptops and very few things in Python take advantage or it unless you make a bit of an effort to make it work.

```python
df_trn = pd.read_csv(LM_PATH / 'train.csv', header=None, chunksize=chunksize)
df_val = pd.read_csv(LM_PATH / 'test.csv', header=None, chunksize=chunksize)

tok_trn, trn_labels = get_all(df_trn, 1)
tok_val, val_labels = get_all(df_val, 1)

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
0
1
2
0

(LM_PATH / 'tmp').mkdir(exist_ok=True)
```
:warning: if you encountered an error: `OSError: [E050] Can't find model 'en'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.`. The problem is due to spaCy unable to find 'en' model. Here's how to fix this:

Try running this command and it should install the model to the correct directory.

```bash
python -m spacy download en

Collecting https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz (37.4MB)
    100% |████████████████████████████████| 37.4MB 81.7MB/s ta 0:00:01
    7%   |██▎                             | 2.6MB 611kB/s eta 0:00:57
Installing collected packages: en-core-web-sm
  Running setup.py install for en-core-web-sm ... done
Successfully installed en-core-web-sm-2.0.0
You are using pip version 9.0.3, however version 10.0.1 is available.
You should consider upgrading via the 'pip install --upgrade pip' command.

    Linking successful
    /home/ubuntu/anaconda3/envs/fastai/lib/python3.6/site-packages/en_core_web_sm
    -->
    /home/ubuntu/anaconda3/envs/fastai/lib/python3.6/site-packages/spacy/data/en

    You can now load the model via spacy.load('en')
```

#### ([0:35:40](https://youtu.be/h5Tz7gZT9Fo?t=35m40s)) Difference between tokens and word - capture semantic of letter case and others

Here is the result at the end [00:35:42]. Beginning of the stream token (`xbos`), beginning of field number 1 token (`xfld 1`), and tokenized text. You'll see that the punctuation is on whole now a separate token.

**`t_up`** : `t_up mgm` — MGM was originally capitalized. But the interesting thing is that normally people either lowercase everything or they leave the case as is. Now if you leave the case as is, then "SCREW YOU" and "screw you" are two totally different sets of tokens that have to be learnt from scratch. Or if you lowercase them all, then there is no difference at all. So how do you fix this so that you both get a semantic impact of "I'M SHOUTING NOW" but not have to learn the shouted version vs. the normal version. So the idea is to come up with a unique token to mean the next thing is all uppercase. Then we lowercase it, so now whatever used to be uppercase is lowercased, and then we can learn the semantic meaning of all uppercase.

**`tk_rep`** : Similarly, if you have 29 `!` in a row, we don't learn a separate token for 29 exclamation marks — instead we put in a special token for "the next thing repeats lots of times" and then put the number 29 and an exclamation mark (i.e. `tk_rep 29 !`). So there are a few tricks like that. If you are interested in NLP, have a look at the tokenizer code for these little tricks that Jeremy added in because some of them are kind of fun.

![Tokenized result](/images/imdb_notebook_002.png)

The nice thing with doing things this way is we can now just `np.save` that and load it back up later [00:37:44]. We don't have to recalculate all this stuff each time like we tend to have to do with torchtext or a lot of other libraries.

```python
np.save(LM_PATH / 'tmp' / 'tok_trn.npy', tok_trn)
np.save(LM_PATH / 'tmp' / 'tok_val.npy', tok_val)

tok_trn = np.load(LM_PATH / 'tmp' / 'tok_trn.npy')
tok_val = np.load(LM_PATH / 'tmp' / 'tok_val.npy')
```

#### ([0:38:05](https://youtu.be/h5Tz7gZT9Fo?t=38m05s)) Numericalize tokens - Python `Counter` class

Now that we got it tokenized, the next thing we need to do is to turn it into numbers which we call numericalizing it.

The way we numericalize it is very simple.

- We make a list list of all the words that appear in some order.
- Then we replace every word with its index into that list.
- The list of all the tokens, we call that the vocabulary.

Here is an example of some of the vocabulary [00:38:28]. The `Counter` class in Python is very handy for this. It basically gives us a list of unique items and their counts. Here are the 25 most common things in the vocabulary. Generally speaking, we don't want every unique token in our vocabulary. If it doesn't appear at least twice then might just be a spelling mistake or a word we can't learn anything about it if it doesn't appear that often. Also the stuff we are going to be learning about so far in this part gets a bit clunky once you've got a vocabulary bigger than 60,000.

:bookmark: Time permitting, we may look at some work Jeremy has been doing recently on handling larger vocabularies, otherwise that might have to come in a future course. But actually for classification, doing more than about 60,000 words doesn't seem to help anyway.

### Pre-trained Language Model - Pre-Training

#### ([0:42:16](https://youtu.be/h5Tz7gZT9Fo?t=42m16s)) Pre-trained language model

_WIP_

#### ([0:47:13](https://youtu.be/h5Tz7gZT9Fo?t=47m13s)) Map IMDb index to wiki text index

_WIP_

#### ([0:53:09](https://youtu.be/h5Tz7gZT9Fo?t=53m09s)) fastai documentation project

_WIP_

#### ([0:58:24](https://youtu.be/h5Tz7gZT9Fo?t=58m24s)) Difference between pre-trained LM and embeddings - word2vec

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

#### ([1:28:44](https://youtu.be/h5Tz7gZT9Fo?t=1h28m44s)) Difference between pre-trained LM and embeddings - Comparison of NLP and CV

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

#### ([1:52:44](https://youtu.be/h5Tz7gZT9Fo?t=1h52m44s)) RNN encoder and `MultiBatchRNN` encoder - BPTT for text classification (BPT3C)

_WIP_

### Tricks to conduct ablation studies

#### ([1:58:35](https://youtu.be/h5Tz7gZT9Fo?t=1h58m35s)) VNC and Google Fire Library

_WIP_

#### ([2:05:10](https://youtu.be/h5Tz7gZT9Fo?t=2h05m10s)) SentencePiece; Tokenize sub-word units

_WIP_
