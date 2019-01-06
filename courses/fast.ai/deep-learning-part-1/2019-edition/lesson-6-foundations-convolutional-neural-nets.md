# Lesson 6 - Foundations of Convolutional Neural Networks

_These are my personal notes from fast.ai Live (the new International Fellowship programme) course and will continue to be updated and improved if I find anything useful and relevant while I continue to review the course to study much more in-depth. Thanks for reading and happy learning!_

Live date: 28 Nov 2018, GMT+8

## Topics

* Regularization
  * Dropout
  * Batch Normalization
  * Data augmentation
* Deep dive into Computer Vision Convolutional Neural Networks (CNNs)
* Platform.ai
* Rossmann Store Sales Kaggle competition
* Data cleaning
* Entity Embeddings
* Categorical variables
* Pre-processing
* Ethics and Data Science

## Lesson Resources

* **Website and video** links will be shared when the MOOC officially released in early 2019.
* [Official resources and updates (Wiki)](https://forums.fast.ai/t/lesson-5-official-resources-and-updates/30863)
* [Forum discussion](https://forums.fast.ai/t/lesson-6-official-resources-and-updates/31441)
* [Advanced forum discussion](https://forums.fast.ai/t/lesson-6-advanced-discussion/31442)
* [FAQ, resources, and official course updates](https://forums.fast.ai/t/faq-resources-and-official-course-updates/27934)
* Jupyter Notebook and code
  * [lesson6-rossmann.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson6-rossmann.ipynb)
  * [rossman_data_clean.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/rossman_data_clean.ipynb)
  * [lesson6-pets-more.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson6-pets-more.ipynb)

### Papers

* Optional reading
  * [Entity Embeddings of Categorical Variables](http://arxiv.org/abs/1604.06737) by Cheng Guo and Felix Berkhahn
  * [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://jmlr.org/papers/v15/srivastava14a.html) by Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever and Ruslan Salakhutdinov.
  * [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167) by Sergey Ioffe and Christian Szegedy.
  * [How Does Batch Normalization Help Optimization](https://arxiv.org/abs/1805.11604) by Shibani Santurkar, et al.

### Blog Posts and Articles

* [CNNs from different viewpoints](https://medium.com/impactai/cnns-from-different-viewpoints-fab7f52d159c)

## Assignments

* Run lesson 6 notebooks.
* Create your own dropout layer in Python.
* Read fastai docs:
  * [Image transforms for data augmentation](https://docs.fast.ai/vision.transform.html)
  * Remember these notebooks you can open up and run thise code yourself and get this output.

# My Notes

Hi everybody, welcome to lesson 6.

We're going to do a deep dive into computer vision convolutional neural networks, what is a convolution and we're also going to learn the final regularization tricks after the last lesson learning about weight decay and/or L2 regularization.

**Software Update**

:exclamation: Always remember to do an update on fastai library and course repo. :exclamation:

```sh
# library update
conda install -c fastai fastai

# course repo update
git pull
```

## Platform.ai

I am really excited about and I have had a small hand and helping to create. For those of you that saw [his talk on TED.com](https://www.ted.com/talks/jeremy_howard_the_wonderful_and_terrifying_implications_of_computers_that_can_learn), you might have noticed this really interesting demo that we did about four years ago showing a way to quickly build models with unlabeled data.

The reason I am mentioning it here is that it's going to let you create models on different types of data sets to what you can do now that is to say data sets that you don't have labels for yet. Platform.ai actually going to help you label them.

**Quick demo about Platform.ai.** [[00:01:21](https://youtu.be/U7c-nYXrKD4?t=81)]

![](../../../../images/fastai_p1_v3/lesson_6/1.png)

If you'd go to platform.ai and choose to get started you'll be able to create a new project and if you create a new project you can either upload your own images.

![](../../../../images/fastai_p1_v3/lesson_6/2.png)

What we're doing here is we're trying to take advantage of the **combination of the human plus machine** the machine is pretty good at quickly doing calculations but as a human, we're pretty good at looking at a lot of things at once and seeing the odd one out. In this case, we're looking for cars that aren't front right and so by laying the one on in front of us we can do that really quickly it's like okay definitely that one so just click on the ones that you don't want all right it's all good so then you can just go back and so then what you can do is you can either put them into a new category but I can create a new label or you can click on one of the existing ones so before I came, I just created a few so here's front right so I just clicks on it here there we go.

> TL;DR: This makes it easier to label image data and build simple models without coding. It is working so well because humans can see a lot of things same time.

---

## [Rossmann Store Sales Kaggle competition](https://www.kaggle.com/c/rossmann-store-sales) [[00:09:43](https://youtu.be/U7c-nYXrKD4?t=583)]

From last week's discussion of regularization specifically in the context of the tabular learner, this was the init method in the tabular learner:

![](../../../../images/fastai_p1_v3/lesson_6/3.png)

Our goal was to understand everything here and we're not quite there yet.

Last week we were looking at the adult dataset which is a really simple dataset that's just for toy purposes. So this week let's look at a dataset that's much more interesting a Kaggle competition dataset. We know kind of what the best in the world and you know Kaggle competition results tend to be much harder to beat than academic state-of-the-art results tend to be because of a lot more people work on Kaggle competitions than most academic data sets. It's a really good challenge to try and do well on a Kaggle competition dataset. The Rossmann dataset is if they've got 3000 drugs in Europe and you're trying to predict how many products they're going to sell in the next couple of weeks.

**Interesting things about this**

- The test set for this is from a time period that is more recent than the training set. This is really common if you want to predict things there's no point predicting things that are in the middle of your training set you want to predict things in the future.
- The evaluation metric they provided is the root mean squared percent error so this is just a normal root mean squared error except we go actual minus prediction divided by actual. It's the percent error that we're taking the root mean square of.

    ![](../../../../images/fastai_p1_v3/lesson_6/4.png)

- It's always interesting to look at the leaderboard. The leaderboard the winner was 0.1. The paper, [Entity Embeddings of Categorical Variables](http://arxiv.org/abs/1604.06737) that we've roughly replicated was point 105 106 and 10th place out of 3,000 was 0.11 ish bit less.

**Additional data**

The data that was provided by Kaggle has a small number of files. But they also let competitors provide additional external data as long as they shared it with all the competitors. In practice, the data the set we're going to use contains 6 or 7 tables. The way that you join tables and stuff isn't really part of a deep learning course so I am going to skip over it and instead refers you to [Introduction to Machine Learning for Coders](https://course.fast.ai/ml) which will take you step-by-step through the data preparation.

For now, it's readily available in course repo [rossman_data_clean.ipynb notebook](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/rossman_data_clean.ipynb). You'll see the whole process there. You'll need to run through that notebook to create these pickle files that we read here. I just wants to mention one particularly interesting part of the Rossmann data clean notebook which is you'll see there's something that says `add_date()` part. I have been mentioning for a while that we're going to look at time series and pretty much everybody who I have spoken to about it has assumed that he's going to do some kind of recurrent neural network but I am not.

Interestingly the kind of the main academic group that studies time series is econometrics and but they tend to study one very specific kind of time series which is where the only data you have is a sequence of time points of one thing like that’s the only thing you have is one sequence. In real life that’s almost never the case. Normally we would have some information about the store that represents or the people that represents, we have metadata, we have sequences of other things measured at similar time periods or different time periods. Most of the time, I find in practice the state-of-the-art results when it comes to competitions on some kind of more real-world datasets don't tend to use recurrent neural networks but instead they tend to take the time piece which in this case it was a date we were given in the data and they add a whole bunch of metadata. So in our case for example we've added day of week. So we were given a date. We've added day of week, year, month, week of year, day of month, day of week, day of year, and a bunch of booleans at the month start or end quarter, year start or end, elapsed time since 1970 so forth.

If you run this one function `add_date` part and pass it a date, it will add all of these columns to your dataset for you. What that means is that let's take a very reasonable behavior example. Purchasing behavior probably changes on pay day. Pay day might be the 15th of the month. So if you have a thing here called this is day of month here, then it will be able to recognize everytime something is a fifteen there and associated it with a higher in this case embedding matrix value.

So this way it basically you can't expect a neural net to do all of our feature engineering for us. We can expect it to kind of find non-linearities and interactions and stuff like that but for something like taking a date like this and figuring out that the fifteen of the month is something when interesting things happen. It's much better if we can provide that information for it.

So this is a really useful function to use and once you've done this you can treat many kinds of time-series problems as regular tabular problems. I say many kinds, not all. If there's very complex kind of state involved in a time series such as equity trading or something like that, this probably won't be the case or this won't be the only thing you need. But in this case, it'll get us a really good result. And in practice most of the time I find this works well.

Tabular data is normally in Pandas so we just stored them as standard Python pickle files. We can read them in. We can take a look at the first 5 records and the key thing here is that we're trying to on a particular date for a particular store ID we want to predict the number of sales. Sales is the dependent variable.

**Pre-processing** [[00:16:58](https://youtu.be/U7c-nYXrKD4?t=1018)]

Transforms are something which we run every time when we take a batch of data. Preprocessing is instead something we run once for all of our data before we start training. What is similar between these is that we need to use the same values for the train, test, and validation sets. When we first preprocess train data with certain values we need to use those same values for test and valid sets.

Pre-processor:
- `Categorify`: it's going to take these strings `Mar,Jun,Sept,Dec`, it's going to find all of the possible unique values of it and it's going to create a list of them and then it's going turn the strings into numbers.
- `FillMissing`: create for everything that's missing, anything that has a missing value, it'll create an additional column with the column name underscore na (i.e: `CompetitionDistance_na`) and it will set it for `True` for any time that was missing and then what we do is we replace `CompetitionDistance` with the median for those.

```python
df.column_name.cat.categories
```

```
Index(['Feb,May,Aug,Nov', 'Jan,Apr,Jul,Oct', 'Mar,Jun,Sept,Dec'], dtype='object')
```

```python
df.column_name.cat.codes
```

```
280   -1
584   -1
588    1
847   -1
896    1
dtype: int8
```

You don’t need to run pre-processes manually. When you create `TabularList` object it will have `procs` parameter where you define pre-processes.

```python
procs=[FillMissing, Categorify, Normalize]
```

![](../../../../images/fastai_p1_v3/lesson_6/5.png)

Fastai assumes that you want to do classification if you pass dependent variables that are in `int` format. That is why you can pass `label_cls` parameter where you can tell that you want these to be floats and that way be handled as a regression problem.

In many cases, it is better to look at percentage differences rather than exact differences and that is why sometimes we need to use RMSPE instead of RMSE. We use this by just setting log true and then taking RMSE. Fastai is using RMSE as default for regression problems.

We can set `y_range` and that ways tell the model to not predict over or under some value. For example, if we are predicting prices of houses we know that price can’t be less than 0.

![](../../../../images/fastai_p1_v3/lesson_6/6.png)

We set intermediate layers to go from 1,000 input activations to 500 output activations so there are 500,000 weights in that matrix. It is a lot for dataset where is only a few hundred thousand rows. This is going to overfit so we need to regularize it. Some beginners might reduce the weights but as we learned from the last lesson it is better to just regularize the model. By default, fastai is using weight decay but for this problem (and often for other problems) we need more regularization. Regularization can be added by passing in `ps` and `emb_drop`.

---

## Dropout [[00:30:03](https://youtu.be/U7c-nYXrKD4?t=1803)]

![](../../../../images/fastai_p1_v3/lesson_6/7.png)

Picture from the paper, [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://jmlr.org/papers/v15/srivastava14a.html).

The picture on the left is normal neural network and the picture on the right is the same network after applying dropout. Every arrow shows multiplications between weights and activations. A circle represents sum.

When we use dropout we throw away some percentage of the activations (a type of parameters that is not weights). For each mini-batch, we throw away different activations. Amount of activations we are going to ignore is `p` which can be `0.5`.

In the picture, some of the inputs are also deleted but that isn’t common practice anymore. When we are overfitting it means that some part of our model is learned to recognize some particular image and not features as it should. When we use dropout it will assure that this can’t happen. Having too much dropout will reduce the capacity of the model.

In fastai, `ps` means that we can add multiple dropouts for different layers same way we can add multiple learning rates.

[[00:34:47](https://youtu.be/U7c-nYXrKD4?t=2087)] We turn off the dropout (and other regularization methods) when we are testing (inference time) the model. But then we have two times more parameters. In the paper, researchers suggested multiplying all weights with `p`. In PyTorch and many other libraries, this multiplying is done during the training so we don’t need to care about it.

:memo: [[00:37:24](https://youtu.be/U7c-nYXrKD4?t=2244)] That'd be a good exercise to try see if you can create your own dropout layer in Python and see if you can replicate the results we get with this dropout there:

```cpp
noise.bernoulli_(1 - p);
noise.div_(1 - p);
return multiply<inplace>(input, noise);
```

[[00:37:49](https://youtu.be/U7c-nYXrKD4?t=2269)] `emb_drop` is dropout for embedding layer. We use special dropout for embedding layer because it can be a little bit higher.

---

## Batch Normalization [[00:42:20](https://youtu.be/U7c-nYXrKD4?t=2540)]

What it is is extremely unclear. Let me describe it to you. It's kind of a bit of regularization, it's kind of a bit of training helper. It's called batch normalization and it comes from this paper, [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167).

![](../../../../images/fastai_p1_v3/lesson_6/8.png)

[[00:45:09](https://youtu.be/U7c-nYXrKD4?t=2709)] What is internal covariate shift? Well, it doesn't matter because this is one of those things where researchers came up with some intuition and some idea about these things they wanted to try, they did it, it worked well, they post-hoc added on some mathematical analysis to try and claim why it worked and it turned out they were totally wrong. In the last 2 months, there's been 2 papers, so it took 3 years for people to really figure this out. In the last 2 months there' been 2 papers that have shown batch normalization doesn't reduce covariate shift at all. And even if it did, that has nothing to do with why it works. That's kind of interesting insight again why we should be focusing on being practitioners and experimentalists and developing an intuition.

[[00:45:57](https://youtu.be/U7c-nYXrKD4?t=2757)] What BatchNorm does is what you see in this picture in this paper, [How Does Batch Normalization Help Optimization](https://arxiv.org/abs/1805.11604):

![](../../../../images/fastai_p1_v3/lesson_6/18.png)

First BatchNorm takes activations. Then it takes mean and variance of those, and using those values it will normalize. Finally, (this is important) we instead of just adding bias, multiply the activations with something which is like bias. By using BatchNorm, loss decrease more smoothly and that way can be trained using higher learning rate.

Why this multiplication bias thing is working so well? Let’s say that we are again predicting movie reviews that are between 1–5. Activations in the last layer are between -1–1. We should make a new set of weights where mean and spread is increased. We can’t just move up the values because the weights are interacting very differently. With bias, we can increase the spread and now with BatchNorm, we can increase scale. Details don’t matter that much but the thing you need to know is that you want to use it. There is some other type of normalizations nowadays but BatchNorm might be the best. Jeremy told that Fastai library is using also something called WeightNorm which is developed in the last couple of months.

[[00:51:50](https://youtu.be/U7c-nYXrKD4?t=3110)] We create for each continues variable own BatchNorm and then run those. One thing Jeremy pointed is that we don’t calculate own mean and standard derivation for every mini-batch but rather take an exponentially weighted moving average of mean and standard derivation. We tune that by changing the momentum parameter (which isn’t the same as momentum regularizer). A smaller value will assure that mean and standard derivation doesn’t change so much and vice versa.

When to use these techniques:
- **weight decay** -  with or without dropout depending on the problem. (Test which is working best)
- **BatchNorm** - always
- **dropout** - with or without weight decay depending on the problem. (Test which is working best)

In general, it is often good to have a little dropout and weight decay.

Next, we are going to look at data augmentation. It is also a regularization technique. It might be the least studied regularization although there is no cost which means that you can do it and get better regularization without needing to train longer or risk to underfit.

**Question:question:**: In what proportion would you use dropout vs. other regularization errors, like, weight decay, L2 norms, etc.? [[00:55:00](https://youtu.be/U7c-nYXrKD4?t=3300)]

:exclamation: Remember that [L2 regularization and weight decay are kind of two ways of doing the same thing](https://bbabenko.github.io/weight-decay/). We should always use weight decay version, not the L2 regularization version. :exclamation:

So there's weight decay, there's batch norm which has a regularizing effect, there's data augmentation which we will see soon and there's dropout. Batch norm you pretty much always want. So that's easy. Data augmentation we will see in a moment. So then it's really between dropout vs. weight decay. I have no idea. I don't think I've seen anybody to find a compelling study on how to combine those two things. Can you always use one instead of the other? Why? Why not? I don't think anybody has figured that out. I think in practice, it seems that you generally want a bit of both. You pretty much always want some weight decay. But you often also want a bit of dropout. But honestly, I don't know why. I've not seen anybody really explain why or how to decide. So this is one of these things you have to try out and kind of get a feel for what tends to work for your kinds of problems. I think the defaults that we provide in most of our learners should work pretty well in most situations but yeah, definitely play around with it.

---

## Data Augmentation [[00:56:46](https://youtu.be/U7c-nYXrKD4?t=3406)]

The next kind of regularization we're going to look at is data augmentation and data augmentation is one of the least well studied types of regularization but it's the kind that I think I'm kind of the most excited about. The reason I'm kind of the most excited about it is that there's basically almost no cost to it. You can do data augmentation and get better generalization without it taking longer to train without underfitting to an extent at least. So let me explain. So what we're going to do now is we're going to come back to computer vision and we're going to come back to our pets dataset again. So, let's load it in.

[lesson6-pets-more.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson6-pets-more.ipynb)

```python
tfms = get_transforms(max_rotate=20, max_zoom=1.3, max_lighting=0.4, max_warp=0.4,
                      p_affine=1., p_lighting=1.)
```

`get_transforms` parameters:
- `p_affine`: probability of affine transform.
- `p_lighting`: probability of lighting transform.
- `max_rotate`: how much rotate (left and right angle).
- `max_zoom`: how much to max zoom in.
- `max_warp`: how much warp the image.

More about these and other parameters, check the [docs](https://docs.fast.ai/vision.transform.html).

[[00:59:03](https://youtu.be/U7c-nYXrKD4?t=3543)] Remember these notebooks you can open up and run thise code yourself and get this output. All of these HTML documentation documents are auto-generated from the notebooks in the `docs_source` directory in the fastai repo.

**Padding mode** [[01:01:30](https://youtu.be/U7c-nYXrKD4?t=3690)]

![](../../../../images/fastai_p1_v3/lesson_6/19.png)

Reflections nearly always better by the way. I don't know that anybody else has really studied this but we have studied it in some depth, haven't actually written a paper about it but just enough for our own purposes to say reflection works best most of the time. So that's the default.

![](../../../../images/fastai_p1_v3/lesson_6/20.png)

**Perspective Warping** [[01:02:04](https://youtu.be/U7c-nYXrKD4?t=3724)]

So the cool thing is as you can see, each of these pictures, as if this cat was being taken kind of from different angles. They're all kind of optically sensible. This is a really great type of data augmentation. It's also one which I don't know of any other library that does it or at least certainly one that does it in a way that's both fast and keep the image crisp as it is in fastai. If you're looking to win a Kaggle competition, this is the kind of thing that's going to get you above the people that aren't using the fastai library.

[[01:03:46](https://youtu.be/U7c-nYXrKD4?t=3826)] Because this is a training dataset, it's going to use data augmentation. So you can see the same doggy using lots of different kinds of data augmentation. You can see why this is going to work really well because these pictures all look pretty different but we didn't have to do any extra hand labeling or anything. It's like free extra data. **Data augmentation is really really great**.

One of the big opportunities for research is to figure out ways to do data augmentation in other domains. So how could you do data augmentation with text data or genomic data or histopathology data or whatever. Almost nobody is looking at that and to me it's one of the biggest opportunities that could let you decrease data requirements like by 5 to 10x.

---

## Convolutional Neural Network (CNN) [[01:05:16](https://youtu.be/U7c-nYXrKD4?t=3916)]

[lesson6-pets-more.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson6-pets-more.ipynb)

Because we are going to study convolutional neural networks, we are going to create a convolutional neural network. You know how to create them.

After training a CNN model we want to see what is happening there. We are going to learn what is happening by creating heatmap.

![](../../../../images/fastai_p1_v3/lesson_6/9.png)

[[01:05:59](https://youtu.be/U7c-nYXrKD4?t=3959)]

There is pre-build function for this in fastai but I am going to show how to make it without fastai.

![](../../../../images/fastai_p1_v3/lesson_6/image_kernels.gif)
http://setosa.io/ev/image-kernels/

[[01:07:41](https://youtu.be/U7c-nYXrKD4?t=4061)]

[[01:15:01](https://youtu.be/U7c-nYXrKD4?t=4501)] Let's take a look at this from another angle or quite a few other angles. We're going to look at a fantastic [post](https://medium.com/impactai/cnns-from-different-viewpoints-fab7f52d159c) from a guy called Matt Kleinsmith who is actually a student in the first year that we did this course and he wrote this as part of his project work back then.

![](../../../../images/fastai_p1_v3/lesson_6/21.png)

This is how we do CNN if we have RGB image. Notice that although the kernel is three dimensional the output for 3x3x3 area is still one pixel.

![](../../../../images/fastai_p1_v3/lesson_6/10.png)

We can add more kernels and combine together. 16 is a common number.

![](../../../../images/fastai_p1_v3/lesson_6/11.png)

Now you are at a point where you start to understand how everything is working and that way you can use some variation of fastai techniques. Often things in the library are designed to work generally well so you might get better results by changing some things.

We can create own kernel. Expand will make tensor 3x3x3 kernel and the first dimension is created because now we can store more than one kernel.

![](../../../../images/fastai_p1_v3/lesson_6/12.png)

First index is the number of kernels.

![](../../../../images/fastai_p1_v3/lesson_6/13.png)

Data we import conv2d need to be in batches and that is why we create one additional dimension.

Average pooling is taking the mean of every layer. Then if we want to have 37 outputs we just multiply the average pooling results with a matrix that is 37 numbers wide. Idea is that all 512 matrices are representing some feature.

![](../../../../images/fastai_p1_v3/lesson_6/15.png)

When we want to create heatmap to the picture best way is to average over 512 dimensions instead of 11x11 area. That way we get 11x11 area where every pixel is average of 512 pixels. Then we can see how much that pixel activated on average.

![](../../../../images/fastai_p1_v3/lesson_6/16.png)

![](../../../../images/fastai_p1_v3/lesson_6/17.png)

---

## Ethics and Data Science [[01:48:56](https://youtu.be/U7c-nYXrKD4?t=6536)]

![](../../../../images/fastai_p1_v3/lesson_6/22.png)

It matters.

If you got this far, you are definitely at a point now where you're ready to make a serious impact on the world. So, I hope we can make sure that that's a positive impact. See you next week. :clap: :clap: :clap:
