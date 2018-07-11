# Lesson 13 - Image Enhancement; Style Transfer; Data Ethics

_These are my personal notes from fast.ai course and will continue to be updated and improved if I find anything useful and relevant while I continue to review the course to study much more in-depth. Thanks for reading and happy learning!_

## Topics

* Image enhancement.
  * Neural style transfer.
    * An interesting approach that allows us to change the style of images in whatever way we like.
    * Optimize pixels, instead of weights, which is an interesting different way of looking at optimization.
* Generative models (and many other techniques we've discussed) can cause harm just as easily as they can benefit society.
  * Ethics in AI.
    * Bias is an important current topic in data ethics.
  * Get a taste of some of the key issues, and ideas for where to learn more.
* fastai library
  * TrainPhase API
    * SGD, SGD with Warm Restarts (SGDR), 1cycle
    * Discriminative learning rates + 1cycle
  * DAWNBench entries
    * ImageNet training
    * CIFAR10 result

## Lesson Resources

* [Website](http://course.fast.ai/lessons/lesson13.html)
* [Video](https://youtu.be/nG3tT31nPmQ)
* [Wiki](http://forums.fast.ai/t/part-2-lesson-14-wiki)
* Jupyter Notebook and code
  * [training_phase.ipynb](https://nbviewer.jupyter.org/github/fastai/fastai/blob/master/courses/dl2/training_phase.ipynb) - demo notebook on how to use the new TrainingPhase API in fastai library by Sylvain Gugger
  * [style-transfer.ipynb](https://nbviewer.jupyter.org/github/fastai/fastai/blob/master/courses/dl2/style-transfer.ipynb)
* Dataset for style transfer notebook
  * ImageNet sample in files.fast.ai/data / [direct download link](http://files.fast.ai/data/imagenet-sample-train.tar.gz) (2.1 GB)
  * Full ImageNet [[faster download from Kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge/data), mirrored over from the [ImageNet Download Site](http://www.image-net.org/download-imageurls)]

## Assignments

### Papers

* Must read
  * [Deep Painterly Harmonization](https://arxiv.org/abs/1804.03189) by Fujun Luan, et. al
  * [Progressive Growing of GANs for Improved Quality, Stability, and Variation](http://research.nvidia.com/publication/2017-10_Progressive-Growing-of) by Tero Karras, et. al
  * [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) by Leon A. Gatys, et. al
* Additional papers (optional)
  * [Large Batch Training of Convolutional Networks](https://arxiv.org/abs/1708.03888) - a training algorithm based on Layer-wise Adaptive Rate Scaling (LARS)

### Other Resources

#### Blog Posts and Articles

* [Adding a cutting-edge deep learning training technique to the fast.ai library](https://medium.com/@hortonhearsafoo/adding-a-cutting-edge-deep-learning-training-technique-to-the-fast-ai-library-2cd1dba90a49) by fast.ai International Fellow, William Horton - averaging weights leads to wider optima and better generalization

#### Other Useful Information

* [Rachel's numerical linear algebra course](https://github.com/fastai/numerical-linear-algebra)
* [Pytorch implementation of Universal Style Transfer via Feature Transforms](https://github.com/sunshineatnoon/PytorchWCT)
* [Gender Shades by MIT Media Lab](http://gendershades.org/) - AI and ethics

## My Notes

### Image enhancement

![](/images/lesson_13_001.png)

![](/images/lesson_13_002.png)

Image enhancement — we'll cover things like this painting that you might be familiar with. However, you might not have noticed before that this painting of an eagle in it. The reason you may not have noticed that before is this painting didn't used to have an eagle in it. By the same token, the painting on the first slide did not used to have Captain America's shield on it either.

#### Deep painterly harmonization paper - style transfer [[00:00:40](https://youtu.be/xXXiC4YRGrQ?t=40)]

![](/images/lesson_13_003.png)

This is a cool new paper that just came out a couple of days ago called [Deep Painterly Harmonization](https://arxiv.org/abs/1804.03189) and it uses almost exactly the technique we are going to learn in this lesson with some minor tweaks. But you can see the basic idea is to take one picture pasted on top of another picture, and then use some kind of approach to combine the two. The approach is called a "style transfer".

---

### Stochastic Weight Averaging [[00:01:10](https://youtu.be/xXXiC4YRGrQ?t=70)]

Before we talk about that, I wanted to mention this really cool contribution by [William Horton who added this stochastic weight averaging technique to the fastai library](https://medium.com/@hortonhearsafoo/adding-a-cutting-edge-deep-learning-training-technique-to-the-fast-ai-library-2cd1dba90a49) that is now all merged and ready to go. He's written a whole post about that which I strongly recommend you check out not just because stochastic weight averaging lets you get higher performance from your existing neural network with basically no extra work (it's as simple as adding two parameters to your fit function: `use_swa`, `swa_start`) but also he's described his process of building this and how he tested it and how he contributed to the library. So I think it's interesting if you are interested in doing something like this. I think William had not built this kind of library before so he describes how he did it.

### Train Phase [[00:02:01](https://youtu.be/xXXiC4YRGrQ?t=2m1s)]

[Notebook](https://nbviewer.jupyter.org/github/fastai/fastai/blob/master/courses/dl2/training_phase.ipynb)

Another very cool contribution to the fastai library is a new Train Phase API. And I'm going to do something I've never done before which is I'm going to present somebody else's notebook. The reason I haven't done it before is because I haven't liked any notebooks enough to think they are worth presenting it, but Sylvain has done a fantastic job here of not just creating this new API but also creating a beautiful notebook describing what it is and how it works and so forth. The background here is as you guys know **we've been trying to train networks faster**, partly as part of this DAWSBench competition and also for a reason that you'll learn about next week. I mentioned on the forum last week it would be really handy for our experiments if we had an easier way to try out different learning rate schedules etc, and I laid out an API that I had in mind as it'd be really cool if somebody could write this because I am going to bed now and I kind of need it by tomorrow. And Sylvain replied on the forum well that sounds like a good challenge and by 24 hours later, it was done and it's been super cool. I want to take you through it because it's going to allow you to research things that nobody has tried before.

It's called the `TrainPhase` API [00:03:32] and the easiest way to show it is to show an example of what it does. Here is an iteration against learning rate chart as you are familiar with seeing. This is one where we train for a while at the learning rate of 0.01 and then we train for a while at the learning rate of 0.001. I actually wanted to create something very much like that learning rate chart because most people that trained ImageNet use this stepwise approach and it's actually not something that's built into fastai because it's not generally something we recommend. But in order to replicate existing papers, I wanted to do it the same way. So rather than writing a number of fit, fit, fit calls with different learning rates, it would be nice to be able to say train for *n* epochs at this learning rate and then *m* epochs at that learning rate.

![](/images/lesson_13_004.png)

So here is how you do that:

```python
phases = [TrainingPhase(epochs=1, opt_fn=optim.SGD, lr=1e-2),
          TrainingPhase(epochs=2, opt_fn=optim.SGD, lr=1e-3)]
```

A phase is a period of training with particular optimizer parameters and `phases` consist of a number of training phase objects. A training phase object says how many epochs to train for, what optimization function to use, and what learning rate amongst other things that we will see. Here, you'll see the two training phases that you just saw on that graph. So now, rather than calling `learn.fit`, you say:

```python
learn.fit_opt_sched(phases)
```

In other words, `learn.fit` with an optimizer scheduler with these phases. From there, most of the things you pass in can just get sent across to the fit function as per usual, so most of the usual parameter will work fine. Generally speaking, we can just use these training phases and you will see it fits in a usual way. Then when you say `plot_lr` you will see the graphs above. Not only does it plot the learning rate, it also plots momentum, and for each phase, it tells you what optimizer it used. You can turn off the printing of the optimizers (`show_text=False`), you can turn off the printing of momentums (`show_moms=False`), and you can do other little things like a training phase could have a `lr_decay` parameter [00:05:47]:

```python
phases = [TrainingPhase(epochs=1, opt_fn=optim.SGD, lr=1e-2),
          TrainingPhase(epochs=1, opt_fn=optim.SGD, lr=(1e-2, 1e-3), lr_decay=DecayType.LINEAR),
          TrainingPhase(epochs=1, opt_fn=optim.SGD, lr=1e-3)]
```

So here is a fixed learning rate, then a linear decay learning rate, and then a fixed learning rate which gives up this picture:

```python
lr_i = start_lr + (end_lr - start_lr) * i/n
```

![](/images/lesson_13_005.png)

This might be quite a good way to train because we know at high learning rates, you get to explore better, and at low learning rates, you get to fine-tune better. And it's probably better to gradually slide between the two. So this actually isn't a bad approach, I suspect.

You can use other decay types such as cosine [00:06:25]:

```python
phases = [TrainingPhase(epochs=1, opt_fn=optim.SGD, lr=1e-2),
          TrainingPhase(epochs=1, opt_fn=optim.SGD, lr=(1e-2, 1e-3), lr_decay=DecayType.COSINE),
          TrainingPhase(epochs=1, opt_fn=optim.SGD, lr=1e-3)]
```

This probably makes even more sense as a genuinely potentially useful learning rate annealing shape.

```python
lr_i = end_lr + (start_lr - end_lr) / 2 * (1 + np.cos(i * np.pi) / n)
```

![](/images/lesson_13_006.png)

Exponential which is super popular approach:

```python
lr_i = start_lr * (end_lr / start_lr)**(i / n)
```

![](/images/lesson_13_007.png)

Polynomial which isn't terribly popular but actually in the literature works better than just about anything else, but seems to have been largely ignored. So polynomial is good to be aware of. And what Sylvain has done is he's given us the formula for each of these curves. So with a polynomial, you get to pick what polynomial to use. I believe `p` of `0.9` is the one I've seen really good results for — FYI.

```python
lr_i = end_lr + (start_lr - end_lr) * (1 - i / n)**p
```

![](/images/lesson_13_008.png)

If you don't give a tuple of learning rates when there is an LR decay, then it will decay all the way down to zero [00:07:26]. And as you can see, you can happily start the next cycle at a different point.

```python
phases = [TrainingPhase(epochs=1, opt_fn=optim.SGD, lr=1e-2),
          TrainingPhase(epochs=1, opt_fn=optim.SGD, lr=1e-2, lr_decay=DecayType.COSINE),
          TrainingPhase(epochs=1, opt_fn=optim.SGD, lr=1e-3)]
```

![](/images/lesson_13_009.png)

### SGDR [[00:07:43](https://youtu.be/xXXiC4YRGrQ?t=7m43s)]

So the cool thing is, now we can replicate all of our existing schedules using nothing but these training phases. So here is a function called `phases_sgdr` which does SGDR using the new training phase API.

```python
def phases_sgdr(lr, opt_fn, num_cycle, cycle_len, cycle_mult):
    phases = [TrainingPhase(epochs=cycle_len / 20, opt_fn=opt_fn, lr=lr / 100),
              TrainingPhase(epochs=cycle_len * 19 / 20,
                            opt_fn=opt_fn, lr=lr, lr_decay=DecayType.COSINE)]

    for i in range(1, num_cycle):
        phases.append(TrainingPhase(epochs=cycle_len * (cycle_mult**i), opt_fn=opt_fn, lr=lr,
                      lr_decay=DecayType.COSINE))
    return phases
```

So you can see, if he runs this schedule, here is what it looks like:

![](/images/lesson_13_010.png)

He's even done the little trick I have where you're training at really low learning rate just for a little bit and then pop up and do a few cycles, and the cycles are increasing in length [00:08:05]. And that's all done in a single function.

### 1cycle [[00:08:20](https://youtu.be/xXXiC4YRGrQ?t=8m20s)]

The new [1cycle policy](https://sgugger.github.io/the-1cycle-policy.html), we can now implement with, again, a single little function.

```python
def phases_1cycle(cycle_len, lr, div, pct, max_mom, min_mom):
    tri_cyc = (1 - pct / 100) * cycle_len

    return [TrainingPhase(epochs=tri_cyc/2, opt_fn=optim.SGD,
                          lr=(lr/div, lr), lr_decay=DecayType.LINEAR,
                          momentum=(max_mom, min_mom),
                          momentum_decay=DecayType.LINEAR),
           TrainingPhase(epochs=tri_cyc/2, opt_fn=optim.SGD,
                         lr=(lr, lr/div), lr_decay=DecayType.LINEAR,
                          momentum=(min_mom, max_mom),
                          momentum_decay=DecayType.LINEAR),
           TrainingPhase(epochs=cycle_len-tri_cyc, opt_fn=optim.SGD,
                         lr=(lr/div, lr/(100*div)),
                         lr_decay=DecayType.LINEAR,
                         momentum=max_mom)]
```

So if we fit with that, we get this triangle followed by a little flatter bit and the momentum is a cool thing — the momentum has a momentum decay. And in the third `TrainingPhase`, we have a fixed momentum. So it's doing the momentum and the learning rate at the same time.

![](/images/lesson_13_011.png)

### Discriminative learning rates + 1cycle [[00:08:53](https://youtu.be/xXXiC4YRGrQ?t=8m53s)]

So something that I haven't tried yet, but I think would be really interesting is to use the combination of discriminative learning rates and 1cycle. No one has tried yet. So that would be really interesting. The only paper I've come across which has discriminative learning rate uses something called [LARS](https://arxiv.org/abs/1708.03888). It was used to train ImageNet with very very large batch sizes by looking at the ratio between the gradient and the mean at each layer and using that to change the learning rate of each layer automatically. They found that they could use much larger batch sizes. That's the only other place I've seen this kind of approach used, but there's lots of interesting things you could try with combining discriminative learning rates and different interesting schedules.

### Customize LR finder / Your own LR finder [[00:10:06](https://youtu.be/xXXiC4YRGrQ?t=10m6s)]

You can now write your own LR finder of different types, specifically because there is now this `stop_div` parameter which basically means that it'll use whatever schedule you asked for but when the loss gets too bad, it'll stop training.

One useful thing that's been added is the `linear` parameter to the `plot` function. If you use linear schedule rather than an exponential schedule in your learning rate finder which is a good idea if you fine-tuned into roughly the right area, then you can use linear to find exactly the right area. Then you probably want to plot it with a linear scale. So that's why you can also pass linear to plot now as well.

You can change the optimizer each phase [00:11:06]. That's more important than you might imagine because actually **the current state-of-the-art for training on really large batch sizes really quickly for ImageNet actually starts with RMSProp for the first bit, then they switch to SGD for the second bit**. So that could be something interesting to experiment more with because at least one paper has now shown that that can work well. :bookmark: Again, it's something that isn't well appreciated as yet.

### Changing data [[00:11:49](https://youtu.be/xXXiC4YRGrQ?t=11m49s)]

Then the bit I find most interesting is you can change your data. Why would we want to change our data? Because you remember from lesson 1 and 2, you could **use small images at the start and bigger images later**. The theory is that you could use that to train the first bit more quickly with smaller images, and remember if you halve the height and halve the width, you've got the quarter of the activations every layer, so it can be a lot faster. It might even generalize better. So you can now create a couple of different sizes, for example, here, we got 28 and 32 sized images. This is CIFAR10 so there's only so much you can do. Then if you pass in an array of data in this `data_list` parameter when you call `fit_opt_sched`, it'll use different dataset for each phase.

```python
data1 = get_data(28, batch_size)
data2 = get_data(32, batch_size)
learn = ConvLearner.from_model_data(ShallowConvNet(), data1)
phases = [TrainingPhase(epochs=1, opt_fn=optim.Adam, lr=1e-2,
                        lr_decay=DecayType.COSINE),
          TrainingPhase(epochs=2, opt_fn=optim.Adam, lr=1e-2,
                        lr_decay=DecayType.COSINE)]
learn.fit_opt_sched(phases, data_list=[data1, data2])
```

:bookmark: *note-to-self: run experiments in `train_phase.ipyn` notebook with my own dataset.*

#### DAWNBench competition for ImageNet

That's really cool because we can use that now like we could use that in our DAWNBench entries and see what happens when we actually increase the size with very little code. So what happens when we do that [00:13:02]? The answer is here in DAWNBench training on ImageNet:

![](/images/lesson_13_012.png)

You can see here that Google has won this with half an hour on a cluster of TPUs. The best non-cluster of TPU result is fast.ai + students under 3 hours beating out Intel on 128 computers, where else, we ran on a single computer. We also beat Google running on a TPU so using this approach, we've shown:

- the fastest GPU result
- the fastest single machine result
- the fastest publicly available infrastructure result

These TPU pods, you can't use unless you're Google. Also the cost is tiny ($72.54), this Intel one costs them $1,200 worth of compute — they haven't even written it here, but that's what you get if you use 128 computers in parallel each one with 36 cores, each one with 140G compare to our single AWS instance. So **this is kind of a breakthrough in what we can do**. The idea that we can train ImageNet on a single publicly available machine and this is $72, by the way, it was actually $25 because we used a spot instance. One of our students Andrew Shaw built this whole system to allow us to throw a whole bunch of spot instance experiments up and run them simultaneously and pretty much automatically, but DAWNBench doesn't quote the actual number we used. So *it's actually $25, not $72*. So this `data_list` idea is super important and helpful.

#### CIFAR10 result on DAWNBench [[00:15:15](https://youtu.be/xXXiC4YRGrQ?t=15m15s)]

Our CIFAR10 results are also now up there officially and you might remember the previous best was a bit over an hour. The trick here was using 1cycle, so all of this stuff that's in Sylvain's training phase API is really all the stuff that we used to get these top results. And another fast.ai student who goes by the name *bkj* has taken that and done his own version, he took a Resnet18 and added the concat pooling that you might remember that we learnt about on top, and used Leslie Smith's 1cycle and so he's got on the leaderboard. So all the top 3 are fast.ai students which wonderful.

![](/images/lesson_13_013.png)

#### CIFAR10 cost result [[00:16:05](https://youtu.be/xXXiC4YRGrQ?t=16m5s)]

Same for cost — the top 3 and you can see, Paperspace. Brett ran this on Paperspace and got the cheapest result just ahead of *bkj*.

![](/images/lesson_13_014.png)

So I think you can see [00:16:25], a lot of the interesting opportunities at the moment for the training stuff more quickly and cheaply are all about learning rate annealing, size annealing, and training with different parameters at different times, and I still think everybody is scratching the surface. I think we can go a lot faster and a lot cheaper. That's really helpful for people in resource constrained environment which is basically everybody except Google, maybe Facebook.

### Conv Architecture Gap - Inception ResNet [[00:17:00](https://youtu.be/xXXiC4YRGrQ?t=17m)]

Architectures are interesting as well though, and one of the things we looked at last week was creating a simpler version of DarkNet architecture. But there's a piece of architecture we haven't talk about which is necessary to understand the [Inception network](https://arxiv.org/abs/1409.4842v1). The Inception network is actually pretty interesting because they use some tricks to make things more efficient. We are not currently using these tricks and I feel that maybe we should try it. The most interesting and most successful Inception network is their [Inception-ResNet-v2](https://ai.googleblog.com/2016/08/improving-inception-and-image.html) network and most of the blocks in that looks something like this:

![](/images/lesson_13_015.png)

It looks a lot like a standard ResNet block in that there's an identity connection, and there's a conv path, and we add them up together [00:17:47]. But it's not quite that. The first is the middle conv path is a 1x1 conv, and it's worth thinking about what a 1x1 conv actually is.

#### 1x1 convolution [[00:18:23](https://youtu.be/xXXiC4YRGrQ?t=18m23s)]

1x1 conv is simply saying for each grid cell in your input, you've got basically a vector. 1 by 1 by number of filters tensor is basically a vector. For each grid cell in your input, you're just doing a dot product with that tensor. Then of course, it's going to be one of those vectors for each of the 192 activations we are creating. So basically do 192 dot products with grid cell (1, 1) and then 192 with grid cell (1, 2) or (1, 3) and so forth. So you will end up with something which has the same grid size as the input and 192 channels in the output. So that's a really **good way to either reduce the dimensionality or increase the dimensionality of an input without changing the grid size**. That's normally what we use 1x1 convs for. Here, we have a 1x1 conv and another 1x1 conv, and then they add it together. Then there is a third path and this third path is not added. It is not explicitly mentioned but this third path is concatenated. There is a form of ResNet which is basically identical to ResNet but we don't do plus, we do concat. That's called a DenseNet. It's just a ResNet where we **do concat instead of plus**. That's an interesting approach because then the kind of the **identity path** is literally being copied. So you get that flow all the way through and so as we'll see next week, that tends to be **good for segmentation and stuff like that where you really want to keep the original pixels**, the first layer of pixels, and the second layer of pixels untouched.

#### Concat in Inception networks

Concatenating rather than adding branches is a very useful thing to do and we are concatenating the middle branch and the right right branch [00:20:22]. The right most branch is doing something interesting, which is, it's doing, first of all, the 1x1 conv, and then a 1x7, and then 7x1. What's going on there? So, what's going on there is basically what we really want to do is do 7x7 conv. The reason we want to do 7x7 conv is that if you have multiple paths (each of which has different kernel sizes), then it's able to look at different amounts of the image. The original Inception network had 1x1, 3x3, 5x5, 7x7 getting concatenated together or something like that. So if we can have a **7x7 filter, then we get to look at a lot of the image at once and create a really rich representation**. So the stem of the Inception network that is the first few layers of the Inception network actually also used this kind fo 7x7 conv because you start out with this 224 by 224 by 3, and you want to turn it into something that's 112 by 112 by 64. **By using a 7x7 conv, you can get a lot of information in each one of those outputs to get those 64 filters**. But the problem is that 7x7 conv is a lot of work. You've got 49 kernel values to multiply by 49 inputs for every input pixel across every channel. So the **compute is crazy**. You can kind of get away with it (maybe) for the very first layer, and in fact, the very first conv of ResNet is a 7x7 conv.

#### Basic idea of Inception networks

But not so for Inception [00:22:30]. They don't do a 7x7 conv, instead, they do a 1x7 followed by 7x1. So to explain, the basic idea of the Inception networks or all the different versions of it that you have a number of separate paths which have different convolution widths. In this case, conceptually the idea is the middle path is 1x1 convolution width, and the right path is going to be a 7 convolution width, so **they are looking at different amount of data and then we combine them together**. But we don't want to have a 7x7 conv through out the network because it's just too computationally expensive.

#### Instead of *A x A* use *A x 1* followed by *1 x A* - Lower rank approximation

But if you think about it [00:23:18], if we've got some input coming in and we have some big filter that we want and it's too big to deal with. What could we do? Let's do 5x5. What we can do is to create two filters — one which is 1x5, one which is 5x1. We take our activations of the previous layer, and we put it through the 1x5. We take the activations out of that, and put it through the 5x1, and something comes out the other end. Now what comes out the other end? Rather than thinking of it as, first of all, we take the activations, then we put it through the 1x5 then we put it through the 5x1, what if instead we think of these two operations together and say what is a 5x1 dot product and a 1x5 dot product do together? Effectively, you could take a 1x5 and 5x1 and the outer product of that is going to give you a 5x5. Now you can't create any possible 5x5 matrix by taking that product, but there's a lot of 5x5 matrices that you can create. So the basic idea here is when you think about the order of operations (if you are interested in more of the theory here, you should check out Rachel's numerical linear algebra course which is basically a whole course about this). **But conceptually, the idea is that very often the computation you want to do is actually more simple than an entire 5x5 convolution. Very often, the term we use in linear algebra is that there's some lower rank approximation.** In other words, that the 1x5 and the 5x1 combined together — that 5x5 matrix is nearly as good as the 5x5 matrix you ideally would have computed if you were able to. So this is very often the case in practice — just because the nature of the real world is that the real world tends to have more structure than randomness.

The cool thing is [00:26:16], if we replace our 7x7 conv with a 1x7 and 7x1, for each cell (grid), it has 14 by input channel by output channel dot products to do, whereas 7x7 one has 49 to do. So it's going to be a lot faster and we have to hope that it's going to be nearly as good. It's certainly capturing as much width of information by definition.

![](/images/lesson_13_016.png)

#### Factored convolutions

If you are interested in learning more about this, specifically in a deep learning area, you can google for **[Factored Convolutions](https://medium.com/@lisulimowicz/dilated-convolutions-and-kronecker-factored-convolutions-b42ed58b2bc7)**. The idea was come up with 3 or 4 years ago now. It's probably been around for longer, but that was when I first saw it. It turned out to work really well and the Inception network uses it quite widely.

#### Stem in backbone

They actually use it in their stem. We've talked before about how we tend to add-on — we tend to say this is main backbone when we have ResNet34, for example. This is main backbone which is all of the convolutions, and then we can add on to it a custom head that tends to be a max pooling or a fully connected layer. It's better to talk about the backbone is containing two pieces: one is the stem and the other is the main backbone. The reason is that the thing that's coming in has only 3 channels, so we want some sequence of operations which is going to expand that out into something richer — generally something like 64 channels.

![](/images/lesson_13_017.png)

In ResNet, the stem is super simple. It's a 7x7 stride 2 conv followed by a stride 2 max pool (I think that's it if memory serves correctly). Inception have a much more complex stem with multiple paths getting combined and concatenated including factored conv (1x7 and 7x1). I'm interested in what would happen if you stacked a standard ResNet on top of an Inception stem, for instance. I think that would be a really interesting thing to try because an Inception stem is quite a carefully engineered thing, and this thing of how you take 3 channel input and turn it into something richer seems really important. And all of that work seems to have gotten thrown away for ResNet. We like ResNet, it works really well. But what if we put a dense net backbone on top of an Inception stem? Or what if we replaced the 7x7 conv with a 1x7 and 7x1 factored conv in standard ResNet? There are lots of things we could try and I think it would be really interesting. :bookmark: So there's some more thoughts about potential research directions.

---

### Image enhancement paper - Progressive GANs

So that was kind of my little bunch of random stuff section [00:29:51]. Moving a little bit closer to the actual main topic of this which is image enhancement. I'm going to talk about a new paper briefly because it really connects what I just discussed with what we are going to discuss next. It's a paper on progressive GANS which came from Nvidia: [Progressive Growing of GANS for Improved Quality, Stability, and Variation](http://research.nvidia.com/publication/2017-10_Progressive-Growing-of). Progressive GANS takes this idea of gradually increasing the image size. It's the only other direction I am aware of that people have actually gradually increase the image size. It surprises me because this paper is actually very popular, well known, and well liked and yet, people haven't taken the basic idea of gradually increasing the image size and use it anywhere else which shows you the general level of creativity you can expect to find in the deep learning research community, perhaps.

#### Progressive GAN - increase image size

![](/images/lesson_13_018.png)

They really go back and they start with 4x4 GAN [00:31:47]. Literally, they are trying to replicate 4x4 pixel, and then 8x8 (the upper left ones above). This is the CelebA dataset so we are trying to recreate pictures of celebrities. Then they go 16x16, 32, 64, 128, then 256. One of the really nifty things they do is that as they increase the size, they also add more layers to the network. Which kind of makes sense because if you are doing more of a ResNet-y type thing, then you are spitting out something which hopefully makes sense at each grid cell size, so you should be able to layer stuff on top. They do another nifty thing where they add a skip connection when they do that, and they gradually change the linear interpolation parameter that moves it more and more away from the old 4x4 network and towards the new 8x8 network. Then once this totally moved it across, they throw away that extra connection. The details don't matter too much but it uses the basic ideas we've talked about, gradually increasing the image size and skip connections. It's a great paper to study because it is one of these rare things where good engineers actually built something that just works in a really sensible way. Now it's not surprising this actually comes from Nvidia themselves. Nvidia don't do a lot of papers and it's interesting that when they do, they build something that is so throughly practical and sensible. :bookmark: So I think **it's a great paper to study if you want to put together lots of the different things we've learned and there aren't many re-implementation of this so it's an interesting thing to project**, and maybe you could build on and find something else.

#### High-res GAN

Here is what happens next [00:33:45]. We eventually go up to 1024x1024, and you'll see that the images are not only getting higher resolution but they are getting better. So I am going to see if you can guess which one of the following is fake:

![](/images/lesson_13_019.png)

They are all fake. That's the next stage. You go up up up up and them BOOM. So GANS and stuff are getting crazy and some of you may have seen this during the week [00:34:16]. This video just came out and it's a speech by Barack Obama and let's check it out:

[![Fake Obama video](https://img.youtube.com/vi/cQ54GDm1eL0/0.jpg)](https://www.youtube.com/watch?v=cQ54GDm1eL0 "Fake Obama video")

As you can see, they've used this kind of technology to literally move Obama's face in the way that Jordan Peele's face was moving. You basically have all the techniques you need now to do that. Is that a good idea?

---

### Ethics in AI [[00:35:31](https://youtu.be/xXXiC4YRGrQ?t=35m31s)]

This is the bit where we talk about what's most important which is now that we can do all this stuff, what should we be doing and how do we think about that? The TL;DR version is I actually don't know. Recently a lot of you saw the founders of the spaCy prodigy folks down at the Explosion AI did a talk, Matthew and Ines, and I went to dinner with them afterwards, and we basically spent the entire evening talking, debating, arguing about what does it mean the companies like ours are building tools that are democratizing access to tools that can be used in harmful ways. They are incredibly thoughtful people and we, I wouldn't say we didn't agree, we just couldn't come to a conclusion ourselves. So I'm just going to lay out some of the questions and point to some of the research, and when I say research, most of the actual literature review and putting this together was done by Rachel, so thanks Rachel.

Let me start by saying the models we build are often pretty crappy in ways which are not immediately apparent [00:36:52]. You won't know how crappy they are unless the people that are building them with you are a range of people and the people that are using them with you are a range of people. For example, a couple of wonderful researchers, [Timnit Gebru](https://twitter.com/timnitGebru) is at Microsoft and [Joy Buolamwini](https://twitter.com/jovialjoy) just finished PhD from MIT, they did this really interesting research where they looked at some off-the-shelf face recognizers, one from FACE++ which is a huge Chinese company, IBM's, and Microsoft's, and they looked for a range of different face types.

![](/images/lesson_13_020.png)

Generally speaking, Microsoft one in particular was incredibly accurate unless the face type happened to be dark-skinned when suddenly it went 25 times worse. IBM got it wrong nearly half the time. For a big company like this to release a product that, for large percentage of the world, doesn't work is more than a technical failure. It's a really deep failure of understanding what kind of team needs to be used to create such a technology and to test such a technology or even an understanding of who your customers are. Some of your customers have dark skin. "I was also going to add that the classifiers all did worse on women than on men" (Rachel). Shocking. It's funny that Rachel tweeted about something like this the other day, and some guy said "What's this all about? What are you saying? Don't you know people made cars for a long time — are you saying you need women to make cars too?" And Rachel pointed out — well actually yes. For most of the history of car safety, women in cars have been far more at risk of death than men in cars because the men created male looking, feeling, sized crash test dummies, so car safety was literally not tested on women size bodies. Crappy product management with a total failure of diversity and understanding is not new to our field.

"I was just going to say that was comparing impacts of similar strength for men and women " (Rachel). I don't know why whenever you say something like this on Twitter, Rachel has to say this because anytime you say something like this on Twitter, there's about 10 people who'll say "oh, you have to compare all these other things" as if we didn't know that.

![](/images/lesson_13_021.png)

Other things our very best most famous systems do like Microsoft's face recognizer or Google's language translator, you turn "She is a doctor. He is a nurse." into Turkish and quite correctly — both pronouns become O because there is no gendered pronouns in Turkish. Go the other direction, what does it get turned into? "He is a doctor. She is a nurse." So we've got these kind of biases built into tools that we are all using every day. And again, people say "oh, it's just showing us what's in the world" and okay, there's lots of problems with that basic assertion, but as you know, machine learning algorithms love to generalize.

![](/images/lesson_13_022.png)

So because they love to generalize, this is one fo the cool things about you guys knowing the technical details now, because they love to generalize when you see something like 60% of people cooking are women in the pictures they used to build this model and then you run the model on a separate set of pictures, then 84% of the people they choose as cooking are women rather than the correct 67%. Which is a really understandable thing for an algorithm to do as it took a biased input and created a more biased output because for this particular loss function, that's where it ended up. This is a really common kind of model amplification.

This stuff matters [00:41:41]. It matters in ways more than just awkward translations or black people's photos not being classified correctly. Maybe there's some wins too as well — like horrifying surveillance everywhere and maybe won't work on black people. "Or it'll be even worse because it's horrifying surveillance and it's flat-out racist and wrong" (Rachel). But let's go deeper. For all we say about human failings, there is a long history of civilization and societies creating layers of human judgement which avoid, hopefully, the most horrible things happening. And sometimes companies which love technology think "let's throw away humans and replace them with technology" like Facebook did. A couple years ago, Facebook literally got rid of their human editors, and this was in the news at the time. And they were replaced with algorithms. So now as algorithms put all the stuff on your news feed and human editors were out of the loop. What happened next?

![](/images/lesson_13_023.png)

Many things happened next. One of which was a massive horrifying genocide in Myanmar. Babies getting torn out of their mothers arms and thrown into fires. Mass rape, murder, and an entire people exiled from their homeland.

![](/images/lesson_13_024.png)

Okay, I'm not gonna say that was because Facebook did this, but what I will say is that when the leaders of this horrifying project are interviewed, they regularly talk about how everything they learnt about the disgusting animal behaviors of Rohingyas that need to be thrown off the earth, they learnt from Facebook. Because the algorithms just want to feed you more stuff that gets you clicking. If you get told these people that don't look like you and you don't know the bad people and here's lots of stories about bad people and then you start clicking on them and then they feed you more of those things. Next thing you know, you have this extraordinary cycle. People have been studying this, so for example, we've been told a few times people click on our fast.ai videos and then the next thing recommended to them is like conspiracy theory videos from Alex Jones, and then continues from there. Because humans click on things that shock us, surprise us, and horrify us. At so many levels, this decision has had extraordinary consequences which we're only beginning to understand. Again, this is not to say this particular consequence is because of this one thing, but to say it's entirely unrelated would be clearly ignoring all of the evidence and information that we have.

#### Unintended consequences [[00:45:04](https://youtu.be/xXXiC4YRGrQ?t=45m4s)]

![](/images/lesson_13_025.png)

The key takeaway is to think what are you building and how could it be used. Lots and lots of effort now being put into face detection including in our course. We've been spending a lot of time thinking about how to recognize stuff and where it is. There's lots of good reasons to want to be good at that for improving crop yields in agriculture, for improving diagnostic and treatment planning in medicine, for improving your LEGO sorting robot system, etc. But it's also being widely used in surveillance, propaganda, and disinformation. Again, the question is what do I do about that? I don't exactly know. But it's definitely at least important to be thinking about it, talking about it.

#### Runaway feedback loops [[00:46:10](https://youtu.be/xXXiC4YRGrQ?t=46m10s)]

![](/images/lesson_13_026.png)

Sometimes you can do really good things. For example, meetup.com did something which I would put in the category of really good thing which is they recognized early a potential problem which is that more men are tending to go to their meet ups. And that was causing their collaborative filtering systems, which you are familiar building now to recommend more technical content to men. And that was causing more men to go to more technical content which was causing the recommendation system to suggest more technical content to men. This kind of runaway feedback loop is extremely common when we interface the algorithm and the human together. So what did Meetup do? They intentionally made the decision to recommend more technical content to women, not because highfalutin idea about how the world should be, but just because that makes sense. Runaway feedback loop was a bug — there are women that want to go to tech meetups, but when you turn up for a tech meet up and it's all men and you don't go, then it recommends more to men and so on and so forth. So Meetup made a really strong product management decision here which was to not do what the algorithm said to do. Unfortunately this is rare. Most of these runaway feedback loops, for example, in predictive policing where algorithms tell policemen where to go which very often is more black neighborhoods which end up crawling with more policemen which leads to more arrests which is assisting to tell more policemen to go to more black neighborhoods and so forth.

### Bias in AI [[00:48:09](https://youtu.be/xXXiC4YRGrQ?t=48m9s)]

![](/images/lesson_13_027.png)

This problem of algorithmic bias is now very wide spread and as algorithms become more and more widely used for specific policy decisions, judicial decisions, day-to-day decisions about who to give what offer to, this just keeps becoming a bigger problem. Some of them are really things that the people involved in the product management decision should have seen at the very start, didn't make sense, and unreasonable under any definition of the term. For example, this stuff Abe Gong pointed out — these were questions that were used for both pretrial so who was required to post bail, so these are people that haven't even been convicted, as well as for sentencing and for who gets parole. This was upheld by the Wisconsin Supreme Court last year despite all the flaws. So whether you have to stay in jail because you can't pay the bail and how long your sentence is for, and how long you stay in jail for depends on what your father did, whether your parents stayed married, who your friends are, and where you live. Now turns out these algorithms are actually terribly terribly bad so some recent analysis showed that they are basically worse than chance. But even if the company's building them were confident on these were statistically accurate correlations, does anybody imagine there's a world where it makes sense to decide what happens to you based on what your dad did?

A lot of this stuff at the basic level is obviously unreasonable and a lot of it just fails in these ways that you can see empirically that these kind of runaway feedback loops must have happened and these over generalizations must have happened. For example, these are the cross tabs that anybody working in any field using these algorithm should be preparing. So prediction of likelihood of reoffending for black vs. white defendants, we can just calculate this very simply. Of the people that were labeled high-risk but didn't reoffend — they were 23.5% white but about twice that African American. Where else, those that were labeled lower risk but did reoffend was half the white people and only 28% of the African American. This is the kind of stuff where at least if you are taking the technologies we've been talking about and putting the production in any way, building an API for other people, providing training for people, or whatever — then at least make sure that what you are doing can be tracked in a way that people know what's going on so at least they are informed. I think it's a mistake in my opinion to assume that people are evil and trying to break society. I think I would prefer to start with an assumption of if people are doing dumb stuff, it's because they don't know better. So at least make sure they have this information. I find very few ML practitioners thinking about what is the information they should be presenting in their interface. Then often I'll talk to data scientists who will say "oh, the stuff I'm working on doesn't have a societal impact." Really? A number of people who think that what they are doing is entirely pointless? Come on. People are paying you to do it for a reason. It's going to impact people in some way. So think about what that is.

### Responsibility in hiring [[00:52:46](https://youtu.be/xXXiC4YRGrQ?t=52m46s)]

![](/images/lesson_13_028.png)

The other thing I know is a lot of people involved here are hiring people and if you are hiring people, I guess you are all very familiar with the fast.ai philosophy now which is the basic premise that, and I thin it comes back to this idea that I don't think people on the whole are evil, I think they need to be informed and have tools. So we are trying to give as many people the tools as possible that they need and particularly we are trying to put those tools in the hands of a more diverse range of people. So if you are involved in hiring decisions, perhaps you can keep this kind of philosophy in mind as well. If you are not just hiring a wider range of people, but also promoting a wider range of people, and providing appropriate career management for a wider range of people, apart from anything else, your company will do better. It actually turns out that more diverse teams are more creative and tend to solve problems more quickly and better than less diverse teams, but also you might avoid these kind of awful screw-ups which, at one level, are bad for the world and another level if you ever get found out, they can destroy your company.

#### IBM & "Death's Calculator" [[00:54:08](https://youtu.be/xXXiC4YRGrQ?t=54m8s)]

![](/images/lesson_13_029.png)

Also they can destroy you or at least make you look pretty bad in history. A couple of examples, one is going right back to the second world war. IBM provided all of the infrastructure necessary to track the Holocaust. These are the forms they used and they had different code — Jews were 8, Gypsies were 12, death in the gas chambers was 6, and they all went on these punch cards. You can go and look at these punch cards in museums now and this has actually been reviewed by a Swiss judge who said that IBM's technical assistance facilitated the task of the Nazis and the commission their crimes against humanity. It is interesting to read back the history from these times to see what was going through the minds of people at IBM at that time. What was clearly going through the minds was the opportunity to show technical superiority, the opportunity to test out their new systems, and of course the extraordinary amount of money that they were making. When you do something which at some point down the line turns out to be a problem, even if you were told to do it, that can turn out to be a problem for you personally. For example, you all remember the diesel emission scandal in VW. Who is the one guy that went to jail? It was the engineer just doing his job. If all of this stuff about actually not messing up the world isn't enough to convince you, it can mess up your life too. If you do something that turns out to cause problems even though somebody told you to do it, you can absolutely be held criminally responsible. Aleksandr Kogan was the guy that handed over the Cambridge Analytica data. He is a Cambridge academic. Now a very famous Cambridge academic the world over for doing his part to destroy the foundations of democracy. This is not how we want to go down in history.

![](/images/lesson_13_030.png)

:question: In one of your tweets, you said dropout is patented [00:56:50]. I think this is about WaveNet patent from Google. What does it mean? Can you please share more insight on this subject? Does it mean that we will have to pay to use dropout in the future?

One of the patent holders is Geoffrey Hinton. So what? Isn't that great? Invention is all about patents, blah blah. My answer is no. Patents have gone wildly crazy. The amount of things that are patentable that we talk about every week would be dozens. It's so easy to come up with a little tweak and then if you turn that into a patent to stop everybody from using that little tweak for the next 14 years and you end up with a situation we have now where everything is patented in 50 different ways. Then you get these patent trolls who have made a very good business out of buying lots of crappy little patents and then suing anybody who accidentally turned out did that thing like putting rounded corners on buttons. So what does it mean for us that a lot of stuff is patented in deep learning? I don't know.

One of the main people doing this is Google and people from Google who replied to this patent tend to assume that Google doing it because they want to have it defensively so if somebody sues them, they can say don't sue us we'll sue you back because we have all these patents. The problem is that as far as I know, they haven't signed what's called a defensive patent pledge so basically you can sign a legally binding document that says our patent portfolio will only be used in defense and not offense. Even if you believe all the management of Google would never turn into a patent troll, you've got to remember that management changes. To give you a specific example I know, the somewhat recent CFO of Google has a much more aggressive stance towards the PNL, I don't know, maybe she might decide that they should start monetizing their patents or maybe the group that made that patent might get spun off and then sold to another company that might end up in private equity hands and decide to monetize the patents or whatever. So I think it's a problem. There has been a big shift legally recently away from software patents actually having any legal standing, so it's possible that these will all end up thrown out of court but the reality is that anything but a big company is unlikely to have the financial ability to defend themselves against one of these huge patent trolls.

You can't avoid using patented stuff if you write code. I wouldn't be surprised if most lines of code you write have patents on them. Actually funnily enough, the best thing to do is not to study the patents because if you do and you infringe knowingly then the penalties are worse. So the best thing to do is to put your hands in your ear, sing a song, and get back to work. So the thing about dropouts patented, forget I said that. You don't know that. You skipped that bit.

---

### Style Transfer [[01:01:28](https://youtu.be/xXXiC4YRGrQ?t=1h1m28s)]

[Notebook](https://nbviewer.jupyter.org/github/fastai/fastai/blob/master/courses/dl2/style-transfer.ipynb)









