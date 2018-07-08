# Lesson 12 - DarkNet; Generative Adversarial Networks (GANs)

_These are my personal notes from fast.ai course and will continue to be updated and improved if I find anything useful and relevant while I continue to review the course to study much more in-depth. Thanks for reading and happy learning!_

## Topics

* Deep dive into the DarkNet architecture used in YOLOv3.
  * Use it to better understand all the details and choices that you can make when implementing a ResNet-ish architecture.
* The basic approach discussed here is what we used to win the DAWNBench competition!
* Generative Adversarial Networks (GANs).
  * At its heart, a different kind of loss function.
  * Generator and a discriminator that battle it out, and in the process combine to create a generative model that can create highly realistic outputs.
  * Wasserstein GAN variant.
    * Easier to train and more resilient to a range of hyperparameters.

## Lesson Resources

* [Website](http://course.fast.ai/lessons/lesson12.html)
* [Video](https://youtu.be/ondivPiwQho)
* [Wiki](http://forums.fast.ai/t/part-2-lesson-12-wiki)
* Jupyter Notebook and code
  * [cifar10-darknet.ipynb](https://nbviewer.jupyter.org/github/fastai/fastai/blob/master/courses/dl2/cifar10-darknet.ipynb)
  * [wgan.ipynb](https://nbviewer.jupyter.org/github/fastai/fastai/blob/master/courses/dl2/wgan.ipynb)
  * [cyclegan.ipynb](https://nbviewer.jupyter.org/github/fastai/fastai/blob/master/courses/dl2/cyclegan.ipynb)
* Dataset
  * [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) / [direct download link](http://files.fast.ai/data/cifar10.tgz) (161 MB)

## Assignments

### Papers

* Must read
  * [Wide Residual Networks](https://arxiv.org/abs/1605.07146) by Sergey Zagoruyko, et. al
  * [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf) by Joseph Redmon, et. al
  * [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (DCGAN)](https://arxiv.org/abs/1511.06434) by Alec Radford, et. al
  * [Wasserstein GAN (WGAN)](https://arxiv.org/abs/1701.07875) by Martin Arjovsky, et. al
  * [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN)](https://junyanz.github.io/CycleGAN/) by Jun-Yan Zhu, et. al
  * [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285) by Vincent Dumoulin, et. al
  * [Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/) by Augustus Odena, et. al
* Additional papers \(optional\)
  * [Multimodal Unsupervised Image-to-Image Translation](https://arxiv.org/abs/1804.04732) by Xun Huang, et. al

### Other Resources

#### Other Useful Information

* [PyTorch implementation of CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

## My Notes

### Generative Adversarial Networks (GANs)

Very hot technology but definitely deserving to be in the cutting edge deep learning part of the course because they are not quite proven to be necessarily useful for anything but they are nearly there and will definitely get there. We are going to focus on the things where they are definitely going to be useful in practice and there is a number of areas where they may turn out to be useful but we don’t know yet. So I think the area that they are definitely going to be useful in practice is the kind of thing you see on the left of the slide — which is for example turning drawing into rendered pictures. This comes from [a paper that just came out 2 days ago](https://arxiv.org/abs/1804.04732), so there’s a very active research going on right now.

From the last lecture [00:01:04]: One of our diversity fellows Christine Payne has a master’s in medicine from Stanford and so she had an interest in thinking what it would look like if we built a language model of medicine. One of the things we briefly touched on back in lesson 4 but didn’t really talk much about last time is this idea that you can actually seed a generative language model which means you’ve trained a language model on some corpus and then you are going to generate some text from that language model. You can start off by feeding it a few words to say "here is the first few words to create the hidden state in the language model and generate from there please. Christine did something clever which was to seed it with a question and repeat the question three times and let it generate from there. She fed a language model lots of different medical texts and fed in questions as you see below:

![](/images/lesson_12_001.png)

What Jeremy found interesting about this is it’s pretty close to being a believable answer to the question for people without master’s in medicine. But it has no bearing on reality whatsoever. He thinks it is an interesting kind of ethical and user experience quandary. Jeremy is involved in a company called doc.ai that’s trying to doing a number of things but in the end provide an app for doctors and patients which can help create a conversational user interface around helping them with their medical issues. He’s been continually saying to the software engineers on that team please **don’t try to create a generative model using LSTM or something because they are going to be really good at creating bad advice that sounds impressive** — kind of like political pundits or tenured professor who can say bullcrap with great authority. So he thought it was really interesting experiment. If you’ve done some interesting experiments, share them in the forum, blog, Twitter. Let people know about it and get noticed by awesome people.

### CIFAR10 [[00:05:26](https://youtu.be/ondivPiwQho?t=5m26s)]

Let’s talk about CIFAR10 and the reason is that we are going to be looking at some more bare-bones PyTorch stuff today to build these generative adversarial models. There is no fastai support to speak up at all for GANs at the moment — there will be soon enough but currently there isn’t so we are going to be building a lot of models from scratch. It’s been a while since we’ve done much serious model building. We looked at CIFAR10 in the part 1 of the course and we built something which was getting about 85% accuracy and took a couple hours to train. Interestingly, there is a competition going on now to see who can actually train CIFAR10 the fastest ([DAWN](https://dawn.cs.stanford.edu/benchmark/#cifar10-train-time)), and the goal is to get it to train to 94% accuracy. It would be interesting to see if we can build an architecture that can get to 94% accuracy because that is a lot better than our previous attempt. Hopefully in doing so we will learn something about creating good architectures, that will be then useful for looking at GANs today. Also it is useful because Jeremy has been looking much more deeply into the last few years’ papers about different kinds of CNN architectures and realizes that a lot of the insights in those papers are not being widely leveraged and clearly not widely understood. So he wants to show you what happens if we can leverage some of that understanding.

#### [cifar10-darknet.ipynb](https://nbviewer.jupyter.org/github/fastai/fastai/blob/master/courses/dl2/cifar10-darknet.ipynb) [[00:07:17](https://youtu.be/ondivPiwQho?t=7m17s)]

The notebook is called [Darknet](https://pjreddie.com/darknet/) because the particular architecture we are going to look at is very close to the Darknet architecture. But you will see in the process that the Darknet architecture as in not the whole YOLO v3 end-to-end thing but just the part of it that they pre-trained on ImageNet to do classification. It’s almost like the most generic simple architecture you could come up with, so it’s a really great starting point for experiments. So we will call it "Darknet" but it’s not quite that and you can fiddle around with it to create things that definitely aren’t Darknet. It’s really just the basis of nearly any modern ResNet based architecture.

CIFAR10 is a fairly small dataset [00:08:06]. The images are only 32 by 32 in size, and it’s a great dataset to work with because:

- You can train it relatively quickly unlike ImageNet
- A relatively small amount of data
- Actually quite hard to recognize the images because 32 by 32 is too small to easily see what’s going on.

It is an under-appreciated dataset because it’s old. Who wants to work with small old dataset when they could use their entire server room to process something much bigger. But it’s is a really great dataset to focus on.

Go ahead and import our usual stuff and we are going to try and build a network from scratch to train this with [00:08:58].

```python
%matplotlib inline
%reload_ext autoreload
%autoreload 2

from fastai.conv_learner import *

PATH = Path('data/cifar10/')
os.makedirs(PATH, exist_ok=True)
torch.backends.cudnn.benchmark = True # cut down training duration from ~3 hr to ~1hr
```

A really good exercise for anybody who is not 100% confident with their broadcasting and PyTorch basic skill is figure out how Jeremy came up with these stats numbers. These numbers are the averages and standard deviations for each channel in CIFAR10. Try and make sure you can recreate those numbers and see if you can do it with no more than a couple of lines of code (no loops!).

Because these are fairly small, we can use a larger batch size than usual and the size of these images is 32 [00:09:46].

```python
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# these numbers are the averages and standard deviations for each channel in CIFAR10
stats = (np.array([ 0.4914 ,  0.48216,  0.44653]), np.array([ 0.24703,  0.24349,  0.26159]))

num_workers = num_cpus() // 2 # num cpus returns 4
bs = 256
sz = 32
```

Transformations [00:09:57], normally we have this standard set of side_on transformations we use for photos of normal objects. We are not going to use that here because these images are so small that trying to rotate a 32 by 32 image a bit is going to introduce a lot of blocky distortions. So the standard transformations that people tend to use is a random horizontal flip and then we add 4 pixels (size divided by 8) of padding on each side. One thing which works really well is by default fastai does not add black padding which many other libraries do. Fastai takes the last 4 pixels of the existing photo and flip it and reflect it, and we find that we get much better results by using reflection padding by default. Now that we have 40 by 40 image, this set of transforms in training will randomly pick a 32 by 32 crops, so we get a little bit of variation but not heaps. Wo we can use the normal `from_paths` to grab our data.

```python
tfms = tfms_from_stats(stats, sz, aug_tfms=[RandomFlip()], pad=sz // 8)
data = ImageClassifierData.from_paths(PATH, val_name='test', tfms=tfms, bs=bs)
```

Now we need an architecture and we are going to create one which fits in one screen [00:11:07]. This is from scratch. We are using predefined `Conv2d`, `BatchNorm2d`, `LeakyReLU` modules but we are not using any blocks or anything. The entire thing is in one screen so if you are ever wondering can I understand a modern good quality architecture, absolutely! Let’s study this one.

```python
def conv_layer(ni, nf, ks=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=ni, out_channels=nf, kernel_size=ks, bias=False, stride=stride, padding=ks // 2),
        nn.BatchNorm2d(num_features=nf, momentum=0.01),
        nn.LeakyReLU(negative_slope=0.1, inplace=True)
    )

class `ResLayer`(nn.Module):
    def __init__(self, ni):
        super().__init__()
        self.conv1 = conv_layer(ni, ni // 2, ks=1)
        self.conv2 = conv_layer(ni // 2, ni, k2=3)

    def forward(self, x):
        return x.add_(self.conv2(self.conv1(x)))

class Darknet(nn.Module):
    def make_group_layer(self, ch_in, num_blocks, stride=1):
        return [conv_layer(ch_in, ch_in * 2, stride=stride)
               ] + [(ResLayer(ch_in * 2)) for i in range(num_blocks)]

    def __init__(self, num_blocks, num_classes, nf=32):
        super().__init__()
        layers = [conv_layer(3, nf, ks=3, stride=1)]
        for i, nb in enumerate(num_blocks):
            layers += self.make_group_layer(nf, nb, stride=2 - (i == 1))
            nf *= 2
        layers += [nn.AdaptiveAvgPool2d(1), Flatten(), nn.Linear(nf, num_classes)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
```

The basic starting point with an architecture is to say it’s a stacked bunch of layers and generally speaking there is going to be some kind of hierarchy of layers [00:11:51]. At the very bottom level, there is things like a convolutional layer and a batch norm layer, but any time you have a convolution, you are probably going to have some standard sequence. Normally it’s going to be:

1. conv
2. batch norm
3. a nonlinear activation (e.g. ReLU)

We will start by determining what our basic unit is going to be and define it in a function (`conv_layer`) so we don’t have to worry about trying to keep everything consistent and it will make everything a lot simpler.

##### Leaky ReLU [[00:12:43](https://youtu.be/ondivPiwQho?t=12m43s)]:

![](/images/lesson_12_002.png)

The gradient of Leaky ReLU (where x < 0) varies but something about 0.1 or 0.01 is common. The idea behind it is that when you are in the negative zone, you don’t end up with a zero gradient which makes it very hard to update it. **In practice, people have found Leaky ReLU more useful on smaller datasets and less useful in big datasets**. But it is interesting that for the [YOLO v3](https://pjreddie.com/media/files/papers/YOLOv3.pdf) paper, they used Leaky ReLU and got great performance from it. It rarely makes things worse and it often makes things better. So it’s probably not bad if you need to create your own architecture to make that your default go-to is to use Leaky ReLU.

You’ll notice that we don’t define PyTorch module in `conv_layer`, we just do nn.Sequential [00:14:07]. This is something if you read other people’s PyTorch code, it’s really underutilized. People tend to write everything as a PyTorch module with `__init__` and `forward`, but if the thing you want is just a sequence of things one after the other, it’s much more concise and easy to understand to make it a `Sequential`.

##### Residual block [[00:14:40](https://youtu.be/ondivPiwQho?t=14m40s)]:

As mentioned before that there is generally a number of hierarchies of units in most modern networks, and we know now that the next level in this unit hierarchy for ResNet is the ResBlock or residual block (see `ResLayer`). Back when we last did CIFAR10, we oversimplified this (cheated a little bit). We had `x` coming in and we put that through a `conv`, then we added it back up to `x` to go out. In the real ResBlock, there are two of them. When we say "conv" we are using it as a shortcut for our `conv_layer` (conv, batch norm, ReLU).

![](/images/lesson_12_003.png)

One interesting insight here is the number of channels in these convolutions [00:16:47]. We have some `ni` coming in (some number of input channels/filters). The way the darknet folks set things up is they make every one of these Res layers spit out the same number of channels that came in, and Jeremy liked that and that’s why he used it in `ResLayer` because it makes life simpler. The first conv halves the number of channels, and then second conv doubles it again. So you have this funneling effect where 64 channels coming in, squished down with a first conv down to 32 channels, and then taken back up again to 64 channels coming out.

:question: Why is `inplace=True` in the LeakyReLU [00:17:54]?

Thanks for asking! A lot of people forget this or don’t know about it, but this is a really important memory technique. If you think about it, this `conv_layer`, it’s the lowest level thing, so pretty much everything in our ResNet once it’s all put together is going to be many `conv_layer`’s. If you do not have `inplace = True`, it’s going to create a whole separate piece of memory for the output of the ReLU so it’s going to allocate a whole bunch of memory that is totally unnecessary. Another example is that the original `forward` in `ResLayer` looked like:

```python
def forward(self, x):
    return x + self.conv2(self.conv1(x))
```

Hopefully some of you might remember that in PyTorch pretty much every every function has an underscore suffix version which tells it to do it in-place. `+` is equivalent to `add` and in-place version of `add` is `add_` so this will reduce memory usage:

```python
def forward(self, x):
    return x.add_(self.conv2(self.conv1(x)))
```

These are really handy little tricks. Jeremy forgot the `inplace = True` at first but he was having to decrease the batch size to much lower amounts and it was driving him crazy — then he realized that that was missing. You can also do that with dropout if you have dropout. Here are what to look out for:

- Dropout
- All the activation functions
- Any arithmetic operation

:question: In ResNet, why is bias usually set to `False` in `conv_layer` [00:19:53]?

Immediately after the `Conv`, there is a `BatchNorm`. Remember, `BatchNorm` has 2 learnable parameters for each activation — the thing you multiply by and the thing you add. If we had bias in `Conv` and then add another thing in `BatchNorm`, we would be adding two things which is totally pointless — that’s two weights where one would do. So if you have a `BatchNorm` after a `Conv`, you can either tell `BatchNorm` not to include the add bit or easier is to tell `Conv` not to include the bias. There is no particular harm, but again, it’s going to take more memory because that is more gradients that it has to keep track of, so best to avoid.

Also another little trick is, most people’s `conv_layer`’s have padding as a parameter [00:21:11]. But generally speaking, you should be able to calculate the padding easily enough. If you have a kernel size of 3, then obviously that is going to overlap by one unit on each side, so we want padding of 1. Or else, if it’s kernel size of 1, then we don’t need any padding. So in general, padding of kernel size "integer divided" by 2 is what you need. There’re some tweaks sometimes but in this case, this works perfectly well. Again, trying to simplify my code by having the computer calculate stuff for me rather than me having to do it myself.

![](/images/lesson_12_004.png)

Another thing with the two `conv_layer`’s [00:22:14]: We had this idea of bottleneck (reducing the channels and then increase them again), there is also what kernel size to use. The first one has 1 by 1 `Conv`. What actually happen in 1 by 1 conv? If we have 4 by 4 grid with 32 filters/channels and we will do 1 by 1 conv, the kernel for the conv looks like the one in the middle. When we talk about the kernel size, we never mention the last piece — but let’s say it’s 1 by 1 by 32 because that’s the part of the filters in and filters out. The kernel gets placed on the first cell in yellow and we get a dot product these 32 deep bits which gives us our first output. We then move it to the second cell and get the second output. So there will be bunch of dot products for each point in the grid. It is allowing us to change the dimensionality in whatever way we want in the channel dimension. We are creating `ni//2` filters and we will have `ni//2` dot products which are basically different weighted averages of the input channels. With very little computation, it lets us add this additional step of calculations and nonlinearities. It is a cool trick to take advantage of these 1 by 1 convs, creating this bottleneck, and then pulling it out again with 3 by 3 convs — which will take advantage of the 2D nature of the input properly. Or else, 1 by 1 conv doesn’t take advantage of that at all.

![](/images/lesson_12_005.png)

These two lines of code, there is not much in it, but it’s a really great test of your understanding and intuition about what is going on [00:25:17] — why does it work? why do the tensor ranks line up? why do the dimensions all line up nicely? why is it a good idea? what is it really doing? It’s a really good thing to fiddle around with. Maybe create some small ones in Jupyter Notebook, run them yourself, see what inputs and outputs come in and out. Really get a feel for that. Once you’ve done so, you can then play around with different things.

####  Wide Residual Networks

One of the really unappreciated papers is this one [00:26:09] —[ Wide Residual Networks](https://arxiv.org/abs/1605.07146). It’s really quite simple paper but what they do is they fiddle around with these two lines of code:

- What if we did `ni*2` instead of `ni//2`?
- What if we added `conv3`?

They come up with this kind of simple notation for defining what the two lines of code can look like and they show lots of experiments. What they show is that this approach of a bottlenecking of decreasing the number of channels which is almost universal in ResNet is probably not a good idea. In fact, from the experiments, definitely not a good idea. Because what happens is it lets you create really deep networks. The guys who created ResNet got particularly famous for creating 1001 layer network. But the thing about 1001 layers is you can’t calculate layer 2 until you are finished layer 1. You can’t calculate layer 3 until you finish calculating layer 2. So it’s sequential. **GPUs don’t like sequential**. So what they showed is that if you have less layers but with more calculations per layer — so one easy way to do that would be to remove `//2`, no other changes:

```python
self.conv1 = conv_layer(ni, ni, ks=1)
self.conv2 = conv_layer(ni, ni, ks=3)
```

:bookmark: Try this at home. Try running CIFAR and see what happens. Even multiply by 2 or fiddle around.

That lets your GPU do more work and **it’s very interesting because the vast majority of papers that talk about performance of different architectures never actually time how long it takes to run a batch through it**. They say "this one takes X number of floating-point operations per batch" but they never actually bother to run it like a proper experimentalists and find out whether it’s faster or slower. A lot of the architectures that are really famous now turn out to be slow as molasses and take crap loads of memory and just totally useless because the researchers never actually bothered to see whether they are fast and to actually see whether they fit in RAM with normal batch sizes.

So Wide ResNet paper is unusual in that it actually times how long it takes as does the YOLO v3 paper which made the same insight. They might have missed the Wide ResNet paper because the YOLO v3 paper came to a lot of the same conclusions but Jeremy is not sure they cited the Wide ResNet paper so they might not be aware that all that work has been done. It’s great to see people are actually timing things and noticing what actually makes sense.

:question: What is your opinion on SELU (scaled exponential linear units)? [00:29:44]

SELU is largely for fully connected layers which allows you to get rid of batch norm and the basic idea is that if you use this different activation function, it’s self normalizing. Self normalizing means it will always remain at a unit standard deviation and zero mean and therefore you don’t need batch norm. It hasn’t really gone anywhere and the reason is because it’s incredibly finicky — you have to use a very specific initialization otherwise it doesn’t start with exactly the right standard deviation and mean. Very hard to use it with things like embeddings, if you do then you have to use a particular kind of embedding initialization which doesn’t make sense for embeddings. And you do all this work, very hard to get it right, and if you do finally get it right, what’s the point? Well, you’ve managed to get rid of some batch norm layers which weren’t really hurting you anyway. It’s interesting because the SELU paper — the main reason people noticed it was because it was created by the inventor of LSTM and also it had a huge mathematical appendix. So people thought "lots of maths from a famous guy — it must be great!" but in practice, Jeremy doesn’t see anybody using it to get any state-of-the-art results or win any competitions.

`Darknet.make_group_layer` contains a bunch of `ResLayer` [00:31:28]. `group_layer` is going to have some number of channels/filters coming in. We will double the number of channels coming in by just using the standard `conv_layer`. Optionally, we will halve the grid size by using a stride of 2. Then we are going to do a whole bunch of ResLayers — we can pick how many (2, 3, 8, etc) because remember ResLayers do not change the grid size and they don’t change the number of channels, so you can add as many as you like without causing any problems. This is going to use more computation and more RAM but there is no reason other than that you can’t add as many as you like. `group_layer`, therefore, is going to end up doubling the number of channels because the initial convolution doubles the number of channels and depending on what we pass in as `stride`, it may also halve the grid size if we put `stride=2`. And then we can do a whole bunch of Res block computations as many as we like.

To define our `Darknet`, we are going to pass in something that looks like this [00:33:13]:

```python
m = Darknet([1, 2, 4, 6, 3], num_classes=10, nf=32)
m = nn.DataParallel(m, [1, 2, 3]) # disabled this line if you have single GPU
```

What this says is create five group layers: the first one will contain 1 extra ResLayer, the second will contain 2, then 4, 6, 3 and we want to start with 32 filters. The first one of ResLayers will contain 32 filters, and there’ll just be one extra ResLayer. The second one, it’s going to double the number of filters because that’s what we do each time we have a new group layer. So the second one will have 64, and then 128, 256, 512 and that’ll be it. Nearly all of the network is going to be those bunches of layers and remember, every one of those group layers also has one convolution at the start. So then all we have is before that all happens, we are going to have one convolutional layer at the very start, and at the very end we are going to do our standard adaptive average pooling, flatten, and a linear layer to create the number of classes out at the end. To summarize [00:34:44], one convolution at one end, adaptive pooling and one linear layer at the other end, and in the middle, these group layers each one consisting of a convolutional layer followed by `n` number of ResLayers.

**Adaptive average pooling** [00:35:02]: Jeremy’s mentioned this a few times, but he’s yet to see any code out there, any example, anything anywhere, that uses adaptive average pooling. Every one he’s seen writes it like `nn.AvgPool2d(n)` where `n` is a particular number — this means that it’s now tied to a particular image size which definitely isn’t what you want. So most people are still under the impression that a specific architecture is tied to a specific size (size here means the input size. i.e image size 32 by 32). That’s a huge problem when people think that because it really limits their ability to use smaller sizes to kick-start their modeling or to use smaller size for doing experiments.

**Sequential** [00:35:53]: A nice way to create architectures is to start out by creating a list, in this case this is a list with just one `conv_layer` in, and `make_group_layer` returns another list. Then we can append that list to the previous list with `+=` and do the same for another list containing `AdaptiveAvgPool2d`. Finally we will call `nn.Sequential` of all those layers. Now the forward is just `self.layers(x)`.

![](/images/lesson_12_006.png)

This is a nice picture of how to make your architectures as simple as possible. There are a lot you can fiddle around with. You can parameterize the divider of `ni` to make it a number that you pass in to pass in different numbers- maybe do times 2 instead. You can also pass in things that change the kernel size, or change the number of convolutional layers. Jeremy has a version of this which he is going to run for you which implements all of the different parameters that were in the Wide ResNet paper, so he could fiddle around to see what worked well.

![](/images/lesson_12_007.png)

```python
lr = 1.3

learn = ConvLearner.from_model_data(m, data)
learn.crit = nn.CrossEntropyLoss()
learn.metrics = [accuracy]
wd = 1e-4
```

Once we’ve got that, we can use `ConvLearner.from_model_data` to take our PyTorch module and a model data object, and turn them into a learner [00:37:08]. Give it a criterion, add a metrics if we like, and then we can fit and away we go.

```python
%time learn.fit(lr, 1, wds=wd, cycle_len=30, use_clr_beta=(20, 20, 0.95, 0.85))
```

![](/images/lesson_12_008.png)

:memo: *On my server with a single Tesla K80, I am able to train to 91% accuracy in 52 minutes 38 seconds.*

:question: Could you please explain adaptive average pooling? How does setting to `1` work [00:37:25]?

Sure. Normally when we are doing average pooling, let’s say we have 4x4 and we did `avgpool((2, 2))` [00:40:35]. That creates 2x2 area (blue in the below) and takes the average of those four. If we pass in `stride=1`, the next one is 2x2 shown in green and take the average. So this is what a normal 2x2 average pooling would be. If we didn’t have any padding, that would spit out 3x3. If we wanted 4x4, we can add padding.

![](/images/lesson_12_009.png)

What if we wanted 1x1? Then we could say `avgpool((4,4), stride=1)` that would do 4x4 in yellow and average the whole lot which results in 1x1. But that’s just one way to do it. Rather than saying the size of the pooling filter, why don’t we instead say "I don’t care what the size of the input grid is. I always want one by one". That’s where you say `adap_avgpool(1)`. In this case, you don’t say what’s the size of the pooling filter, you instead say what the size of the output we want. We want something that’s one by one. If you put a single integer `n`, it assumes you mean `n` by `n`. In this case, adaptive average pooling 1 with a 4x4 grid coming in is the same as average pooling (4, 4). If it was 7x7 grid coming in, it would be the same as average pooling (7, 7). It is the same operation, it’s just expressing it in a way that regardless of the input, we want something of that sized output.

#### [DAWNBench](https://dawn.cs.stanford.edu/benchmark/index.html) [[00:37:43](https://youtu.be/ondivPiwQho?t=37m43s)]

Let’s see how we go with our simple network against these state-of-the-art results. Jeremy has the command ready to go. We’ve taken all that stuff and put it into a simple Python script, and he modified some of the parameters he mentioned to create something he called `wrn_22` network which doesn’t officially exist but it has a bunch of changes to the parameters we talked about based on Jeremy’s experiments. It has bunch of cool stuff like:

- Leslie Smith’s one cycle
- Half-precision floating-point implementation

![](/images/lesson_12_010.png)

This is going to run on AWS p3 which has 8 GPUs and Volta architecture GPUs which have special support for half-precision floating-point. Fastai is the first library to actually integrate the Volta optimized half-precision floating-point into the library, so you can just do `learn.half()` and get that support automatically. And it’s also the first to integrate one cycle.

What this actually does is it’s using PyTorch’s multi-GPU support [00:39:35]. Since there are eight GPUs, it is actually going to fire off eight separate Python processes and each one is going to train on a little bit and then at the end it’s going to pass the gradient updates back to the master process that is going to integrate them all together. So you will see lots of progress bars pop up together.

You can see it’s training three or four seconds when you do it this way. Where else, when Jeremy was training earlier, he was getting 30 seconds per epoch. So doing it this way, we can train things ~10 times faster which is pretty cool.

**Checking on the status** [[00:43:19](https://youtu.be/ondivPiwQho?t=43m19s)]:

![](/images/lesson_12_011.png)

It’s done! We got to 94% and it took 3 minutes and 11 seconds. Previous state-of-the-art was 1 hour 7 minutes. Was it worth fiddling around with those parameters and learning a little bit about how these architectures actually work and not just using what came out of the box? Well, holy crap. We just used a publicly available instance (we used a spot instance so it costs us $8 per hour — for 3 minutes, 40 cents) to train this from scratch 20 times faster than anybody has ever done it before. So that is one of the craziest state-of-the-art result. We’ve seen many but this one just blew it out of the water. This is partly thanks to fiddling around with those parameters of the architecture, mainly frankly about using Leslie Smith’s one cycle. Reminder of what it is doing [00:44:35], for learning rate, it creates upward path that is equally long as the downward path so it’s true triangular cyclical learning rate (CLR). As per usual, you can pick the ratio of x and y (i.e. starting LR / peak LR).

![](/images/lesson_12_012.png)

In this case, we picked 50 for the ratio. So we started out with much smaller learning rate. Then it has this cool idea where you get to say what percentage of your epochs is spent going from the bottom of the triangle all the way down pretty much to zero — that is the second number. So 15% of the batches are spent going from the bottom of our triangle even further.

![](/images/lesson_12_013.png)

That is not the only thing one cycle does, we also have momentum. Momentum goes from .95 to .85. In other words, when learning rate is really low, we use a lot of momentum and when the learning rate is really high, we use very little momentum which makes a lot of sense but until Leslie Smith showed this in the paper, Jeremy has never seen anybody do it before. It’s a really cool trick. You can now use that by using `use-clr-beta` parameter in fastai ([forum post by Sylvain](http://forums.fast.ai/t/using-use-clr-beta-and-new-plotting-tools/14702)) and you should be able to replicate the state-of-the-art result. You can use it on your own computer or your Paperspace, the only thing you won’t get is the multi-GPU piece, but that makes it a bit easier to train anyway.

:question: `make_group_layer` contains stride equals 2, so this means stride is one for layer one and two for everything else. What is the logic behind it?

Usually the strides I have seen are odd [00:46:52]. Strides are either one or two. I think you are thinking of kernel sizes. So `stride=2` means that I jump two across which means that you halve your grid size. So I think you might have got confused between stride and kernel size there. If you have a stride of one, the grid size does not change. If you have a stride of two, then it does. In this case, because this is CIFAR10, 32 by 32 is small and we don’t get to halve the grid size very often because pretty quickly we are going to run out of cells. So that is why the first layer has a stride of one so we don’t decrease the grid size straight away. It is kind of a nice way of doing it because that’s why we have a low number at first `Darknet([1, 2, 4, 6, 3], …)` . We can start out with not too much computation on the big grid, and then we can gradually doing more and more computation as the grids get smaller and smaller because the smaller grid the computation will take less time.
