# Lesson 10 - Multi-modal & GANs

Topics:

* Multi-modal models; models which can combine multiple types of data
* Combine text and images in a single model using a technique called DeVISE
* Handling large datasets; training a model using the whole ImageNet dataset
* Generative Adversarial Networks \(GANs\)

Lesson

* [Website](http://course17.fast.ai/lessons/lesson10.html)
* [Video](https://youtu.be/uv0gmrXSXVg)
* [Wiki](http://forums.fast.ai/t/lesson-10-wiki)
* [Forum discussion](http://forums.fast.ai/t/lesson-10-discussion)

## Coursework

### Jupyter Notebook Used

* [imagenet-processing.ipynb](https://github.com/fastai/courses/blob/master/deeplearning2/imagenet_process.ipynb)
* [neural-sr.ipynb](https://github.com/fastai/courses/blob/master/deeplearning2/neural-sr.ipynb)
* Keras [DCGAN.ipynb](https://github.com/fastai/courses/blob/master/deeplearning2/DCGAN.ipynb)
* [pytorch-tut.ipynb](https://github.com/fastai/courses/blob/master/deeplearning2/pytorch-tut.ipynb)
* [wgan-pytorch.ipynb](https://github.com/fastai/courses/blob/master/deeplearning2/wgan-pytorch.ipynb)

### Reading: Paper \[TODO\]

* \[time: 01:42:54\] [Wasserstein GAN \(WGAN\)](https://arxiv.org/abs/1701.07875) by Martin Arjovsky, Soumith Chintala, Léon Bottou
  * Math of GAN
  * \[time: 02:09:34\] As per usual, you should go back and look at the papers. The original GAN paper is a fairly easy read. There's a section called Theoretical Results which is kind of like the pointless math bit, like here's some theoretical stuff.

### Reading: Blog posts and articles \[TODO\]

* [Picking an optimizer for Style Transfer](https://blog.slavv.com/picking-an-optimizer-for-style-transfer-86e7b8cba84b)
  * Comparison of different optimization algos like L-FBGS, Adam, GD, Adadelta, etc. for artistic style transfer.
* [How to Train a GAN? Tips and tricks to make GANs work](https://github.com/soumith/ganhacks)
* [From GAN to WGAN](https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html)
* [WGAN TensorFlow CelebA dataset training took approx. 18 hrs](https://github.com/shekkizh/WassersteinGAN.tensorflow) and probably not the most converged result.
* [Conditional Generative Adversarial Nets \(Conditional GANs\)](https://arxiv.org/abs/1411.1784)
* Best GANs Demos
  * [Cycle GAN \(Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks\)](https://arxiv.org/abs/1703.10593)
  * [StarGAN](https://github.com/yunjey/StarGAN)
  * [Giant list of GANs applications](https://github.com/nashory/gans-awesome-applications)

### Reading: Forum posts

* [http://forums.fast.ai/t/wasserstein-gan-discussion-clarification/1633](http://forums.fast.ai/t/wasserstein-gan-discussion-clarification/1633) \[DONE\]
* [http://forums.fast.ai/t/improved-training-of-wasserstein-gans/2337](http://forums.fast.ai/t/improved-training-of-wasserstein-gans/2337)
* [http://forums.fast.ai/t/lesson-10-wiki/1937](http://forums.fast.ai/t/lesson-10-wiki/1937) \[DONE\]

### Projects \[TODO\]

* SIMD for resizing large dataset of images like ImageNet
  * compare speedup with and without pillow-SIMD
* Parallel processing vs serial processing
  * local.threading\(\) variable
  * Python GIL
  * multi-threading vs multi-process performance \(experiments\)
  * thread pool, process pool, workers
* Study transfer learning from ResNet \(and write blog post\)
* Briefly learn about cosine distance on the web
* Nearest Neighbours
  * different ways
    * brute-force approach
    * n-squared time
    * approximate nearest neighbors
    * approximately log-n time --&gt; orders of magnitude faster
  * Locality Sensitive Hashing \(LSH\)
* Look at Dask
* \[time: 02:08:28\] Jeremy recommend GAN as something where anybody who's interested in a project

### Datasets

* \[time: 01:56:49\] [LSUN dataset website](http://lsun.cs.princeton.edu/2016)

## My Notes

* Here is the result of serial vs parallel. The serial without simd version is 6X bigger than this, 2000 images. With SIMD, it's 25 seconds. With the process pool it's 8 seconds for 3 workers, for 6 workers it's 5 seconds, so on and so forth. For the thread pool, it's even better, 3.6 seconds for 12 workers, 3.2 seconds for 16 workers. Now your mileage will vary, depending on what CPU you have. Given that quite a lot of you are using the P2 still \(unless you've got your deep learning box up and running\), you'll have the same performance as other people using the P2, but you should try something like this which is to try different numbers of workers and see what's the optimal for that particular CPU. Once you've done that, you know.
* So that's the general approach here. Run through something in parallel. Each time append it to my bcolz array. At the end of that, I've got a bcolz array that I can use again and again. So I don't rerun that code very often anymore. I've got all of ImageNet resized into each of 72x72, 224 and 288. I give them different names and I just use them.
* That's my theory and I'd be fascinated to see somebody do a really in-depth analysis of like black borders vs center cropping vs squishing in ImageNet.
* When you're working on really big datasets, you don't want to process things any more than necessary, any more times than necessary.
* There's a couple of really important intricacies to be aware of though. The first one is you'll notice that ResNet and Inception are not used very often in transfer learning.
* So when you think about transfer learning from ResNet, you kind of need to think about, Okay, Should I transfer learn from an identity block, or after, or from a bottleneck block, or after? Again I don't think anybody's studied this, or at least I haven't seen anybody write it down. I've played around with it a bit and I'm not sure I have a totally decisive suggestion for you. My guess is that the best point to grab in ResNet is the end of the block immediately before the bottleneck block. And the reason for that is that at that level of receptive field \(obviously because each bottleneck block is changing the receptive field\) and at that level of semantic complexity this is the most sophisticated version of it.
* My belief is you want to get just before that bottleneck is the best place to transfer learn from. So that's what this is. This is the spot just before the last bottleneck layer in ResNet.
* Again, this is one of those things where I'm skipping over something where you could probably spend a week in undergrad studying. There's heaps of information about cosine distance on the web. So for those of you who are very familiar with it, I won't waste your time. For those of you not, it's a very very good idea to become familiar with this.
* My favorite kind of algorithms are these approximate algorithms. In data science, you almost never need to know something exactly. Yet nearly every algorithm that people learn at university \(and certainly high school\) are exact.
* So I generally use LSHForest when I'm doing nearest neighbors, because it's arbitrarily close and much faster when you've got word vectors.
* But now that we want to shuffle, it would. So what we've done is, somebody actually on the Kaggle forum, provided something called a bcolz array iterator. And the bcolz array iterator, which was kindly discovered on the forums actually by somebody named MPJansen
* So it's like sometimes deep learning is so magic, you go, How can that possibly work? ... Answer: Yeah, yeah. Only a little bit. Maybe in a future course we might look at that. Maybe even in your numerical linear algebra course we might be looking at that. I don't think we'll cover it in this course. But do look at Dask, it's super-cool.
* Starting next week we're going to be learning about sequence-to-sequence models and memory and attention methods.
* I always like to start with something that works and make small little changes so it keeps working at every point.
* Don't forget, lambda layers are great for this kind of thing. Whatever code you can write, tuck it into a lambda layer and suddenly it's a Keras layer.
* So we're going to learn about adversarial networks, generative adversarial networks. And this will kind of close off our deep dive into generative models as applied to images. Just to remind you, the purpose of this has been to learn about generative models, not just to specifically learn about super-resolution or artistic style.
* \[time: 01:40:15 Wasserstein GAN\]
  * Wasserstein GAN got rid of all these problems.
  * And here is the Wasserstein GAN paper. And this paper is quite an extraordinary paper. It's particularly extraordinary because \(and I think I mentioned this in the first class of this part\), most papers either tend to be math theory which goes nowhere or kind of nice experiments and engineering, where the theory bit is kind of hacked on at the end and kind of meaningless.
  * This paper is entirely driven by theory. And then the theory, they go on to show this is what the theory means, this is what we do, and suddenly all the problems go away. The loss curves are going to actually mean something and we're going to be able to do what I said we wanted to do right at the start of this GAN section which is to train the discriminator a whole bunch of steps, and then do a generator, and then the discriminator a whole bunch of steps, and then do a generator. And all that is going to suddenly start working.
  * there's actually only two things you need to do.
    * One is remove the log from the loss file. Rather than using cross-entropy loss, we're just going to use mean square error. That's one change.
    * And the second change is we're going to constrain the weights so that they lie between -.01 and +.01. We're going to constrain the weights to make them small.
  * Now in the process of saying that's all we're going to do is to not give credit to this paper. Because what this paper is they figured out that's what we need to do. And on the forums, some of you have been reading through this paper. I've already given you some tips, there's a really great walk-through, I'll put it on our wiki, that explains all the math from scratch, but basically what the math says is this.
  * \[time: 01:45:50\] Wasserstein GAN in PyTorch notebook \(`wgan-pytorch.ipynb`\)
* \[time: 01:46:36\] First look at PyTorch. PyTorch tutorial notebook \(`pytorch-tut.ipynb`\)
* \[time: 01:56:49\] [LSUN dataset website](http://lsun.cs.princeton.edu/2016) \[DONE, read on 2018-05-06\]
* \[time: 01:57:47\] [`dcgan.py` file](https://github.com/fastai/courses/blob/master/deeplearning2/dcgan.py)
  * We're going to start with CIFAR10, so we've got 47 thousand of those images. So I have just put the definitions of the discriminator and generator architectures into a separate Python file, `dcgan.py`.
* \[time: 02:03:16\] So this is where we go ahead and train the discriminator. And you'll see here we clamp \(this is the same as clip\) the weights in the discriminator to fall in this range. And if you're interested in reading the paper, the paper explains that basically the reason for this is that their assumptions are only true in this kind of small area. So that's why we have to make sure the weights stay in this small area.
* So during the week you can look at these two different versions and you're going to see the PyTorch and the Keras version are basically the same thing. The only difference is ... two things. One is the presence of this clamping and the second is that the loss function is mean-square-error, rather than cross-entropy.
* So here's the other thing, the loss function for these actually makes sense. The discriminator and the generator loss functions actually decrease as they get better. So you can actually tell if your thing is training properly. You can't exactly compare two different architectures to each other still, but you can certainly see that the training curves are working.
* So you can take any old paper that produces 3D outputs or segmentations or vector outputs or colorization and add this and it would be great to see what happens because none of that's been done before. \[time: 02:08:28\] It's not been done before because we haven't had a good way to train GANs before.
  * You know I think this is kind of something where anybody who's interested in a project, yeah, this would be a great project. And something that maybe you can do reasonably quickly.
  * Another thing you could do as a project is to convert this into Keras. You can take the Keras DCGAN notebook that we've already got and change the loss function and the weight clipping and try training on this LSUN bedroom dataset and you should get the same results.

