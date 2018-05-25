# Lesson 14 - Memory Networks

Topics:

* CNN improvements
* Clustering in PyTorch
* Memory networks

Lesson

* [Website](http://course17.fast.ai/lessons/lesson11.html)
* [Video](https://youtu.be/bZmJvmxfH6I)
* [Wiki](http://forums.fast.ai/t/lesson-11-wiki)
* [Forum discussion](http://forums.fast.ai/t/lesson-11-discussion)

## Coursework

### Jupyter Notebook Used \[TODO\]

* [time: 00:53:00] [kmeans_test.ipynb](https://github.com/fastai/courses/blob/master/deeplearning2/kmeans_test.ipynb)
* [time: 01:28:34] [babi-memnn.ipynb](https://github.com/fastai/courses/blob/master/deeplearning2/babi-memnn.ipynb) [TODO]

### Reading: Paper

Recommended to do but optional.

- Anybody who's interested in exploring this multi-modal images:
    - [time: 00:08:40] Zero-Shot Learning by Convex Combination of Semantinc Embeddings (arXiv, by Andre Frome, also the author of DeViSE paper)
    - [time: 00:09:21] Attention for Fine-Grained Categorization (by Andre Frome, et. al)
- [time: 00:10:00] Systematic evaluation of CNN advances on ImageNet
    - One key insight, which is very much the kind of thing I appreciate, is that they compared what the difference between the kind of original CaffeNet plus AlexNet versus GoogLeNet versus VGGNet on two different sized images, training on the original 227 or 128.
    - And so they then used this insight to do all of their experiments from then on using a smaller 128x128 ImageNet model, which the said was 10X faster.
- [time: 00:21:27] An analysis of Deep Neural Network Models for Practical Applications, Canziani, Culurciello, Adam Paszke.
    - Modern architectures are both more accurate and more efficient (memory and FLOPS)
- [time: 00:23:47] Deep Conv Neural Net Design Patterns by Leslie Smith, Nicholay Topin.
- [time: 00:24:20] Cyclical Learning Rates for Training Neural Networks by Leslie Smith
- [time: 00:48:31] Shape-Based CT Lung Nodule Segmentation Using Five-Dimensional Mean Shift Clustering and MEM With Shape Information by Xujiong Ye, et. al
    - A non-deep-learning approach to find lung nodules (research)
- [time: 01:28:56] Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks by Jason Weston et. al
- [time: 1:57:45] Recurrent Entity Networks
- [00:40:58] U-Net Segmentation approach to cancer diagnosis (based on the Kaggle DSB 2017 tutorial)
- [00:41:38] More recently, there's actually been something called DenseNet, which takes U-Net even a little bit further, and maybe that would be the new winner for newer Kaggle competitions, when they happen.

### Kaggle Competitions

- Kaggle Data Science Bowl (DSB) 2017:
    - in one step, you will find the lung nodule.
    - in the second step, zoom into a possible nodule & figure out is this a malignant tumor or something else, a false positive.
    - the bad news is that the DSB dataset does not give you any information at all for the training set where are the cancer nodules. This is a terrible terrible idea because that information actually exists. The dataset they got this is from the National Lung Screening Trial (NLST) which actually has that info or something close to it. So the fact they didn't provide it, I just think it is horrible, you know for competition which could save lives. The good news though, is that there is a dataset which does have this info. The original dataset was called LIDC-IDRI. But interestingly that dastaset was recently used for another competition, a non-Kaggle competition called LUNA. What you could do, in fact what you have to do to do well in this competition is download the LUNA dataset. Use that to build a nodule detection algorithm. So for LUNA dataset includes files saying this lung has nodules here, here and here. Then run that nodule detection algo on the Kaggle dataset. Find the nodules and then use that to do some classification. There are some tricky things with that. The biggest tricky thing is that most of the CT scans in LUNA dataset are what called contrast studies. A contrast scan means that the patient had a radioactive die injected into them so the things they are looking for are easy to see. For the NSLT dataset, none of them use contrast. When I look into it, I didn't find that a terribly a difficult problem.
- Related to DSB 2017, some Kaggle competitions to try for learning segmentation technique:
    - Fishery competition
        - to do well, first find the fish. Then zoom into the fish and figure out what kind of fish it is.
    - White whale competition
        - same approach as Fishery certainly in this competition.

## My Notes

- [time: 00:21:20] Still lots to be done...
    - data augmentation
    - skip connections
    - initialization methods
    - depth
    - impact on ResNet
    - Most important: transfer learning effectiveness
        - E.g.: are FC layers better for transfer learning?
- [time: 00:27:14] Data Science Bowl 2017 (Cancer Diagnosis) on Kaggle
    - if you can find lung cancer earlier, the probability of survival is 10X higher. So here is something where you can have a real impact by doing this well.
- [time: 00:36:30] DSB 2017: full preprocessing tutorial, + others.
- Why work on medical diagnosis using deep learning?
    - I personally care about this because my previous startup, Enlitic was the first organization to use Deep Learning to tackle this exact problem which is trying to find lung cancer in CT scans. The reason I made that and look at first problem with mainly because I learned that if you can find lung cancer earlier, the probability or survival is 10 times higher. So, here is something where you can have real impact by doing this well which not to say a million dollar is not a big impact as well.
    - Note: Jeremy uses RadiAnt DICOM Viewer 3.4.2 (evaluation) on Windows 10.
- Kaggle DSB 2017 competition.
- [time: 00:48:31] So to finalize this discussion, I wanted to refer to this paper, which I'm guessing not that many people have read yet. It's a medical imaging paper. What it is is a non-deep-learning approach to trying to find nodules, so that's where they use this thing here, nodule segmentation.
- [time: 00:52:30] But what I did want to talk about was the mean shift clustering, a particular approach to clustering which they talk about.
- [time: 00:53:00] So clustering is something which for a long time I've been kind of an anti-fan of. It belongs to this group of unsupervised learning algorithms which always seem to be kind of looking for a problem to solve. But I realized recently that there are some specific problems that can be solved well them. [`kmeans_test.ipynb`]
- [time: 00:56:46] Now, K-means it's a shame it's so popular because it kind of sucks, right. Sucky Thing #1 is that you have to decide how many clusters there are, and at all points we don't know how many nodules there are. And then Sucky Thing #2 is without some changes (this thing called Kernel K-means), it only works if the things are the same shape, they're all kind of nicely Gaussian shaped.
- [time: 00:57:15] So we're going to talk about something way cooler, which I kind of came across quite recently, much less well-know, which is called mean-shift clustering.
- [time: 01:08:08] So one challenge with this is that it's kind of slow. So I thought let's try and accelerate it with a GPU. And because meanshift's not very cool, nobody's implemented it in a GPU yet, or maybe it's not a good idea. So I thought I would use PyTorch.
- [time: 01:13:44] unsqueeze is the same as expand_dims in NumPy...
- [time: 01:15:30] Question: I get how batching helps with locality and cache, but I do not quite follow how it helps otherwise, especially with respect to accelerating the for loop.
- [time: 01:17:23] So basically the idea here is we figure there are 2 steps we need to figure out where the nodules are in something like this. Step 1 is to find the things that may be kind of nodule-ish, zoom in to them and create a little cropped version. And then Step 2 would be where your deep learning particularly comes in, which is to figure out is that cancerous or not.
- [time: 01:19:22] The other approach that you might find interesting to think about is called tri-planar. What tri-planar means is that you take a slice through the x and the y and the z axes and so you basically end up with 3 images, one that goes through x, y and z, and so you could treat those as different channels, if you like even. They probably use pretty standard neural net libraries that expect 3 channels. So there's a couple of ideas for how you can deal with the 3D aspect of it.
- [time: 01:21:17] Question: Last class, you mentioned that you would explain when and why to use Keras versus PyTorch. If you only had brain space for one (in the same way some have only brain space for vi or emacs) which would you pick? Answer: I would pick PyTorch. It kind of does everything Keras does but gives you the flexibility to play around a lot more. I'm sure you've got brain space for both.
- [time: 01:22:25] There's a whole kernel on Kaggle for Candidate Generation and LUNA16 something something, which shows how to use LUNA to build a nodule finder. This is one of the highest rated Kaggle kernels.
- [time: 01:22:39] I mentioned an opportunity to improve this meanshift algorithm.
- [time: 01:25:00] So we learnt very briefly about a particular approach which is Locality Sensitive Hashing. I think I mentioned also there's another approach which I'm really fond of, called spill-trees. I really really want us as a team to take this algorithm and add approximate nearest neighbors to it and release it to the community as the first ever super-fast GPU-accelerated approximate nearest neighbor accelerated meanshift clustering algorithm. That would be a really good deal.
- [time: 01:26:35] here's a whole paper about how to write K-means with cuda.
- [time: 01:27:15] So big change. We're going to learn about chatbots. So we're going to start here with Slate, Facebook Thinks It Has Found the Secret to Making Bots Less Dumb. Okay, so this talks about a new thing called memory networks, which was demonstrated by Facebook.
- [time: 01:28:34] So we're going to implement this paper, and this paper is called end-to-end memory networks. The paper was actually not shown on Lord of the Rings, but was actually shown on something called babi-memnn. It's a paper describing a synthetic dataset, Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks.
- [time: 01:30:08] There's a number of different structures. This is called a 1-supporting-fact structure, which is to say you only have to go back and find one sentence in the story to grab the answer. We're also going to look at 2-supporting-facts stories, which is ones where you're going to have to look twice.
- [time: 01:33:19] So to get this into Keras we need to get this into a tensor in which everything is the same size so we use pad_sequences for that, like we did in the last part of the course, which will add 0's to make sure that everything's the same size. The other thing we will do is we will create a dictionary from words to integers, to turn every word into an index. So we're going to turn every word into an index and then pad them so that they're all the same length.
- [time: 01:37:09] so what is the memory network, or more specifically the end-to-end memory network?
- [time: 01:52:05] So this is kind of an unusual class for me to be teaching, particularly compared to Part 1, where it was like best practices. Clearly this is anything but. I'm kind of showing you something which was maybe the most popular request was teach us about chatbots.
- [time: 01:57:45] I had a good chat with Stephen Merity the other day, who's a researcher I respect a lot and also somebody I like. I asked him what he thought was the most exciting research in this direction at the moment, and he mentioned something that I was also very excited about, which is called Recurrent Entity Networks. And the Recurrent Entity Network paper is the first to solve all of the babi tasks with 100% accuracy. Take of that what you will. I don't know how much that means; they're synthetic tasks.
- [time: 01:58:40] Having said all that, one of the key reasons I wanted to look at memory networks was not only was it the largest request from the forums for this part of the course, but also because it introduces something that's going to be critical for the next couple of lessons, which is the concept of Attention.
- [time: 01:59:16] Attention, or Attentional Models, are models where we have to do exactly what we just looked at, which is basically find out at each time which part of a story to look at next, or which part of an image to look at next, or which part of a sentence to look at next.
- [time: 01:59:48] And so the task that we're going to be trying to get at over the next lesson or two is going to be to translate French into English.
- [time: 02:01:05] And so, interestingly during this time that memory networks and neural Turing Machines and stuff were getting all this huge amount of press attention, very quietly in the background at exactly the same time, Attentional Models were appearing as well. And it's the Attentional Models for language that have really turned out to be critical.
- [time: 02:03:54] So I guess one other thing to mention about the memory network is that Keras actually comes with an end-to-end memory network example in the Keras github. ... ... I find this quite surprising to discover that once you start getting in to some of these more recent advances, not just a standard CNN or whatever, it's less and less common that you actually find code that's correct and actually works.
