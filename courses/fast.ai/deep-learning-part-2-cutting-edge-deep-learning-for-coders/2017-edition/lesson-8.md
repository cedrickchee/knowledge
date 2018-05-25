# Lesson 8 - Artistic Style

Topics:

* Introduction
* From Theano to TensorFlow
* New developments in TensorFlow
* Project: build your own deep learning box
* Learn how to read academic papers
* Creative and generative applications, with artistic style transfer

Lesson

* [Website](http://course17.fast.ai/lessons/lesson13.html)
* [Video](https://youtu.be/cRjPVN3oo4s)
* [Wiki](http://forums.fast.ai/t/lesson-8-wiki)
* [Forum discussion](http://forums.fast.ai/t/lesson-8-discussion)

## Coursework

### Jupyter Notebook Used

* [neural_style.ipynb](https://github.com/fastai/courses/blob/master/deeplearning2/neural-style.ipynb)

### Reading: Paper

* [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) by Leon A. Gatys et. al
* [Demystifying Neural Style Transfer](https://arxiv.org/abs/1701.01036) by Yanghao Li et. al

### Reading: Blog posts and articles

\[WIP\]

### Projects

\[WIP\]

### Datasets

\[WIP\]

## My Notes

* Remember that:
  * Convolutional layers are slower, dense layers are bigger.
* In general more modern networks tend not to have any dense layers.
* Transfer Learning
  * So what's the best way to do transfer learning? I'm going to leave that as an open question for now. We're going to look into it a bit during this class, but it's not a question that anybody has answered to my satisfaction.
  * transfer learning to get us a long way.
  * the next thing we have to get us a long way is to try and create an architecture which suits our problem, both our data and our loss function.
* 5 steps to avoid overfitting
  * get more data
  * "faking" more data with data augmentation
  * use more generalizable architectures
    * batch normalization
    * regularization techniques
      * use as few as we can because by definition they destroy some data
    * dropout
  * if we have to, we can look at reducing the complexity of architecture
* Question in forum to look out for
  * Could you go through steps for underfitting?
* Introductory part. This mostly focused on TensorFlow Dev Summit and interesting things came out from it.
* Wrapping up this introductory part, I wanted to kind of change your expectations about how you've learnt so far and how you're going to learn in the future. Part 1, to me, was about showing you best practices. Here's a library, here's a problem. Use this library and these steps to solve this problem, do it this way and lo and behold we've gotten in the top 10 in this Kaggle competition.
* I tried to select things that had best practices. You now know everything I know about best practices, I don't really have anything else to tell you. So we're now up to stuff I haven't quite figured out yet \(nor has anybody else\), but you probably need to know.
* They're kind of the 3 main categories. Generally at the end of each class, it won't be like that's it, that's how you do this thing. It will be more like here are the things you can explore.
* The homework will be pick one of these interesting things and dig into it. Generally speaking, that homework will get you to a point that probably no one's done before. Or at least no one's written down before.
* Generally speaking, there's going to be lots of opportunities if you're interested to write a little blog post about the things you tried and what worked and what didn't. You'll generally find that there's no other post like that out there. Particularly if you pick a dataset that's in your domain area, it's very unlikely that someone's written it up.
* it's time to start reading papers. That is an extract from the Adam paper, and you all know how to do Adam in Microsoft Excel.
* It's amazing how most papers manage to make simple things incredibly complex. A lot of that is because academics need to show other academics how worthy they are of a conference spot, which means showing off all their fancy math skills. So if you really need a proof of the convergence of your optimizer, rather than just running it and seeing if it works, you can study Theorem 4.1 and Corollary 4.2 and blah blah blah.
* Ideally finding out in the meantime that somebody else has written a blog post in simple English like this example, Adam. Don't be disheartened when you start reading deep learning papers unless you have a math background.
* Talking about Greek symbols. It's very hard to read and remember things you can't pronounce. So if you don't know how to read the Greek letters, google the Greek alphabet and learn how to say them.
* The reason we need to read papers is as of now, a lot of the things we're doing only exist in very recent paper form.
* I really think writing is a good idea. I hope all of your projects end up in at least one blog.
* I think the most important tip here is don't wait to be perfect before you start writing. Rachel's tip is you should think of your target audience as the person who's one step behind you, maybe your target audience is just someone who's working through the Part 1 MOOC right now. Your target audience is not Geoffrey Hinton, it's you 6 months ago. There will be far more people in that target audience than in the Geoffrey Hinton target audience.
* We're staring with CNN generative models today.
* The general topic areas in Part 2 will be CNNs and NLP beyond classification. We're going to now be talking more about generative models.
* Then finally, something I'm pretty excited about because I've done a lot of work recently finding some interesting stuff about using deep learning for structured data and for time-series. For example, we heard about fraud. So fraud is both of those things, time-series and transaction histories and structured data, customer information. Traditionally that's not being tackled with deep learning, but I've actually found some state-of-the-art, world-class approaches to solving those with deep learning.
* We're going to learn generative models today for images.
  * In particular, we're going to start by looking at what's called either neural style transfer, or the original paper was called Artistic Style.
  * The idea is that we're going to take a photo and make it look like it was painted in the style of some painter.
  * That's actually enough for us to get started. Let's try and build something that optimizes pixels using a loss function of the VGG network's some convolutional layer.
  * This is the `neural-style.ipynb` notebook and much of what we're going to look at is going to look very similar.
  * So I need to make sure I run `limit_mem` very soon, as soon as I start running the notebook.
* What arXiv is
* Mendeley, arxiv-sanity, another great place for finding papers is twitter, the other place which I find extremely helpful is reddit machine learning.
* Question: Is it better to calculate `f_content` for a higher layer for VGG and use a lower layer for `f_style` since the higher layer abstracts are captured in the higher layer and the lower layer captures textures and "style"?
* Broadcasting
  * The reason I'm talking about this now is because we're going to be using this a lot.
* We've now basically got the data that we want, so the next thing we need is a VGG model. Here's the thing though. When we're doing generative models, we want to be very careful of throwing away information. And one of the main ways to throw away information is to use MaxPooling.
* Slightly better is to use AveragePooling instead of MaxPooling. At least with AveragePooling, we're using all of the data to create an average. We're still kind of throwing away 3/4 of it, but at least it's all been incorporated into calculating that average.
* Question: Shouldn't we use something like ResNet instead of VGG since the residual blocks carry more context?
  * It's a lot harder to use ResNet for anything beyond kind of basic classification, for a number of reasons. One is that just the structure of ResNet blocks is much more complex. So if you're not careful, you're going to end up picking something that's on like one of those little arms of the ResNet rather than on the additive merge of the ResNet and it's not going to give you any meaningful information.
* More generally, is `BatchNorm` helpful for generative models? I'm not sure that we have a great answer to that. Try it!
* Question: Will the pre-trained weights change if we're using `AvePooling` instead of `MaxPooling`?
  * Again, this would be an interesting thing to try
* So what we're going to do is we need to create our target
* So this is now a model, when we call `.predict` it will return this set of activations.
* Use symbolically in a computation graph, we wrap it with `K.variable`
* Define a loss function
* Of course, any time you have a computation graph, you can get its gradients.
* Normally when we run an optimizer we use some kind of SGD. Now the "S" in SGD is for stochastic. In this case, there's nothing stochastic. We're not creating lots of random batches and getting different gradients every time. So why use stochastic gradient descent when we don't have a stochastic problem to solve.
* Instead with a deterministic optimization
* Once the slope changes sign \(it's called "bracketing"\), we know that we've bracketed the minimum of that function.
* Line search
* Saddle point
* Finding a conjugate direction
* The good news is that you don't need to know any of those details, all you need to know is that there is a module called `scipy.optimize`.
* The particular version we're going to use is a limited memory bfgs.
* In our case, we're using mean-square-error, which is a nice smooth objective, so we can use the much faster convex optimization.
* When we do our artistic style, we can choose which layer will be our `f_content`.
* Question: There's a question about the checkerboard artifact, the geometric pattern that's appearing.
* Question: Would it be useful to use a tool like Quiver12 to figure out which VGG layer to use for this?
  * Answer: It's so easy just to try a few and see what works.
* We're now going to do `f_style`.
  * So we're going to create our target as before, but we're going to use a different loss function. The loss function is called style\_loss. Just like before, it's going to use MSE. But rather than use the MSE on the activations, it's the MSE on the `gram_matrix` of the activations.
  * What is a `gram_matrix`? A `gram_matrix` is very simply the dot product of a matrix with its own transpose.
    * So when you take the dot product of something with the transpose of itself, what you're basically doing is creating something a lot like a correlation matrix. You're saying, How much is each row similar to each other row?
    * You can think about it in a number of ways. You can think about it like a cosine; the cosine is basically just a dot product. You can think of it as a correlation matrix, basically a normalized version of this.
  * So this `style_loss` says that for two different images, how do these fingerprints differ. How similar are these fingerprints? The answer is nobody knows.
    * So a paper just came out two weeks ago, "Demystifying Neural Style Transfer" with a mathematical treatment where they claim to have an answer to this question.
* Question: Since the publication of that paper, has anyone else used any other loss functions for `f(style)` that achieved similar results?
  * Answer: As I mentioned, just a couple of weeks ago there was a paper \(I'll put it on the forum\) that tries to generalize this loss function. It turns out that this particular loss function seems to be about the best they could come up with.
* Basically what we're going to learn next time, a sense of where we're going to head.
  * And we're also going to learn about adversarial networks.
  * Then finally we're going to learn about a particular thing that came out three weeks ago, the Wasserstein GAN.
  * Generative Adversarial Networks basically didn't work very well at all until about 3 weeks ago.

