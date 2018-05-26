# Lesson 14 - Time Series & Segmentation

Topics:

* Time series \(structured data\) and neural network
* Embeddings beyond collaborative filtering and word encodings
* Tiramisu: fully convolutional DenseNets architecture for image segmentation
* Outro

Lesson

* [Website](http://course17.fast.ai/lessons/lesson14.html)
* [Video](https://youtu.be/1-NYPQw5THU)
* [Wiki](http://forums.fast.ai/t/lesson-14-wiki)
* [Forum discussion](http://forums.fast.ai/t/lesson-14-discussion)

## Coursework

### Jupyter Notebook Used \[TODO\]

* \[00:18:22\] [rossmann.ipynb](https://github.com/fastai/courses/blob/master/deeplearning2/rossman.ipynb) \[TODO\]
* \[01:44:01\] Tiramisu paper notebook \[TODO\]
  * Jeremy's Keras implementation: [tiramisu-keras.ipynb](https://github.com/fastai/courses/blob/master/deeplearning2/tiramisu-keras.ipynb)
  * Brandon Fortuner's PyTorch implementation: [tiramisu-pytorch.ipynb](https://github.com/fastai/courses/blob/master/deeplearning2/tiramisu-pytorch.ipynb)
    * [Forum dicussion](http://forums.fast.ai/t/one-hundred-layers-tiramisu/2266)

### Reading: Paper \[TODO\]

* \[00:01:26\] [Dynamic Mortality Risk Predictions in Pediatric Critical Care Using Recurrent Neural Networks](https://arxiv.org/abs/1701.06675) by Melissa Aczon et. al
  * time series and structured data
* \[00:09:52\] [Entity Embeddings of Categorical Variables](https://arxiv.org/abs/1604.06737) by Cheng Guo and Felix Berkhahn
* \[01:27:32\] [Artificial Neural Networks Applied to Taxi Destination Prediction](https://arxiv.org/abs/1508.00021) by Alexandre de Brebisson, Yoshua Bengio, et. al
* \[01:38:11\] [Fully Character-Level Neural Translation without Explicit Segmentation](https://arxiv.org/abs/1610.03017) by Jason Lee, Kyunghyun Cho, et. al
* \[01:49:09\] [ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation](https://arxiv.org/abs/1606.02147) by Adam Paszke, Eugenio Culurciello, et. al
* \[02:03:32\] [LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation](https://arxiv.org/abs/1707.03718) by Abhishek Chaurasia and Eugenio Culurciello

### Reading: Blog posts and articles

* \[00:00:44\] Forbes's Why We Need To Democratize Artificial Intelligence Education

### Kaggle Competitions

* \[00:07:38\] [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) - forecast sales using store, promotion, and competitor data
* \[01:25:42\] [Taxi Destination](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i)

### Projects \[TODO\]

* \[02:05:08\] Let's do good shit. But most importantly, write code! Please, write code, build apps, take your work projects and try doing them with deep learning. Build libraries and try to make things easier. ... Maybe go back to stuff from Part 1 of the course and go back and think, Oh why didn't we do it this other way, maybe I can make this simpler. Write papers. I showed you that amazing result of that new style transfer from Vincent last week; hopefully that might turn into a paper. Write blog posts.

### Datasets

* [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
  * [Download](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid)

### Tools

* \[00:27:48\] fantastic R package called [vtreat](https://cran.r-project.org/web/packages/vtreat/vignettes/vtreat.html), which has a bunch of state-of-the-art approaches to dealing with stuff like categorical variable encoding.
* \[00:54:09\] sklearn\_pandas package

## My Notes

* \[00:00:44\] One of the things that was great to see this week was this terrific article in Forbes that talked about deep learning education and it was written by one of our terrific students, Mariya, and focuses on some of the great work of some of the students that have come through this course.
* \[00:01:26\] So today we are going to be talking about a couple of things, but we're going to start with time series and structured data.
* \[00:03:36\] Now this kind of time series data is what I'm going to refer to as signal time series data.
* \[00:04:36\] In statistical terms, we would refer to that as auto-correlation. Auto-correlation means correlation with previous time periods. For this kind of signal, I think it's very likely that an RNN is the way to go.
* \[time: 00:05:25\] The only thing which was quite clever was that their sensor readings were not necessarily equally spaced. For example, did they receive some particular medical intervention, clearly they're very widely spaced and they're not equally spaced.
* \[00:07:38\] the other kind of time series data. For example, there was a Kaggle competition which was looking at forecasting sales for this big company in Europe, Rossmann, based on the date and what promotions are going on, and what the competitors are doing and so forth.
* \[00:09:20\] So it turns out that the state-of-the-art for this kind of approach does not necessarily use and RNN. I'm actually going to look at the third place result from this competition, because the 3rd place result was nearly as good as places 1 and 2, but way way way simple. And it also turns out that there's stuff we can build on top of for almost every model of this kind.
* \[00:09:52\] Basically \(surprise, surprise!\) it turns out that the answer is to use a neural network. I need to warn you again, what I'm going to teach you here is very very uncool. You'll never read about it from DeepMind or OpenAI. It doesn't involve any robot arms. It doesn't involve thousands of GPUs. It's the kind of boring stuff that normal companies use to make more, or spend less money, or satisfy their customers. I apologize deeply for that oversight.
* \[00:18:22\] Rossmann notebook
* \[00:20:00\] So I'm not sure if we've used Pandas much, if at all yet, so let's talk a bit about Pandas.
* \[00:27:48\] One of my favorite data scientists \(a pair of them actually\) who are very nearly neighbors of Rachel and I, have this fantastic R package called vtreat, which has a bunch of state-of-the-art approaches to dealing with stuff like categorical variable encoding.
* \[00:54:09\] There's a very little-known package called sklearn\_pandas, and actually I contributed some new stuff to it for this course, to make this even easier to use. If you use this data frame mapper from sklearn\_pandas, as you'll see it makes life very easy. Without it, life is very hard. And because very few people know about it, the vast majority of code you'll find on the Internet makes life look very hard. So use this code, not the other code.
* \[01:16:51\] By the way, XGBoost is fantastic.
* \[01:22:35\] First of all, I spent days screwing around with experiments in a notebook by hand, continually forgetting what I had just done until eventually I just, it took me like an hour to write this. ... And then of course I pasted it into Excel. And here it is. Chucked it into a pivot table, used conditional formatting, and here's my results. You can see all my different combinations, with and without normalization, with my special function versus a dictionary, using a single dense matrix versus putting everything together, using my init versus their lack of init. And here is, this dark blue here, is what they did. It's full of weird to me. But as you can see, it's actually the darkest blue; it is the best.
* \[01:25:42\] I'm going to very briefly mention another competition which is the Kaggle Taxi Destination Competition. The taxi competition was won by the team with this unicode name, pretty cool. It actually turned out to be a team run by Yoshua Bengio, who's one of the people who stuck it out through the AI winter and is now one of the leading lights in deep learning.
* \[01:27:08\] Interestingly, the thing I just showed you, the Rossmann competition, this paper they wrote in the Rossmann competition they claimed to have invented this idea of categorical embeddings. But actually, Yoshua Bengio's team won this competition a year earlier, with this same technique. Again, it's so uncool that nobody noticed, even though it's Yoshua Bengio.
* \[01:27:32\] I want to quickly show you what they did. This is the paper they wrote. And their approach to picking an embedding size was very simple -- use 10.
* \[01:37:55\] Question: This p\_i \* c\_i is very similar to what happens in the memory network paper \(Babi memn2n\). In that case, the output embeddings are weighted by the "attention" probability vector. ... Answer: Yes, or it's a lot like a regular attentional language model.
* \[01:38:11\] Question: Can you talk more about the idea you have about first having the convolutional layer and passing that to an RNN? What do you mean exactly by that? ... Answer: So here is a fantastic paper. We looked at these sub-word encodings last week for language models. I don't know if any of you thought about this and wondered, What if we just had individual characters? There's a really fascinating paper, Fully Character-Level Neural Translation without Explicit Segmentation, from November of last year. They actually get fantastic results on just character level, beating pretty much everything including the BPE approach we saw last time.
* \[01:43:05\] We talked last week about DenseNet and I mentioned that DenseNet is like ass-kickingly good on doing image classification with a small number of data points. Like crazily good. But I also mentioned that it's the basis of this thing, The 100 Layer Tiramisu, which is an approach to segmentation.
* \[01:44:01\] `tiramisu-keras.ipynb` Let me set the scene. Brandon \(one of our students, many of you have seen his blog posts\), he has successfully got a PyTorch of this working so I've shared that on our files.fast.ai, and I've got the Keras version working.
* \[01:49:09\] There's a really nice paper called ENet. ENet is not only an incredibly accurate model for segmentation, but it's also incredibly fast. It actually can run in real time, you can actually run it on the video. But the mistakes it makes, look at this chair. This chair has a big gap here and here and here, but ENet gets it totally wrong. And the reason why is that they use a very traditional downsampling upsampling approach, and by the time they get to the bottom, they've just lost track of the fine detail. So the trick are these connections here. What we do is we start with out input, we do a standard initial convolution \(just like we did with style transfer\), we then have a DenseNet block \(which we learned about last week\). We keep going down, we do a Max Pooling type thing, DenseNet block, Max Pooling type thing, keep going down. And then as we go up the other side, we do a deconvolution, dense block, deconvolution, dense block, we take the output from the dense block on the way down and we actually copy it over to here and concatenate the two together.
* \[01:52:20\] These things here, they're called skip connection and they were really inspired by this paper called U-Net, which has won many Kaggle competitions. But they're using dense blocks, rather than normal fully connected blocks.
* \[02:02:54\] So I briefly mention that there's a model which doesn't have any skip connections, called ENet, which is actually better than Tiramisu on everything except for Tree. But on the tree it's terrible -- it's 77.3 versus ... I take that back. I'm sure it was less good than this model, but now I can't find that data. The reason I wanted to mention this is that Eugenio Culurciello is about to release a new model which combines these approaches with skip connections, it's called LinkNet. So keep an eye out on the forums, I'll be looking into that quite shortly.
* \[02:04:10\] Really I actually wanted to talk about this briefly. A lot of you have come up to me and been like: Aagh! We're finishing! What do we do now?
* \[02:05:08\] So what's next? The forums will continue forever. We all know each other. Let's do good shit. But most importantly, write code! Please, write code, build apps, take your work projects and try doing them with deep learning. Build libraries and try to make things easier.
* \[02:06:18\] People don't believe that what you've done is possible. I know that because as recently as yesterday, the highest ranked hackernews comment on a story about deep learning was about how it's pointless trying to do deep learning unless you have years of mathematical background, and you know C++, and you're an expert in machine learning techniques across the board. Otherwise, there's no way you're going to be able to do anything useful on a real-world project. ... That today is what everybody believes. We now know that's not true.
* \[02:06:18\] Rachel and I are going to start up a podcast where we're going to try to help deep learning learners. ... Absolutely, so Rachel and I really just want to spend the next 6-12 months focused on supporting your projects. So I'm very interested in working on this lung cancer stuff, but I'm also interested in every project that you guys are working on. I want to help with that.
* \[02:09:18\] The experiment has worked, you guys are all here, you're reading papers, you're writing code, you're understanding the most cutting-edge research level deep learning that exists today. We've gone beyond some of the cutting-edge research in many situations. Some of you have gone beyond the cutting-edge research.

