# Lesson 13 - Neural Translation

Topics:

* Neural Machine Translation
* Densenets

Lesson

* [Website](http://course17.fast.ai/lessons/lesson13.html)
* [Video](https://youtu.be/-lx2shfA-5s)
* [Wiki](http://forums.fast.ai/t/lesson-13-wiki)
* [Forum discussion](http://forums.fast.ai/t/lesson-13-discussion)

## Coursework

### Jupyter Notebook Used \[TODO\]

* \[time: 00:51:31\] spelling\_bee\_RNN.ipynb \[DONE ON 2018-04-13\]
* \[time: 01:18:09\] translate-pytorch.ipynb \[TODO\]
* \[time: 02:07:06\] densenet-keras.ipynb \[TODO\]

### Reading: Paper \[TODO\]

* \[time: 00:07:01\] Cyclical Learning Rate for Training Neural Networks by Leslie Smith
* \[time: 00:09:26\] Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization by Xun Huang et. al
* \[time: 00:10:35\] [BEGAN: Boundary Equilibrium Generative Adversarial Networks](https://arxiv.org/abs/1703.10717)
* \[time: 00:11:39\] [CycleGAN](https://github.com/junyanz/CycleGAN)
* \[time: 00:37:00\] [Grammar as a Foreign Language](https://arxiv.org/abs/1412.7449) by Oriol Vinyals et. al \[continue from last lesson\]
* \[time: 01:48:39\] Neural Machine Translation of Rare Words with Subword Units by Rico Sennrich et. al
* \[time: 01:57:14\] [The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/abs/1611.09326) by Simon Jegou, Yoshua Bengio, et. al

### Reading: Blog posts and articles

* \[time: 00:22:15\] [Facebook's Faiss](https://code.facebook.com/posts/1373769912645926/faiss-a-library-for-efficient-similarity-search/)
* \[time: 00:35:14\] [Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/)
* Beam search and viterbi tutorial \([slides](http://www.phontron.com/slides/nlp-programming-en-13-search.pdf)\)
* Chris Manning’s [talk](https://simons.berkeley.edu/talks/christopher-manning-2017-3-27) at the Simons' Institute, Representations for Language: From Word Embeddings to Sentence Meanings.

### Projects \[TODO\]

* \[time: 00:09:26\] Hats off again to Brad for drawing my attention to this paper, which is a new style transfer paper which can transfer to any style in real time. So you don't have to build a separate network for each one. This is the kind of thing which you could absolutely turn into an app. Obviously no one's done it yet because this paper's just come out. So you could be the first one to say here's an app which create any photo into any style you like.

### Datasets

* \[time: 01:23:10\] giga french corpus, which I'll put a link to on the wiki.
  * It's a French-English corpus:
    * [Website](http://www.statmt.org/wmt10/translation-task.html)
    * [Direct download link](http://www.statmt.org/wmt10/training-giga-fren.tar)
* [French word embeddings models.](http://fauconnier.github.io/index.html#wordembeddingmodels) pre-trained word2vec models for French from Jean-Philippe Fauconnier.

### Tools

* \[time: 01:52:09\] [Subword Neural Machine Translation](https://github.com/rsennrich/subword-nmt) \(Byte Pair Encoding/BPE\)

## My Notes

* \[time: 00:07:01\] Talking of great work, I also wanted to mention the work of another student who also has the great challenge of being a fast.ai intern, Brad. Brad took up the challenge I set out two weeks ago in implementing cyclical learning rates. You might remember that the cyclical learning rates paper showed faster training of neural nets and also more automated. Brad actually had it coded up super quickly.
* \[time: 00:07:41\] This is exactly the kind of thing Rachel and I at fast.ai are trying to do if this really works out, get rid of the whole question of how to set learning rates, which currently is such an artisanal thing.
* We've been talking about meanshift clustering from time to time \(we'll talk about it more next week in terms of applications\), but the main application we've talked about so far is using it for a kind of faster preprocessing of really large data items, like CT scans in order to find objects of interest, in this case, lung nodules that might be cancers.
* \[time: 00:22:15\] So very interestingly, today or maybe yesterday, Facebook just announced that they have implemented an enormous improvement in the state-of-the-art in approximate nearest neighbors. So you can check this out if you like, Faiss.
* \[time: 00:28:29\] Now we are going to talk about the BiLSTM Hegemony.
* \[time: 00:34:03\] So NLP is by no means solved.
* \[time: 00:58:48\] The only thing remaining is how to do this attention layer.
* \[time: 01:03:32\] A minor note, for those of you that are playing with PyTorch, I'm sure you've discovered how cool it is that you can go self.add\_module and it keeps track of everything you need to train, all the parameters. ... Keras is not so clever, that's why we have to go self.add\_weight for every one of these weights because Keras has to know what to train, when you optimize what do you train. ... Keras really doesn't have a convenient user-facing API for creating custom RNN code. In fact, this is something which nobody's really figured out yet. TensorFlow has just release a new custom RNN API, but there isn't any documentation for it. ... This probably shouldn't have been this hard in the end, but it's probably the nature of the Keras API for this stuff, it doesn't really exist, we had to go in and create it. All we really wanted to do was say, Okay, Keras, before you run the 3 decoder RNNs, take your input and modify it in this way. But we have to do it for every step. That's basically what was missing, some way in Keras to easily say like change the step. I've spoken to Francois \(the Keras author\), he's well aware that this is not convenient right now and he really wants to fix it but it's difficult to get right and no one's quite done it yet.
* \[time: 01:16:41\] Here is a photo and here is a painting, and here is the regular style transfer result \(that's not bad\). And here is what happens when you use his new mathematical technique, that's so much better! So hopefully by next week I will understand this enough so that I can explain it to you. But I know one of the key differences is he's using the Earth-Mover distance, which is the basis of the Wasserstein GAN. I've managed to avoid teaching you about eigenvalues and eigenvectors so far. I don't know how we're going to do that. ... This is got to be a paper, he's created a whole new technique. This is super-exciting. Congratulations, I look forward to learning more about it. ... People just keep doing cool stuff, I love it. You guys are just zipping along.
* \[time: 01:18:09\] So let's translate English into French.
* \[time: 01:20:36\] I'm passing in the previous letter's correct answer for every step, but in real life I don't know that. What we actually need to do is to at inference time you don't use feature forcing, but instead you take the predicted previous step's result and feed it in to the next step. I have no idea how to do that in Keras. It drove me crazy trying to figure out how to do that in Keras. And that was the thing that pushed me to PyTorch. I was so sick of this attention layer, the idea of going back and trying to put this thing in drove me crazy and I don't know how to do it.
* \[time: 01:22:33\] Let's look at the PyTorch version. Interestingly, the PyTorch version in terms of like the attention model itself turns out to be way easier. But we have to write more code because there's less structure for NLP models. For computer vision stuff, there's the PyTorch vision project which has all that loading, models and blah blah blah that we don't seem to have in NLP. So there's a bit more code to write here.
* \[time: 01:23:10\] I downloaded this giga french corpus, which I'll put a link to on the wiki.
* \[time: 01:25:59\] So the next step is tokenization. Tokenization is taking a sentence and turning it basically into words. This is not quite straight-forward, because like what's a word? So is that a word? Is that a word? Or is that a word? Or is that a word? So I have to make some decisions based on my view of what is likely a word. It's basically like, Okay I think that's a word. So I just wrote some regular expressions for doing heuristic tokenization. So you can use NLTK that has a lot of tokenizers in it.
* \[time: 01:44:10\] BEAM search and A\* search slides. I'm actually going to steal some slides from Graham Neubig from Nara Institute of Science and Technology, who has shown a fantastic simple example of what you could do instead.
  * Beam search is faster
    * Remove some candidates from consideration -&gt; faster speed!
    * What is the time complexity?
      * T = Number of tags
      * N = Length of sentence
      * B = beam width
  * BEAM search algo - Jeremy encourage we write this code on our own.
* \[time: 01:48:39\] So the technique I'm going to show you next is described in this paper, Neural Machine Translation of Rare Words with Subword Units.
* \[time: 01:49:51\] ... how do I translate language when somebody uses a word I haven't seen before. Or more generally, maybe I don't want to use 160,000 words in my vocabulary. That's a huge embedding matrix. It take a lot of memory, it takes a lot of time. It's going to be hard to train. So what do you do? The answer is you use something called BPE, which is basically an encoder. What it's going to do is it's going to take a sentence like, "Hello, I am Jeremy", and it's going to basically say I'm going to try and turn this into a list of tokens that aren't necessarily the same as the words.
* \[time: 01:52:09\] Now the cool thing is that you can do this by going to this github site \([https://github.com/rsennrich/subword-nmt](https://github.com/rsennrich/subword-nmt)\), downloading this and running it on your file of prose and it will spit out the exact same thing but it will stick "@@ " between every one of these BPE codes.
* \[01:55:17\] Segmentation
  * Will learn about Tiramisu network next week. But, the prerequisite for this is the Densenet paper.
  * Densenet vs ResNet: replace addition with concat.
  * Densenet paper - a great place to start implementing paper cause it's an easy paper to read \(no math in it\).
* \[02:00:16\] I can actually describe it to you in a single sentence: A DenseNet is a ResNet where you replace addition with concatenation. That's actually the entirety of what a DenseNet is.
  * So because we keep concatenating, the number of filters is getting bigger and bigger. So we're going to have to be careful not to add too many filters at each layer.
  * So the number of filters that are added at each layer they call the growth rate, and for some reason they use the letter "k" for growth rate. They tend to use the values of 12 or 24; in the Tiramisu paper they tend to use the value 16.
  * \[time: 02:06:41\] So really if you're using something that's more of the 100 to 100,000 images range, you probably want to be using DenseNet. If it's more than 100,000 images maybe it doesn't matter so much.
* \[02:07:06\] densenet-keras.ipynb So let's see the code. So interestingly, this turned out to be something that suited Keras really well. These kind of things where you're using standard kind of layers, connected in different ways, Keras is fantastically great for.
* \[time: 02:11:41\] Question: Can we do transfer learning on DenseNet? ... Answer: Absolutely you can. And in fact PyTorch, just came out yesterday or today, and has some pre-trained DenseNet models.

