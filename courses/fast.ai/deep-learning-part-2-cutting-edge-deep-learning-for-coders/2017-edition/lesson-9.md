# Lesson 9 - Generative Models

Topics:

* Generative models
* Fast style transfer
* Super resolution (improve photos)

Lesson

* [Website](http://course17.fast.ai/lessons/lesson9.html)
* [Video](https://youtu.be/I-P363wSv0Q)
* [Wiki](http://forums.fast.ai/t/lesson-9-wiki)
* [Forum discussion](http://forums.fast.ai/t/lesson-9-discussion)

## Coursework

### Jupyter Notebook Used

* [neural-style.ipynb](https://github.com/fastai/courses/blob/master/deeplearning2/neural-style.ipynb)
* [imagenet-processing.ipynb](https://github.com/fastai/courses/blob/master/deeplearning2/imagenet_process.ipynb)

### Reading: Paper

- [time: 00:33:06] [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) by Justin Johnson, Alexandre Alahi, Li Fei-Fei
- [time: 01:33:57] [Deep Visual-Semantic Embedding model (DeViSE)](https://papers.nips.cc/paper/5204-devise-a-deep-visual-semantic-embedding-model) by Andrea Frome et. al

## My Notes

- Artistic style transfer part 2 (cont' from last lesson 8)
    - sharing work done by students, things happening in the forums
    - style loss plus content loss
- how to read paper, tips. we are reading the "A Neural Algorithm for Artistic Style" paper.
    - [math notation wikipedia](https://en.wikipedia.org/wiki/List_of_mathematical_symbols)
- [time: 00:23:09] the next step
- [time: 00:30:45] super resolution
- [time: 00:33:06] So this is the paper we're going to look at today, Perceptual Losses for Real-Time Style Transfer and Super-Resolution.
- As you know from things like the Fisheries Competition, segmentation can be really important as a part of solving other bigger problems.
- [time: 00:39:14] Let's look at how to create this super-resolution idea.
- Part of your homework this week will be to create the new approach to style transfer. I'm going to build the super-resolution version (which is a slightly simpler version) and then you're going to try to build on top of that to create the style transfer version.
- [time: 00:39:40] continue where we left off in `neural-style.ipynb` notebook
- So I've already created a folder with a sample of 20,000 ImageNet images. I've created two sizes; one is 288x288 and one is 72x72, and they're available as bcolz arrays. I actually posted the link to these last week, it's on platform.ai [now files.fast.ai]. So we'll open up those bcolz arrays. One trick you might have (hopefully) learned in Part 1 is that you can turn a bcolz array into a Numpy array by slicing it with everything. Anytime you slice a bcolz array, you get back a Numpy array. So if your slice is everything, then this turns it into a Numpy array. This is just a convenient way of sharing Numpy arrays.
    - the [link to the data files](http://forums.fast.ai/t/lesson-8-discussion/1522):
- fast style transfer
- next steps (in the bottom-most of the `neural-style.ipynb`)
    - some ideas for things to try:
        - iGAN
        - papers
- [time: 01:31:34] I want to talk about going big. Going big can mean two things.
- Imagenet processing in parallel (`imagenet_process.ipynb`)
- To handle this data that doesn't fit in RAM, we need some tricks. So I thought we would try some interesting project that involves looking at the whole ImageNet Competition dataset.
- you can go ahead and download ImageNet and you can start working through this project.
- This project is about implementing a paper called DeViSE. DeViSE is a really, really interesting paper.
- [time: 01:33:57] Deep Visual-Semantic Embedding model (DeViSE)
- I generally think it's a good idea to define the path for both. One path to the mount point that has my big, slow, cheap, spinning disks, and this path happens to live somewhere which is fast SSDs. That way when I'm doing my code, anytime I've got something I'm going to be accessing a lot, particularly if it's in a random order I'm going to want to make sure that that thing (as long as it's not too big) sits in this path. Anytime I'm accessing something generally sequentially, or if it's really big, I can put it in this path.
