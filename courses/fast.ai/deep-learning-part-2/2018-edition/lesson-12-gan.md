# Lesson 11 - Generative Adversarial Networks (GANs)

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
