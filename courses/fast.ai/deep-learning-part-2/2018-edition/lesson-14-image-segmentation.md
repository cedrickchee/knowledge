# Lesson 14 - Super Resolution; Image Segmentation with U-Net

_These are my personal notes from fast.ai course and will continue to be updated and improved if I find anything useful and relevant while I continue to review the course to study much more in-depth. Thanks for reading and happy learning!_

## Topics

* Super resolution.
  * A technique that allows us to restore high resolution detail in our images, based on a convolutional neural network.
  * In the process, we’ll look at a few modern techniques for faster and more reliable training of generative convnets.
* Image segmentation.
  * U-Net architecture.
    * A state of the art technique that has won many Kaggle competitions and is widely used in industry.
  * Image segmentation models allow us to precisely classify every part of an image, right down to pixel level.

## Lesson Resources

* [Website](http://course.fast.ai/lessons/lesson14.html)
* [Video](https://youtu.be/nG3tT31nPmQ)
* [Wiki](http://forums.fast.ai/t/part-2-lesson-14-wiki)
* Jupyter Notebook and code
  * [enhance.ipynb](https://nbviewer.jupyter.org/github/fastai/fastai/blob/master/courses/dl2/enhance.ipynb)
  * [style-transfer-net.ipynb](https://nbviewer.jupyter.org/github/fastai/fastai/blob/master/courses/dl2/style-transfer-net.ipynb)
  * [carvana.ipynb](https://nbviewer.jupyter.org/github/fastai/fastai/blob/master/courses/dl2/carvana.ipynb)
  * [carvana-unet.ipynb](https://nbviewer.jupyter.org/github/fastai/fastai/blob/master/courses/dl2/carvana-unet.ipynb)
* Dataset
  * ImageNet sample in files.fast.ai/data / [direct download link](http://files.fast.ai/data/imagenet-sample-train.tar.gz) (2.1 GB)
  * Full ImageNet [[faster download from Kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge/data), mirrored over from the [ImageNet Download Site](http://www.image-net.org/download-imageurls)]
  * [Kaggle Carvana Image Masking competition](https://www.kaggle.com/c/carvana-image-masking-challenge/data) - you can download it with Kaggle API as usual

## Assignments

### Papers

* Must read
  * [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) by Justin Johnson, et. al
  * [Enhanced Deep Residual Networks for Single Image Super-Resolution (EDSR)](https://arxiv.org/abs/1707.02921) by Bee Lim, et. al
  * [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158) by Wenzhe Shi, et. al
  * [Checkerboard artifact free sub-pixel convolution: A note on sub-pixel convolution, resize convolution and convolution resize](https://arxiv.org/abs/1707.02937) by Andrew Aitken, et. al
  * [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) by Olaf Ronneberger, et. al

## My Notes