# Lesson 13 - Image Enhancement; Style Transfer; Data Ethics

_These are my personal notes from fast.ai course and will continue to be updated and improved if I find anything useful and relevant while I continue to review the course to study much more in-depth. Thanks for reading and happy learning!_

## Topics

* Image enhancement.
  * Style transfer.
    * An interesting approach that allows us to change the style of images in whatever way we like.
    * Optimize pixels, instead of weights, which is an interesting different way of looking at optimization.
* Generative models (and many other techniques we’ve discussed) can cause harm just as easily as they can benefit society.
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
