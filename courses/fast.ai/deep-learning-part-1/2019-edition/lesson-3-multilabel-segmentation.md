# Lesson 3 - Multi-label, Segmentation, Image Regression, and More

_These are my personal notes from fast.ai Live (the new International Fellowship programme) course and will continue to be updated and improved if I find anything useful and relevant while I continue to review the course to study much more in-depth. Thanks for reading and happy learning!_

Live date: 9 Nov 2018, GMT+8

## Topics

* Multi-label classification
* Kaggle Planet Amazon dataset of satellite images
* fastai Data Block API
* fastai DataBunch class
* Image segmentation with CamVid
* U-Net
* Learning rate annealing
* Mixed precision training
* Image regression with BIWI head pose dataset
* NLP classification
* Universal approximation theorem

## Lesson Resources

* **Website and video** links will be shared when the MOOC officially released in early 2019.
* [Official resources and updates (Wiki)](https://forums.fast.ai/t/lesson-3-official-resources-and-updates/29732)
* [Forum discussion](https://forums.fast.ai/t/lesson-3-chat/29733)
* [FAQ, resources, and official course updates](https://forums.fast.ai/t/faq-resources-and-official-course-updates/27934)
* Jupyter Notebook and code
  * [lesson3-planet.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson3-planet.ipynb)
  * [lesson3-camvid.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson3-camvid.ipynb)
  * [lesson3-head-pose.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson3-head-pose.ipynb)
  * [lesson3-imdb.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson3-imdb.ipynb)

## Assignments

* Run lesson 3 notebooks.
* Replicate lesson 3 notebooks with your own dataset.
* Dig into the Data Block API

## Other Resources

### Blog Posts and Articles

* [Universal Language Model Fine-tuning (ULMFiT) for Text Classification](http://nlp.fast.ai/category/classification.html) used in [`language_model_learner`](https://docs.fast.ai/text.html)
* [Michael Nielsen's book](http://neuralnetworksanddeeplearning.com/)
* Quick and easy model deployment on Zeit Now guide
* [Data Block API](https://docs.fast.ai/data_block.html)

### Other Useful Information

* Useful online courses for machine learning background:
  * [Machine Learning](https://www.coursera.org/learn/machine-learning) taught by Andrew Ng (Coursera)
  * [Introduction to Machine Learning for Coders](https://course.fast.ai/ml) taught by Jeremy Howard
* [Python partials](https://docs.python.org/3/library/functools.html#functools.partial)
* Nov 14 Meetup - [Conversation between Jeremy Howard and Leslie Smith](https://www.meetup.com/sfmachinelearning/events/255566613/)
* [List of vision transforms](https://docs.fast.ai/vision.transform.html#List-of-transforms)

### Useful Tools and Libraries

* Fast.ai Video Viewer with searchable transcript
* [MoviePy](https://zulko.github.io/moviepy) Python module for video editing mentioned by Rachel
* [WebRTC example for web video](https://github.com/etown/dl1/blob/master/face/static/index.html) from Ethan Sutin

### Papers

* Optional reading
  * [Cyclical Learning Rates for Training Neural Networks by Leslie Smith](https://arxiv.org/abs/1506.01186)

# My Notes