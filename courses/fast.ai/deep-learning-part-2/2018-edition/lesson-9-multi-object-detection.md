# Lesson 9 - Multi-object Detection

_These are my personal notes from fast.ai course and will continue to be updated and improved if I find anything useful and relevant while I continue to review the course to study much more in-depth. Thanks for reading and happy learning!_

## Topics

* Move from single object to multi-object detection.
* **Main focus** is on the single shot multibox detector \(SSD\).
  * Multi-object detection by using a loss function that can combine losses from multiple objects, across both localization and classification.
  * Custom architecture that takes advantage of the difference receptive fields of different layers of a CNN.
* YOLO v3
* Simple but powerful trick called focal loss.

## Lesson Resources

* [Website](http://course.fast.ai/lessons/lesson9.html)
* [Video](https://youtu.be/0frKXR-2PBY)
* [Wiki](http://forums.fast.ai/t/part-2-lesson-9-wiki)
* [Forum discussion](http://forums.fast.ai/t/part-2-lesson-9-in-class/14028/1)
* Jupyter Notebook and code
  * [pascal.ipynb](https://nbviewer.jupyter.org/github/fastai/fastai/blob/master/courses/dl2/pascal.ipynb)
  * [pascal-multi.ipynb](https://nbviewer.jupyter.org/github/fastai/fastai/blob/master/courses/dl2/pascal-multi.ipynb)

## Assignments

### Papers

* Must read
  * [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
  * [Scalable Object Detection using Deep Neural Networks](https://arxiv.org/abs/1312.2249)
  * [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
  * [You Only Look Once \(YOLO\): Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
  * [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) \(RetinaNet\)
  * [YOLO version 3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
* Additional papers \(optional\)
  * [Understanding the Effective Receptive Field in Deep Convolutional Neural Networks](https://arxiv.org/abs/1701.04128)

### Other Resources

#### Blog Posts and Articles

* [Understanding SSD MultiBox — Real-Time Object Detection In Deep Learning](https://towardsdatascience.com/understanding-ssd-multibox-real-time-object-detection-in-deep-learning-495ef744fab)

#### Other Useful Information

* [Understanding Anchors](https://docs.google.com/spreadsheets/d/1ci7KMggF-_4kv8zRTE0B_u7z-mbrKEzgvqXXKy4-KYQ/edit#gid=0)

