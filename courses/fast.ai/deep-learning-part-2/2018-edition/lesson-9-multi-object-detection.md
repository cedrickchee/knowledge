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

## My Notes

### Review

You should understand this by now:

- Pathlib; JSON
- Dictionary comprehensions
- `defaultdict`
- How to jump around fastai source
- matplotlib Object Oriented API
- Lambda functions
- Bounding box coordinates
- Custom head; bounding box regression

#### Data Augmentation and Bounding Box

[pascal.ipynb](https://nbviewer.jupyter.org/github/fastai/fastai/blob/master/courses/dl2/pascal.ipynb)

A classifier is anything with dependent variable is categorical or binomial. As opposed to regression which is anything with dependent variable is continuous. Naming is a little confusing but will be sorted out in future. Here, `continuous` is `True` because our dependent variable is the coordinates of bounding box — hence this is actually a regressor data.

```Python
tfms = tfms_from_model(f_model, sz, crop_type=CropType.NO, aug_tfms=augs)
md = ImageClassifierData.from_csv(PATH, JPEGS, BB_CSV, tfms=tfms, continuous=True, bs=4)
```

##### Data Augmentation

```Python
augs = [RandomFlip(),
        RandomRotate(30),
        RandomLighting(0.1,0.1)]
```

```Python
tfms = tfms_from_model(f_model, sz, crop_type=CropType.NO, aug_tfms=augs)
md = ImageClassifierData.from_csv(PATH, JPEGS, BB_CSV, tfms=tfms, continuous=True, bs=4)

idx = 3
fig, axes = plt.subplots(3, 3, figsize=(9, 9))

for i, ax in enumerate(axes.flat):
    x, y = next(iter(md.aug_dl))
    ima = md.val_ds.denorm(to_np(x))[idx]
    b = bb_hw(to_np(y[idx]))
    print('b:', b)
    show_img(ima, ax=ax)
    draw_rect(ax, b)
```

```Python
b: [  1.  89. 499. 192.]
b: [  1.  89. 499. 192.]
b: [  1.  89. 499. 192.]
b: [  1.  89. 499. 192.]
b: [  1.  89. 499. 192.]
b: [  1.  89. 499. 192.]
b: [  1.  89. 499. 192.]
b: [  1.  89. 499. 192.]
b: [  1.  89. 499. 192.]
```

![Bounding box problem when using data augmentation](/images/pascal_notebook_data_aug_wrong_bbox.png)

As you can see, the image gets rotated and lighting varies, but bounding box is not moving and is in a wrong spot [00:06:17]. This is the problem with data augmentations when your dependent variable is pixel values or in some way connected to the independent variable — they need to be augmented together.

The dependent variable needs to go through all the geometric transformation as the independent variables.

To do this [00:07:10], every transformation has an optional `tfm_y` parameter:

```Python
augs = [RandomFlip(tfm_y=TfmType.COORD),
        RandomRotate(30, tfm_y=TfmType.COORD),
        RandomLighting(0.1,0.1, tfm_y=TfmType.COORD)]

tfms = tfms_from_model(f_model, sz, crop_type=CropType.NO, aug_tfms=augs, tfm_y=TfmType.COORD)
md = ImageClassifierData.from_csv(PATH, JPEGS, BB_CSV, tfms=tfms, continuous=True, bs=4)
```

`TrmType.COORD` indicates that the `y` value represents coordinate. This needs to be added to all the augmentations as well as `tfms_from_model` which is responsible for cropping, zooming, resizing, padding, etc.

```Python
idx = 3
fig, axes = plt.subplots(3, 3, figsize=(9, 9))

for i, ax in enumerate(axes.flat):
    x, y = next(iter(md.aug_dl))
    ima = md.val_ds.denorm(to_np(x))[idx]
    b = bb_hw(to_np(y[idx]))
    print(b)
    show_img(ima, ax=ax)
    draw_rect(ax, b)
```

```Python
[  1.  60. 221. 125.]
[  0.  12. 224. 211.]
[  0.   9. 224. 214.]
[  0.  21. 224. 202.]
[  0.   0. 224. 223.]
[  0.  55. 224. 135.]
[  0.  15. 224. 208.]
[  0.  31. 224. 182.]
[  0.  53. 224. 139.]
```

![Bounding box moves with the image and is in the right spot](/images/pascal_notebook_obj_det_one_img_bbox_plot.png)

##### `custom_head`

`learn.summary()` will run a small batch of data through a model and prints out the size of tensors at every layer. As you can see, right before the `Flatten` layer, the tensor has the shape of 512 by 7 by 7. So if it were a rank 1 tensor (i.e. a single vector) its length will be 25088 (512 * 7 * 7)and that is why our custom header’s input size is 25088. Output size is 4 since it is the bounding box coordinates.

![Model summary](/images/pascal_notebook_model_summary.png)
