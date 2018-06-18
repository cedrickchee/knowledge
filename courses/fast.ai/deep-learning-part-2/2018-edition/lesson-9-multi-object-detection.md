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

### Single Object Detection

We combine the two to create something that can classify and localize the largest object in each image.

There are 3 things that we need to do to train a neural network:

1. Data
2. Architecture
3. Loss function

#### Data

We need a `ModelData` object whose independent variable is the images, and dependent variable is a tuple of bounding box coordinates and class label.

There are several ways to do this, but here's a particularly 'lazy' and convenient way that is to create two `ModelData` objects representing the two different dependent variables we want:
1. bounding boxes coordinates
2. class

```Python
f_model=resnet34
sz=224
bs=64

# Split dataset for validation set
val_idxs = get_cv_idxs(len(trn_fns))

tfms = tfms_from_model(f_model, sz, crop_type=CropType.NO, tfm_y=TfmType.COORD, aug_tfms=augs)
```

`ModelData` for bounding box of the largest object:

```Python
md = ImageClassifierData.from_csv(PATH, JPEGS, BB_CSV, tfms=tfms,
                                  bs=bs, continuous=True, val_idxs=val_idxs)
```

`ModelData` for classification of the largest object:

```Python
md2 = ImageClassifierData.from_csv(PATH, JPEGS, CSV, tfms=tfms_from_model(f_model, sz))
```

Let's break that down a bit.

```Python
CSV_FILES = PATH / 'tmp'

!ls {CSV_FILES}

bb.csv	lrg.csv
```

**`BB_CSV`** is the CSV file for bounding boxes of the largest object. This is simply a regression with 4 outputs (predicted values). So we can use a CSV with multiple 'labels'.

```Python
!head -n 10 {CSV_FILES}/bb.csv

fn,bbox
008197.jpg,186 450 226 496
008199.jpg,84 363 374 498
008202.jpg,110 190 371 457
008203.jpg,187 37 359 303
000012.jpg,96 155 269 350
008204.jpg,144 142 335 265
000017.jpg,77 89 335 402
008211.jpg,181 77 499 281
008213.jpg,125 291 166 330
```

**`CSV`** is the CSV file for large object classification. It contains the CSV data of image filename and class of the largest object (from annotations JSON).

```Python
!head -n 10 {CSV_FILES}/lrg.csv

fn,cat
008197.jpg,car
008199.jpg,person
008202.jpg,cow
008203.jpg,sofa
000012.jpg,car
008204.jpg,person
000017.jpg,horse
008211.jpg,person
008213.jpg,chair
```

A **dataset** can be anything with `__len__` and `__getitem__`. Here's a dataset that adds a second label to an existing dataset:

```Python
class ConcatLblDataset(Dataset):
    """
    A dataset that adds a second label to an existing dataset.
    """
    
    def __init__(self, ds, y2):
        """
        Initialize

        ds: contains both independent and dependent variables
        y2: contains the additional dependent variables
        """
        self.ds, self.y2 = ds, y2
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, i):
        x, y = self.ds[i]

        # returns an independent variable and the combination of two dependent variables.
        return (x, (y, self.y2[i]))
```

We'll use it to add the classes to the bounding boxes labels.

```Python
trn_ds2 = ConcatLblDataset(md.trn_ds, md2.trn_y)
val_ds2 = ConcatLblDataset(md.val_ds, md2.val_y)
```

Here is an example dependent variable:

```Python
# Grab the two 'label' (bounding box & class) from a record in the validation dataset.
val_ds2[0][1] # record at index 0. labels at index 1, input image(x) at index 0 (we are not grabbing this)
```

```Python
(array([  0.,   1., 223., 178.], dtype=float32), 14)
```

We can replace the dataloaders' datasets with these new ones.

```Python
md.trn_dl.dataset = trn_ds2
md.val_dl.dataset = val_ds2
```

We have to `denorm`alize the images from the dataloader before they can be plotted.

```Python
idx = 9

x, y = next(iter(md.val_dl)) # x is image array, y is labels
ima = md.val_ds.ds.denorm(to_np(x))[idx] # reverse the normalization done to a batch of images.
b = bb_hw(to_np(y[0][idx]))
b
```

```Python
array([134., 148.,  36.,  48.])
```

Plot image and object bounding box.

```Python
ax = show_img(ima)
draw_rect(ax, b)
draw_text(ax, b[:2], md2.classes[y[1][idx]])
```

![Single image object detection](/images/pascal_notebook_single_obj_det.png)

Let's break that code down a bit.

- Inspect `y` variable:

```Python
print(f'type of y: {type(y)}, y length: {len(y)}')
print(y[0].size()) # bounding box top-left coord & bottom-right coord values
print(y[1].size()) # object category (class)

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
type of y: <class 'list'>, y length: 2
torch.Size([64, 4])
torch.Size([64])
```

```Python
# y[0] returns 64 set of bounding boxes (labels).
# Here's we only grab the first 2 images' bounding boxes. The returned data type is PyTorch FloatTensor in GPU.
print(y[0][:2])

# Grab the first 2 images' object classes. The returned data type is PyTorch LongTensor in GPU.
print(y[1][:2])

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
   0    1  223  178
   7  123  186  194
[torch.cuda.FloatTensor of size 2x4 (GPU 0)]


 14
  3
[torch.cuda.LongTensor of size 2 (GPU 0)]
```

- Inspect `x` variable:
  - data from GPU

    ```Python
    x.size() # batch of 64 images, each image with 3 channels and size of 224x224

    # -----------------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------------
    torch.Size([64, 3, 224, 224])
    ```
  - data from CPU

    ```Python
    to_np(x).shape

    # -----------------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------------
    (64, 3, 224, 224)  
    ```