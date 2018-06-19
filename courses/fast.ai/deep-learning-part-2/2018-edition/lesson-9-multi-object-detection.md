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

#### 1. Data

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

#### 2. Architecture

The architecture will be the same as the one we used for the classifier and bounding box regression, but we will just combine them. In other words, if we have `c` classes, then the number of activations we need in the final layer is 4 plus `c`. 4 for bounding box coordinates and `c` probabilities (one per class).

We’ll use an extra linear layer this time, plus some dropout, to help us train a more flexible model. In general, we want our custom head to be capable of solving the problem on its own if the pre-trained backbone it is connected to is appropriate. So in this case, we are trying to do quite a bit — classifier and bounding box regression, so just the single linear layer does not seem enough.

If you were wondering why there is no `BatchNorm1d` after the first `ReLU`, ResNet backbone already has `BatchNorm1d` as its final layer.

```Python
head_reg4 = nn.Sequential(
    Flatten(),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(25088, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(0.5),
    nn.Linear(256, 4 + len(cats))
)
models = ConvnetBuilder(f_model, 0, 0, 0, custom_head=head_reg4)

learn = ConvLearner(md, models)
learn.opt_fn = optim.Adam
```

Inspect what's inside `cats`:

```Python
print(type(cats))
print(len(cats))
print('%s, %s' % (cats[1], cats[2]))

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
<class 'dict'>
20
aeroplane, bicycle
```

#### 3. Loss Function

The loss function needs to look at these `4 + len(cats)` activations and decide if they are good — whether these numbers accurately reflect the position and class of the largest object in the image. We know how to do this. For the first 4 activations, we will use L1Loss just like we did before (L1Loss is like a Mean Squared Error — instead of sum of squared errors, it uses sum of absolute values). For rest of the activations, we can use cross entropy loss.

```Python
def detn_loss(input, target):
    """
    Loss function for the position and class of the largest object in the image.
    """    
    bb_t, c_t = target
    # bb_i: the 4 values for the bbox
    # c_i: the 20 classes `len(cats)`
    bb_i, c_i = input[:, :4], input[:, 4:]
    bb_i = F.sigmoid(bb_i) * 224 # scale bbox values to stay between 0 and 224 (224 is the max img width or height)
    bb_l = F.l1_loss(bb_i, bb_t) # bbox loss
    clas_l = F.cross_entropy(c_i, c_t) # object class loss
    # I looked at these quantities separately first then picked a multiplier
    # to make them approximately equal
    return bb_l + clas_l * 20

def detn_l1(input, target):
    """
    Loss function for the first 4 activations.

    L1Loss is like a Mean Squared Error — instead of sum of squared errors, it uses sum of absolute values
    """
    bb_t, _ = target
    bb_i = input[:, :4]
    bb_i = F.sigmoid(bb_i) * 224
    return F.l1_loss(V(bb_i), V(bb_t)).data

def detn_acc(input, target):
    """
    Accuracy
    """
    _, c_t = target
    c_i = input[:, 4:]
    return accuracy(c_i, c_t)
```

- `input` : activations.
- `target` : ground truth.
- `bb_t, c_t = target` : our custom dataset returns a tuple containing bounding box coordinates and classes. This assignment will destructure them.
- `bb_i, c_i = input[:, :4]`, `input[:, 4:]` : the first `:` is for the batch dimension. e.g.: 64 (for 64 images).
- `b_i = F.sigmoid(bb_i) * 224` : we know our image is 224 by 224. `Sigmoid` will force it to be between 0 and 1, and multiply it by 224 to help our neural net to be in the range of what it has to be.

:question: **Question:** As a general rule, is it better to put BatchNorm before or after ReLU [00:18:02]?

Jeremy would suggest to put it after a ReLU because BatchNorm is meant to move towards zero-mean one-standard deviation. So if you put ReLU right after it, you are truncating it at zero so there is no way to create negative numbers. But if you put ReLU then BatchNorm, it does have that ability and gives slightly better results. Having said that, it is not too big of a deal either way. You see during this part of the course, most of the time, Jeremy does ReLU then BatchNorm but sometimes does the opposite when he wants to be consistent with the paper.

:question: **Question:** What is the intuition behind using dropout after a BatchNorm? Doesn't BatchNorm already do a good job of regularizing [00:19:12]?

BatchNorm does an okay job of regularizing but if you think back to part 1 when we discussed a list of things we do to avoid overfitting and adding BatchNorm is one of them as is data augmentation. But it's perfectly possible that you'll still be overfitting. One nice thing about dropout is that is it has a parameter to say how much to drop out. Parameters are great specifically parameters that decide how much to regularize because it lets you build a nice big over parameterized model and then decide on how much to regularize it. Jeremy tends to always put in a drop out starting with p=0 and then as he adds regularization, he can just change the dropout parameter without worrying about if he saved a model he want to be able to load it back, but if he had dropout layers in one but no in another, it will not load anymore. So this way, it stays consistent.

Now we have out inputs and targets, we can calculate the L1 loss and add the cross entropy [00:20:39]:

```
bb_l = F.l1_loss(bb_i, bb_t)
clas_l = F.cross_entropy(c_i, c_t)
return bb_l + clas_l * 20
```

This is our loss function. Cross entropy and L1 loss may be of wildly different scales — in which case in the loss function, the larger one is going to dominate. In this case, Jeremy printed out the values and found out that if we multiply cross entropy by 20, that makes them about the same scale.

```Python
learn.crit = detn_loss
learn.metrics = [detn_acc, detn_l1]

# Set learning rate and train
lr = 1e-2
learn.fit(lr, 1, cycle_len=3, use_clr=(32, 5))

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
epoch      trn_loss   val_loss   detn_acc   detn_l1       
    0      71.055205  48.157942  0.754      33.202651 
    1      51.411235  39.722549  0.776      26.363626     
    2      42.721873  38.36225   0.786      25.658993     
[array([38.36225]), 0.7860000019073486, 25.65899333190918]
```

It is nice to print out information as you train, so we grabbed L1 loss and added it as metrics.

```Python
learn.save('reg1_0')

learn.freeze_to(-2)

lrs = np.array([lr/100, lr/10, lr])

learn.fit(lrs/5, 1, cycle_len=5, use_clr=(32, 10))

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
epoch      trn_loss   val_loss   detn_acc   detn_l1       
    0      36.650519  37.198765  0.768      23.865814 
    1      30.822986  36.280846  0.776      22.743629     
    2      26.792856  35.199342  0.756      21.564384     
    3      23.786961  33.644777  0.794      20.626075     
    4      21.58091   33.194585  0.788      20.520627     
[array([33.19459]), 0.788, 20.52062666320801]
```

```Python
learn.unfreeze()

learn.fit(lrs/10, 1, cycle_len=10, use_clr=(32, 10))

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
epoch      trn_loss   val_loss   detn_acc   detn_l1       
    0      19.133272  33.833656  0.804      20.774298 
    1      18.754909  35.271939  0.77       20.572007     
    2      17.824877  35.099138  0.776      20.494296     
    3      16.8321    33.782667  0.792      20.139132     
    4      15.968     33.525141  0.788      19.848904     
    5      15.356815  33.827995  0.782      19.483242     
    6      14.589975  33.49683   0.778      19.531291     
    7      13.811117  33.022376  0.794      19.462907     
    8      13.238251  33.300647  0.794      19.423868     
    9      12.613972  33.260653  0.788      19.346758     
[array([33.26065]), 0.7880000019073486, 19.34675830078125]
```

A detection accuracy is in the low 80's which is the same as what it was before. This is not surprising because ResNet was designed to do classification so we wouldn't expect to be able to improve things in such a simple way. It certainly wasn’t designed to do bounding box regression. It was explicitly actually designed in such a way to not care about geometry — it takes the last 7 by 7 grid of activations and averages them all together throwing away all the information about where everything came from.

Interestingly, when we do accuracy (classification) and bounding box at the same time, the L1 seems a little bit better than when we just do bounding box regression [00:22:46].

:memo: If that is counterintuitive to you, then this would be one of the main things to think about after this lesson since it is a really important idea.

The idea is this — figuring out what the main object in an image is, is kind of the hard part. Then figuring out exactly where the bounding box is and what class it is is the easy part in a way. So when you have a single network that’s both saying what is the object and where is the object, it’s going to share all the computation about finding the object. And all that shared computation is very efficient. When we back propagate the errors in the class and in the place, that’s all the information that is going to help the computation around finding the biggest object. So anytime you have multiple tasks which share some concept of what those tasks would need to do to complete their work, it is very likely they should share at least some layers of the network together. Later, we will look at a model where most of the layers are shared except for the last one.

Here are the result [00:24:34]. As before, it does a good job when there is single major object in the image.

![Training results](/images/pascal_notebook_single_obj_det_train_results.png)

### Multi Label Classification

[pascal-multi.ipynb](https://nbviewer.jupyter.org/github/fastai/fastai/blob/master/courses/dl2/pascal-multi.ipynb)

We want to keep building models that are slightly more complex than the last model so that if something stops working, we know exactly where it broke.

#### Setup

Global scope variables:

```Python
PATH = Path('data/pascal')
trn_j = json.load((PATH / 'pascal_train2007.json').open())
IMAGES, ANNOTATIONS, CATEGORIES = ['images', 'annotations', 'categories']
FILE_NAME, ID, IMG_ID, CAT_ID, BBOX = 'file_name', 'id', 'image_id', 'category_id', 'bbox'

cats = dict((o[ID], o['name']) for o in trn_j[CATEGORIES])
trn_fns = dict((o[ID], o[FILE_NAME]) for o in trn_j[IMAGES])
trn_ids = [o[ID] for o in trn_j[IMAGES]]

JPEGS = 'VOCdevkit/VOC2007/JPEGImages'
IMG_PATH = PATH / JPEGS
```

Define common functions.

Very similar to the first Pascal notebook, a model (single object detection).

```Python
def hw_bb(bb):
    # Example, bb = [155, 96, 196, 174]
    return np.array([ bb[1], bb[0], bb[3] + bb[1] - 1, bb[2] + bb[0] - 1 ])

def get_trn_anno():
    trn_anno = collections.defaultdict(lambda:[])

    for o in trn_j[ANNOTATIONS]:
        if not o['ignore']:
            bb = o[BBOX] # one bbox. looks like '[155, 96, 196, 174]'.
            bb = hw_bb(bb)
            trn_anno[o[IMG_ID]].append( (bb, o[CAT_ID]) )
    return trn_anno

trn_anno = get_trn_anno()

def show_img(im, figsize=None, ax=None):
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.set_xticks(np.linspace(0, 224, 8))
    ax.set_yticks(np.linspace(0, 224, 8))
    ax.grid()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return ax

def draw_outline(o, lw):
    o.set_path_effects( [patheffects.Stroke(linewidth=lw, foreground='black'),
                          patheffects.Normal()] )

def draw_rect(ax, b, color='white'):
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor=color, lw=2))
    draw_outline(patch, 4)

def draw_text(ax, xy, txt, sz=14, color='white'):
    text = ax.text(*xy, txt, verticalalignment='top', color=color, fontsize=sz, weight='bold')
    draw_outline(text, 1)

def bb_hw(a):
    return np.array( [ a[1], a[0], a[3] - a[1] + 1, a[2] - a[0] + 1 ] )

def draw_im(im, ann):
    # im is image, ann is annotations
    ax = show_img(im, figsize=(16, 8))
    for b, c in ann:
        # b is bbox, c is class id
        b = bb_hw(b)
        draw_rect(ax, b)
        draw_text(ax, b[:2], cats[c], sz=16)

def draw_idx(i):
    # i is image id
    im_a = trn_anno[i] # training annotations
    im = open_image(IMG_PATH / trn_fns[i]) # trn_fns is training image file names
    draw_im(im, im_a) # im_a is an element of annotation
```

#### Multi class

Setup.

```Python
MC_CSV = PATH / 'tmp/mc.csv'

trn_anno[12]

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
[(array([ 96, 155, 269, 350]), 7)]

mc = [ set( [cats[p[1]] for p in trn_anno[o] ] ) for o in trn_ids ]
mcs = [ ' '.join( str(p) for p in o ) for o in mc ] # stringify mc

print('mc:', mc[1])
print('mcs:', mcs[1])

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
mc: {'horse', 'person'}
mcs: horse person

df = pd.DataFrame({ 'fn': [trn_fns[o] for o in trn_ids], 'clas': mcs }, columns=['fn', 'clas'])
df.to_csv(MC_CSV, index=False)
```

:memo: One of the students pointed out that by using Pandas, we can do things much simpler than using `collections.defaultdict` and shared [this gist](https://gist.github.com/binga/1bc4ebe5e41f670f5954d2ffa9d6c0ed). The more you get to know Pandas, the more often you realize it is a good way to solve lots of different problems.

##### Model

Setup ResNet model and train.

```Python
f_model = resnet34
sz = 224
bs = 64

tfms = tfms_from_model(f_model, sz, crop_type=CropType.NO)
md = ImageClassifierData.from_csv(PATH, JPEGS, MC_CSV, tfms=tfms, bs=bs)

learn = ConvLearner.pretrained(f_model, md)
learn.opt_fn = optim.Adam

lr = 2e-2

learn.fit(lr, 1, cycle_len=3, use_clr=(32, 5))

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
epoch      trn_loss   val_loss   <lambda>                  
    0      0.319539   0.139347   0.9535    
    1      0.172275   0.080689   0.9724                    
    2      0.116136   0.075965   0.975                     

[array([0.07597]), 0.9750000004768371]

# Define learning rates to search
lrs = np.array([lr/100, lr/10, lr])

# Freeze the model till the last 2 layers as before
learn.freeze_to(-2)

# Refit the model
learn.fit(lrs/10, 1, cycle_len=5, use_clr=(32, 5))

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
epoch      trn_loss   val_loss   <lambda>                   
    0      0.071997   0.078266   0.9734    
    1      0.055321   0.082668   0.9737                     
    2      0.040407   0.077682   0.9757                     
    3      0.027939   0.07651    0.9756                     
    4      0.019983   0.07676    0.9763                     
[array([0.07676]), 0.9763000016212463]

# Save the model
learn.save('mclas')
learn.load('mclas')
```

##### Evaluate the model

```Python
y = learn.predict()
x, _ = next(iter(md.val_dl))
x = to_np(x)

fig, axes = plt.subplots(3, 4, figsize=(12, 8))

for i, ax in enumerate(axes.flat):
    ima = md.val_ds.denorm(x)[i]
    ya = np.nonzero(y[i] > 0.4)[0]
    b = '\n'.join(md.classes[o] for o in ya)
    ax = show_img(ima, ax=ax)
    draw_text(ax, (0, 0), b)
plt.tight_layout()
```

![Multi-class classification](/images/pascal_multi_notebook_img_classification_plot.png)

Multi-class classification is pretty straight forward [00:28:28]. One minor tweak is the use of `set` in this line so that each object type appear once:

```Python
mc = [ set( [cats[p[1]] for p in trn_anno[o] ] ) for o in trn_ids ]
```
