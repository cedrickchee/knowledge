# Lesson 8 - Object Detection

_These are my personal notes from fast.ai course and will continue to be updated and improved if I find anything useful and relevant while I continue to review the course to study much more in-depth. Thanks for reading and happy learning!_

## Topics

* A quick recap of what we learned in part 1.
* Introduces the new focus of this part of the course: cutting edge research.
* We’ll show you how to read an academic paper in a way that you don’t get overwhelmed by the notation and writing style.
* Another difference in this part is that we’ll be digging deeply into the source code of the fastai and PyTorch libraries.
* We’ll see how to use Python’s debugger to deepen your understand of what’s going on, as well as to fix bugs.
* The **main topic** of this lesson is object detection, which means getting a model to draw a box around every key object in an image, and label each one correctly.
  * Two main tasks: find and localize the objects, and classify them; we’ll use a single model to do both these at the same time.
  * Such multi-task learning generally works better than creating different models for each task—which many people find rather counter-intuitive.
  * To create this custom network whilst leveraging a pre-trained model, we’ll use fastai's flexible custom head architecture.

## Lesson Resources

* [Website](http://course.fast.ai/lessons/lesson8.html)
* [Video](https://youtu.be/Z0ssNAbe81M)
* [Wiki](http://forums.fast.ai/t/part-2-lesson-8-wiki)
* [Forum discussion](http://forums.fast.ai/t/part-2-lesson-8-in-class/13556)
* Jupyter Notebook and Code
  * [pascal.ipynb](https://nbviewer.jupyter.org/github/fastai/fastai/blob/master/courses/dl2/pascal.ipynb)

## Assignments

### Papers

* Must read
  * _WIP_
* Additional papers \(optional\)
  * _WIP_

### Other Resources

#### Blog Posts and Articles

* _WIP_

#### Other Useful Information

* _WIP_

#### Tips and Tricks

* _WIP_

### Code Snippets

* _WIP_

### Useful Tools and Libraries

* _WIP_

## My Notes

### Pascal Notebook

```Python
from pathlib import Path
import json
from PIL import ImageDraw, ImageFont
from matplotlib import patches, patheffects
torch.cuda.set_device(3)
```

You may find a line `torch.cuda.set_device(3)` left behind which will give you an error if you only have one GPU. This is how you select a GPU when you have multiple, so just set it to zero (`torch.cuda.set_device(0)`) or take out the line entirely.

#### Pascal VOC

We will be looking at the [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) dataset. It's quite slow, so you may prefer to download from [this mirror](https://pjreddie.com/projects/pascal-voc-dataset-mirror/). There are two different competition/research datasets, from 2007 and 2012. **We'll be using the 2007 version.** You can use the larger 2012 for better results, or even combine them (but be careful to avoid data leakage between the validation sets if you do this).

:memo: How to download the dataset:

```bash
# Make new directory to keep our downloaded files
cd ~/fastai/courses/dl2
ln -s ~/data data && cd $_
mkdir pascal && cd $_

# Download images
aria2c --file-allocation=none -c -x 5 -s 5 http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar

# Download annotations - bounding boxes
curl -OL https://storage.googleapis.com/coco-dataset/external/PASCAL_VOC.zip

# Extract tar and zip files
tar -xf VOCtrainval_06-Nov-2007.tar
unzip PASCAL_VOC.zip

mv PASCAL_VOC/*.json .
rmdir PASCAL_VOC
```

#### Pathlib

Unlike previous lessons, we are using the python 3 standard library `pathlib` for our paths and file access. Note that it returns an OS-specific class (on Linux, `PosixPath`) so your output may look a little different. Most libraries than take paths as input can take a pathlib object - although some (like `cv2`) can't, in which case you can use `str()` to convert it to a string.

:memo: [pathlib cheat sheet](http://pbpython.com/pathlib-intro.html)

```Python
PATH = Path('data/pascal')
list(PATH.iterdir())
```

```Python
[PosixPath('data/pascal/pascal_train2012.json'),
 PosixPath('data/pascal/VOCtrainval_06-Nov-2007.tar'),
 PosixPath('data/pascal/pascal_train2007.json'),
 PosixPath('data/pascal/VOCdevkit'),
 PosixPath('data/pascal/pascal_val2007.json'),
 PosixPath('data/pascal/pascal_test2007.json'),
 PosixPath('data/pascal/pascal_val2012.json'),
 PosixPath('data/pascal/PASCAL_VOC.zip')]
```

##### Python 3 Generators

Generators are a simple and powerful tool for creating iterators. They are written like regular functions but use the `yield` statement whenever they want to return data.

```Python
# Turn a generator into a list
list(PATH.iterdir())

# List comprehension
[i for i in PATH.iterdir()]

# Loop over a list
for i in PATH.iterdir():
    print(i)
```

The reason that things generally return generators is that if the directory had 10 million items in, you don’t necessarily want 10 million long list. Generator lets you do things "lazily".

#### Loading Pascal VOC Annotations

```Python
trn_j = json.load( (PATH / 'pascal_train2007.json').open() )
trn_j.keys()
```

`pascal_train2007.json` file contains not the images but the bounding boxes and the classes of the objects.

:memo: `json.load()` deserialize file-like object containing a JSON document to a Python object.

##### Images

```Python
IMAGES, ANNOTATIONS, CATEGORIES = ['images', 'annotations', 'categories']
trn_j[IMAGES][:5]
```

```Python
[{'file_name': '000012.jpg', 'height': 333, 'width': 500, 'id': 12},
 {'file_name': '000017.jpg', 'height': 364, 'width': 480, 'id': 17},
 {'file_name': '000023.jpg', 'height': 500, 'width': 334, 'id': 23},
 {'file_name': '000026.jpg', 'height': 333, 'width': 500, 'id': 26},
 {'file_name': '000032.jpg', 'height': 281, 'width': 500, 'id': 32}]
```

##### Annotations

```Python
trn_j[ANNOTATIONS][:2]
```

```Python
[{'segmentation': [[155, 96, 155, 270, 351, 270, 351, 96]],
  'area': 34104,
  'iscrowd': 0,
  'image_id': 12,
  'bbox': [155, 96, 196, 174],
  'category_id': 7,
  'id': 1,
  'ignore': 0},
 {'segmentation': [[184, 61, 184, 199, 279, 199, 279, 61]],
  'area': 13110,
  'iscrowd': 0,
  'image_id': 17,
  'bbox': [184, 61, 95, 138],
  'category_id': 15,
  'id': 2,
  'ignore': 0}]
```

Schema:
- `bbox` : column (x coord, origin of top left), row (y coord, origin of top left), height, width
- `image_id` : you'd have join this up with `trn_j[IMAGES]` (above) to find `file_name` etc.
- `category_id` : see `trn_j[CATEGORIES]` (below)
- `segmentation` : polygon segmentation (we will not be using them)
- `ignore` : we will ignore the `ignore` flags
- `iscrowd` : specifies that it is a crowd of that object, not just one of them

##### Categories

```Python
trn_j[CATEGORIES][:8]
```

```Python
[{'supercategory': 'none', 'id': 1, 'name': 'aeroplane'},
 {'supercategory': 'none', 'id': 2, 'name': 'bicycle'},
 {'supercategory': 'none', 'id': 3, 'name': 'bird'},
 {'supercategory': 'none', 'id': 4, 'name': 'boat'},
 {'supercategory': 'none', 'id': 5, 'name': 'bottle'},
 {'supercategory': 'none', 'id': 6, 'name': 'bus'},
 {'supercategory': 'none', 'id': 7, 'name': 'car'},
 {'supercategory': 'none', 'id': 8, 'name': 'cat'}]
```

#### Convert VOC's Bounding Box

Convert VOC's height/width into top-left/bottom-right, and switch x/y coords to be consistent with numpy.

```Python
# bb is an example of VOC's bbox
# x, y, width, height -> top-left coord (y, x), bottom-right coord (y+height-1, x+width-1) 
#ix   0    1   2    3
bb = [155, 96, 196, 174]
bb[1], bb[0], bb[3] + bb[1] - 1, bb[2] + bb[0] - 1 # (96, 155, 269, 350)
```

```Python
def hw_bb(bb):
    return np.array([ bb[1], bb[0], bb[3] + bb[1] - 1, bb[2] + bb[0] - 1 ])
```

```Python
trn_anno = collections.defaultdict(lambda:[])

for o in trn_j[ANNOTATIONS]:
    if not o['ignore']:
        bb = o[BBOX]  # one bbox. looks like '[155, 96, 196, 174]'.
        bb = hw_bb(bb) # output '[96 155 269 350]'.
        trn_anno[o[IMG_ID]].append( (bb, o[CAT_ID]) )
```

```Python
list(trn_anno.values())[0]
```

```Python
[(array([ 96, 155, 269, 350]), 7)] # [bbox], class id
```

##### Python's defaultdict

A `defaultdict` is useful any time you want to have a default dictionary entry for new keys [00:55:05]. If you try and access a key that doesn’t exist, it magically makes itself exist and it sets itself equal to the return value of the function you specify (in this case `lambda:[]`).

#### Covert Back to VOC Bounding Box Format

Some libs take VOC format bounding boxes, so this let’s us convert back when required [1:02:23]:

```Python
def bb_hw(a):
    return np.array( [ a[1], a[0], a[3] - a[1] + 1, a[2] - a[0] + 1 ] )
```
