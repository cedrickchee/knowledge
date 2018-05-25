# Lesson 8 - Object Detection

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

