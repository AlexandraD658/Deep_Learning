---
title: Talk slides template
tags: Templates, Talk
description: View the slide with "Slide Mode".
---

# Deep Learning - Notebook 2 

<!-- Put the link to this slide here so people can follow -->
Complete notebook: https://www.kaggle.com/vaillant/dfdc-3d-2d-inc-cutmix-with-3d-model-fix

---

Summary :

- Competition description 
- Data details
- Evaluation description
- Notebook details
- Concept details

---

## Competition description 

- The goal of this Kaggle competition is to develop a deep learning algorithm capable of identifying tricked videos and real videos. 
But what exactly is Deep Fake?
Deepfake is an artificial intelligence technique that consists in generating very realistic synthetic data. It can be applied to different types of data, for example an image, a video, a sound (music, voice), or even writing.
We can thus generate a realistic image from a drawing, colorize images, transfer a style, restore an image, or even give a face expressions, change the gender of the person, and even exchange faces.

---

## Data details - Kaggle 

* **train_sample_videos.zip** - a ZIP file containing a sample set of training videos and a metadata.json with labels. the full set of training videos is available through the links provided above.
* **sample_submission.csv** - a sample submission file in the correct format.
* **test_videos.zip** - a zip file containing a small set of videos to be used as a public validation set.


> the colums are difined as below



| filename | label | original | split  |
| -------- | -------- | -------- | -------- |
|  video's filename| REAL/FAKE     | Original video in case that a train set video is fake     | always equal to train     |


---


## Evaluation description

Submissions are scored on **log loss**:

![](https://i.imgur.com/iDQgB6j.png)

*     n is the number of videos being predicted
*     y^i is the predicted probability of the video being FAKE
*     yi is 1 if the video is FAKE, 0 if REAL
*     log() is the natural (base e) logarithm



# 

> Log Loss is a loss function used in (multinomial) logistic regression based on probabilities. It returns y_pred probabilities for its training data y_true.

> The log loss is only defined for two or more labels. For any given problem, a lower log-loss value means better predictions. 

> Log Loss is a slight twist on something called the Likelihood Function. In fact, Log Loss is -1 * the log of the likelihood function.

> Each prediction is between 0 and 1. If you multiply enough numbers in this range, the result gets so small that computers can't keep track of it. So, as a clever computational trick, we instead keep track of the log of the Likelihood. This is in a range that's easy to keep track of. We multiply this by negative 1 to maintain a common convention that lower loss scores are better.





---




## Notebook details

### Librairies used across all the file 

- **cv2**

> It mainly focuses on image processing, video capture and analysis including features like face detection and object detection. It is used here for video capture, cout the number of frames in the video file, get the height and width of the frame, resize...

```
cap = cv2.VideoCapture(filename)
n_frames_in = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
```


- **MTCNN from facenet_pytorch** 

> detects faces and facial landmarks on images
```
mtcnn = MTCNN(margin=0, keep_all=True, post_process=False, select_largest=False, device='cuda:0', thresholds=MTCNN_THRESHOLDS, factor=MMTNN_FACTOR)
```
- **Image from PIL** 
> PIL is the Python Imaging Library which provides the python interpreter with image editing capabilities. It used for opening, manipulating, and saving many different image file formats. The Image module provides a class with the same name which is used to represent a PIL image.
```
from PIL import Image
pil_img = Image.fromarray(frame_in)
```

- **torchvision**
> The models subpackage contains definitions of models for addressing different tasks, including: image classification, pixelwise semantic segmentation, object detection, instance segmentation, person keypoint detection and video classification.

> It's provide here models for action recognition pre-trained on Kinetics-400.

> mc3_18 : constructor for 18 layer Mixed Convolution network.

> r2plus1d_18 : constructor for the 18 layer deep R(2+1)D network
```
from torchvision.models.video import mc3_18, r2plus1d_18
model1 = mc3_18(num_classes=2)

model2 = r2plus1d_18(num_classes=2)
```
- **tqdm** 
> TQDM is a progress bar library with good support for nested loops and Jupyter/IPython notebooks.

> here is a good example of what it does : https://www.geeksforgeeks.org/python-how-to-make-a-terminal-progress-bar-using-tqdm/
```
from tqdm.notebook import tqdm 

for videopath, faces in tqdm(faces_by_videopath.items(), total=len(faces_by_videopath)):
    ...........
```


- **glob** 
> In Python, the glob module is used to retrieve files/pathnames matching a specified pattern. The pattern rules of glob follow standard Unix path expansion rules.
```
videopaths = sorted(glob(os.path.join(INPUT_DIR, "*.mp4")))
```


- **defaultdict from collections** 
> Defaultdict is a sub-class of the dict class that returns a dictionary-like object. The functionality of both dictionaries and defualtdict are almost same except for the fact that defualtdict never raises a KeyError . It provides a default value for the key that does not exists.
```
predictions = defaultdict(list)
```

- **partial from functools** 
> Partial functions allow one to derive a function with x parameters to a function with fewer parameters and fixed values set for the more limited function.

>Here is a good example of what it does : https://www.learnpython.org/en/Partial_functions#:~:text=You%20can%20create%20partial%20functions,for%20the%20more%20limited%20function.
```
from functools import partial

downsample = partial(downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
```

### Understand main functions 

1. **load_video function** 
> * Extract frames from each video using cv2
> * select frames and resize them.


2. **get_roi_for_each_face function** 
> after detecting face by the MTCNN model, the function extract each **ROI** (Region Of Interest) corresponding to all faces. 


![](https://i.imgur.com/WhGP0ia.jpg)


3. **Find faces functions** 
> * Get cordinates for boxes and ROI
> * Doing linear interpolation for data augmentation
> * Conditions on frame if detecting face of not
> * Get all ROI for unique face

4. **3D CNNs functions** 
> * Build a **InceptionI3D** model for classification

![](https://i.imgur.com/PHHUsuK.png)


> * **ResNet** : A residual neural network is an artificial neural network based on known constructions of the pyramidal cells of the cerebral cortex. Residual neural networks do this by using jump connections or shortcuts to jump over certain layers.

![](https://i.imgur.com/oaGxFsr.png)







---
## Concept details : CutMix

> Data augmentation is a very powerful technique used to artificially create variations in existing images to expand an existing image data set. **It acts as a regularizer and helps reduce overfitting** when training a model which **improve the model's performance**. 
> Data augmentation is also used for **generating more data** without collecting new data which helps in increasing the diversity of the dataset

> As data augmentation techniques tend to increase the performance of models, efforts were also made to improve them. One such technique introduced recently is **CutMix**.

> In CutMix augmentation we cut and paste random patches between the training images. The ground truth labels are mixed in proportion to the area of patches in the images.

![](https://i.imgur.com/wQsukDi.png)

1. > ImageNet Cls : ImageNet classification
1. > ImageNet Loc : ImageNet localization
1. > VOC Det : Object detection
 

>   CutMix increases localization ability by making the model to focus on less discriminative parts of the object being classified and hence is also well suited for tasks like object detection.






---
> Thanks for reading! :100: 
