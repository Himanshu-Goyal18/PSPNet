# Sementic Segmentation using PSPNet Model
## Overview
PSPNet (Pyramid Scene Parsing Network) is a state-of-the-art deep learning model for semantic segmentation, designed to understand the context of a scene by aggregating information at different scales. It leverages a pyramid pooling module to capture global context and refine the feature map for precise segmentation.

Key Features

- Global Context Understanding: Utilizes pyramid pooling to capture information from different regions of the image.
- Accurate Segmentation: Provides detailed and accurate segmentation results, even for complex scenes.
- Robust Architecture: Based on the ResNet architecture, known for its strong feature extraction capabilities.
  
Dataset:

The PSPNet model in this project is trained and tested using the Cityscapes dataset, which is a high-quality dataset widely used for semantic segmentation tasks. The Cityscapes data consists of labeled videos captured from vehicles driving through various cities in Germany. This specific version is a processed subsample from the Pix2Pix paper, containing still images from the original videos.

- Content: The dataset includes 2,975 training images and 500 validation images.
- Resolution: Each image is 256x512 pixels.
- Format: Each file is a composite image with the original photo on the left half and the corresponding semantic segmentation label on the right half.
This dataset is particularly valued for its detailed annotations and is one of the best resources for training and evaluating semantic segmentation models.






## Inference
### Import necessary libraries:
```python 
# Importing necessary libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

```

### Dataset Loading and Preprocessing
```python
train_folder="/kaggle/input/cityscapes-image-pairs/cityscapes_data/cityscapes_data/train/"
valid_folder="/kaggle/input/cityscapes-image-pairs/cityscapes_data/cityscapes_data/val/"

def get_images_masks(path):
    names=os.listdir(path)
    img_g,img_m=[],[]
    for name in names:
        img=cv2.imread(path+name)
        img=cv2.normalize(img,None,0,1,cv2.NORM_MINMAX,cv2.CV_32F)
        img=img[:,:,::-1]
        img_g.append(img[:,:256])
        img_m.append(np.reshape(img[:,256:],(256*256*3)))
        del img
    del names
    return img_g,img_m
        
train_imgs,train_masks=get_images_masks(train_folder)
valid_imgs,valid_masks=get_images_masks(valid_folder)

#train_len=len(train_imgs)
#valid_len=len(valid_imgs)
#print(f'Train Images:{train_len}\nValid Images:{valid_len}')
```






