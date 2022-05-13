---
title:  "**EfficientNet 구현하기**"
excerpt: "EfficientNet 구현"

categories:
 - reproduce
tags:
 - [CNN]
 - [Deep Learning]
 - [Transfer Learning]
 - [SOTA]

toc: true
toc_sticky: true

date: 2022-05-12
last_modified_at: 2022-05-12

published: true
---

## 개요

- EfficientNet은 SOTA 중에서 획기적인 방법으로 파라미터 수를 줄이고 정확도를 높인 방법이다.
- EfficientNet의 특징과 성능을 알아보고 Keras에 저장되어 있는 EfficientNet을 활용하여 Cifar100 이미지셋을 분류해보자.

## EfficientNet이란?

- CNN의 정확도를 높일 때 우린 주로 3가지 요소를 주목한다. 모델의 깊이, 너비, 그리고 입력 데이터의 해상도이다. 
- 다음 그림은 순서대로 모델의 깊이, 너비, 입력 데이터의 해상도에 따른 모델의 정확도 변화를 보여준다. 
![image-20220512180708250](/assets/images/image-20220512180708250.png)

- 최적의 깊이, 너비, 해상도를 찾기 힘든 가장 큰 원인은, 각 파라미터들이 서로 영향을 주는 관계에 놓여 있다는 것이다. 이러한 제약 때문에 대부분의 Conventional methods는 이 파라미터들 중 하나만을 골라 진행되었다. 

- 다음은 각 변수들의 특징과 trade-off들이다. 
1. Depth(d): 깊이에 대해서 직관적으로 말할 수 있는 것은, 모델의 깊이가 깊어질 수록 네트워크가 더 풍부하고 많은 feature들을 포착할 수 있을 것이라는 기대이다. 하지만 모델의 깊이는 vanishing gradient 문제로부터 자유롭지 못하다. 이를 극복하기 위한 다양한 방법이 출현하였지만, 여전히 매우 깊은 모델의 정확도는 떨어진다. 따라서 최고의 정확도를 위해서는 적절한 수준의 깊이가 요구된다.

2. Width(w): 모델의 너비를 스케일링하는 것은 작은 사이즈의 모델에서 주로 사용된다. 모델의 너비가 넓으면 넓을수록 더 미세한 수준의 feature들을 잘 잡아내는 경향을 가지며, 학습시키기에도 용이하다. 하지만 너비가 너무 넓어져버리면 더 높은 차원의 feature를 포착하기 힘들어진다. 이 또한 적절한 수준을 찾는 것이 필요하다. 

3. Resolution(r): 입력 데이터가 더 높은 해상도를 가질수록 네트워크는 더 미세한 수준의 feature를 잡아내기 용이해진다. 과거 224x224의 이미지를 사용하던 ConvNet에서 지금은 무려 600x600 수준의 이미지 해상도까지 늘릴 정도로 해상도는 정확도를 높이는데 큰 기여를 한다. 해상도를 높이는 것은 좋지만 위 그림을 보면 알 수 있듯이, 특정 수준을 넘어가면 정확도 향상의 정도가 작아져서 계산량의 증가량 대비 효율성이 매우 떨어지게 된다. FLOPS를 중시하는 최신 기류를 미루어보아 해상도는 적절한 수준을 유지해야할 것이다. 


## EfficientNet의 원리

![image-20220512181709344](/assets/images/image-20220512181709344.png)
- EfficientNet 개발진들은 모델의 깊이, 너비, 해상도 사이에 수식적인 관계가 존재할 것이라고 판단하였고 위와 같은 식을 정립하였다. 
- N은 전체 네트워크를 의미하고, 네트워크는 F layer들의 곱으로 표현 가능하다.
- 각 F layer 내부에는 입력 데이터 X의 텐서가 존재하고, 이 텐서는 깊이, 너비, 해상도에 따라 변화한다. 
- 결국, 위 식을 최대화하는 깊이, 너비, 해상도를 찾는 것이 EfficientNet의 목표이다. 
- 이 방법은 평소의 Model scaling과 같이 최고의 F구조를 찾는 것이 아니라, F를 고정시킨 채로 네트워크의 깊이, 너비, 해상도를 바꾼다는 점에서 획기적이다. 

- Compound Scaling Method
- CSM은 compound 계수 Φ를 통해서 네트워크의 깊이, 너비, 해상도를 한 번에 스케일링 할 수 있다. 
![image-20220513021526410](/assets/images/image-20220513021526410.png)

- 여기서 α, β, Г는 Grid search를 통해서 정할 수 있는 상수이다. Φ는 하이퍼파라미터로, 얼마나 많은 resource를 모델 스케일링에 할당할 수 있는지를 조절하는 파라미터이다.
- FLOPS는 αβ²Г²의 Φ승 만큼 증가하므로 αβ²Г²=2로 설정하여 FLOPS가 2배 이상으로 증가하지 않도록 억제하였다. 

- **Step 1** Φ = 1로 고정한 후에 α,β,Г에 대해서 그리드 서치를 하여 최적의 α,β,Г를 찾는다.
- **Step 2** α,β,Г를 상수로 고정한 후에 <Equation 2>를 이용해서 Φ에 변화를 주며 최적의 accuracy를 주는 Φ를 찾는다. 



## EfficientNet 구현

- Cifar100 Image set과 EfficientNet을 Keras Model application을 통해 불러온다. 해당 모델은 학습이 안되도록 잠궈둔다. 
- B0 ~ B7 순으로 모델의 크기가 커지며, 더 많은 파라미터를 보유한다. 
- 불러온 base model의 최후단에 Batch Normalization layer와 Drop out layer를 붙인다. 이 작업은 Gradient Vanishing problem을 예방해줄 것이다.
- Global average pooling 2D를 통해서 flatten 해준 다음, softmax를 통해서 100개의 확률을 출력하는 dense layer를 마지막으로 모델을 완성시킨다. 
- Transfer learning을 진행한다. 



- **주의사항**
1. 학습이 완료된 모델의 baseline 부분의 일정부분을 학습 가능하도록 할 때, Batch Normalization layer는 계속 inference mode로 유지시키기 위해 피해서 unfreeze한다. 
2. 해당 모델 내부에는 ResNet처럼 layer들을 건너뛰어서 연결되는 layer들이 존재하므로 block단위로 unfreeze해준다. 


```python
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from collections import defaultdict

from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import glorot_normal, RandomNormal, Zeros
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation, Dropout
from keras import backend as K
```


```python
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
tf.debugging.set_log_device_placement(False)
```


```python
tf.config.list_physical_devices('GPU')
```




    [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]




```python
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
```




    [name: "/device:CPU:0"
     device_type: "CPU"
     memory_limit: 268435456
     locality {
     }
     incarnation: 5378870955904894258
     xla_global_id: -1,
     name: "/device:GPU:0"
     device_type: "GPU"
     memory_limit: 5738397696
     locality {
       bus_id: 1
       links {
       }
     }
     incarnation: 2061542315244942730
     physical_device_desc: "device: 0, name: NVIDIA GeForce RTX 3070, pci bus id: 0000:08:00.0, compute capability: 8.6"
     xla_global_id: 416903419]




```python
!nvidia-smi
```

    Fri May 13 01:48:00 2022       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 497.09       Driver Version: 497.09       CUDA Version: 11.5     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  NVIDIA GeForce ... WDDM  | 00000000:08:00.0  On |                  N/A |
    |  0%   49C    P2    44W / 270W |   1626MiB /  8192MiB |      2%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |    0   N/A  N/A      1424    C+G   Insufficient Permissions        N/A      |
    |    0   N/A  N/A      6880    C+G   ...root\Office16\WINWORD.EXE    N/A      |
    |    0   N/A  N/A     11004    C+G   C:\Windows\explorer.exe         N/A      |
    |    0   N/A  N/A     12516    C+G   ...2txyewy\TextInputHost.exe    N/A      |
    |    0   N/A  N/A     12756    C+G   ...8bbwe\Microsoft.Notes.exe    N/A      |
    |    0   N/A  N/A     12920    C+G   ...wekyb3d8bbwe\Video.UI.exe    N/A      |
    |    0   N/A  N/A     13636    C+G   ...5n1h2txyewy\SearchApp.exe    N/A      |
    |    0   N/A  N/A     13732    C+G   ...artMenuExperienceHost.exe    N/A      |
    |    0   N/A  N/A     15772    C+G   ...ekyb3d8bbwe\YourPhone.exe    N/A      |
    |    0   N/A  N/A     16584    C+G   ...\LeagueClientUxRender.exe    N/A      |
    |    0   N/A  N/A     17172    C+G   ...perience\NVIDIA Share.exe    N/A      |
    |    0   N/A  N/A     17340    C+G   ...perience\NVIDIA Share.exe    N/A      |
    |    0   N/A  N/A     17476    C+G   ...7.0.5.0\GoogleDriveFS.exe    N/A      |
    |    0   N/A  N/A     18108    C+G   ...me\Application\chrome.exe    N/A      |
    |    0   N/A  N/A     21968    C+G   ...qxf38zg5c\Skype\Skype.exe    N/A      |
    |    0   N/A  N/A     23060    C+G   ...\Programs\OP.GG\OP.GG.exe    N/A      |
    |    0   N/A  N/A     25668      C   ...Tensorflow-gpu\python.exe    N/A      |
    |    0   N/A  N/A     25944    C+G   ...qxf38zg5c\Skype\Skype.exe    N/A      |
    |    0   N/A  N/A     26456    C+G   ...y\ShellExperienceHost.exe    N/A      |
    |    0   N/A  N/A     27468    C+G   ...ram Files\LGHUB\lghub.exe    N/A      |
    |    0   N/A  N/A     29704    C+G   ...8wekyb3d8bbwe\GameBar.exe    N/A      |
    |    0   N/A  N/A     30308    C+G   ...bbwe\Microsoft.Photos.exe    N/A      |
    +-----------------------------------------------------------------------------+


# 데이터 불러오기


```python

import tensorflow as tf
from tensorflow.keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(
    label_mode='fine'
)
```

# 사용되는 라이브러리

# 사용되는 함수들


```python
import matplotlib.pyplot as plt

def plot_image(i, predictions_array, true_label, img):
  plt.figure(figsize = (3,6))
  plt.subplot(2,1,1)
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img)

  true_label = np.argmax(true_label)

  predicted_label_index = np.argpartition(predictions_array, -5)[-5:]
  predictions = predictions_array[predicted_label_index]
  prediction_index_n_score = dict(zip(predicted_label_index, predictions))
  prediction_index_n_score = sorted(prediction_index_n_score.items(), key=lambda x: x[1], reverse=True)

  top1_index = prediction_index_n_score[0][0]
  top2_index = prediction_index_n_score[1][0]
  top3_index = prediction_index_n_score[2][0]
  top4_index = prediction_index_n_score[3][0]
  top5_index = prediction_index_n_score[4][0]
  print(predicted_label_index)
  if top1_index == true_label:
    color = 'blue'
  elif true_label in predicted_label_index:
    color = 'green'
  else:
    color = 'red'

  plt.xlabel(
      f'{class_names[top1_index]}, {predictions_array[top1_index]*100:.2f}% - {class_names[true_label]}\n\
      {class_names[top2_index]}, {predictions_array[top2_index]*100:.2f}% - {class_names[true_label]}\n\
      {class_names[top3_index]}, {predictions_array[top3_index]*100:.2f}% - {class_names[true_label]}\n\
      {class_names[top4_index]}, {predictions_array[top4_index]*100:.2f}% - {class_names[true_label]}\n\
      {class_names[top5_index]}, {predictions_array[top5_index]*100:.2f}% - {class_names[true_label]}', color = color)
  
  plt.show()

  
```


```python

```

# 데이터 Normalize & Categorize


```python
from tensorflow.keras import preprocessing

Eff_x_train = tf.keras.applications.efficientnet.preprocess_input(
    x_train, data_format=None
)
Eff_x_test = tf.keras.applications.efficientnet.preprocess_input(
    x_test, data_format=None
)
Dense_x_train = tf.keras.applications.densenet.preprocess_input(
    x_train, data_format=None
)
Dense_x_test = tf.keras.applications.densenet.preprocess_input(
    x_test, data_format=None
)


x_train_proc = x_train/255.0
x_test_proc = x_test/255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

image_gen = preprocessing.image.ImageDataGenerator(rotation_range = 40, 
                                                   width_shift_range = 0.2,
                                                   height_shift_range = 0.2, 
                                                   shear_range = 0.2,
                                                   zoom_range = 0.2, 
                                                   horizontal_flip = True)

train_gen = image_gen.flow(x_train, y=y_train, batch_size=32)
Eff_train_gen = image_gen.flow(Eff_x_train, y=y_train, batch_size=32)
Dense_train_gen = image_gen.flow(Dense_x_train, y=y_train, batch_size=32)
```

# 학습 데이터의 Lablels


```python
class_names = [
'apple', # id 0
'aquarium_fish',
'baby',
'bear',
'beaver',
'bed',
'bee',
'beetle',
'bicycle',
'bottle',
'bowl',
'boy',
'bridge',
'bus',
'butterfly',
'camel',
'can',
'castle',
'caterpillar',
'cattle',
'chair',
'chimpanzee',
'clock',
'cloud',
'cockroach',
'couch',
'crab',
'crocodile',
'cup',
'dinosaur',
'dolphin',
'elephant',
'flatfish',
'forest',
'fox',
'girl',
'hamster',
'house',
'kangaroo',
'computer_keyboard',
'lamp',
'lawn_mower',
'leopard',
'lion',
'lizard',
'lobster',
'man',
'maple_tree',
'motorcycle',
'mountain',
'mouse',
'mushroom',
'oak_tree',
'orange',
'orchid',
'otter',
'palm_tree',
'pear',
'pickup_truck',
'pine_tree',
'plain',
'plate',
'poppy',
'porcupine',
'possum',
'rabbit',
'raccoon',
'ray',
'road',
'rocket',
'rose',
'sea',
'seal',
'shark',
'shrew',
'skunk',
'skyscraper',
'snail',
'snake',
'spider',
'squirrel',
'streetcar',
'sunflower',
'sweet_pepper',
'table',
'tank',
'telephone',
'television',
'tiger',
'tractor',
'train',
'trout',
'tulip',
'turtle',
'wardrobe',
'whale',
'willow_tree',
'wolf',
'woman',
'worm',
]
```


```python
print(x_train.shape)
print(y_train.shape)
```

    (50000, 32, 32, 3)
    (50000, 100)


# Image Resizing


```python
from __future__ import absolute_import, division, print_function, unicode_literals
import os

IMG_SIZE = 456
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

input_tensor = tf.keras.Input(shape=(32,32,3))
resized_image = tf.keras.layers.Lambda(lambda image: tf.image.resize(image, (IMG_SIZE, IMG_SIZE)))(input_tensor)


```


```python
x_train.shape
```




    (50000, 32, 32, 3)



# Base Model Importing


```python
with tf.device('/CPU:0'):

    Eff_model = tf.keras.applications.EfficientNetB5(
        input_tensor=resized_image,
        include_top=False,
        weights="imagenet"
    )
    
    Eff_model.trainable = False

```


```python
#EfficientNet 모델의 형태를 보여준다.
#tf.keras.utils.plot_model(Eff_model, show_shapes=True)
```

# Base Model에 내가 만든 Layer 결합


```python
# # Mobile_model 끝단에 내가 만든 Layers 결합

inputs = Eff_model.get_layer('input_1').input
x = layers.BatchNormalization()(Eff_model.output)
x = layers.Dropout(0.2)(x)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(100, activation = 'softmax')(x)

my_Eff_model = tf.keras.Model(inputs, outputs, name="EfficientNet")



my_Eff_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(learning_rate=0.0001),
              metrics=["accuracy", "top_k_categorical_accuracy"])

#각 Layer들의 형태, 파라미터 수 등을 알려준다.
#my_Eff_model.summary()
```

# 모델 구조 시각화

#### EfficeintNet


```python
#최종 형태 확인
#tf.keras.utils.plot_model(my_Eff_model, show_shapes=True)
```

# Check Point 생성


```python
from tensorflow.keras.callbacks import ModelCheckpoint

Eff_mc = ModelCheckpoint('/content/drive/MyDrive/University/인공지능이론및프로그래밍/term_project/EfficientNet_model4',
                     monitor='val_accuracy',
                     mode='max',
                     verbose=1,
                     save_best_only=True)


```

# 모델 학습


```python
Eff_history = my_Eff_model.fit(x_train,
                               y_train,
                               validation_split=0.3,
                               epochs=10,
                               callbacks=[Eff_mc])
```

    Epoch 1/10
    1094/1094 [==============================] - ETA: 0s - loss: 2.8672 - accuracy: 0.4260 - top_k_categorical_accuracy: 0.7100
    Epoch 1: val_accuracy improved from -inf to 0.58600, saving model to /content/drive/MyDrive/University/인공지능이론및프로그래밍/term_project\EfficientNet_model4
    INFO:tensorflow:Assets written to: /content/drive/MyDrive/University/인공지능이론및프로그래밍/term_project\EfficientNet_model4\assets
    1094/1094 [==============================] - 816s 730ms/step - loss: 2.8672 - accuracy: 0.4260 - top_k_categorical_accuracy: 0.7100 - val_loss: 1.8113 - val_accuracy: 0.5860 - val_top_k_categorical_accuracy: 0.8611
    Epoch 2/10
    1094/1094 [==============================] - ETA: 0s - loss: 1.6047 - accuracy: 0.6200 - top_k_categorical_accuracy: 0.8790
    Epoch 2: val_accuracy improved from 0.58600 to 0.63933, saving model to /content/drive/MyDrive/University/인공지능이론및프로그래밍/term_project\EfficientNet_model4
    INFO:tensorflow:Assets written to: /content/drive/MyDrive/University/인공지능이론및프로그래밍/term_project\EfficientNet_model4\assets
    1094/1094 [==============================] - 811s 742ms/step - loss: 1.6047 - accuracy: 0.6200 - top_k_categorical_accuracy: 0.8790 - val_loss: 1.3827 - val_accuracy: 0.6393 - val_top_k_categorical_accuracy: 0.8895
    Epoch 3/10
    1094/1094 [==============================] - ETA: 0s - loss: 1.3148 - accuracy: 0.6615 - top_k_categorical_accuracy: 0.9015
    Epoch 3: val_accuracy improved from 0.63933 to 0.66553, saving model to /content/drive/MyDrive/University/인공지능이론및프로그래밍/term_project\EfficientNet_model4
    INFO:tensorflow:Assets written to: /content/drive/MyDrive/University/인공지능이론및프로그래밍/term_project\EfficientNet_model4\assets
    1094/1094 [==============================] - 804s 735ms/step - loss: 1.3148 - accuracy: 0.6615 - top_k_categorical_accuracy: 0.9015 - val_loss: 1.2296 - val_accuracy: 0.6655 - val_top_k_categorical_accuracy: 0.9013
    Epoch 4/10
    1094/1094 [==============================] - ETA: 0s - loss: 1.1769 - accuracy: 0.6848 - top_k_categorical_accuracy: 0.9150
    Epoch 4: val_accuracy improved from 0.66553 to 0.67900, saving model to /content/drive/MyDrive/University/인공지능이론및프로그래밍/term_project\EfficientNet_model4
    INFO:tensorflow:Assets written to: /content/drive/MyDrive/University/인공지능이론및프로그래밍/term_project\EfficientNet_model4\assets
    1094/1094 [==============================] - 797s 728ms/step - loss: 1.1769 - accuracy: 0.6848 - top_k_categorical_accuracy: 0.9150 - val_loss: 1.1474 - val_accuracy: 0.6790 - val_top_k_categorical_accuracy: 0.9094
    Epoch 5/10
    1094/1094 [==============================] - ETA: 0s - loss: 1.0862 - accuracy: 0.7061 - top_k_categorical_accuracy: 0.9232
    Epoch 5: val_accuracy improved from 0.67900 to 0.69193, saving model to /content/drive/MyDrive/University/인공지능이론및프로그래밍/term_project\EfficientNet_model4
    INFO:tensorflow:Assets written to: /content/drive/MyDrive/University/인공지능이론및프로그래밍/term_project\EfficientNet_model4\assets
    1094/1094 [==============================] - 806s 737ms/step - loss: 1.0862 - accuracy: 0.7061 - top_k_categorical_accuracy: 0.9232 - val_loss: 1.0974 - val_accuracy: 0.6919 - val_top_k_categorical_accuracy: 0.9148
    Epoch 6/10
    1094/1094 [==============================] - ETA: 0s - loss: 1.0175 - accuracy: 0.7191 - top_k_categorical_accuracy: 0.9308
    Epoch 6: val_accuracy improved from 0.69193 to 0.70153, saving model to /content/drive/MyDrive/University/인공지능이론및프로그래밍/term_project\EfficientNet_model4
    INFO:tensorflow:Assets written to: /content/drive/MyDrive/University/인공지능이론및프로그래밍/term_project\EfficientNet_model4\assets
    1094/1094 [==============================] - 798s 730ms/step - loss: 1.0175 - accuracy: 0.7191 - top_k_categorical_accuracy: 0.9308 - val_loss: 1.0615 - val_accuracy: 0.7015 - val_top_k_categorical_accuracy: 0.9193
    Epoch 7/10
    1094/1094 [==============================] - ETA: 0s - loss: 0.9677 - accuracy: 0.7311 - top_k_categorical_accuracy: 0.9372
    Epoch 7: val_accuracy improved from 0.70153 to 0.70580, saving model to /content/drive/MyDrive/University/인공지능이론및프로그래밍/term_project\EfficientNet_model4
    INFO:tensorflow:Assets written to: /content/drive/MyDrive/University/인공지능이론및프로그래밍/term_project\EfficientNet_model4\assets
    1094/1094 [==============================] - 794s 726ms/step - loss: 0.9677 - accuracy: 0.7311 - top_k_categorical_accuracy: 0.9372 - val_loss: 1.0355 - val_accuracy: 0.7058 - val_top_k_categorical_accuracy: 0.9211
    Epoch 8/10
    1094/1094 [==============================] - ETA: 0s - loss: 0.9295 - accuracy: 0.7403 - top_k_categorical_accuracy: 0.9410
    Epoch 8: val_accuracy improved from 0.70580 to 0.71033, saving model to /content/drive/MyDrive/University/인공지능이론및프로그래밍/term_project\EfficientNet_model4
    INFO:tensorflow:Assets written to: /content/drive/MyDrive/University/인공지능이론및프로그래밍/term_project\EfficientNet_model4\assets
    1094/1094 [==============================] - 792s 724ms/step - loss: 0.9295 - accuracy: 0.7403 - top_k_categorical_accuracy: 0.9410 - val_loss: 1.0160 - val_accuracy: 0.7103 - val_top_k_categorical_accuracy: 0.9229
    Epoch 9/10
    1094/1094 [==============================] - ETA: 0s - loss: 0.8918 - accuracy: 0.7501 - top_k_categorical_accuracy: 0.9444
    Epoch 9: val_accuracy improved from 0.71033 to 0.71547, saving model to /content/drive/MyDrive/University/인공지능이론및프로그래밍/term_project\EfficientNet_model4
    INFO:tensorflow:Assets written to: /content/drive/MyDrive/University/인공지능이론및프로그래밍/term_project\EfficientNet_model4\assets
    1094/1094 [==============================] - 796s 728ms/step - loss: 0.8918 - accuracy: 0.7501 - top_k_categorical_accuracy: 0.9444 - val_loss: 0.9999 - val_accuracy: 0.7155 - val_top_k_categorical_accuracy: 0.9248
    Epoch 10/10
    1094/1094 [==============================] - ETA: 0s - loss: 0.8608 - accuracy: 0.7594 - top_k_categorical_accuracy: 0.9475
    Epoch 10: val_accuracy improved from 0.71547 to 0.71640, saving model to /content/drive/MyDrive/University/인공지능이론및프로그래밍/term_project\EfficientNet_model4
    INFO:tensorflow:Assets written to: /content/drive/MyDrive/University/인공지능이론및프로그래밍/term_project\EfficientNet_model4\assets
    1094/1094 [==============================] - 802s 733ms/step - loss: 0.8608 - accuracy: 0.7594 - top_k_categorical_accuracy: 0.9475 - val_loss: 0.9857 - val_accuracy: 0.7164 - val_top_k_categorical_accuracy: 0.9266


# 모델 저장


```python
my_Eff_model.save('/content/drive/MyDrive/University/인공지능이론및프로그래밍/term_project/EfficientNet_model4')
```

    INFO:tensorflow:Assets written to: D:/정석-한양대/4학년 2학기/인공지능이론및프로그래밍/term_project/EfficientNet_model3\assets


    C:\Users\user\anaconda3\lib\site-packages\keras\engine\functional.py:1410: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      layer_config = serialize_layer_fn(layer)
    C:\Users\user\anaconda3\lib\site-packages\keras\saving\saved_model\layer_serialization.py:112: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      return generic_utils.serialize_keras_object(obj)


# 모델 평가


```python
loss, acc, k_acc = my_Eff_model.evaluate(Eff_x_test,  y_test, verbose=2)

print('EfficientNet모델의 정확도 (top-1-error): {:5.2f}%'.format(100*acc))
print('EfficientNet모델의 정확도 (top-5-error): {:5.2f}%'.format(100*k_acc))
```

    313/313 - 220s - loss: 0.9701 - accuracy: 0.7206 - top_k_categorical_accuracy: 0.9319 - 220s/epoch - 701ms/step
    EfficientNet모델의 정확도 (top-1-error): 72.06%
    EfficientNet모델의 정확도 (top-5-error): 93.19%



```python
pd.DataFrame(Eff_history.history).plot()
```




    <AxesSubplot:>




![image-20220513174643547](/assets/images/image-20220513174643547.png)
    


# 모델 Fine Tuning

- Transfer learning 진행해서 fine-tuning 진행


```python
def unfreeze_model(model):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-51:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", 
        metrics=["accuracy", "top_k_categorical_accuracy"]
        )

```


```python
unfreeze_model(my_Eff_model)
#my_Eff_model.summary()
```


```python
Eff_history2 = my_Eff_model.fit(x_train,
                               y_train,
                               validation_split=0.3,
                               epochs=11,
                               callbacks=[Eff_mc])


```

    Epoch 1/10
    1094/1094 [==============================] - ETA: 0s - loss: 0.7744 - accuracy: 0.7766 - top_k_categorical_accuracy: 0.9529
    Epoch 00001: val_accuracy improved from 0.71873 to 0.73233, saving model to D:/정석-한양대/4학년 2학기/인공지능이론및프로그래밍/term_project\EfficientNet_model4
    INFO:tensorflow:Assets written to: D:/정석-한양대/4학년 2학기/인공지능이론및프로그래밍/term_project\EfficientNet_model4\assets


    C:\Users\user\anaconda3\lib\site-packages\keras\engine\functional.py:1410: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      layer_config = serialize_layer_fn(layer)
    C:\Users\user\anaconda3\lib\site-packages\keras\saving\saved_model\layer_serialization.py:112: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      return generic_utils.serialize_keras_object(obj)


    1094/1094 [==============================] - 1369s 1s/step - loss: 0.7744 - accuracy: 0.7766 - top_k_categorical_accuracy: 0.9529 - val_loss: 0.9174 - val_accuracy: 0.7323 - val_top_k_categorical_accuracy: 0.9362
    Epoch 2/10
    1094/1094 [==============================] - ETA: 0s - loss: 0.7078 - accuracy: 0.7940 - top_k_categorical_accuracy: 0.9604
    Epoch 00002: val_accuracy improved from 0.73233 to 0.74120, saving model to D:/정석-한양대/4학년 2학기/인공지능이론및프로그래밍/term_project\EfficientNet_model4
    INFO:tensorflow:Assets written to: D:/정석-한양대/4학년 2학기/인공지능이론및프로그래밍/term_project\EfficientNet_model4\assets


    C:\Users\user\anaconda3\lib\site-packages\keras\engine\functional.py:1410: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      layer_config = serialize_layer_fn(layer)
    C:\Users\user\anaconda3\lib\site-packages\keras\saving\saved_model\layer_serialization.py:112: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      return generic_utils.serialize_keras_object(obj)


    1094/1094 [==============================] - 1333s 1s/step - loss: 0.7078 - accuracy: 0.7940 - top_k_categorical_accuracy: 0.9604 - val_loss: 0.8852 - val_accuracy: 0.7412 - val_top_k_categorical_accuracy: 0.9403
    Epoch 3/10
    1094/1094 [==============================] - ETA: 0s - loss: 0.6514 - accuracy: 0.8113 - top_k_categorical_accuracy: 0.9653
    Epoch 00003: val_accuracy improved from 0.74120 to 0.74480, saving model to D:/정석-한양대/4학년 2학기/인공지능이론및프로그래밍/term_project\EfficientNet_model4
    INFO:tensorflow:Assets written to: D:/정석-한양대/4학년 2학기/인공지능이론및프로그래밍/term_project\EfficientNet_model4\assets


    C:\Users\user\anaconda3\lib\site-packages\keras\engine\functional.py:1410: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      layer_config = serialize_layer_fn(layer)
    C:\Users\user\anaconda3\lib\site-packages\keras\saving\saved_model\layer_serialization.py:112: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      return generic_utils.serialize_keras_object(obj)


    1094/1094 [==============================] - 1327s 1s/step - loss: 0.6514 - accuracy: 0.8113 - top_k_categorical_accuracy: 0.9653 - val_loss: 0.8619 - val_accuracy: 0.7448 - val_top_k_categorical_accuracy: 0.9433
    Epoch 4/10
    1094/1094 [==============================] - ETA: 0s - loss: 0.6098 - accuracy: 0.8236 - top_k_categorical_accuracy: 0.9691
    Epoch 00004: val_accuracy improved from 0.74480 to 0.74927, saving model to D:/정석-한양대/4학년 2학기/인공지능이론및프로그래밍/term_project\EfficientNet_model4
    INFO:tensorflow:Assets written to: D:/정석-한양대/4학년 2학기/인공지능이론및프로그래밍/term_project\EfficientNet_model4\assets


    C:\Users\user\anaconda3\lib\site-packages\keras\engine\functional.py:1410: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      layer_config = serialize_layer_fn(layer)
    C:\Users\user\anaconda3\lib\site-packages\keras\saving\saved_model\layer_serialization.py:112: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      return generic_utils.serialize_keras_object(obj)


    1094/1094 [==============================] - 1379s 1s/step - loss: 0.6098 - accuracy: 0.8236 - top_k_categorical_accuracy: 0.9691 - val_loss: 0.8453 - val_accuracy: 0.7493 - val_top_k_categorical_accuracy: 0.9455
    Epoch 5/10
    1094/1094 [==============================] - ETA: 0s - loss: 0.5655 - accuracy: 0.8364 - top_k_categorical_accuracy: 0.9741
    Epoch 00005: val_accuracy improved from 0.74927 to 0.75527, saving model to D:/정석-한양대/4학년 2학기/인공지능이론및프로그래밍/term_project\EfficientNet_model4
    INFO:tensorflow:Assets written to: D:/정석-한양대/4학년 2학기/인공지능이론및프로그래밍/term_project\EfficientNet_model4\assets


    C:\Users\user\anaconda3\lib\site-packages\keras\engine\functional.py:1410: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      layer_config = serialize_layer_fn(layer)
    C:\Users\user\anaconda3\lib\site-packages\keras\saving\saved_model\layer_serialization.py:112: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      return generic_utils.serialize_keras_object(obj)


    1094/1094 [==============================] - 1386s 1s/step - loss: 0.5655 - accuracy: 0.8364 - top_k_categorical_accuracy: 0.9741 - val_loss: 0.8307 - val_accuracy: 0.7553 - val_top_k_categorical_accuracy: 0.9474
    Epoch 6/10
    1094/1094 [==============================] - ETA: 0s - loss: 0.5295 - accuracy: 0.8473 - top_k_categorical_accuracy: 0.9765
    Epoch 00006: val_accuracy improved from 0.75527 to 0.75773, saving model to D:/정석-한양대/4학년 2학기/인공지능이론및프로그래밍/term_project\EfficientNet_model4
    INFO:tensorflow:Assets written to: D:/정석-한양대/4학년 2학기/인공지능이론및프로그래밍/term_project\EfficientNet_model4\assets


    C:\Users\user\anaconda3\lib\site-packages\keras\engine\functional.py:1410: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      layer_config = serialize_layer_fn(layer)
    C:\Users\user\anaconda3\lib\site-packages\keras\saving\saved_model\layer_serialization.py:112: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      return generic_utils.serialize_keras_object(obj)


    1094/1094 [==============================] - 1345s 1s/step - loss: 0.5295 - accuracy: 0.8473 - top_k_categorical_accuracy: 0.9765 - val_loss: 0.8204 - val_accuracy: 0.7577 - val_top_k_categorical_accuracy: 0.9494
    Epoch 7/10
    1094/1094 [==============================] - ETA: 0s - loss: 0.4933 - accuracy: 0.8577 - top_k_categorical_accuracy: 0.9803
    Epoch 00007: val_accuracy improved from 0.75773 to 0.76020, saving model to D:/정석-한양대/4학년 2학기/인공지능이론및프로그래밍/term_project\EfficientNet_model4
    INFO:tensorflow:Assets written to: D:/정석-한양대/4학년 2학기/인공지능이론및프로그래밍/term_project\EfficientNet_model4\assets


    C:\Users\user\anaconda3\lib\site-packages\keras\engine\functional.py:1410: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      layer_config = serialize_layer_fn(layer)
    C:\Users\user\anaconda3\lib\site-packages\keras\saving\saved_model\layer_serialization.py:112: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      return generic_utils.serialize_keras_object(obj)


    1094/1094 [==============================] - 1335s 1s/step - loss: 0.4933 - accuracy: 0.8577 - top_k_categorical_accuracy: 0.9803 - val_loss: 0.8116 - val_accuracy: 0.7602 - val_top_k_categorical_accuracy: 0.9499
    Epoch 8/10
    1094/1094 [==============================] - ETA: 0s - loss: 0.4561 - accuracy: 0.8707 - top_k_categorical_accuracy: 0.9831
    Epoch 00008: val_accuracy improved from 0.76020 to 0.76140, saving model to D:/정석-한양대/4학년 2학기/인공지능이론및프로그래밍/term_project\EfficientNet_model4
    INFO:tensorflow:Assets written to: D:/정석-한양대/4학년 2학기/인공지능이론및프로그래밍/term_project\EfficientNet_model4\assets


    C:\Users\user\anaconda3\lib\site-packages\keras\engine\functional.py:1410: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      layer_config = serialize_layer_fn(layer)
    C:\Users\user\anaconda3\lib\site-packages\keras\saving\saved_model\layer_serialization.py:112: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      return generic_utils.serialize_keras_object(obj)


    1094/1094 [==============================] - 1341s 1s/step - loss: 0.4561 - accuracy: 0.8707 - top_k_categorical_accuracy: 0.9831 - val_loss: 0.8060 - val_accuracy: 0.7614 - val_top_k_categorical_accuracy: 0.9521
    Epoch 9/10
    1094/1094 [==============================] - ETA: 0s - loss: 0.4269 - accuracy: 0.8806 - top_k_categorical_accuracy: 0.9851
    Epoch 00009: val_accuracy improved from 0.76140 to 0.76313, saving model to D:/정석-한양대/4학년 2학기/인공지능이론및프로그래밍/term_project\EfficientNet_model4
    INFO:tensorflow:Assets written to: D:/정석-한양대/4학년 2학기/인공지능이론및프로그래밍/term_project\EfficientNet_model4\assets


    C:\Users\user\anaconda3\lib\site-packages\keras\engine\functional.py:1410: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      layer_config = serialize_layer_fn(layer)
    C:\Users\user\anaconda3\lib\site-packages\keras\saving\saved_model\layer_serialization.py:112: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      return generic_utils.serialize_keras_object(obj)


    1094/1094 [==============================] - 1390s 1s/step - loss: 0.4269 - accuracy: 0.8806 - top_k_categorical_accuracy: 0.9851 - val_loss: 0.7980 - val_accuracy: 0.7631 - val_top_k_categorical_accuracy: 0.9517
    Epoch 10/10
    1094/1094 [==============================] - ETA: 0s - loss: 0.3961 - accuracy: 0.8887 - top_k_categorical_accuracy: 0.9877
    Epoch 00010: val_accuracy improved from 0.76313 to 0.76407, saving model to D:/정석-한양대/4학년 2학기/인공지능이론및프로그래밍/term_project\EfficientNet_model4
    INFO:tensorflow:Assets written to: D:/정석-한양대/4학년 2학기/인공지능이론및프로그래밍/term_project\EfficientNet_model4\assets


    C:\Users\user\anaconda3\lib\site-packages\keras\engine\functional.py:1410: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      layer_config = serialize_layer_fn(layer)
    C:\Users\user\anaconda3\lib\site-packages\keras\saving\saved_model\layer_serialization.py:112: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      return generic_utils.serialize_keras_object(obj)


    1094/1094 [==============================] - 1419s 1s/step - loss: 0.3961 - accuracy: 0.8887 - top_k_categorical_accuracy: 0.9877 - val_loss: 0.7928 - val_accuracy: 0.7641 - val_top_k_categorical_accuracy: 0.9524



```python
def plot_history(hist):
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'r', linestyle ='--', label='val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    #loss_ax.set_yticks(np.linspace(0, 2, 9))
    #loss_ax.set_ylim(-0.1,2.1)
    loss_ax.legend(loc='upper left')

    acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
    acc_ax.plot(hist.history['val_accuracy'], 'g', linestyle ='--', label='val acc')
    acc_ax.set_ylabel('accuracy')
    #acc_ax.set_yticks(np.linspace(0, 1, 11))
    #acc_ax.set_ylim(-0.1, 1.1)
    acc_ax.legend(loc='upper right')

plot_history(Eff_history2)
```


​    
![image-20220513174739066](/assets/images/image-20220513174739066.png)
​    


## 성능 확인

- 위 그림을 보면 알 수 있듯이 더 많은 epoch를 진행했다면 더 좋은 결과를 얻을 수 있었을 것이다. 마지막 epoch에서도 validation set의 가파른 정확도 상승률과 loss 하강률을 볼 수 있다. 


```python
loss, acc, k_acc = my_Eff_model.evaluate(x_test,  y_test, verbose=2)

print('EfficientNet모델의 정확도 (top-1-error): {:5.2f}%'.format(100*acc))
print('EfficientNet모델의 정확도 (top-5-error): {:5.2f}%'.format(100*k_acc))
```

    313/313 - 220s - loss: 0.7724 - accuracy: 0.7695 - top_k_categorical_accuracy: 0.9547 - 220s/epoch - 704ms/step
    EfficientNet모델의 정확도 (top-1-error): 76.95%
    EfficientNet모델의 정확도 (top-5-error): 95.47%



```python
my_Eff_model.save('/content/drive/MyDrive/University/인공지능이론및프로그래밍/term_project/EfficientNet_model4')
```
