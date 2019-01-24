# RefineNet: a Keras implementation

Paper: https://arxiv.org/abs/1611.06612

This implementation is based on [GeorgeSeif's Semantic Segmentation Suite in TensorFlow](https://github.com/GeorgeSeif/Semantic-Segmentation-Suite).

The ResNet-101 frontend model is based on [flyyufelix's implementation](https://gist.github.com/flyyufelix/65018873f8cb2bbe95f429c474aa1294).

![Results](results.png)

---

# Usage
ResNet-101 weights can be downloaded [here](https://my.syncplicity.com/share/m1qj80sthgfalaz/resnet101_weights_tf).
Pre-trained weights for CityScapes can be downloaded [here](https://my.syncplicity.com/share/cptvaesdqgw49vf/refinenet_baseline).

## Dataset directory structure
Image labels should be provided in RGB format, accompanied by a class dictionary.
Structure your dataset in the following way:
- `Dataset name`
  - `class_dict.csv`
  - `training`
    - `images`
	- `labels`
  - `validation`
    - `images`
	- `labels`
  - `testing`
    - `images`
	- `labels`
	
The `class_dict.csv` file should have the following structure (example for Cityscapes dataset):
```
name,r,g,b
road,128,64,128
sidewalk,244,35,232
building,70,70,70
wall,102,102,156
fence,190,153,153
pole,153,153,153
traffic_light,250,170,30
traffic_sign,220,220,0
vegetation,107,142,35
terrain,152,251,152
sky,70,130,180
person,220,20,60
rider,255,0,0
car,0,0,142
truck,0,0,70
bus,0,60,100
on_rails,0,80,100
motorcycle,0,0,230
bicycle,119,11,32
void,0,0,0
```
The class corresponding to (0,0,0) will be ignored during both training and evaluation.

## Training model
1. Specify paths to `resnet101_weights_tf.h5` and your dataset base directory in `train.py`.
1. Run `train.py`. Logs, weights and all other files will be generated in a new `runs` directory.

## Inference
1. Obtain a pre-trained weights file: either download one [here](https://my.syncplicity.com/share/cptvaesdqgw49vf/refinenet_baseline) (CityScapes) or train your own network.
1. Specify paths to `resnet101_weights_tf.h5`, RefineNet weights file and your dataset base directory in `inference.py`.
1. Run `inference.py`. Prediction results and original images will be placed into a new `predictions` directory.

---

# Performance
Performance of first trial evaluated on the CityScapes dataset.

## Overall
| Metric | Score |
| --- | --- |
| IoU | 0.631 |
| nIoU | 0.370 |

## Class-specific
| Class | IoU | nIoU |
| --- | --- | --- |
| bicycle | 0.626 | 0.397 |
| building | 0.883 | NaN |
| bus | 0.693 | 0.393 |
| car | 0.906 | 0.789 |
| fence | 0.408 | NaN |
| motorcycle | 0.378 | 0.143 |
| person | 0.667 | 0.442 |
| pole | 0.446 | NaN |
| rider | 0.421 | 0.224 |
| road | 0.969 | NaN |
| sidewalk | 0.759 | NaN |
| sky | 0.926 | NaN |
| terrain | 0.537 | NaN |
| traffic light | 0.414 | NaN |
| traffic sign | 0.573 |  NaN |
| train | 0.517 | 0.340 |
| truck | 0.558 | 0.235 |
| vegetation | 0.890 | NaN |
| wall | 0.411 | NaN |
