# RefineNet: a Keras implementation

Paper: https://arxiv.org/abs/1611.06612

This implementation is based on [GeorgeSeif's Semantic Segmentation Suite in TensorFlow](https://github.com/GeorgeSeif/Semantic-Segmentation-Suite).

The ResNet-101 frontend model is based on [flyyufelix's implementation](https://gist.github.com/flyyufelix/65018873f8cb2bbe95f429c474aa1294).

Additional bits of code were borrowed from [zhixuhao's UNet tes](https://github.com/zhixuhao/unet) and [aurora95's Keras-FCN](https://github.com/aurora95/Keras-FCN).

---
Currently training / evaluating, results and weights to be provided soon.

## Usage
1. Download ResNet-101 weights [here](https://drive.google.com/file/d/0Byy2AcGyEVxfTmRRVmpGWDczaXM/view?usp=sharing)
1. Specify ResNet-101 and dataset paths in `main.ipynb`
1. Build and train model.
