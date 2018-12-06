# -*- coding: utf-8 -*-
"""
Based on https://github.com/aurora95/Keras-FCN/blob/master/utils/BilinearUpSampling.py
"""

from keras.layers import Layer, InputSpec
import keras.backend as K
import tensorflow as tf

# Model definition 
class Upsampling(Layer):
    def __init__(self, scale = 1, **kwargs):
        self.scale = scale
        self.input_spec = [InputSpec(ndim=4)]
        super(Upsampling, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        width = int(self.scale * input_shape[1] if input_shape[1] is not None else None)
        height = int(self.scale * input_shape[2] if input_shape[2] is not None else None)
        return (input_shape[0],width,height,input_shape[3])

    def call(self, X, mask=None):
        original_shape = K.int_shape(X)
        new_shape = tf.shape(X)[1:3]
        new_shape *= tf.constant(self.scale)
        X = tf.image.resize_bilinear(X, new_shape)
        X.set_shape((None, original_shape[1] * self.scale, original_shape[2] * self.scale, None))
        return X
    
    def get_config(self):
        config = {'scale': self.scale}
        base_config = super(Upsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))