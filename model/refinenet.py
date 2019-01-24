"""
Based on https://github.com/GeorgeSeif/Semantic-Segmentation-Suite
"""

from keras.models import Model
from model.resnet_101 import resnet101_model
from model.layers.Upsampling import Upsampling
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, ReLU, Add, MaxPooling2D

def ConvBlock(inputs, n_filters, kernel_size=[3, 3]):
    """
    Basic conv block for Encoder-Decoder
    Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
    """
    
    net = Conv2D(n_filters, kernel_size, padding='same')(inputs)
    net = BatchNormalization()(net)
    net = ReLU()(net)
    
    # TODO: Check order!
    
#    net = tf.nn.relu(slim.batch_norm(inputs, fused=True))
#    net = slim.conv2d(net, n_filters, kernel_size, activation_fn=None, normalizer_fn=None)
    return net

def ConvUpscaleBlock(inputs, n_filters, kernel_size=[3, 3], scale=2):
    """
    Basic conv transpose block for Encoder-Decoder upsampling
    Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
    """
    
    net = Conv2DTranspose(n_filters, kernel_size, strides = (scale,scale))(inputs)
    net = BatchNormalization()(net)
    net = ReLU()(net)
    
    # TODO: Check order!
    
#    net = tf.nn.relu(slim.batch_norm(inputs, fused=True))
#    net = slim.conv2d_transpose(net, n_filters, kernel_size=[3, 3], stride=[scale, scale], activation_fn=None)
    return net


def ResidualConvUnit(inputs,n_filters=256,kernel_size=3):
    """
    A local residual unit designed to fine-tune the pretrained ResNet weights

    Arguments:
      inputs: The input tensor
      n_filters: Number of output feature maps for each conv
      kernel_size: Size of convolution kernel

    Returns:
      Output of local residual block
    """
    
    net = ReLU()(inputs)
    net = Conv2D(n_filters, kernel_size, padding='same')(net)
    net = ReLU()(net)
    net = Conv2D(n_filters, kernel_size, padding='same')(net)
    net = Add()([net, inputs])
    
    return net

def ChainedResidualPooling(inputs,n_filters=256):
    """
    Chained residual pooling aims to capture background 
    context from a large image region. This component is 
    built as a chain of 2 pooling blocks, each consisting 
    of one max-pooling layer and one convolution layer. One pooling
    block takes the output of the previous pooling block as
    input. The output feature maps of all pooling blocks are 
    fused together with the input feature map through summation 
    of residual connections.

    Arguments:
      inputs: The input tensor
      n_filters: Number of output feature maps for each conv

    Returns:
      Double-pooled feature maps
    """
    
    net_relu = ReLU()(inputs)
    net = MaxPooling2D(pool_size = (5,5), strides = 1, padding = 'same')(net_relu)
    net = Conv2D(n_filters, 3, padding='same')(net_relu)
    net_sum_1 = Add()([net,net_relu])
    
    net = MaxPooling2D(pool_size = (5,5), strides = 1, padding = 'same')(net)
    net = Conv2D(n_filters, 3, padding='same')(net)
    net_sum_2 = Add()([net,net_sum_1])

    return net_sum_2


def MultiResolutionFusion(high_inputs=None,low_inputs=None,n_filters=256):
    """
    Fuse together all path inputs. This block first applies convolutions
    for input adaptation, which generate feature maps of the same feature dimension 
    (the smallest one among the inputs), and then up-samples all (smaller) feature maps to
    the largest resolution of the inputs. Finally, all features maps are fused by summation.

    Arguments:
      high_inputs: The input tensors that have the higher resolution
      low_inputs: The input tensors that have the lower resolution
      n_filters: Number of output feature maps for each conv

    Returns:
      Fused feature maps at higher resolution
    
    """

    
    
    if high_inputs is None: # RefineNet block 4
        
        fuse = Conv2D(n_filters, 3, padding='same')(low_inputs)

        return fuse

    else:

        conv_low = Conv2D(n_filters, 3, padding='same')(low_inputs)
        conv_high = Conv2D(n_filters, 3, padding='same')(high_inputs)
        
        conv_low_up = Upsampling(scale = 2)(conv_low)
        
        return Add()([conv_low_up, conv_high])


def RefineBlock(high_inputs=None,low_inputs=None):
    """
    A RefineNet Block which combines together the ResidualConvUnits,
    fuses the feature maps using MultiResolutionFusion, and then gets
    large-scale context with the ResidualConvUnit.

    Arguments:
      high_inputs: The input tensors that have the higher resolution
      low_inputs: The input tensors that have the lower resolution

    Returns:
      RefineNet block for a single path i.e one resolution
    
    """

    if low_inputs is None: # block 4
        rcu_new_low = ResidualConvUnit(high_inputs, n_filters=512)
        rcu_new_low = ResidualConvUnit(rcu_new_low, n_filters=512)

        fuse = MultiResolutionFusion(high_inputs=None, low_inputs=rcu_new_low, n_filters=512)
        fuse_pooling = ChainedResidualPooling(fuse, n_filters=512)
        output = ResidualConvUnit(fuse_pooling, n_filters=512)
        return output
    else:
        rcu_high= ResidualConvUnit(high_inputs, n_filters=256)
        rcu_high = ResidualConvUnit(rcu_high, n_filters=256)

        fuse = MultiResolutionFusion(rcu_high, low_inputs,n_filters=256)
        fuse_pooling = ChainedResidualPooling(fuse, n_filters=256)
        output = ResidualConvUnit(fuse_pooling, n_filters=256)
        return output



def build_refinenet(input_shape, num_class, resnet_weights, frontend_trainable, upscaling_method='bilinear'):
    """
    Builds the RefineNet model. 

    Arguments:
      input_shape: Size of input image, including number of channels
      num_classes: Number of classes
      resnet_weights: Path to pre-trained weights for ResNet-101
      frontend_trainable: Whether or not to freeze ResNet layers during training
      upscaling_method: Either 'bilinear' or 'conv'

    Returns:
      RefineNet model
    """
    
    # Build ResNet-101
    model_base = resnet101_model(input_shape, resnet_weights)
    model_base.trainable = frontend_trainable

    # Get ResNet block output layers
    high = [model_base.get_layer('res5c_relu').output,
            model_base.get_layer('res4b22_relu').output,
            model_base.get_layer('res3b3_relu').output,
            model_base.get_layer('res2c_relu').output]

    low = [None, None, None]

    # Get the feature maps to the proper size with bottleneck
    high[0] = Conv2D(512, 1, padding='same')(high[0])
    high[1] = Conv2D(256, 1, padding='same')(high[1])
    high[2] = Conv2D(256, 1, padding='same')(high[2])
    high[3] = Conv2D(256, 1, padding='same')(high[3])

    # RefineNet
    low[0]=RefineBlock(high_inputs=high[0],low_inputs=None) # Only input ResNet 1/32
    low[1]=RefineBlock(high[1],low[0]) # High input = ResNet 1/16, Low input = Previous 1/16
    low[2]=RefineBlock(high[2],low[1]) # High input = ResNet 1/8, Low input = Previous 1/8
    net=RefineBlock(high[3],low[2]) # High input = ResNet 1/4, Low input = Previous 1/4.

    # g[3]=Upsampling(g[3],scale=4)

    net = ResidualConvUnit(net)
    net = ResidualConvUnit(net)

    if upscaling_method.lower() == 'conv':
        net = ConvUpscaleBlock(net, 128, kernel_size=[3, 3], scale=2)
        net = ConvBlock(net, 128, padding='same')
        net = ConvUpscaleBlock(net, 64, kernel_size=[3, 3], scale=2)
        net = ConvBlock(net, 64, padding='same')
    elif upscaling_method.lower() == 'bilinear':
        net = Upsampling(scale = 4)(net)
        
    net = Conv2D(num_class, 1, activation = 'softmax')(net)
    
    model = Model(model_base.input,net)
    
    return model