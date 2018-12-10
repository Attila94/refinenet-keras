'''
Based on https://github.com/zhixuhao/unet
'''
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans

from utils.helpers import one_hot_it, reverse_one_hot, colour_code_segmentation

def preprocImage(img):
    # Subtract training means
    img[:,:,:,0] -= 103.939
    img[:,:,:,1] -= 116.779
    img[:,:,:,2] -= 123.68
    img = img / 255
    return img

def preprocMask(mask, num_class = 0, mask_colors = None):
    if mask.shape[3] == 1:
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            new_mask[mask == i,i] = 1
        mask = new_mask
    elif mask.shape[3] == 3:
        mask = one_hot_it(mask[0,:,:,:], mask_colors)
        mask = np.expand_dims(mask, axis=0)
#        np.save(r'C:\Projects\MSc Thesis\git\code\image_segmentation\refinenet-keras\img\numpymask',mask) # save for debugging
    return mask
        

def trainGenerator(batch_size,
                   train_path,
                   image_folder,
                   mask_folder,
                   num_class,
                   target_size,
                   aug_dict,
                   mask_colors = None,
                   image_color_mode = 'rgb',
                   mask_color_mode = 'rgb',
                   image_save_prefix  = 'image',
                   mask_save_prefix  = 'mask',
                   save_to_dir = None,
                   seed = 1):
    '''
    Data generator for training and validation.
    
    Arguments:
        batch_size: Batch size.
        image_folder: Name of directory containing images.
        mask_folder: Name of directory containing masks.
        aug_dict: Data augmentation options.
        num_class: Number of segmentation classes.
        target_size: Size to which input images are resized before being fed
            into RefineNet.
        image_color_mode: One of "grayscale", "rbg", "rgba". Default: "rgb".
            Whether the images will be converted to have 1, 3, or 4 channels.
        mask_color_mode: see image_color_mode
        image_save_prefix: Filename prefix when saving augmented input.
        mask_save_prefix: Filename prefix when saving augmented input.
        save_to_dir: If you want to visualize the generator output, set this
            to the desired output directory.
        seed: Define seed to use same image transformations for both images
            and masks.
    '''
    
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size[:2],
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size[:2],
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img = preprocImage(img)
        mask = preprocMask(mask, mask_colors = mask_colors)
        yield (img, mask)

import skimage.io as io
import numpy as np
from utils.data import preprocImage, labelVisualize
import skimage.transform as trans

def testGenerator(test_path,
                  target_size,
                  out_dir = None,
                  num_images = 30):
    '''
    Data generator for testing.
    
    Arguments: 
        test_path: Path to directory containing test images.
        target_size: Size to which input images are resized before being fed
            into RefineNet.
        out_dir: Path to output directory.
        num_image: Number of images to process
    '''
    
    files = next(os.walk(test_path))[2]
    for i, file in enumerate(files):
        if i == num_images:
            break
        file_path = os.path.join(test_path,file)
        img = io.imread(file_path)
        if out_dir:
          out_path = os.path.join(out_dir,'{}_image.png'.format(i))
          io.imsave(out_path,img)
        img = trans.resize(img,target_size[:2])
        img = np.expand_dims(img, axis=0).astype('float64')
        img = preprocImage(img)
        yield img


def labelVisualize(img, mask_colors):
    img = reverse_one_hot(img)
    img = colour_code_segmentation(img, mask_colors)
    return img

def saveResult(results, save_path, out_size, mask_colors):
    for i, item in enumerate(results):
        img = labelVisualize(item,mask_colors)
        img = trans.resize(img,out_size[:2])
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)