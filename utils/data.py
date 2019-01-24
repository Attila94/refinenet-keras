'''
Based on https://github.com/zhixuhao/unet
'''
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import cv2

from utils.helpers import one_hot_it, reverse_one_hot, colour_code_segmentation

def preprocImage(img):
    # Subtract training means
    img[:,:,:,0] -= 103.939
    img[:,:,:,1] -= 116.779
    img[:,:,:,2] -= 123.68
    return img/255

def preprocMask(mask, num_class, mask_colors = None):
    batch_size, height, width = mask.shape[0:3]
    if mask.shape[3] == 1: #class-encoded masks
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            new_mask[mask == i,i] = 1
        mask = new_mask
    elif mask.shape[3] == 3: #color-encoded masks
        # TODO: throw exception if mask_colors undefined
        mask_out = np.zeros((batch_size,height,width,num_class))
        for i in range(batch_size):
            mask_out[i,:,:,:] = one_hot_it(mask[i,:,:,:], mask_colors)
        mask = mask_out
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
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img = preprocImage(img)
        mask = preprocMask(mask, num_class, mask_colors = mask_colors)
        yield (img, mask)

def testGenerator(test_path,
                  target_size,
                  batch_size,
                  out_dir = None):
    '''
    Data generator for testing.
    
    Arguments: 
        test_path: Path to directory containing test images.
        target_size: Size to which input images are resized before being fed
            into RefineNet.
        batch_size: Images per batch.
        out_dir: Path to output directory.
    '''
    image_datagen = ImageDataGenerator()
    
    image_generator = image_datagen.flow_from_directory(
        os.path.dirname(test_path),
        classes = [os.path.basename(test_path)],
        class_mode = None,
        target_size = target_size,
        batch_size = batch_size,
        shuffle = False)

    for batch in image_generator:
        idx = ((image_generator.batch_index - 1)%len(image_generator)) * image_generator.batch_size
        file_names = image_generator.filenames[idx : idx + image_generator.batch_size]
        if out_dir:
            for img, file_name in zip(batch,file_names):
                cv2.imwrite(os.path.join(out_dir,os.path.basename(file_name)), img[:,:,::-1])
        batch = preprocImage(batch)
        if image_generator.batch_index == 0:
            yield batch, file_names
            return
        yield batch, file_names

def labelVisualize(img, mask_colors):
    img = reverse_one_hot(img)
    img = colour_code_segmentation(img, mask_colors).astype('uint8')
    return img

def saveResult(results, save_path, file_names, mask_colors):
    for img, file_name in zip(results, file_names):
        img = labelVisualize(img, mask_colors)
        cv2.imwrite(os.path.join(save_path, os.path.basename(file_name)), img[:,:,::-1])