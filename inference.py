import os
import numpy as np
from time import localtime, strftime

from utils.data import testGenerator, saveResult
from utils.helpers import get_label_info
from model.refinenet import build_refinenet

# Define parameters
input_dir = '/home/lengyel/data/Cityscapes/validation/images'
#input_dir = '/home/lengyel/data/DarkModel/NighttimeDrivingTest/leftImg8bit/test/night'
class_dict = '/home/lengyel/data/Cityscapes/class_dict.csv'

resnet_weights = '/home/lengyel/data/resnet101_weights_tf.h5'
refinenet_weights = '/home/lengyel/refinenet-keras/runs/20190114-120056/weights/weights.25-0.19.hdf5'

input_shape = (512,1024,3) # height x width x channels

# Generate output directories
output_dir = os.path.join('predictions',strftime("%Y%m%d-%H%M%S", localtime()))
if not os.path.exists(output_dir):
    org_dir = os.path.join(output_dir,'input') # input images save dir
    pred_dir = os.path.join(output_dir,'pred') # output predictions save dir
    os.makedirs(org_dir)
    os.makedirs(pred_dir)
	
with open(os.path.join(output_dir,'settings.txt'), 'w') as f:
    f.write('RefineNet weights: {}\n'.format(refinenet_weights))

# Import classes from csv file
class_names_list, mask_colors, num_class, class_names_string = get_label_info(class_dict)

# Define model and load weights
model = build_refinenet(input_shape, num_class, resnet_weights, False)
model.load_weights(refinenet_weights)

myTestGenerator = testGenerator(input_dir, input_shape[:2], 2, out_dir = org_dir)
for batch, file_names in myTestGenerator:
    results = model.predict_on_batch(batch)
    saveResult(results, pred_dir, file_names, mask_colors)