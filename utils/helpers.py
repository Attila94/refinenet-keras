'''
Based on https://github.com/GeorgeSeif/Semantic-Segmentation-Suite/
'''
import numpy as np
import os, csv, math
from time import localtime, strftime
from contextlib import redirect_stdout
import keras
from keras.losses import categorical_crossentropy

def step_decay(epoch):
    """
    Define custom learning rate schedule
    """
    initial_lrate = 1e-5
    drop = 0.5
    epochs_drop = 5.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
    
def ignore_unknown_xentropy(ytrue, ypred):
    """
    Define custom loss function to ignore void class (https://github.com/keras-team/keras/issues/6261)
    Assuming last class is void
    """
    return (1-ytrue[:, :, :, -1])*categorical_crossentropy(ytrue, ypred)

def get_label_info(csv_path):
    """
    Retrieve the class names and label values for the selected dataset.
    Must be in CSV format!

    # Arguments
        csv_path: The file path of the class dictionairy
        
    # Returns
        Two lists: one for the class names and the other for the label values
    """
    filename, file_extension = os.path.splitext(csv_path)
    if not file_extension == ".csv":
        return ValueError("File is not a CSV!")

    class_names = []
    label_values = []
    with open(csv_path, 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        header = next(file_reader)
        for row in file_reader:
            class_names.append(row[0])
            label_values.append([int(row[1]), int(row[2]), int(row[3])])
    
    class_names_string = ""
    for class_name in class_names:
        if not class_name == class_names[-1]:
            class_names_string = class_names_string + class_name + ", "
        else:
            class_names_string = class_names_string + class_name
    
    return class_names, label_values, len(label_values),class_names_string


def one_hot_it(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes

    # Arguments
        label: The 2D array segmentation image label
        label_values
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    imsize = label.shape[:2]
    semantic_map = np.zeros((imsize[0],imsize[1],len(label_values)))
    for i,colour in enumerate(label_values):
        # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map[class_map,i] = 1
    return semantic_map
    
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.

    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    x = np.argmax(image, axis = -1)
    return x


def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        label_values
        
    # Returns
        Colour coded image for segmentation visualization
    """
    
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x
    
class LossHistory(keras.callbacks.Callback):
    def __init__(self, batch_log_path, epoch_log_path):
        self.epoch = 0
        self.batch_log_path = batch_log_path
        self.epoch_log_path = epoch_log_path
        with open(self.batch_log_path, 'a') as batch_log:
            batch_log.write('epoch,epoch_step,loss,acc\n')
        with open(self.epoch_log_path, 'a') as epoch_log:
            epoch_log.write('epoch,loss,acc,val_loss,val_acc\n')

    def on_batch_end(self, batch, logs={}):
        with open(self.batch_log_path, 'a') as batch_log:
            batch_log.write('{},{},{},{}\n'.format(self.epoch,batch,logs.get('loss'),logs.get('acc')))

    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1
        with open(self.epoch_log_path, 'a') as epoch_log:
            epoch_log.write('{},{},{},{},{}\n'.format(epoch,logs.get('loss'),logs.get('acc'),logs.get('val_loss'),logs.get('val_acc')))

def gen_dirs():
    """
    Generate directory structure for storing files produced during current run.
    """
    date_time = strftime("%Y%m%d-%H%M%S", localtime())
    run_dir = os.path.join('runs',date_time)
    summary_path = os.path.join(run_dir,'RefineNet_summary.txt')
    settings_path = os.path.join(run_dir,'settings.txt')
    epoch_log_path = os.path.join(run_dir,'epoch_log_'+date_time+'.csv')
    batch_log_path = os.path.join(run_dir,'batch_log_'+date_time+'.csv')
    weight_dir = os.path.join(run_dir,'weights')
    tb_dir = os.path.join(run_dir,'log')
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
        os.makedirs(weight_dir)
        os.makedirs(tb_dir)
    dirs = {'date_time' : date_time,
            'run_dir' : run_dir,
            'summary_path' : summary_path,
            'settings_path' : settings_path,
            'epoch_log_path' : epoch_log_path,
            'batch_log_path' : batch_log_path,
            'weight_dir' : weight_dir,
            'tb_dir' : tb_dir}
    return dirs

def save_settings(settings_path,
                  summary_path,
                  resnet_trainable = None,
                  datagen_args = None,
                  batch_size = None,
                  input_shape = None,
                  dataset_basepath = None,
                  resnet_weights = None,
                  steps_per_epoch = None,
                  epochs = None,
                  pre_trained_weights = None,
                  model = None):
    """
    Save summary of training settings used in current run.
    """
    
    with open(summary_path, 'w') as f:
        with redirect_stdout(f):
            model.summary()
    
    with open(settings_path, 'w') as f:
        f.write('ResNet weights trainable: {}\n'.format(resnet_trainable))
        f.write('Data augmentation settings: {}\n'.format(datagen_args))
        f.write('Batch size: {}\n'.format(batch_size))
        f.write('Input shape: {}\n'.format(input_shape))
        f.write('Dataset path: {}\n'.format(dataset_basepath))
        f.write('ResNet weights path: {}\n'.format(resnet_weights))
        f.write('Steps per epoch: {}\n'.format(steps_per_epoch))
        f.write('Epochs: {}\n'.format(epochs))
        f.write('Pre-trained weights: {}\n'.format(pre_trained_weights))