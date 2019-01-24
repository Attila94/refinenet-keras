from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler

import os

from utils.data import trainGenerator, testGenerator, saveResult
from utils.helpers import get_label_info, gen_dirs, step_decay, ignore_unknown_xentropy, save_settings, LossHistory
from model.refinenet import build_refinenet
  
#### Define parameters
# Dataset
dataset_basepath = 'path_to_dataset' 
class_dict = 'class_dict.csv'
# ResNet
resnet_weights = 'path_to_resnet_weights/resnet101_weights_tf.h5'
resnet_trainable = True
# Network architecture
input_shape = (512,1024,3) # height x width x channels
# Train settings
pre_trained_weights = None
batch_size = 2
steps_per_epoch = 2975//batch_size + 1
epochs = 50
data_gen_args = dict(rotation_range=0.1,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

# Import classes from csv file
class_names_list, mask_colors, num_class, class_names_string = get_label_info(os.path.join(dataset_basepath,class_dict))
   
dirs = gen_dirs()

# Data generators for training
myTrainGenerator = trainGenerator(batch_size,
                                  os.path.join(dataset_basepath,'training'),
                                  'images',
                                  'labels',
                                  num_class,
                                  input_shape[:2],
                                  data_gen_args,
                                  mask_colors = mask_colors)
myValGenerator = trainGenerator(batch_size,
                                os.path.join(dataset_basepath,'validation'),
                                'images',
                                'labels',
                                num_class,
                                input_shape[:2],
                                data_gen_args,
                                mask_colors = mask_colors)

# Define callbacks
model_checkpoint = ModelCheckpoint(dirs['weight_dir']+'/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                   monitor = 'val_loss',
                                   verbose = 1,
                                   save_best_only = False)
tbCallBack = TensorBoard(log_dir=dirs['tb_dir'], histogram_freq=0,
                         write_graph=True,
                         write_grads=True,
                         batch_size=batch_size,
                         write_images=True)
history = LossHistory(dirs['batch_log_path'], dirs['epoch_log_path'])
lrate = LearningRateScheduler(step_decay)

# Build and compile RefineNet
model = build_refinenet(input_shape, num_class, resnet_weights, resnet_trainable)
model.compile(optimizer = Adam(lr=1e-5), loss = ignore_unknown_xentropy, metrics = ['accuracy'])

if pre_trained_weights:
    model.load_weights(pre_trained_weights)

save_settings(dirs['settings_path'],
                dirs['summary_path'],
                resnet_trainable = resnet_trainable,
                datagen_args = data_gen_args,
                batch_size = batch_size,
                input_shape = input_shape,
                dataset_basepath = dataset_basepath,
                resnet_weights = resnet_weights,
                steps_per_epoch = steps_per_epoch,
                epochs = epochs,
                pre_trained_weights = pre_trained_weights,
                model = model)

# Start training
model.fit_generator(myTrainGenerator,
                    steps_per_epoch = steps_per_epoch,
                    validation_data = myValGenerator,
                    validation_steps = 50//batch_size,
                    epochs = epochs,
                    callbacks = [model_checkpoint, tbCallBack, lrate, history])