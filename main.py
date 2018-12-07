from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler

import os, math, fnmatch
from contextlib import redirect_stdout

from utils.data import trainGenerator, testGenerator, saveResult
from utils.helpers import get_label_info
from utils.utils import step_decay
from model.refinenet import build_refinenet

# Parameters
dataset_basepath = r'C:\Projects\MSc Thesis\data\Cityscapes' 
class_dict = 'class_dict.csv'

resnet_weights = 'model/resnet101_weights_tf.h5'

input_shape = (768,768,3)
batch_size = 1

#data_gen_args = dict(rotation_range=0.1,
#                    width_shift_range=0.05,
#                    height_shift_range=0.05,
#                    shear_range=0.05,
#                    zoom_range=0.05,
#                    horizontal_flip=True,
#                    fill_mode='nearest')

data_gen_args = dict()

save_summary = True

steps_per_epoch = math.floor(len(fnmatch.filter(os.listdir(os.path.join(dataset_basepath,'training','images')), '*.png'))/batch_size)

# Import classes from csv file
class_names_list, mask_colors, num_class, class_names_string = get_label_info(os.path.join(dataset_basepath,class_dict))

# Data generators for training
myTrainGenerator = trainGenerator(batch_size,os.path.join(dataset_basepath,
            'training'),'images','labels',num_class,input_shape,
            data_gen_args,mask_colors=mask_colors)
myValGenerator = trainGenerator(batch_size,os.path.join(dataset_basepath,
            'validation'),'images','labels',num_class,input_shape,
            data_gen_args,mask_colors=mask_colors)

# Define callbacks
model_checkpoint = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                   monitor = 'val_loss',
                                   verbose = 1,
                                   save_best_only = True)

tbCallBack = TensorBoard(log_dir='log', histogram_freq=0,
                         write_graph=True,
                         write_grads=True,
                         batch_size=batch_size,
                         write_images=True)

lrate = LearningRateScheduler(step_decay)

# Build and compile RefineNet
model = build_refinenet(input_shape, num_class, resnet_weights)
sgd = SGD(lr = 1e-5, momentum = 0.9, nesterov = True) # TODO: Tune optimizer
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

if save_summary:
    with open('RefineNet_summary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()


# Start training
model.fit_generator(myTrainGenerator,
                    steps_per_epoch = 2000,
                    validation_data = myValGenerator,
                    validation_steps = 30,
                    epochs = 50,
                    callbacks = [model_checkpoint, tbCallBack, lrate])

# Test
myTestGenerator = testGenerator(os.path.join(dataset_basepath,'data/test'))
results = model.predict_generator(myTestGenerator, 30, verbose=1)
saveResult('testing', results, mask_colors)