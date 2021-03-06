# -*- coding: utf-8 -*-
"""
Tutorial Keras: Transfer Learning with ResNet50 for image classification on Cats & Dogs dataset

This kernel is intended to be a tutorial on Keras around image files handling for Transfer Learning using pre-trained weights from ResNet50 convnet.

Though loading all train & test images resized (224 x 224 x 3) in memory would have incurred ~4.9GB of memory, the plan was to batch source image data during the training, validation & testing pipeline. Keras ImageDataGenerator supports batch sourcing image data for all training, validation and testing. Actually, it is quite clean and easy to use Keras ImageDataGenerator except few limitations (listed at the end).
"""

import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import tensorflow as tf

tf.test.gpu_device_name()

# KungFu imports
from kungfu import current_cluster_size, current_rank
from kungfu.tensorflow.initializer import BroadcastGlobalVariablesCallback
from kungfu.tensorflow.ops import broadcast
from kungfu.tensorflow.optimizers import (PairAveragingOptimizer,
                                          SynchronousAveragingOptimizer,
                                          SynchronousSGDOptimizer)
from kungfu.tensorflow.initializer import BroadcastGlobalVariablesCallback

# tf.compat.v1.disable_eager_execution()


# Global Constants

# Training data directory 
TRAIN_DIR = '../data/train_dst'
VAL_DIR = '../data/val_dst'

# Fixed for our Cats & Dogs classes
NUM_CLASSES = 2
CLASS_NAMES = np.array(['cats', 'dogs'])

# Fixed for Cats & Dogs color images
CHANNELS = 3

IMG_RESIZE = 224
RESNET50_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'

# Common accuracy metric for all outputs, but can use different metrics for different output
LOSS_METRICS = ['accuracy']

# EARLY_STOP_PATIENCE must be < NUM_EPOCHS
NUM_EPOCHS = 10
EARLY_STOP_PATIENCE = 3

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# Training images processed in each step would be no.-of-train-images / STEPS_PER_EPOCH_TRAINING
STEPS_PER_EPOCH_TRAINING = 10
STEPS_PER_EPOCH_VALIDATION = 10

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# NOTE that these BATCH* are for Keras ImageDataGenerator batching to fill epoch step input
BATCH_SIZE_TRAINING = 100
BATCH_SIZE_VALIDATION = 100

# Using 1 to easily manage mapping between test_generator & prediction for submission preparation
BATCH_SIZE_TESTING = 1

LEARNING_RATE = 0.01


"""### ResNet50
* Notice that resnet50 folder has 2 pre-trained weights files... xyz_tf_kernels.h5 & xyz_tf_kernels_NOTOP.h5
* The xyz_tf_kernels.h5 weights is useful for pure prediction of test image and this prediction will rely completely on ResNet50 pre-trained weights, i.e., it does not expected any training from our side
* Out intention in this kernel is Transfer Learning by using ResNet50 pre-trained weights except its TOP layer, i.e., the xyz_tf_kernels_NOTOP.h5 weights... Use this weights as initial weight for training new layer using train images
"""

# resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

"""### Define Our Transfer Learning Network Model Consisting of 2 Layers

Here, we are preparing specification or blueprint of the TensorFlow DAG (directed acyclcic graph) for just the MODEL part.
"""


def build_model(n_shards):
    model = Sequential()

    # 1st layer as the lumpsum weights from resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
    # NOTE that this layer will be set below as NOT TRAINABLE, i.e., use it as is
    model.add(ResNet50(include_top=False,
                       pooling=RESNET50_POOLING_AVERAGE, weights='imagenet'))

    # 2nd layer as Dense for 2-class classification, i.e., dog or cat using SoftMax activation
    model.add(Dense(NUM_CLASSES, activation=DENSE_LAYER_ACTIVATION))

    # Say not to train first layer (ResNet) model as it is already trained
    model.layers[0].trainable = False
    model.summary()

    sgd = optimizers.SGD(lr=LEARNING_RATE*n_shards, decay=1e-6, momentum=0.9, nesterov=True)
    sgd_kungfu = SynchronousSGDOptimizer(sgd, use_locking=True)
    model.compile(optimizer=sgd_kungfu, loss=OBJECTIVE_FUNCTION, metrics=LOSS_METRICS)

    return model


"""### Prepare Keras Data Generators

Keras *ImageDataGenerator(...)* generates batches of tensor image data with real-time data augmentation. The data will be looped over (in batches). It is useful with large dataset to source, pre-process (resize, color conversion, image augmentation, batch normalize) & supply resulting images in batches to downstream Keras modeling components, namely *fit_generator(...)* & *predict_generator(...)* -vs- *fit(...)* & *predict(...)* for small dataset.

Kaggle competition rule expects Dog & Cat to be labeled as 1 & 0. Keras >> ImageDataGenerator >> flow_from_directory takes in 'classes' list for mapping it to LABEL indices otherwise treats sub-folders enumerated classes in alphabetical order, i.e., Cat is 0 & Dog is 1.
"""

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2] == CLASS_NAMES


def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    # img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_RESIZE, IMG_RESIZE])
    return img


def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def process_data(data_dir, sharding=True):
    ds = tf.data.Dataset.list_files(str(data_dir + "/*/*"))

    if sharding:
        ds = ds.shard(num_shards = current_cluster_size(),
                index=current_rank())

    ds = ds.map(process_path, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH_SIZE_TRAINING)
    return ds


def train_model(model, train_dataset, val_dataset,  epochs):
    """### Train Our Model With Cats & Dogs Train (splitted) Data Set"""

    # Early stopping & checkpointing the best model in ../working dir & restoring that as our model for prediction
#    cb_early_stopper = EarlyStopping(
#        monitor='val_loss', patience=EARLY_STOP_PATIENCE)
#    cb_checkpointer = ModelCheckpoint(
#        filepath='../working/best.hdf5', monitor='val_loss', save_best_only=True, mode='auto')

    # Accumulate history of all permutations (may be for viewing trend) and keep watching for lowest val_loss as final model

    model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=[BroadcastGlobalVariablesCallback()]
    )
#    model.load_weights("../working/best.hdf5")


if __name__ == "__main__":
    n_shards = current_cluster_size()
    model = build_model(n_shards)
    train_dataset = process_data(TRAIN_DIR)
    val_dataset = process_data(VAL_DIR, sharding=False)
    train_model(model, train_dataset, val_dataset, epochs=NUM_EPOCHS)


"""### Keras Limitations

* [10/02/2018] The *validation_split* is not supported in *fit_generator*, hence its expects ImageDataGenerator for pre-splitted train & valid.
* [10/02/2018] Model learning through *fit_generator* is not compatible for Sklearn *GridSearchCV* again *mostly* due to no support for *validation_split*.

### Followup Plan

1. Scale and pad and avoid aspect ratio change of original image through Keras ImageDataGenerator pre-processing insfrastructure
2. Image augmentation
3. Pipeline
4. Distributed ML for Grid Search on Spark Cluster

### References

1. [Transfer Learning by Dan B](https://www.kaggle.com/dansbecker/transfer-learning)
"""
