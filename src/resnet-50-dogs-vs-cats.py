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

"""### Global Constants"""

# Fixed for our Cats & Dogs classes
NUM_CLASSES = 2

# Fixed for Cats & Dogs color images
CHANNELS = 3

IMAGE_RESIZE = 224
RESNET50_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'

# Common accuracy metric for all outputs, but can use different metrics for different output
LOSS_METRICS = ['accuracy']

# EARLY_STOP_PATIENCE must be < NUM_EPOCHS
NUM_EPOCHS = 1
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


"""### ResNet50
* Notice that resnet50 folder has 2 pre-trained weights files... xyz_tf_kernels.h5 & xyz_tf_kernels_NOTOP.h5
* The xyz_tf_kernels.h5 weights is useful for pure prediction of test image and this prediction will rely completely on ResNet50 pre-trained weights, i.e., it does not expected any training from our side
* Out intention in this kernel is Transfer Learning by using ResNet50 pre-trained weights except its TOP layer, i.e., the xyz_tf_kernels_NOTOP.h5 weights... Use this weights as initial weight for training new layer using train images
"""

# resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

"""### Define Our Transfer Learning Network Model Consisting of 2 Layers

Here, we are preparing specification or blueprint of the TensorFlow DAG (directed acyclcic graph) for just the MODEL part.
"""


def build_model():
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

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=OBJECTIVE_FUNCTION, metrics=LOSS_METRICS)

    return model


"""### Prepare Keras Data Generators

Keras *ImageDataGenerator(...)* generates batches of tensor image data with real-time data augmentation. The data will be looped over (in batches). It is useful with large dataset to source, pre-process (resize, color conversion, image augmentation, batch normalize) & supply resulting images in batches to downstream Keras modeling components, namely *fit_generator(...)* & *predict_generator(...)* -vs- *fit(...)* & *predict(...)* for small dataset.

Kaggle competition rule expects Dog & Cat to be labeled as 1 & 0. Keras >> ImageDataGenerator >> flow_from_directory takes in 'classes' list for mapping it to LABEL indices otherwise treats sub-folders enumerated classes in alphabetical order, i.e., Cat is 0 & Dog is 1.
"""


def process_data():
    image_size = IMAGE_RESIZE

    # preprocessing_function is applied on each image but only after re-sizing & augmentation (resize => augment => pre-process)
    # Each of the keras.application.resnet* preprocess_input MOSTLY mean BATCH NORMALIZATION (applied on each batch) stabilize the inputs to nonlinear activation functions
    # Batch Normalization helps in faster convergence
    data_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input)

    # flow_From_directory generates batches of augmented data (where augmentation can be color conversion, etc)
    # Both train & valid folders must have NUM_CLASSES sub-folders
    train_generator = data_generator.flow_from_directory(
        '../data/train_dst',
        target_size=(image_size, image_size),
        batch_size=BATCH_SIZE_TRAINING,
        class_mode='categorical')

    validation_generator = data_generator.flow_from_directory(
        '../data/valid_dst',
        target_size=(image_size, image_size),
        batch_size=BATCH_SIZE_VALIDATION,
        class_mode='categorical')
    return train_generator, validation_generator


def train_model(model, train_generator, validation_generator, epochs):
    """### Train Our Model With Cats & Dogs Train (splitted) Data Set"""

    # Early stopping & checkpointing the best model in ../working dir & restoring that as our model for prediction
    cb_early_stopper = EarlyStopping(
        monitor='val_loss', patience=EARLY_STOP_PATIENCE)
    cb_checkpointer = ModelCheckpoint(
        filepath='../working/best.hdf5', monitor='val_loss', save_best_only=True, mode='auto')

    # Accumulate history of all permutations (may be for viewing trend) and keep watching for lowest val_loss as final model

    model.fit_generator(
        train_generator,
        epochs=NUM_EPOCHS,
        validation_data=validation_generator,
        callbacks=[cb_checkpointer, cb_early_stopper]
    )
    model.load_weights("../working/best.hdf5")


if __name__ == "__main__":
    model = build_model()
    train_generator, validation_generator = process_data()
    train_model(model, train_generator,
                validation_generator, epochs=NUM_EPOCHS)


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

