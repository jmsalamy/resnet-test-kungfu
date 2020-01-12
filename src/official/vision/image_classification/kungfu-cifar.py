# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs a ResNet model on the Cifar-10 dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import app as absl_app
import tensorflow as tf

from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.utils.misc import distribution_utils
from official.utils.misc import keras_utils
from official.vision.image_classification import cifar_preprocessing
from official.vision.image_classification import common
from official.vision.image_classification import resnet_cifar_model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Activation, AveragePooling2D,
                                     BatchNormalization, Conv2D, Dense,
                                     Dropout, Flatten, Input, MaxPooling2D)

from tensorflow.compat.v1 import logging

# KungFu imports
from kungfu import current_cluster_size, current_rank
from kungfu.tensorflow.initializer import BroadcastGlobalVariablesCallback
from kungfu.tensorflow.ops import broadcast
from kungfu.tensorflow.optimizers import (PairAveragingOptimizer,
                                          SynchronousAveragingOptimizer,
                                          SynchronousSGDOptimizer)
from kungfu.tensorflow.initializer import BroadcastGlobalVariablesCallback


num_classes = 10
LR_SCHEDULE = [  # (multiplier, epoch to start) tuples
    (0.1, 20), (0.01, 30), (0.001, 40)
]

def Conv4_model(x_train, num_classes):

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=x_train.shape[1:], name="conv_1"))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), name="conv_2"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', name="conv_3"))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), name="conv_4"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model


def learning_rate_schedule(current_epoch,
                           current_batch,
                           batches_per_epoch,
                           batch_size, 
                           cluster_size):

  """Handles linear scaling rule and LR decay.

  Scale learning rate at epoch boundaries provided in LR_SCHEDULE by the
  provided scaling factor.

  Args:
    current_epoch: integer, current epoch indexed from 0.
    current_batch: integer, current batch in the current epoch, indexed from 0.
    batches_per_epoch: integer, number of steps in an epoch.
    batch_size: integer, total batch sized.

  Returns:
    Adjusted learning rate.
  """
  del current_batch, batches_per_epoch  # not used
  initial_learning_rate = common.BASE_LEARNING_RATE * batch_size / 128
  learning_rate = initial_learning_rate
  for mult, start_epoch in LR_SCHEDULE:
    if current_epoch >= start_epoch:
      learning_rate = initial_learning_rate * mult
    else:
      break
  return learning_rate


def run(flags_obj):
  """Run ResNet Cifar-10 training and eval loop using native Keras APIs.

  Args:
    flags_obj: An object containing parsed flag values.

  Raises:
    ValueError: If fp16 is passed as it is not currently supported.

  Returns:
    Dictionary of training and eval stats.
  """
  keras_utils.set_session_config(
      enable_eager=flags_obj.enable_eager,
      enable_xla=flags_obj.enable_xla)

  # Execute flag override logic for better model performance
  if flags_obj.tf_gpu_thread_mode:
    keras_utils.set_gpu_thread_mode_and_count(
        per_gpu_thread_count=flags_obj.per_gpu_thread_count,
        gpu_thread_mode=flags_obj.tf_gpu_thread_mode,
        num_gpus=flags_obj.num_gpus, datasets_num_private_threads=flags_obj.datasets_num_private_threads)
  common.set_cudnn_batchnorm_mode()

  dtype = flags_core.get_tf_dtype(flags_obj)
  if dtype == 'fp16':
    raise ValueError('dtype fp16 is not supported in Keras. Use the default '
                     'value(fp32).')

  data_format = flags_obj.data_format
  if data_format is None:
    data_format = ('channels_first'
                   if tf.test.is_built_with_cuda() else 'channels_last')
  tf.keras.backend.set_image_data_format(data_format)

  strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy=flags_obj.distribution_strategy,
      num_gpus=flags_obj.num_gpus,
      num_workers=distribution_utils.configure_cluster(),
      all_reduce_alg=flags_obj.all_reduce_alg,
      num_packs=flags_obj.num_packs)

  if strategy:
    # flags_obj.enable_get_next_as_optional controls whether enabling
    # get_next_as_optional behavior in DistributedIterator. If true, last
    # partial batch can be supported.
    strategy.extended.experimental_enable_get_next_as_optional = (
        flags_obj.enable_get_next_as_optional
    )

  strategy_scope = distribution_utils.get_strategy_scope(strategy)

  if flags_obj.use_synthetic_data:
    distribution_utils.set_up_synthetic_data()
    input_fn = common.get_synth_input_fn(
        height=cifar_preprocessing.HEIGHT,
        width=cifar_preprocessing.WIDTH,
        num_channels=cifar_preprocessing.NUM_CHANNELS,
        num_classes=cifar_preprocessing.NUM_CLASSES,
        dtype=flags_core.get_tf_dtype(flags_obj),
        drop_remainder=True)
  else:
    distribution_utils.undo_set_up_synthetic_data()
    input_fn = cifar_preprocessing.input_fn

  #train_input_dataset = input_fn(
  #    is_training=True,
  #    data_dir=flags_obj.data_dir,
  #    batch_size=flags_obj.batch_size,
  #    num_epochs=flags_obj.train_epochs,
  #    parse_record_fn=cifar_preprocessing.parse_record,
  #    datasets_num_private_threads=flags_obj.datasets_num_private_threads,
  #    dtype=dtype,
  #    # Setting drop_remainder to avoid the partial batch logic in normalization
  #    # layer, which triggers tf.where and leads to extra memory copy of input
  #    # sizes between host and GPU.
  #    drop_remainder=(not flags_obj.enable_get_next_as_optional))

  # eval_input_dataset = None
  # if not flags_obj.skip_eval:
  #   eval_input_dataset = input_fn(
  #       is_training=False,
  #       data_dir=flags_obj.data_dir,
  #       batch_size=flags_obj.batch_size,
  #       num_epochs=flags_obj.train_epochs,
  #       parse_record_fn=cifar_preprocessing.parse_record)
  
  (x_train, y_train) , (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /=255
  x_test /=255
  y_train = tf.keras.utils.to_categorical(y_train, num_classes)
  y_test = tf.keras.utils.to_categorical(y_test, num_classes)


  # optimizer = common.get_optimizer()

  opt = tf.keras.optimizers.SGD(learning_rate=0.1)

  logging.info(opt.__dict__)
  optimizer = SynchronousSGDOptimizer(opt, use_locking=True) 
  optimizer._hyper = opt._hyper
  					   
  logging.info(optimizer.__dict__)

  model = Conv4_model(x_train, num_classes)
  
  # TODO(b/138957587): Remove when force_v2_in_keras_compile is on longer
  # a valid arg for this model. Also remove as a valid flag.
  if flags_obj.force_v2_in_keras_compile is not None:
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=(['accuracy']),
        run_eagerly=flags_obj.run_eagerly,
        experimental_run_tf_function=flags_obj.force_v2_in_keras_compile)
  else:
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=(['accuracy']),
        run_eagerly=flags_obj.run_eagerly)
  

  cluster_size = current_cluster_size()
  steps_per_epoch = (
  cifar_preprocessing.NUM_IMAGES['train'] // flags_obj.batch_size)
  steps_per_epoch = steps_per_epoch // cluster_size
  train_epochs = flags_obj.train_epochs
  
  callbacks = common.get_callbacks(steps_per_epoch, current_rank(), cluster_size, learning_rate_schedule)
  callbacks.append(BroadcastGlobalVariablesCallback())  

  if flags_obj.train_steps:
     steps_per_epoch = min(flags_obj.train_steps, steps_per_epoch)
 
  num_eval_steps = (cifar_preprocessing.NUM_IMAGES['validation'] //
                    flags_obj.batch_size)

  # validation_data = eval_input_dataset
  if flags_obj.skip_eval:
    if flags_obj.set_learning_phase_to_train:
      # TODO(haoyuzhang): Understand slowdown of setting learning phase when
      # not using distribution strategy.
      tf.keras.backend.set_learning_phase(1)
    num_eval_steps = None
    validation_data = None


  tf.compat.v1.logging.info(x_train.shape)
  history = model.fit(x_train, y_train,
                      batch_size=flags_obj.batch_size,
                      epochs=train_epochs,
                      steps_per_epoch=steps_per_epoch,
                      callbacks=callbacks,
                      validation_steps=num_eval_steps,
                      validation_data=(x_test, y_test),
                      validation_freq=flags_obj.epochs_between_evals,
                      verbose=2)
  eval_output = None
  if not flags_obj.skip_eval:
    eval_output = model.evaluate((x_test, y_test),
                                 steps=num_eval_steps,
                                 verbose=2)
  stats = common.build_stats(history, eval_output, callbacks)
  return stats


def define_cifar_flags():
  common.define_keras_flags(dynamic_loss_scale=False)

  flags_core.set_defaults(data_dir='/tmp/cifar10_data/cifar-10-batches-bin',
                          model_dir='/tmp/cifar10_model',
                          epochs_between_evals=1,
                          batch_size=128)


def main(_):
  with logger.benchmark_context(flags.FLAGS):
    return run(flags.FLAGS)


if __name__ == '__main__':
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  define_cifar_flags()
  absl_app.run(main)
