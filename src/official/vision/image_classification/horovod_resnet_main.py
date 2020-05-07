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
"""Runs a ResNet model on the ImageNet dataset using Horovod."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from official.benchmark.models import trivial_model
from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.utils.misc import distribution_utils
from official.utils.misc import keras_utils
from official.utils.misc import model_helpers
from official.vision.image_classification import common
from official.vision.image_classification import imagenet_preprocessing
from official.vision.image_classification import resnet_model

#Horovod Import
import horovod.tensorflow as hvd
# Horovod: initialize Horovod.
hvd.init()
# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

def run(flags_obj):
  """Run ResNet ImageNet training and eval loop using native Keras APIs.

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
        num_gpus=flags_obj.num_gpus,
        datasets_num_private_threads=flags_obj.datasets_num_private_threads)
  common.set_cudnn_batchnorm_mode()

  dtype = flags_core.get_tf_dtype(flags_obj)
  if dtype == tf.float16:
    loss_scale = flags_core.get_loss_scale(flags_obj, default_for_fp16=128)
    policy = tf.compat.v2.keras.mixed_precision.experimental.Policy(
        'mixed_float16', loss_scale=loss_scale)
    tf.compat.v2.keras.mixed_precision.experimental.set_policy(policy)
    if not keras_utils.is_v2_0():
      raise ValueError('--dtype=fp16 is not supported in TensorFlow 1.')
  elif dtype == tf.bfloat16:
    policy = tf.compat.v2.keras.mixed_precision.experimental.Policy(
        'mixed_bfloat16')
    tf.compat.v2.keras.mixed_precision.experimental.set_policy(policy)

  data_format = flags_obj.data_format
  if data_format is None:
    data_format = ('channels_first'
                   if tf.test.is_built_with_cuda() else 'channels_last')
  tf.keras.backend.set_image_data_format(data_format)

  # pylint: disable=protected-access
  if flags_obj.use_synthetic_data:
    distribution_utils.set_up_synthetic_data()
    input_fn = common.get_synth_input_fn(
        height=imagenet_preprocessing.DEFAULT_IMAGE_SIZE,
        width=imagenet_preprocessing.DEFAULT_IMAGE_SIZE,
        num_channels=imagenet_preprocessing.NUM_CHANNELS,
        num_classes=imagenet_preprocessing.NUM_CLASSES,
        dtype=dtype,
        drop_remainder=True)
  else:
    distribution_utils.undo_set_up_synthetic_data()
    input_fn = imagenet_preprocessing.input_fn

  # When `enable_xla` is True, we always drop the remainder of the batches
  # in the dataset, as XLA-GPU doesn't support dynamic shapes.
  drop_remainder = flags_obj.enable_xla

  train_input_dataset = input_fn(
      is_training=True,
      data_dir=flags_obj.data_dir,
      batch_size=flags_obj.batch_size,
      num_epochs=flags_obj.train_epochs,
      parse_record_fn=imagenet_preprocessing.parse_record,
      datasets_num_private_threads=flags_obj.datasets_num_private_threads,
      dtype=dtype,
      drop_remainder=drop_remainder,
      tf_data_experimental_slack=flags_obj.tf_data_experimental_slack,
      training_dataset_cache=flags_obj.training_dataset_cache,
  )

  eval_input_dataset = None
  if not flags_obj.skip_eval:
    eval_input_dataset = input_fn(
        is_training=False,
        data_dir=flags_obj.data_dir,
        batch_size=flags_obj.batch_size,
        num_epochs=flags_obj.train_epochs,
        parse_record_fn=imagenet_preprocessing.parse_record,
        dtype=dtype,
        drop_remainder=drop_remainder)

  lr_schedule = 0.1
  if flags_obj.use_tensor_lr:
    lr_schedule = common.PiecewiseConstantDecayWithWarmup(
        batch_size=flags_obj.batch_size,
        epoch_size=imagenet_preprocessing.NUM_IMAGES['train'],
        warmup_epochs=common.LR_SCHEDULE[0][1],
        boundaries=list(p[1] for p in common.LR_SCHEDULE[1:]),
        multipliers=list(p[0] for p in common.LR_SCHEDULE),
        compute_lr_on_cpu=True)

  fp16_allreduce = False
  # Horovod: (optional) compression algorithm.
  compression = hvd.Compression.fp16 if fp16_allreduce else hvd.Compression.none

  # Horovod: adjust learning rate based schedule + number GPUs.
  opt = common.get_optimizer(lr_schedule * hvd.size())

  # Horovod: add Horovod Distributed Optimizer.
  optimizer = hvd.DistributedOptimizer(opt, compression=compression)

  if flags_obj.fp16_implementation == 'graph_rewrite':
    # Note: when flags_obj.fp16_implementation == "graph_rewrite", dtype as
    # determined by flags_core.get_tf_dtype(flags_obj) would be 'float32'
    # which will ensure tf.compat.v2.keras.mixed_precision and
    # tf.train.experimental.enable_mixed_precision_graph_rewrite do not double
    # up.
    optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(
        optimizer)

  # TODO(hongkuny): Remove trivial model usage and move it to benchmark.
  if flags_obj.use_trivial_model:
    model = trivial_model.trivial_model(
        imagenet_preprocessing.NUM_CLASSES)
  else:
    model = resnet_model.resnet50(
        num_classes=imagenet_preprocessing.NUM_CLASSES)

  # TODO(b/138957587): Remove when force_v2_in_keras_compile is on longer
  # a valid arg for this model. Also remove as a valid flag.

    metrics = (['sparse_categorical_accuracy'])
    metrics.append('sparse_top_k_categorical_accuracy')

  if flags_obj.force_v2_in_keras_compile is not None:
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=metrics,
        run_eagerly=flags_obj.run_eagerly,
        experimental_run_tf_function=flags_obj.force_v2_in_keras_compile)
  else:
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=metrics,
        run_eagerly=flags_obj.run_eagerly)

  # adjust number of steps
  cluster_size = hvd.size()
  steps_per_epoch = (
      imagenet_preprocessing.NUM_IMAGES['train'] // flags_obj.batch_size)
  steps_per_epoch = steps_per_epoch // cluster_size

  train_epochs = flags_obj.train_epochs

  callbacks = [
      # Horovod: broadcast initial variable states from rank 0 to all other processes.
      # This is necessary to ensure consistent initialization of all workers when
      # training is started with random weights or restored from a checkpoint.
      hvd.callbacks.BroadcastGlobalVariablesCallback(0),

      # Horovod: average metrics among workers at the end of every epoch.
      #
      # Note: This callback must be in the list before the ReduceLROnPlateau,
      # TensorBoard, or other metrics-based callbacks.
      hvd.callbacks.MetricAverageCallback(),

      # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
      # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
      # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
      hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=args.warmup_epochs, verbose=verbose),

      # Horovod: after the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
      hvd.callbacks.LearningRateScheduleCallback(start_epoch=args.warmup_epochs, end_epoch=30, multiplier=1.),
      hvd.callbacks.LearningRateScheduleCallback(start_epoch=30, end_epoch=60, multiplier=1e-1),
      hvd.callbacks.LearningRateScheduleCallback(start_epoch=60, end_epoch=80, multiplier=1e-2),
      hvd.callbacks.LearningRateScheduleCallback(start_epoch=80, multiplier=1e-3),

      common.get_callbacks(steps_per_epoch, hvd.rank(), hvd.size(), common.learning_rate_schedule),
  ]

  if flags_obj.enable_checkpoint_and_export and hvd.rank() == 0:
    ckpt_full_path = os.path.join(
      flags_obj.model_dir, 'model.ckpt-{epoch:04d}')
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(ckpt_full_path,
                                                        save_weights_only=True))

  if flags_obj.train_steps:
    steps_per_epoch = min(flags_obj.train_steps, steps_per_epoch)

  num_eval_steps = (
      imagenet_preprocessing.NUM_IMAGES['validation'] // flags_obj.batch_size)

  validation_data = eval_input_dataset
  if flags_obj.skip_eval:
    # Only build the training graph. This reduces memory usage introduced by
    # control flow ops in layers that have different implementations for
    # training and inference (e.g., batch norm).
    if flags_obj.set_learning_phase_to_train:
      # TODO(haoyuzhang): Understand slowdown of setting learning phase when
      # not using distribution strategy.
      tf.keras.backend.set_learning_phase(1)
    num_eval_steps = None
    validation_data = None

  history = model.fit(train_input_dataset,
                      epochs=train_epochs,
                      steps_per_epoch=steps_per_epoch,
                      callbacks=callbacks,
                      validation_steps=num_eval_steps,
                      validation_data=validation_data,
                      validation_freq=flags_obj.epochs_between_evals,
                      verbose=2)

  # Checkpoint only on 0th worker
  if flags_obj.enable_checkpoint_and_export and hvd.rank() == 0:
      if dtype == tf.bfloat16:
          logging.warning(
              "Keras model.save does not support bfloat16 dtype.")
      else:
          # Keras model.save assumes a float32 input designature.
          export_path = os.path.join(flags_obj.model_dir, 'saved_model')
          model.save(export_path, include_optimizer=False)

  eval_output = None
  if not flags_obj.skip_eval:
    eval_output = model.evaluate(eval_input_dataset,
                                  steps=num_eval_steps,
                                  verbose=2)

  stats = common.build_stats(history, eval_output, callbacks)
  return stats


def define_imagenet_keras_flags():
  common.define_keras_flags()
  flags_core.set_defaults()
  flags.adopt_module_key_flags(common)


def main(_):
  model_helpers.apply_clean(flags.FLAGS)
  with logger.benchmark_context(flags.FLAGS):
    stats = run(flags.FLAGS)
  logging.info('Run stats:\n%s', stats)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  define_imagenet_keras_flags()
  app.run(main)
