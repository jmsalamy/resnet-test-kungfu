from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

from official.vision.image_classification import common
from official.vision.image_classification import imagenet_preprocessing
from official.vision.image_classification import resnet_model
from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.utils.misc import keras_utils
from official.utils.misc import model_helpers

from tensorflow.keras.applications.resnet import ResNet50

data_dir = "../../imagenet/data/imagenet/data"
batch_size = 128
train_epochs = 1


def load_eval_data(flags_obj): 
    dtype = flags_core.get_tf_dtype(flags_obj)
    input_fn = imagenet_preprocessing.input_fn
    return input_fn(
        is_training=False,
        data_dir=data_dir,
        batch_size=batch_size,
        num_epochs=train_epochs,
        parse_record_fn=imagenet_preprocessing.parse_record,
        dtype=dtype,
        drop_remainder=True)
    

def load_model(): 

    # model = resnet_model.resnet50(
    #         num_classes=imagenet_preprocessing.NUM_CLASSES)

    model = ResNet50(include_top=True, weights='imagenet')
    lr_schedule = 0.1
    # optimizer = common.get_optimizer(lr_schedule)
    optimizer = tf.keras.optimizers.Adam(0.1)

    model.compile(
            loss='sparse_categorical_crossentropy', 
            optimizer=optimizer,
            metrics=(['sparse_categorical_accuracy']))
    return model


def evaluate_model(flags_obj, ckpt):
    val_data = load_eval_data(flags_obj)
    model = load_model()
    # loss, acc = model.evaluate(val_data , verbose=1)
    # print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

    # model.load_weights(ckpt)

    loss,acc = model.evaluate(val_data, verbose=1)
    print("Pretrained model : , accuracy: {:5.2f}%".format(100*acc))


def define_imagenet_keras_flags():
    common.define_keras_flags()
    flags_core.set_defaults()
    flags.adopt_module_key_flags(common)


def main(_):
    model_helpers.apply_clean(flags.FLAGS)
    ckpt = "./models/model.ckpt-0060"
    with logger.benchmark_context(flags.FLAGS):
        evaluate_model(flags.FLAGS, ckpt)


if __name__ == '__main__':
    # logging.set_verbosity(logging.INFO)
    define_imagenet_keras_flags()
    app.run(main)