from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import sys

import numpy as np
from . import model
from . import utils

import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
from tensorflow import keras


def get_args():
    """Argument parser.

      Returns:
        Dictionary of arguments.
      """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='GCS location to write checkpoints and export models')
    parser.add_argument(
        '--train-file',
        type=str,
        help='Training file local or GCS')
    parser.add_argument(
        '--train-labels-file',
        type=str,
        help='Training labels file local or GCS')
    parser.add_argument(
        '--test-file',
        type=str,
        help='Test file local or GCS')
    parser.add_argument(
        '--test-labels-file',
        type=str,
        help='Test file local or GCS')
    parser.add_argument(
        '--num-epochs',
        type=float,
        default=5,
        help='number of times to go through the data, default=5')
    parser.add_argument(
        '--batch-size',
        default=128,
        type=int,
        help='number of records to read during each training step, default=128')
    parser.add_argument(
        '--learning-rate',
        default=.01,
        type=float,
        help='learning rate for gradient descent, default=.001')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')
    return parser.parse_args()


def train_and_evaluate(hparams):
    """Helper function: Trains and evaluates model.

    Args:
      hparams: (dict) Command line parameters passed from task.py
    """
    # Loads data.
    base_dir = '/Users/lusenii/Developer/MachineLearning/cloudml-samples-master/gender/keras/data'
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')

    train_women_dir = os.path.join(train_dir, 'women')
    validation_women_dir = os.path.join(validation_dir, 'women')
    test_women_dir = os.path.join(test_dir, 'women')

    train_men_dir = os.path.join(train_dir, 'men')
    validation_men_dir = os.path.join(validation_dir, 'men')
    test_men_dir = os.path.join(test_dir, 'men')

    # All images will be rescaled by 1./255
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    # decreased batch size from 10 to 5
    train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x225
        target_size=(80, 120),
        batch_size=16,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(80, 120),
        batch_size=5,
        class_mode='binary')

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: model.input_fn(
            train_generator,
            None,
            16,
            mode=tf.estimator.ModeKeys.TRAIN),
        max_steps=1000)

    # Create EvalSpec.
    exporter = tf.estimator.LatestExporter('exporter', model.serving_input_fn)
    # Shape numpy array.
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: model.input_fn(
            validation_generator,
            None,
            5,
            mode=tf.estimator.ModeKeys.EVAL),
        steps=None,
        exporters=exporter,
        start_delay_secs=10,
        throttle_secs=10)

    # Define running config.
    run_config = tf.estimator.RunConfig(save_checkpoints_steps=500)
    # Create estimator.
    estimator = model.my_model(
        model_dir=hparams.job_dir,
        config=run_config)
    # Start training
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == '__main__':
    args = get_args()
    tf.logging.set_verbosity(args.verbosity)

    hparams = hparam.HParams(**args.__dict__)
    train_and_evaluate(hparams)
