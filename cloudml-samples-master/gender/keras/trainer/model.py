from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras import optimizers


def keras_estimator(model_dir, config, learning_rate):
    """Creates a Keras Sequential model with layers.

    Args:
      model_dir: (str) file path where training files will be written.
      config: (tf.estimator.RunConfig) Configuration options to save model.
      learning_rate: (int) Learning rate.

    Returns:
      A keras.Model
    """
    model = models.Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dense(10, activation=tf.nn.softmax))

    # Compile model with learning parameters.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    estimator = tf.keras.estimator.model_to_estimator(
        keras_model=model, model_dir=model_dir, config=config)
    return estimator


def my_model(model_dir, config):
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation='relu',
                            input_shape=(150, 225, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    estimator = tf.keras.estimator.model_to_estimator(
        keras_model=model, model_dir=model_dir, config=config)
    return estimator


def input_fn(features, labels, batch_size, mode):
    """Input function.

    Args:
      features: (numpy.array) Training or eval data.
      labels: (numpy.array) Labels for training or eval data.
      batch_size: (int)
      mode: tf.estimator.ModeKeys mode

    Returns:
      A tf.estimator.
    """
    # Default settings for training.
    if labels is None:
        inputs = features
    else:
        # Change numpy array shape.
        inputs = (features, labels)
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(100).repeat().batch(batch_size)
    if mode in (tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT):
        dataset = dataset.batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


def serving_input_fn():
    """Defines the features to be passed to the model during inference.

    Expects already tokenized and padded representation of sentences

    Returns:
      A tf.estimator.export.ServingInputReceiver
    """
    feature_placeholder = tf.placeholder(tf.float32, [None, 9600])
    features = feature_placeholder
    return tf.estimator.export.TensorServingInputReceiver(features,
                                                          feature_placeholder)
