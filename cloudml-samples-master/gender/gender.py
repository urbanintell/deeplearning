
# coding: utf-8

# In[1]:

import os
import shutil
from tensorflow.keras import layers
from tensorflow.keras import models
from PIL import ImageFile
from tensorflow import keras
from tensorflow.keras import optimizers


base_dir = '/Users/lusenii/Developer/MachineLearning/cloudml-samples-master/gender/data'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')


train_women_dir = os.path.join(train_dir, 'women')
validation_women_dir = os.path.join(validation_dir, 'women')
test_women_dir = os.path.join(test_dir, 'women')


train_men_dir = os.path.join(train_dir, 'men')
validation_men_dir = os.path.join(validation_dir, 'men')
test_men_dir = os.path.join(test_dir, 'men')


print('total training male images:', len(os.listdir(train_men_dir)))
print('total training female images:', len(os.listdir(train_women_dir)))
print('total validation male images:', len(os.listdir(validation_men_dir)))
print('total validation female images:', len(os.listdir(validation_women_dir)))
print('total test male images:', len(os.listdir(test_men_dir)))
print('total test women images:', len(os.listdir(test_women_dir)))


# parameters should not exceed number of datapoints in training per label by a factor of 10.
# small dataste NN was able to memorize was not able to generalize
# model can capture variations
# smaller means number of parameters
# increasing layers m decrease parameters
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


sgd = optimizers.SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
# rmsprop = optimizers.RMSprop(lr=1e-2, decay=1e-6)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)
# decreased batch size from 10 to 5
train_generator = train_datagen.flow_from_directory(
    # This is the target directory
    train_dir,
    # All images will be resized to 150x225
    target_size=(150, 225),
    batch_size=16,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 225),
    batch_size=5,
    class_mode='binary')


keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0.1,
                              patience=3,
                              verbose=0, mode='auto')


keras.callbacks.TensorBoard(log_dir='./logs',
                            histogram_freq=0,
                            batch_size=32,
                            write_graph=True,
                            write_grads=False,
                            write_images=False,
                            embeddings_freq=0,
                            embeddings_layer_names=None,
                            embeddings_metadata=None,
                            embeddings_data=None)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=10,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=40)


# In[46]:

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[48]:

keras.callbacks.TensorBoard(log_dir='./logs',
                            histogram_freq=0,
                            batch_size=32,
                            write_graph=True,
                            write_grads=False,
                            write_images=False,
                            embeddings_freq=0,
                            embeddings_layer_names=None,
                            embeddings_metadata=None,
                            embeddings_data=None)


# In[112]:

model.save('thesatorialist.h5')
