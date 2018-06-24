
# coding: utf-8

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dropout, Activation, Flatten, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
import h5py
from keras import backend as K
import numpy as np
import itertools
import matplotlib.pyplot as plt


# In[ ]:


training_path = "/workspace/dog-vs-cat/train"
valid_path = "/workspace/dog-vs-cat/val"
output_path = "/workspace/dog-vs-cat/output"
step_per_epoch = 2000
step_validation = 400
epochs = 500
batch_size = 32
width = 224
height = 224

if K.image_data_format() == "channels_first":
    input_shape = (3, width, height)
else:
    input_shape = (width, height, 3)

print(input_shape)


# # Generate Batch Images
# ## Batch Training

# In[ ]:


# train_data = ImageDataGenerator(
#     rescale = 1./255,
#     shear_range = 0.2,
#     zoom_range = 0.2,
#     horizontal_flip = True
# )


# In[ ]:


train_batches = ImageDataGenerator().flow_from_directory(
    training_path,
    classes = ['Cats', 'Dogs'],
    batch_size = batch_size,
    target_size = (width,height)
)


# ## Batch Validating

# In[ ]:


# val_data = ImageDataGenerator(rescale=1./255)


# In[ ]:


validation_batches = ImageDataGenerator().flow_from_directory(
    valid_path,
    classes = ['Cats', 'Dogs'],
    batch_size = batch_size,
    target_size = (width,height)
)


# # Model Definition

# In[ ]:


# model = Sequential()
# model.add(Conv2D(32,(3,3), input_shape = input_shape))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size = (2,2)))

# model.add(Conv2D(32,(3,3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size = (2,2)))

# model.add(Conv2D(64, (3,3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size = (2,2)))

# model.add(Flatten())
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.4))
# model.add(Dense(1))
# model.add(Activation('sigmoid'))

# model.compile(
#     loss = 'binary_crossentropy',
#     optimizer = 'rmsprop',
#     metrics = ['accuracy']
# )


# # Model Definition (VGG-16)

# In[ ]:


model = Sequential()

# Group: Conv Block 1
model.add(ZeroPadding2D((1,1),input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

# Group: Conv Block 2
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

# Group: Conv Block 3
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

# Group: Conv Block 4
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

# Group: Conv Block 5
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Flatten())

# Group: Dense Block
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(
    Adam(lr=.0001),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

model.summary()


# # Build CNN

# In[ ]:


model.fit_generator(
    train_batches,
    steps_per_epoch = step_per_epoch,
    epochs = epochs,
    validation_data = validation_batches,
    validation_steps = step_validation
)


# # Save Weights

# In[ ]:


model.save(output_path + "/weights.hdf5")
model.summary()

