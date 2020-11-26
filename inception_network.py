"""

Inception model function. This Inception network was implemented by FAIZAN SHAIKH, 
and was modified to meet the requirements of this project. The original implementation
can be found in the following link:
https://www.analyticsvidhya.com/blog/2018/10/understanding-inception-network-from-scratch/

Modified by Nicolas Gachancipa to meet the requirements of the project.

"""
# Imports
from keras.models import Model
from tensorflow.keras.layers import (Conv2D, MaxPool2D, Dropout, Dense, Input, 
                                     concatenate, GlobalAveragePooling2D, 
                                     AveragePooling2D, Flatten)
from tensorflow import keras

# Model definition.
def inception_model(input_shape):
  """
  Inception Model V1 (GoogLeNet)
  """
  b = keras.initializers.Constant(value=0.2)

  # Input layers.
  input_layer  = Input(shape=input_shape)
  x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', bias_initializer=b)(input_layer)
  x = MaxPool2D((3, 3), padding='same', strides=(2, 2))(x)
  x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu')(x)
  x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu')(x)
  x = MaxPool2D((3, 3), padding='same', strides=(2, 2))(x)

  # Inception 3a
  conv_1x1 = Conv2D(64, (1, 1), padding='same', activation='relu', bias_initializer=b)(x)
  conv_3x3 = Conv2D(96, (1, 1), padding='same', activation='relu', bias_initializer=b)(x)
  conv_3x3 = Conv2D(128, (3, 3), padding='same', activation='relu', bias_initializer=b)(conv_3x3)
  conv_5x5 = Conv2D(16, (1, 1), padding='same', activation='relu', bias_initializer=b)(x)
  conv_5x5 = Conv2D(32, (5, 5), padding='same', activation='relu', bias_initializer=b)(conv_5x5)
  pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
  pool_proj = Conv2D(32, (1, 1), padding='same', activation='relu',  bias_initializer=b)(pool_proj)
  x = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3)

  # Inception 3b
  conv_1x1 = Conv2D(128, (1, 1), padding='same', activation='relu', bias_initializer=b)(x)
  conv_3x3 = Conv2D(128, (1, 1), padding='same', activation='relu', bias_initializer=b)(x)
  conv_3x3 = Conv2D(192, (3, 3), padding='same', activation='relu', bias_initializer=b)(conv_3x3)
  conv_5x5 = Conv2D(32, (1, 1), padding='same', activation='relu', bias_initializer=b)(x)
  conv_5x5 = Conv2D(96, (5, 5), padding='same', activation='relu', bias_initializer=b)(conv_5x5)
  pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
  pool_proj = Conv2D(64, (1, 1), padding='same', activation='relu', bias_initializer=b)(pool_proj)
  x = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3)
  x = MaxPool2D((3, 3), padding='same', strides=(2, 2))(x)

  # Inception 4a
  conv_1x1 = Conv2D(192, (1, 1), padding='same', activation='relu', bias_initializer=b)(x)
  conv_3x3 = Conv2D(96, (1, 1), padding='same', activation='relu', bias_initializer=b)(x)
  conv_3x3 = Conv2D(208, (3, 3), padding='same', activation='relu', bias_initializer=b)(conv_3x3)
  conv_5x5 = Conv2D(16, (1, 1), padding='same', activation='relu', bias_initializer=b)(x)
  conv_5x5 = Conv2D(48, (5, 5), padding='same', activation='relu', bias_initializer=b)(conv_5x5)
  pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
  pool_proj = Conv2D(64, (1, 1), padding='same', activation='relu', bias_initializer=b)(pool_proj)
  x = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3)

  # Auxiliary output 1
  x1 = AveragePooling2D((5, 5), strides=3)(x)
  x1 = Conv2D(128, (1, 1), padding='same', activation='relu')(x1)
  x1 = Flatten()(x1)
  x1 = Dense(1024, activation='relu')(x1)
  x1 = Dropout(0.7)(x1)
  x1 = Dense(5, activation='softmax')(x1)

  # Inception 4b
  conv_1x1 = Conv2D(160, (1, 1), padding='same', activation='relu', bias_initializer=b)(x)
  conv_3x3 = Conv2D(112, (1, 1), padding='same', activation='relu', bias_initializer=b)(x)
  conv_3x3 = Conv2D(224, (3, 3), padding='same', activation='relu', bias_initializer=b)(conv_3x3)
  conv_5x5 = Conv2D(24, (1, 1), padding='same', activation='relu', bias_initializer=b)(x)
  conv_5x5 = Conv2D(64, (5, 5), padding='same', activation='relu', bias_initializer=b)(conv_5x5)
  pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
  pool_proj = Conv2D(64, (1, 1), padding='same', activation='relu', bias_initializer=b)(pool_proj)
  x = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3)

  # Inception 4c
  conv_1x1 = Conv2D(128, (1, 1), padding='same', activation='relu', bias_initializer=b)(x)
  conv_3x3 = Conv2D(128, (1, 1), padding='same', activation='relu', bias_initializer=b)(x)
  conv_3x3 = Conv2D(256, (3, 3), padding='same', activation='relu', bias_initializer=b)(conv_3x3)
  conv_5x5 = Conv2D(24, (1, 1), padding='same', activation='relu', bias_initializer=b)(x)
  conv_5x5 = Conv2D(64, (5, 5), padding='same', activation='relu', bias_initializer=b)(conv_5x5)
  pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
  pool_proj = Conv2D(64, (1, 1), padding='same', activation='relu', bias_initializer=b)(pool_proj)
  x = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3)

  # Inception 4d
  conv_1x1 = Conv2D(112, (1, 1), padding='same', activation='relu', bias_initializer=b)(x)
  conv_3x3 = Conv2D(144, (1, 1), padding='same', activation='relu', bias_initializer=b)(x)
  conv_3x3 = Conv2D(288, (3, 3), padding='same', activation='relu', bias_initializer=b)(conv_3x3)
  conv_5x5 = Conv2D(32, (1, 1), padding='same', activation='relu', bias_initializer=b)(x)
  conv_5x5 = Conv2D(64, (5, 5), padding='same', activation='relu', bias_initializer=b)(conv_5x5)
  pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
  pool_proj = Conv2D(64, (1, 1), padding='same', activation='relu', bias_initializer=b)(pool_proj)
  x = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3)

  # Ausiliary output 2
  x2 = AveragePooling2D((5, 5), strides=3)(x)
  x2 = Conv2D(128, (1, 1), padding='same', activation='relu')(x2)
  x2 = Flatten()(x2)
  x2 = Dense(1024, activation='relu')(x2)
  x2 = Dropout(0.7)(x2)
  x2 = Dense(5, activation='softmax')(x2)

  # Inception 4e
  conv_1x1 = Conv2D(256, (1, 1), padding='same', activation='relu', bias_initializer=b)(x)
  conv_3x3 = Conv2D(160, (1, 1), padding='same', activation='relu', bias_initializer=b)(x)
  conv_3x3 = Conv2D(320, (3, 3), padding='same', activation='relu', bias_initializer=b)(conv_3x3)
  conv_5x5 = Conv2D(32, (1, 1), padding='same', activation='relu', bias_initializer=b)(x)
  conv_5x5 = Conv2D(128, (5, 5), padding='same', activation='relu', bias_initializer=b)(conv_5x5)
  pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
  pool_proj = Conv2D(128, (1, 1), padding='same', activation='relu', bias_initializer=b)(pool_proj)
  x = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3)
  x = MaxPool2D((3, 3), padding='same', strides=(2, 2))(x)

  # Inception 5a
  conv_1x1 = Conv2D(256, (1, 1), padding='same', activation='relu', bias_initializer=b)(x)
  conv_3x3 = Conv2D(160, (1, 1), padding='same', activation='relu', bias_initializer=b)(x)
  conv_3x3 = Conv2D(320, (3, 3), padding='same', activation='relu', bias_initializer=b)(conv_3x3)
  conv_5x5 = Conv2D(32, (1, 1), padding='same', activation='relu', bias_initializer=b)(x)
  conv_5x5 = Conv2D(128, (5, 5), padding='same', activation='relu', bias_initializer=b)(conv_5x5)
  pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
  pool_proj = Conv2D(128, (1, 1), padding='same', activation='relu', bias_initializer=b)(pool_proj)
  x = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3)

  # Inception 5e
  conv_1x1 = Conv2D(384, (1, 1), padding='same', activation='relu', bias_initializer=b)(x)
  conv_3x3 = Conv2D(192, (1, 1), padding='same', activation='relu', bias_initializer=b)(x)
  conv_3x3 = Conv2D(384, (3, 3), padding='same', activation='relu', bias_initializer=b)(conv_3x3)
  conv_5x5 = Conv2D(48, (1, 1), padding='same', activation='relu', bias_initializer=b)(x)
  conv_5x5 = Conv2D(128, (5, 5), padding='same', activation='relu', bias_initializer=b)(conv_5x5)
  pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
  pool_proj = Conv2D(128, (1, 1), padding='same', activation='relu', bias_initializer=b)(pool_proj)
  x = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3)

  x = GlobalAveragePooling2D()(x)
  x = Dropout(0.4)(x)
  x = Dense(5, activation='softmax', name='output')(x)

  return Model(input_layer, [x, x1, x2])