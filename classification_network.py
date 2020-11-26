"""

Classification using the Inception convolutional neural network.

Nicolas Gachancipa
Embry-Riddle Aeronautical University

Inputs:
    load(boolean): True if you want to load a pre-trained model.
    train(boolean): True if you want to train the model.
    train_dir(str): Directory to the training data.
    validation_dir(str): Directory to the validation data.
    input_shape(tuple): Image dimensions (Input images will be reshaped to 
                                          this target size).
    h5_file(str): Directory to the pre-trained h5 file (If the load option is
                                                        selected).
    save_h5_file(str): Directory where to save the h5 file during and after 
                       training the model.

In order for this file to work, you must add training and validation images
to the respective folders. Those were not added to the Github repository due
to size. However, training images can be downloaded from the COCO dataset or 
any other open-source image datatset.

The inception model is imported from the inception_network.py file, saved 
under the same directory as this file. The Inception network in that file was 
originally implemented by FAIZAN SHAIKH, and can be found in the following 
link: https://www.analyticsvidhya.com/blog/2018/10/understanding-inception-network-from-scratch/
The model was modified to meet the requirements of this project.
    
"""
# Imports.
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from inception_network import inception_model
from keras.callbacks import ModelCheckpoint 

# Inputs
load = True
train = False
train_dir = "Dataset/train"
validation_dir = 'Dataset/validation'
input_shape = (224, 224, 3)
h5_file = 'Output/H5_Files/best_weights_inception_5_classes.h5'
save_h5_file = 'Output/H5_Files/best_weights_inception_5_classes.h5'
    
# Data generators for TensorFlow.
training_datagen =  ImageDataGenerator(rescale = 1./255,
                                        rotation_range=60,
                                        width_shift_range=0.3,
                                        height_shift_range=0.3,
                                        shear_range=0.3,
                                        zoom_range=0.3,
                                        horizontal_flip=True,
                                        vertical_flip=True,
                                        fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale = 1./255)

# Flow data from directories.
train_generator = training_datagen.flow_from_directory(train_dir,
                                                	target_size=(input_shape[0], input_shape[0]),
                                                	class_mode='categorical',
                                                    batch_size=10)
validation_generator = validation_datagen.flow_from_directory(
                                                	validation_dir,
                                                	target_size=(input_shape[0], input_shape[0]),
                                                	class_mode='categorical',
                                                    batch_size=10)

# Define model.
model = inception_model(input_shape)
model.summary()

# Load or train.
if load:
    model.load_weights(h5_file)
if train:
    checkpoint = ModelCheckpoint(save_h5_file,  verbose=1, monitor='loss', 
                                 save_best_only=True, mode='auto')
    model.compile(loss ='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
    history = model.fit(train_generator, validation_data = validation_generator, 
                        epochs=50, batch_size = 10, callbacks=[checkpoint])
      
    
