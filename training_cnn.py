"""
Convolutional Neural Network
"""

#Pytorch for deep learning created by Facebook & TensorFlow created by Google

#Importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator #Preprocessing images


"""
Data Preprocessing
"""

#Preprocessing the Training set

# 1) Apply from transformations to training set alone to avoid overfitting: Rotate, horizontal flip, zoom in or zoom out etc all called image augmentation so that your CNN doesn't overlearn and get a perfect image


#Check Keras API Data Preprocessing ImageDataGenerator  documentation online
#rescale is used to apply feature scaling by dividing the pixel values by 255 to put the values between 0 and 1

#Preprocessing Training set

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True) #To avoid overfitting

#Apply ImageDataGenerator to training dataset
training_set = train_datagen.flow_from_directory(directory = "C:/Users/SOK Consulting/Documents/Udemy Deep Learning Course Kiril/Class Coding/CNN_dataset/training_set",
                                                    target_size = (64,64),
                                                    batch_size = 32,
                                                    class_mode = "binary")

#class_mode can be = categorical


#Preprocessing the test set

test_datagen = ImageDataGenerator(rescale=1./255) #Don't apply transformations just rescale them

test_set = test_datagen.flow_from_directory(directory = "C:/Users/SOK Consulting/Documents/Udemy Deep Learning Course Kiril/Class Coding/CNN_dataset/test_set",
                                                    target_size = (64,64),
                                                    batch_size = 32,
                                                    class_mode = "binary")


"""
Building the CNN
"""

#Initialising the CNN
cnn = tf.keras.models.Sequential()

#Step 1 - Convolution Layer
#number of filters or feature detectors, kernel size no of rows and columns 3 is 3,3 feauture detectors
#input_shape=(64,64,3) 64,64 is the size we reshaped the training and test images to while 3 means the image is colored and 1 means black and white 
#classic architecture is filter = 32

#Add the first convolutional layer 
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=[64,64,3]))


#Step 2 - Apply Pooling (This time is Max pooling)

cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2 , strides=2 )) #reccommended values


#Add a second convolutional layer

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu")) #No need for input_shape for second covolutional layer because you only need that when you are passing your input images to your first conv layer
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2 , strides=2 ))


#Step 3 - Flattening: To convert the output of pooling to 1 dimensional array which would be passed to the ANN for training

cnn.add(tf.keras.layers.Flatten())



#Step 4 - Full Connection

#Input layer and hidden layer
cnn.add(tf.keras.layers.Dense(units=128, activation = "relu")) #use 128 neurons because we are working with images


#Step 5 - Output Layer

cnn.add(tf.keras.layers.Dense(units=1, activation = "sigmoid"))



"""
Training the CNN
"""

#Compiling the CNN
 

cnn.compile(optimizer = "adam", loss= "binary_crossentropy" , metrics = ["accuracy"] ) 
#Stochastic gradient descent optimizer:adam is the best (per iteration weight is updated)
#Binary classification must always have loss = "binary_crossentropy", non binary classification loss = "categorical_crossentropy"
#loss in regression is better with mean_squared_error not cross entropy

#Training the CNN on the Training set and evaluating it on the test set

cnn.fit(x = training_set, validation_data = test_set , epochs= 25 ) 
 

"""
Making a SIngle Prediction
"""

from tensorflow.keras.preprocessing import image 


#Test Image loaded as PIL
test_image = image.load_img("C:/Users/SOK Consulting/Documents/Udemy Deep Learning Course Kiril/Class Coding/CNN_dataset/single_prediction/cat_or_dog_5.JFIF", target_size = (64,64))

#Format the PIL image into an array that can be passed to the neural network for prediction

test_image = image.img_to_array(test_image)

#Convert the test image to a batch_size of 32 stated during the training of the CNN

test_image = np.expand_dims(test_image, axis = 0) #The batch of image will be in the first dimension. The dimension of the batch we are adding to our image will be the first dimension


result = cnn.predict(test_image/255.0)

#Convert result 0 and 1 to Dog or Cat

training_set.class_indices


#View your result as dog or cat instead of the probability of 1
if result[0][0] > 0.5:
    prediction = "Dog"

else:
    prediction = "Cat"


print(prediction)
