#Python 3.7 is required 
import sys
#assert sys.version>="3.7"

#Tensorflow >=2.0 is required 
from keras.models import load_model 
import keras
#assert tf.__version__>="2.0"

#chek matplotlib version
import matplotlib.pyplot as plt
import matplotlib as mpl
#assert mpl.__version__>"3.2"

import os
import time
import math
import h5py
import numpy as np
import seaborn as sns
from datetime import timedelta

plt.rcParams['figure.figsize'] = (16.0, 4.0) # Set default figure size


#load test data
h5f = h5py.File('C:/Users/rashi/Desktop/Elevator/svhn-multi-digit/data/SVHN_single_grey_2.h5', 'r')

# Load the training, test and validation set
X_train = h5f['X_train'][:]
y_train = h5f['y_train'][:]
X_test = h5f['X_test'][:]
y_test = h5f['y_test'][:]
X_test2 = h5f['X_test2'][:]
y_test2 = h5f['y_test2'][:]
X_val = h5f['X_val'][:]
y_val = h5f['y_val'][:]
# Close this file
h5f.close()

print('Test set', X_test.shape, y_test.shape)
print('Test set2', X_test2.shape, y_test2.shape)

# We know that SVHN images have 32 pixels in each dimension
img_size = X_train.shape[1]
# Greyscale images only have 1 color channel
num_channels = X_train.shape[-1]
# Number of classes, one class for each of 10 digits
num_classes = y_train.shape[1]
# Calculate the mean on the training data
train_mean = np.mean(X_train, axis=0)
# Calculate the std on the training data
train_std = np.std(X_train, axis=0)

# Subtract it equally from all splits
X_train = (X_train - train_mean) / train_std
X_test = (X_test - train_mean)  / train_std
X_test2 = (X_test2 - train_mean)  / train_std

#keras.backend.clear_session()
#tf.random.set_seed(42)
#np.random.seed(42)

model = load_model('C:/Users/rashi/Desktop/Elevator/svhn-multi-digit/digit_model_tf2.h5')
# Show the model architecture
model.summary()
# Evaluate the model
loss,acc = model.evaluate(X_test2, y_test2, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


