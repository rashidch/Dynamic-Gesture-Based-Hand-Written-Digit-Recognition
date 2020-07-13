#Python 3.7 is required 
import sys
assert sys.version>="3.7"

#Tensorflow >=2.0 is required 
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense,Conv2D, Flatten, MaxPooling2D, Dropout
assert tf.__version__>="2.0"

#chek matplotlib version
import matplotlib.pyplot as plt
import matplotlib as mpl
assert mpl.__version__>"3.2"

import os
import time
import math
import h5py
import numpy as np
import seaborn as sns
from datetime import timedelta

plt.rcParams['figure.figsize'] = (16.0, 4.0) # Set default figure size

class classifier:

    def __init__(self):

        self.model = 0

    def create_classifier(self):
        tf.keras.backend.clear_session()
        tf.random.set_seed(42)
        np.random.seed(42)

        self.model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
                tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
                tf.keras.layers.MaxPool2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(10, activation="softmax")])
        return self.model