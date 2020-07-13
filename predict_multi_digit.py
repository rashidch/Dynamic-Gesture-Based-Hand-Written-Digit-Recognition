from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import numpy as np
import time
import os
from datetime import timedelta
from sklearn.utils import shuffle
print("Tensorflow version: " + tf.__version__)

def load_data():
    # Open the HDF5 file containing the datasets
    h5f = h5py.File('C:/Users/rashi/Desktop/Elevator/multi-digit-leap-motion/data/mlt_leap_grey.h5','r')
    # Extract the datasets
    X_train = h5f['train_dataset'][:]
    y_train = h5f['train_labels'][:]
    X_test = h5f['test_dataset'][:]
    y_test = h5f['test_labels'][:]
    # Close the file
    h5f.close()
    # Randomly shuffle the training data
    X_train, y_train = shuffle(X_train, y_train)
    #X_test, y_test = shuffle(X_test, y_test)
    
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_data()
# Get the data dimensions
_, img_height, img_width, num_channels = X_train.shape
# ... and label information
num_digits, num_labels = y_train.shape[1], len(np.unique(y_train))

print('Training set', X_train.shape, y_train.shape)
print('Test set', X_test.shape, y_test.shape)

def subtract_mean(a):
    """ Helper function for subtracting the mean of every image
    """
    for i in range(a.shape[0]):
        a[i] -= a[i].mean()
    return a

# Subtract the mean from every image
X_train = subtract_mean(X_train)
X_test = subtract_mean(X_test)

# Helper function for plotting images
# Helper function that will help us plot nrows*ncols images and their true and predicted labels.

def plot_images(images, nrows, ncols, cls_true, cls_pred=None):
    
    # Initialize figure
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2*nrows))
    
    # Randomly select nrows * ncols images
    rs = np.random.choice(images.shape[0], nrows*ncols)
    
    # For every axes object in the grid
    for i, ax in zip(rs, axes.flat): 
        
        # Pretty string with actual number
        true_number = ''.join(str(x) for x in cls_true[i] if x != 10)
        
        if cls_pred is None:
            title = "True: {0}".format(true_number)
        else:
            # Pretty string with predicted number
            pred_number = ''.join(str(x) for x in cls_pred[i] if x != 10)
            title = "True: {0}, Pred: {1}".format(true_number, pred_number) 
            
        ax.imshow(images[i,:,:,0], cmap='binary')
        ax.set_title(title)   
        ax.set_xticks([]); ax.set_yticks([])

# Plot some images from the training set
plot_images(X_train, 2, 8, y_train)
plt.show()

### Helper functions for creating new variables
tf.reset_default_graph() 
def init_conv_weights(shape, name):
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())

def init_fc_weights(shape, name):
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())

def init_biases(shape):
    return tf.Variable(tf.constant(0.0, shape=shape))

### Helper function for stacking CONV-RELU layers followed by an optional POOL layer
def conv_layer(input_tensor,    # The input or previous layer
                filter_size,    # Width and height of each filter
                in_channels,    # Number of channels in previous layer
                num_filters,    # Number of filters
                layer_name,     # Layer name
                pooling):       # Use 2x2 max-pooling?
    
    # Add layer name scopes for better graph visualization
    with tf.name_scope(layer_name):
    
        # Shape of the filter-weights for the convolution
        shape = [filter_size, filter_size, in_channels, num_filters]

        # Create weights and biases
        weights = init_conv_weights(shape, layer_name + '/weights')
        biases = init_biases([num_filters])
        
        # Add histogram summaries for weights
        tf.summary.histogram(layer_name + '/weights', weights)
        
        # Create the TensorFlow operation for convolution, with S=1 and zero padding
        activations = tf.nn.conv2d(input_tensor, weights, [1, 1, 1, 1], 'SAME') + biases

        # Rectified Linear Unit (ReLU)
        activations = tf.nn.relu(activations)

        # Do we insert a pooling layer?
        if pooling:
            # Create a pooling layer with F=2, S=1 and zero padding
            activations = tf.nn.max_pool(activations, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        # Return the resulting layer
        return activations

### Helper function for reshaping the CONV layers to FC layers

def flatten_tensor(input_tensor):
    """ Helper function for transforming a 4D tensor to 2D
    """
    # Get the shape of the input_tensor.
    input_tensor_shape = input_tensor.get_shape()

    # Calculate the volume of the input tensor
    num_activations = input_tensor_shape[1:4].num_elements()
    
    # Reshape the input_tensor to 2D: (?, num_activations)
    input_tensor_flat = tf.reshape(input_tensor, [-1, num_activations])

    # Return the flattened input_tensor and the number of activations
    return input_tensor_flat, num_activations

### Helper function for stacking FC-RELU layers
def fc_layer(input_tensor,  # The previous layer,         
             input_dim,     # Num. inputs from prev. layer
             output_dim,    # Num. outputs
             layer_name,    # The layer name
             relu=False):         # Use ReLU?

    # Add layer name scopes for better graph visualization
    with tf.name_scope(layer_name):
    
        # Create new weights and biases.
        weights = init_fc_weights([input_dim, output_dim], layer_name + '/weights')
        biases = init_biases([output_dim])
        
        # Add histogram summaries for weights
        tf.summary.histogram(layer_name + '/weights', weights)

        # Calculate the layer activation
        activations = tf.matmul(input_tensor, weights) + biases

        # Use ReLU?
        if relu:
            activations = tf.nn.relu(activations)

        return activations

## Tensorflow Model
# The configuration of the Convolutional Neural Network and data dimensions are defined here for convenience, 
# so you can easily find and change these numbers and re-run the Notebook.
# Block 1
filter_size1 = filter_size2 = 5          
num_filters1 = num_filters2 = 32        
# Block 2
filter_size3 = filter_size4 = 5          
num_filters3 = num_filters4 = 64
# Block 3
filter_size5 = filter_size6 = filter_size7 = 5          
num_filters5 = num_filters6 = num_filters7 = 128  
# Fully-connected layers
fc1_size = fc2_size = 256

### Placeholder Variables
# Placeholder variables serve as the input to the graph that we may change each time we execute the graph

with tf.name_scope("input"):
    
    # Placeholders for feeding input images
    x = tf.placeholder(tf.float32, shape=(None, img_height, img_width, num_channels), name='x')
    y_ = tf.placeholder(tf.int64, shape=[None, num_digits], name='y_')

with tf.name_scope("dropout"):
    
    # Dropout rate applied to the input layer
    p_keep_1 = tf.placeholder(tf.float32)
    tf.summary.scalar('input_keep_probability', p_keep_1)

    # Dropout rate applied after the pooling layers
    p_keep_2 = tf.placeholder(tf.float32)
    tf.summary.scalar('conv_keep_probability', p_keep_2)

    # Dropout rate using between the fully-connected layers
    p_keep_3 = tf.placeholder(tf.float32)
    tf.summary.scalar('fc_keep_probability', p_keep_3)

### Model
# We implement the following ConvNet architecture
# INPUT -> [[CONV -> RELU]*2 -> POOL]*2 -> [[CONV -> RELU]*3 -> POOL] -> [FC -> RELU]*2 -> OUTPUT

# Apply dropout to the input layer
drop_input = tf.nn.dropout(x, p_keep_1) 

# Block 1
conv_1 = conv_layer(drop_input, filter_size1, num_channels, num_filters1, "conv_1", pooling=False)
conv_2 = conv_layer(conv_1, filter_size2, num_filters1, num_filters2, "conv_2", pooling=True)
drop_block1 = tf.nn.dropout(conv_2, p_keep_2) # Dropout

# Block 2
conv_3 = conv_layer(conv_2, filter_size3, num_filters2, num_filters3, "conv_3", pooling=False)
conv_4 = conv_layer(conv_3, filter_size4, num_filters3, num_filters4, "conv_4", pooling=True)
drop_block2 = tf.nn.dropout(conv_4, p_keep_2) # Dropout

# Block 3
conv_5 = conv_layer(drop_block2, filter_size5, num_filters4, num_filters5, "conv_5", pooling=False)
conv_6 = conv_layer(conv_5, filter_size6, num_filters5, num_filters6, "conv_6", pooling=False)
conv_7 = conv_layer(conv_6, filter_size7, num_filters6, num_filters7, "conv_7", pooling=True)
flat_tensor, num_activations = flatten_tensor(tf.nn.dropout(conv_7, p_keep_3)) # Dropout

# Fully-connected 1
fc_1 = fc_layer(flat_tensor, num_activations, fc1_size, 'fc_1', relu=True)
drop_fc2 = tf.nn.dropout(fc_1, p_keep_3) # Dropout

# Fully-connected 2
fc_2 = fc_layer(drop_fc2, fc1_size, fc2_size, 'fc_2', relu=True)

# Paralell softmax layers
logits_1 = fc_layer(fc_2, fc2_size, num_labels, 'softmax1')
logits_2 = fc_layer(fc_2, fc2_size, num_labels, 'softmax2')
#logits_3 = fc_layer(fc_2, fc2_size, num_labels, 'softmax3')
#logits_4 = fc_layer(fc_2, fc2_size, num_labels, 'softmax4')
#logits_5 = fc_layer(fc_2, fc2_size, num_labels, 'softmax5')

y_pred = tf.stack([logits_1, logits_2])
print(y_pred.shape)

# The class-number is the index of the largest element
y_pred_cls = tf.transpose(tf.argmax(y_pred, axis=2))

### Loss Function
# We calculate the loss by taking the average loss of every individual example for each of our 2 digits 
# and adding them together. Using tf.nn.sparse_softmax_cross_entropy_with_logits allows us to skip 
# using OneHotEncoding on our label values.

with tf.name_scope('loss'):
    
    # Calculate the loss for each individual digit in the sequence
    loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_[:, 0], logits=logits_1))
    loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_2, labels=y_[:, 1]))
    #loss3 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_3,labels= y_[:, 2]))
    #loss4 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_4, labels=y_[:, 3]))
    #loss5 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_5, labels=y_[:, 4]))

    # Calculate the total loss for all predictions
    #loss = loss1 + loss2 + loss3 + loss4 + loss5
    loss = loss1 + loss2
    tf.summary.scalar('loss', loss)
# Launch the graph in a session
session = tf.Session()
#load checkpoints for multi-digit model
saver = tf.train.Saver()
save_path = os.path.join('C:/Users/rashi/Desktop/Elevator/multi-digit-leap-motion/checkpoints/', 'mlt_leap_v1-6000')
try:
    print("Restoring checkpoint ...")
    # Try and load the data in the checkpoint.
    saver.restore(session, save_path=save_path)
    print("Restored checkpoint from:", save_path.split('/')[-1])
    
# If the above failed - initialize all the variables
except:
    print("Failed to restore checkpoint - initializing variables")

# Feed the test set with dropout disabled
feed_dict={
    x: X_test,
    y_: y_test,
    p_keep_1: 1.,
    p_keep_2: 1.,
    p_keep_3: 1.
}

# Generate predictions for the testset
test_pred = session.run(y_pred_cls, feed_dict=feed_dict)

# Display the predictions
print('Prediction on test set:')
print(test_pred)
print()
idx=700
plt.imshow(np.squeeze(X_test[idx,:,:,0]), cmap='binary')
plt.xticks([]), plt.yticks([])
plt.show()

img = np.expand_dims(X_test[idx], axis=0)
img_yt = np.expand_dims(y_test[idx], axis=0)
print('Original Label: {}'.format(img_yt))
pred = session.run(y_pred_cls, feed_dict={x:img , y_:img_yt , p_keep_1: 1., p_keep_2: 1., p_keep_3: 1.})
print('Predicted Label: {}'.format(img_yt))