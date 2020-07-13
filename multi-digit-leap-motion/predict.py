import os
import time
import math
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import cv2 


plt.rcParams['figure.figsize'] = (12.0, 4.0) # Set default figure size

print("Tensorflow version", tf.__version__)

# Open the file as readonly
h5f = h5py.File('data/SVHN_single_grey_9.h5', 'r')

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


def conv_weight_variable(layer_name, shape):
    """ Retrieve an existing variable with the given layer name 
    """
    return tf.get_variable(layer_name, shape=shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())

def fc_weight_variable(layer_name, shape):
    """ Retrieve an existing variable with the given layer name
    """
    return tf.get_variable(layer_name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def bias_variable(shape):
    """ Creates a new bias variable
    """
    return tf.Variable(tf.constant(0.0, shape=shape))

def conv_layer(input,               # The previous layer
                layer_name,         # Layer name
                num_input_channels, # Num. channels in prev. layer
                filter_size,        # Width and height of each filter
                num_filters,        # Number of filters
                pooling=True):      # Use 2x2 max-pooling

    # Shape of the filter-weights for the convolution
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new filters with the given shape
    weights = conv_weight_variable(layer_name, shape=shape)
    
    # Create new biases, one for each filter
    biases = bias_variable(shape=[num_filters])

    # Create the TensorFlow operation for convolution
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME') # with zero padding

    # Add the biases to the results of the convolution
    layer += biases
    
    # Rectified Linear Unit (RELU)
    layer = tf.nn.relu(layer)

    # Down-sample the image resolution?
    if pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Return the resulting layer and the filter-weights
    return layer, weights

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The number of features is: img_height * img_width * num_channels
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    layer_flat = tf.reshape(layer, [-1, num_features])

    # Return the flattened layer and the number of features.
    return layer_flat, num_features


def fc_layer(input,        # The previous layer
             layer_name,   # The layer name
             num_inputs,   # Num. inputs from prev. layer
             num_outputs,  # Num. outputs
             relu=True):   # Use RELU?

    # Create new weights and biases.
    weights = fc_weight_variable(layer_name, shape=[num_inputs, num_outputs])
    biases = bias_variable(shape=[num_outputs])

    # Calculate the layer activation
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if relu:
        layer = tf.nn.relu(layer)

    return layer

def plot_images(images, nrows, ncols, cls_true, cls_pred=None):
    """ Plot nrows * ncols images from images and annotate the images
    """
    # Initialize the subplotgrid
    fig, axes = plt.subplots(nrows, ncols)
    
    # Randomly select nrows * ncols images
    rs = np.random.choice(images.shape[0], nrows*ncols)
    
    # For every axes object in the grid
    for i, ax in zip(rs, axes.flat): 
        
        # Predictions are not passed
        if cls_pred is None:
            title = "True: {0}".format(np.argmax(cls_true[i]))
        
        # When predictions are passed, display labels + predictions
        else:
            title = "True: {0}, Pred: {1}".format(np.argmax(cls_true[i]), cls_pred[i])  
            
        # Display the image
        ax.imshow(images[i,:,:,0], cmap='binary')
        
        # Annotate the image
        ax.set_title(title)
        
        # Do not overlay a grid
        ax.set_xticks([])
        ax.set_yticks([])
# Plot 2 rows with 9 images each from the test2 set
plot_images(X_test2, 2, 6, y_test2);
plt.show()

def create_model():
    # Convolutional Layer 1.
    filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
    num_filters1 = 32         # There are 16 of these filters.

    # Convolutional Layer 2.
    filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
    num_filters2 = 64         # There are 36 of these filters.

    # Fully-connected layer.
    fc_size = 256            # Number of neurons in fully-connected layer.

    x = tf.placeholder(tf.float32, shape=(None, img_size, img_size, num_channels), name='x')

    y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
    y_true_cls = tf.argmax(y_true, axis=1)

    keep_prob = tf.placeholder(tf.float32)

    conv_1, w_c1 = conv_layer(input=x,
                          layer_name="conv_1",
                          num_input_channels=num_channels,
                          filter_size=filter_size1,
                          num_filters=num_filters1, pooling=True)
    
    conv_2, w_c2 = conv_layer(input=conv_1,
                          layer_name="conv_2",
                          num_input_channels=num_filters1,
                          filter_size=filter_size2,
                          num_filters=num_filters2,
                          pooling=True)

    # Apply dropout after the pooling operation
    dropout = tf.nn.dropout(conv_2, keep_prob)

    layer_flat, num_features = flatten_layer(dropout)

    fc_1 = fc_layer(input=layer_flat,
                layer_name="fc_1",
                num_inputs=num_features,
                num_outputs=fc_size,
                relu=True)
    fc_2 = fc_layer(input=fc_1,
                layer_name="fc_2",
                num_inputs=fc_size,
                num_outputs=num_classes,
                relu=False)
    
    y_pred = tf.nn.softmax(fc_2)

    # The class-number is the index of the largest element.
    y_pred_cls = tf.argmax(y_pred, dimension=1)

    return x, y_pred, y_pred_cls, y_true, y_true_cls, keep_prob 

def predict():

    # create model for inference
    x, y_pred, y_pred_cls, y_true, y_true_cls, keep_prob  = create_model()

    # Predicted class equals the true class of each image?
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)

    # Cast predictions to float and calculate the mean
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Start-time used for printing time-usage below.
    start_time = time.time()

    session = tf.Session()
    session.run(tf.initialize_all_variables())

    saver = tf.train.Saver()
    save_dir = 'checkpoints/'
    # Create directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'svhn_single_greyscale_2')

    saver.restore(sess=session, save_path=save_path)


    # Difference between start and end-times.
    time_diff = time.time() - start_time

    # Calculate the accuracy on the test-set
    test_accuracy = session.run(accuracy, {x: X_test, y_true: y_test, keep_prob: 1.0})

    print("Test accuracy: %.4f" % test_accuracy)
    print("Time usage: " + str(timedelta(seconds=int(round(time_diff)))))

    # Generate predictions for the testset
    test_pred = session.run(y_pred_cls, {x: X_test2, y_true: y_test2, keep_prob: 1.0})

    # Find the incorrectly classified examples
    incorrect = test_pred != np.argmax(y_test2, axis=1)
    # Select the incorrectly classified examples
    images = X_test2[incorrect]
    cls_true = y_test2[incorrect]
    cls_pred = test_pred[incorrect]
    # Plot the mis-classified examples
    plot_images(images, 2, 5, cls_true, cls_pred)
    plt.show()

    # Find the incorrectly classified examples
    correct = np.invert(incorrect)

    # Select the correctly classified examples
    images = X_test2[correct]
    cls_true = y_test2[correct]
    cls_pred = test_pred[correct]

    # Plot the mis-classified examples
    plot_images(images, 2, 5, cls_true, cls_pred)
    plt.show()

def rgb2gray(image):
    """Convert images from rbg to grayscale
    """
    return np.expand_dims(np.dot(image, [0.2989, 0.5870, 0.1140]), axis=3)

def process_image(image, label=None):

    image = cv2.resize(image, dsize=(32,32),interpolation = cv2.INTER_AREA)
    image = rgb2gray(image).astype(np.float32)
    # Subtract it equally from image
    image = (image - train_mean) / train_std
    image = np.expand_dims(image, axis=0)
    if label is not None:
        label = np.expand_dims(y_test2[0], axis=0)

    return image

def predict_single_image(image, label=None):

    # create model for inference
    x, y_pred, y_pred_cls, y_true, y_true_cls, keep_prob  = create_model()

    # Predicted class equals the true class of each image?
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)

    # Cast predictions to float and calculate the mean
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Start-time used for printing time-usage below.
    start_time = time.time()

    session = tf.Session()
    session.run(tf.initialize_all_variables())

    saver = tf.train.Saver()
    save_dir = 'checkpoints/'
    # Create directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'svhn_single_greyscale_9')
    saver.restore(sess=session, save_path=save_path)

    # Calculate the accuracy on the test-set
    prediction = session.run(y_pred_cls, {x: image, keep_prob: 1.0})

    # Difference between start and end-times.
    time_diff = time.time() - start_time

    print("Test accuracy: %.4f" % prediction)
    print("Time usage: " + str(timedelta(seconds=int(round(time_diff)))))

    return prediction 

if __name__ == "__main__":
    predict()