"""
    Author : Rashid Ali
"""

# Import packages and modules from Python Standard library and Third Party libraries!
from __future__ import division
import numpy as np
import os
import cv2
import h5py
from sklearn.utils import shuffle
import tensorflow as tf

print("Tensorflow Version:", tf.__version__)

# Set the images data dimensions used to train the model
img_height, img_width, num_channels = 112, 112, 1

# set Max lenth of digit sequence and total number of labels
len_digitSeq, num_labels = 2, 11


def subtract_mean(dataset):
    """
    Helper function for subtracting mean of every image
    """
    for i in range(dataset.shape[0]):
        dataset[i] -= dataset[i].mean()

    return dataset


# reser the default graph
tf.reset_default_graph()


def init_conv_weights(shape, name):
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())


def init_fc_weights(shape, name):
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())


def init_biases(shape):
    return tf.Variable(tf.constant(0.0, shape=shape))


def conv_layer(
    input_tensor,  # the input or output of previoud layer
    filter_size,  # width and height of each filter
    in_channel,  # number of channels in previous layers or input
    num_filters,  # number of filter (depth of filter)
    layer_name,  # Layer name
    pooling,
):  # 2x2 max_pooling?
    """
    Function for applying convolution to input_tensors
    """

    with tf.name_scope(layer_name):

        # Shape of filter-weights
        shape = [filter_size, filter_size, in_channel, num_filters]

        # initialize weights and biases of conv-layer
        weights = init_conv_weights(shape=shape, name=layer_name + "/weights")
        biases = init_biases([num_filters])

        # Add histogram summary for weights
        tf.summary.histogram(layer_name + "/weights", weights)

        # create tensorflow operation for convolution with stride=1 and zero padding
        activations = tf.nn.conv2d(input_tensor, weights, [1, 1, 1, 1], "SAME") + biases

        # apply rectified linear unit (Relu)
        activations = tf.nn.relu(activations)

        # Do we insert a pooling layer?
        if pooling:
            # create max pooling operation with F=2 and S=2 and zero padding
            activations = tf.nn.max_pool(activations, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

    return activations


def flatten_tensor(input_tensor):
    """
    Helper function for flattening the 4D tensor to 2D
    """
    # get shape of input_tensor
    input_tensor_shape = input_tensor.get_shape()

    # get total number of elements in height, width and channel dimensions(excluding batch dimension)
    num_activations = input_tensor_shape[1:4].num_elements()

    # Reshpe the input tensor into dimensions (batch,num_activations)
    input_tensor_flat = tf.reshape(input_tensor, [-1, num_activations])

    return input_tensor_flat, num_activations


def fc_layer(input_tensor, input_dim, output_dim, layer_name, relu=False):
    """
    Helper function for applying fully-connected operation on 2D tensor
    """
    with tf.name_scope(layer_name):
        # create weights anf biases
        weights = init_fc_weights(shape=[input_dim, output_dim], name=layer_name + "/weights")
        biases = init_biases([output_dim])

        # create histogram of weihgts for tensorboard visualizations
        tf.summary.histogram(layer_name + "/weights", weights)

        # multiply input_tensor and weights to get activations
        activations = tf.matmul(input_tensor, weights) + biases
        # apply Relu unit to activations if relu?
        if relu:
            activations = tf.nn.relu(activations)

    return activations


"""
    Build Tensorflow Model
"""
# Block1
filter_size1 = filter_size2 = 5
num_filters1 = num_filters2 = 32

# Block2
filter_size3 = filter_size4 = 5
num_filters3 = num_filters4 = 64

# Block3
filter_size5 = filter_size6 = filter_size7 = 5
num_filters5 = num_filters6 = num_filters7 = 128

# fully connected layers
fc1_size = fc2_size = 256


# Creating Placeholders for input data and dropout  keep_prob
with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, shape=[None, img_height, img_width, num_channels], name="x")
    y_ = tf.placeholder(tf.int64, shape=[None, len_digitSeq], name="y_")

with tf.name_scope("dropout"):
    # Dropout rate applied to input layer
    p_keep_1 = tf.placeholder(tf.float32)
    tf.summary.scalar("input_keep_probability", p_keep_1)

    # Dropout rate applied after pooling layers
    p_keep_2 = tf.placeholder(tf.float32)
    tf.summary.scalar("conv_keep_probability", p_keep_2)

    # Dropout rate applied between fully connected layers
    p_keep_3 = tf.placeholder(tf.float32)
    tf.summary.scalar("fc_keep_probability", p_keep_3)


# Create model by stacking layers
def model():
    """
    Helper function for defining the model by stacking Conv, Relu and pooling and FC laeyrs
    """
    # apply dropout to the input
    drop_input = tf.nn.dropout(x, p_keep_1)

    # Block1
    conv_1 = conv_layer(
        input_tensor=drop_input,
        filter_size=filter_size1,
        in_channel=num_channels,
        num_filters=num_filters1,
        layer_name="conv_1",
        pooling=False,
    )
    conv_2 = conv_layer(
        input_tensor=conv_1,
        filter_size=filter_size2,
        in_channel=num_filters1,
        num_filters=num_filters2,
        layer_name="conv_2",
        pooling=True,
    )
    # apply dropout after pooling
    drop_block1 = tf.nn.dropout(conv_2, p_keep_2)

    # Block2
    conv_3 = conv_layer(
        input_tensor=conv_2,
        filter_size=filter_size3,
        in_channel=num_filters2,
        num_filters=num_filters3,
        layer_name="conv_3",
        pooling=False,
    )
    conv_4 = conv_layer(
        input_tensor=conv_3,
        filter_size=filter_size4,
        in_channel=num_filters3,
        num_filters=num_filters4,
        layer_name="conv_4",
        pooling=True,
    )
    # apply dropout after pooling
    drop_block2 = tf.nn.dropout(conv_4, p_keep_2)

    # Block3
    conv_5 = conv_layer(
        input_tensor=conv_4,
        filter_size=filter_size5,
        in_channel=num_filters4,
        num_filters=num_filters5,
        layer_name="conv_5",
        pooling=False,
    )
    conv_6 = conv_layer(
        input_tensor=conv_5,
        filter_size=filter_size6,
        in_channel=num_filters5,
        num_filters=num_filters6,
        layer_name="conv_6",
        pooling=False,
    )
    conv_7 = conv_layer(
        input_tensor=conv_6,
        filter_size=filter_size7,
        in_channel=num_filters6,
        num_filters=num_filters7,
        layer_name="conv_7",
        pooling=True,
    )
    print(conv_7.shape)
    # apply dropout after pooling
    drop_block3 = tf.nn.dropout(conv_7, p_keep_3)
    # flatten the tensor
    flat_tensor, num_activations = flatten_tensor(drop_block3)

    # Fully connected layer1
    fc_1 = fc_layer(
        input_tensor=flat_tensor, input_dim=num_activations, output_dim=fc1_size, layer_name="fc_1", relu=True
    )
    drop_fc2 = tf.nn.dropout(fc_1, p_keep_3)

    # Fully connected layer2
    fc_2 = fc_layer(input_tensor=drop_fc2, input_dim=fc1_size, output_dim=fc2_size, layer_name="fc_2", relu=True)

    # parallel softmax layers
    logits_1 = fc_layer(input_tensor=fc_2, input_dim=fc2_size, output_dim=num_labels, layer_name="softmax1")
    logits_2 = fc_layer(input_tensor=fc_2, input_dim=fc2_size, output_dim=num_labels, layer_name="softmax2")

    y_pred = tf.stack([logits_1, logits_2])

    y_pred_cls = tf.transpose(tf.argmax(y_pred, axis=2))

    return logits_1, logits_2, y_pred_cls


# create model
logits_1, logits_2, y_pred_cls = model()

# launch the graph in session
session = tf.Session()

# Creat Saver Object to Save and Restore Tensorflow graph variables
saver = tf.train.Saver()
save_path = os.path.join("C:/Users/rashi/Desktop/Elevator/multi-digit-leap-motion/checkpoints/", "new_model_v1-6000-3")
try:
    print("Restoring Checkpoints .....")

    # restore checkpoint using restore function
    saver.restore(session, save_path)
    print("Restored checkpoint from:", save_path)

except:
    print("Failed to restore checkpoints - initializing variables")
    session.run(tf.global_variables_initializer())


def predict_multi_image(image, label=None):
    image = cv2.resize(np.float32(image), dsize=(112, 112), interpolation=cv2.INTER_AREA)
    image = subtract_mean(image)
    image = np.expand_dims(np.dot(image, [0.2989, 0.5870, 0.1140]), axis=3).astype(np.float32)
    img = np.expand_dims(image, axis=0)
    pred = session.run(y_pred_cls, feed_dict={x: img, p_keep_1: 1.0, p_keep_2: 1.0, p_keep_3: 1.0})
    return pred
