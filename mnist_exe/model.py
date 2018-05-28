import tensorflow as tf
import numpy as np

def conv_layer(
    x,
    input_channels, 
    filter_height, 
    filter_width, 
    num_filters, 
    stride_y, 
    stride_x, 
    padding, 
    name):

    with tf.variable.scope(name):

        weights = tf.get_variable(
            name='weights', 
            shape=[filter_height, filter_width, input_channels, num_filters])

        biases = tf.get_variable(
            name='biases', 
            shape=[num_filters])
        
        conv = tf.nn.conv2d(
            input=x,
            filter=weights,
            strides=[1, stride_y, stride_x, 1], 
            padding=padding)

        bias = tf.nn.bias_add(
            value=conv, 
            bias=biases)

        relu = tf.nn.relu(
            features=bias)        

        return relu

def fc_layer(
    x,
    num_input,
    num_output, 
    name,
    is_relu):
    
    with tf.variable.scope(name):

        weights = tf.get_variable(
            name='weights', 
            shape=[num_input, num_output])

        biases = tf.get_variable(
            name='biases', 
            shape=[num_output])

        bias = tf.nn.wx_plus_b(
            x=x,
            weights=weights, 
            biases=biases)

        if is_relu:
            return tf.nn.relu(
                features=bias)
        else:
            return bias

def max_pool(
    x,
    filter_height,
    filter_width, 
    stride_y, 
    stride_x, 
    padding, 
    name):

    return tf.nn.max_pool(
        value=x, 
        ksize=[1, filter_height, filter_width, 1], 
        strides=[1, stride_y, stride_x, 1], 
        padding=padding, 
        name=name)

def dropout(
    x,
    keep_rate,
    train_mode):
    
    return tf.cond(
        pred=train_mode,
        true_fn=lambda: tf.dropout(
            x=x,
            keep_prob=keep_rate),
        false_fn=lambda: x)

