import numpy as npimport tensorflow as tf

class My_AlexNet():

    def __init__(self, x, 
                 keep_rate, 
                 num_class,
                 reinit_layer,
                 weight_path,
                 is_training):

        """
        Inputs:
        - x: Input images, tf.placeholder
        - keep_rate: Probability to keep data in dropout, tf.placeholder
        - num_class: Number of new class, int
        - reinit_layer: Names of the layers to be reinitialized, list of strings
        - weight_path: Path to the pretrained weights, string
        - is_training: If retraining the model, boolean 
        """

        self.X = x
        self.KEEP_RATE = keep_rate
        self.NUM_CLASS = num_class
        self.REINIT_LAYER = reinit_layer
        self.WEIGHT_PATH = weight_path
        self.IS_TRAINING = is_training

    def build(self):
        pass

    def load_initial_weights(self):
        pass
   
    def conv_layer(self, x,
                   filter_height,
                   filter_width,
                   num_filters, 
                   stride_y,
                   stride_x,
                   name,
                   padding='SAME'):

        # Get input channels
        input_channels = int(x.get_shape()[-1])

        # Since the weights do not have grouping, grouping function is not
        # included.
        with tf.variable.scope(name):

            # Set up tf variables for weights and biases 
            weights = tf.get_variable(name='weights',
                                      shape=[filter_height,
                                             filter_width, 
                                             input_channels, 
                                             num_filters])
                
            biases = tf.get_variable(name='biases', shape=[num_filters])

            # Convolution
            conv = tf.nn.conv2d(input=x,
                                filter=weights, 
                                stride=[1, stride_y, stride_x, 1], 
                                padding=padding)

            # Add biases
            bias = tf.nn.bias_add(value=conv, bias=biases)

            # Relu
            relu = tf.nn.relu(features=bias)

        return relu

    def fc_layer(self, x, num_input, num_output, name, relu=True):

        with tf.variable.scope(name):

            # Set up tf variables for weights and biases 
            weights = tf.get_variable(name='weights',
                                      shape=[num_input, num_output])
                
            biases = tf.get_variable(name='biases', shape=[num_output])

            # Matix multiplication and add biases
            bias = tf.nn.xw_plus_b(x=x, weights=weights, biases=biases)

            if relu == True:
                relu = tf.nn.relu(features=bias)
                return relu
            else:
                return bias

    def max_pool(self, x, 
                 filter_height, 
                 filter_width, 
                 stride_y,
                 stride_x, 
                 name, 
                 padding='SAME'):

        return tf.nn.max_pool(value=x,
                              ksize=[1, filter_height, filter_width, 1],
                              stride=[1, stride_y, stride_x, 1],
                              padding=padding,
                              name=name)

    def lrn(self, input, depth_radius, bias, alpha, beta, name):

        return tf.nn.local_response_normalization(input=input,
                                                  depth_radius=depth_radius,
                                                  bias=bias,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  name=name) 
        
    def dropout(self, x, keep_rate):
        return tf.nn.dropout(x=x, keep_prob=keep_rate) 
