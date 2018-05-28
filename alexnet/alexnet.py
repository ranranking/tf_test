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
        # First conv layer
        self.conv1 = self.conv_layer(self.X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        self.norm1 = self.lrn(self.conv1, 2, 1e-05, 0.75, name='norm1') 
        self.pool1 = self.max_pool(self.norm1, 3, 3, 2, 2, padding='VALID', name='pool1')

        # Second conv layer
        self.conv2 = self.conv_layer(self.pool1, 5, 5, 256, 1, 1, padding='SAME', name='conv2')
        self.norm2 = self.lrn(self.conv2, 2, 1e-05, 0.75, name='norm2') 
        self.pool2 = self.max_pool(self.norm2, 3, 3, 2, 2, padding='VALID', name='pool2')

        # Third conv layer
        self.conv3 = self.conv_layer(self.pool2, 3, 3, 384, 1, 1, padding='SAME', name='conv3')

        # Fourth conv layer
        self.conv4 = self.conv_layer(self.conv3, 3, 3, 384, 1, 1, padding='SAME', name='conv4')

        # Fifth conv layer
        self.conv5 = self.conv_layer(self.conv4, 3, 3, 256, 1, 1, padding='SAME', name='conv5')
        self.pool5 = self.max_pool(self.conv5, 3, 3, 2, 2, padding='VALID', name='pool5')
       
        # Sixth fc layer, flatten first
        self.flatten = tf.reshape(self.pool5, [-1, 256*6*6])        
        self.fc6 = self.fc_layer(self.flatten, 256*6*6, 4096, relu=True, name='fc6')
        
        if self.IS_TRAINING:
            self.dropout6 = self.dropout(self.fc6, self.KEEP_RATE)
        else:
            self.dropout6 = self.fc6

        # Seventh fc layer 
        self.fc7 = self.fc_layer(self.dropout6, 4096, 4096, relu=True, name='fc7')
            
        if self.IS_TRAINING:
            self.dropout7 = self.dropout(self.fc7, self.KEEP_RATE)
        else:
            self.dropout7 = sefl.fc7

        # Eighth fc layer
        self.fc8 = self.fc_layer(self.dropout7, 4096, self.NUM_CLASS, relu=FALSE, name='fc8')

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

    def fc_layer(self, x, num_input, num_output, name):

        with tf.variable.scope(name):

            # Set up tf variables for weights and biases 
            weights = tf.get_variable(name='weights',
                                      shape=[num_input, num_output])
                
            biases = tf.get_variable(name='biases', shape=[num_output])

            # Matix multiplication and add biases
            bias = tf.nn.xw_plus_b(x=x, weights=weights, biases=biases)

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

    def lrn(self, input, depth_radius, alpha, beta, name, bias=1.0):

        return tf.nn.local_response_normalization(input=input,
                                                  depth_radius=depth_radius,
                                                  bias=bias,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  name=name) 
        
    def dropout(self, x, keep_rate):

        return tf.nn.dropout(x=x, keep_prob=keep_rate) 
