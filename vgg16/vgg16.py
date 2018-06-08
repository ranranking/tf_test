import tensorflow as tf
import numpy as np

class MY_VGG16:

    def __init__ (
            self, 
            x, 
            keep_rate,
            num_classes,
            mean_image,
            skip_layers=[],
            weights_path=None,
            retrain=True):

        self.X = x
        self.KEEP_RATE = keep_rate
        self.NUM_CLASSES = num_classes
        self.MEAN_IMAGE = mean_image
        self.SKIP_LAYERS = skip_layers
        self.WEIGHTS_PATH = weights_path
        self.RETRAIN = retrain

    def conv_layer(
            self, 
            x,
            input_channels, 
            filter_height, 
            filter_width, 
            num_filters, 
            stride_y, 
            stride_x, 
            padding,
            name):
    
        with tf.variable_scope(name):
    
            weights = tf.get_variable(
                name='weights', 
                shape=[filter_height, filter_width,
                       input_channels, num_filters])
    
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
    
            # Summaries
            tf.summary.histogram("weights", weights)
            tf.summary.histogram("biases", biases)
            tf.summary.histogram("activations", relu)
    
            return relu

    def fc_layer(
            self,
            x,
            num_input,
            num_output, 
            name,
            is_relu):
        
        with tf.variable_scope(name):
    
            weights = tf.get_variable(
                name='weights', 
                shape=[num_input, num_output])
    
            biases = tf.get_variable(
                name='biases', 
                shape=[num_output])
    
            bias = tf.nn.xw_plus_b(
                x=x,
                weights=weights, 
                biases=biases)
    
            # Summaries
            tf.summary.histogram("weights", weights)
            tf.summary.histogram("biases", biases)
    
            if is_relu:
                relu = tf.nn.relu(
                    features=bias)
                tf.summary.histogram("relu", relu)
                return relu
            else:
                return bias

    def max_pool(
            self,
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
            self,
            x,
            keep_rate,
            name):
        
        return tf.nn.dropout(
            x=x,
            keep_prob=keep_rate,
            name=name)

    def build(
            self):
        
        # Convert to BGR <-- Because the weights were trained from opencv images
        self.bgr = tf.reverse(
            tensor=self.X, 
            axis=[-1])
                
        # Subtract from mean
        self.bgr_sub = self.bgr - self.MEAN_IMAGE

        # First stack
        self.conv1_1 = self.conv_layer(
            x=self.bgr_sub,
            input_channels=3, 
            filter_height=3,
            filter_width=3, 
            num_filters=64, 
            stride_y=1, 
            stride_x=1, 
            padding='SAME', 
            name='conv1_1')
        self.conv1_2 = self.conv_layer(
            x=self.conv1_1,
            input_channels=64, 
            filter_height=3,
            filter_width=3, 
            num_filters=64, 
            stride_y=1, 
            stride_x=1, 
            padding='SAME', 
            name='conv1_2')
        self.pool1 = self.max_pool(
            x=self.conv1_2,
            filter_height=2,
            filter_width=2, 
            stride_y=2, 
            stride_x=2, 
            padding='SAME', 
            name='pool1')

        # Second stack
        self.conv2_1 = self.conv_layer(
            x=self.pool1,
            input_channels=64, 
            filter_height=3,
            filter_width=3, 
            num_filters=128, 
            stride_y=1, 
            stride_x=1, 
            padding='SAME', 
            name='conv2_1')
        self.conv2_2 = self.conv_layer(
            x=self.conv2_1,
            input_channels=128, 
            filter_height=3,
            filter_width=3, 
            num_filters=128, 
            stride_y=1, 
            stride_x=1, 
            padding='SAME', 
            name='conv2_2')
        self.pool2 = self.max_pool(
            x=self.conv2_2,
            filter_height=2,
            filter_width=2, 
            stride_y=2, 
            stride_x=2, 
            padding='SAME', 
            name='pool2')

        # Third stack
        self.conv3_1 = self.conv_layer(
            x=self.pool2,
            input_channels=128, 
            filter_height=3,
            filter_width=3, 
            num_filters=256, 
            stride_y=1, 
            stride_x=1, 
            padding='SAME', 
            name='conv3_1')
        self.conv3_2 = self.conv_layer(
            x=self.conv3_1,
            input_channels=256, 
            filter_height=3,
            filter_width=3, 
            num_filters=256, 
            stride_y=1, 
            stride_x=1, 
            padding='SAME', 
            name='conv3_2')
        self.conv3_3 = self.conv_layer(
            x=self.conv3_2,
            input_channels=256, 
            filter_height=3,
            filter_width=3, 
            num_filters=256, 
            stride_y=1, 
            stride_x=1, 
            padding='SAME', 
            name='conv3_3')
        self.pool3 = self.max_pool(
            x=self.conv3_3,
            filter_height=2,
            filter_width=2, 
            stride_y=2, 
            stride_x=2, 
            padding='SAME', 
            name='pool3')

        # Fourth stack
        self.conv4_1 = self.conv_layer(
            x=self.pool3,
            input_channels=256, 
            filter_height=3,
            filter_width=3, 
            num_filters=512, 
            stride_y=1, 
            stride_x=1, 
            padding='SAME', 
            name='conv4_1')
        self.conv4_2 = self.conv_layer(
            x=self.conv4_1,
            input_channels=512, 
            filter_height=3,
            filter_width=3, 
            num_filters=512, 
            stride_y=1, 
            stride_x=1, 
            padding='SAME', 
            name='conv4_2')
        self.conv4_3 = self.conv_layer(
            x=self.conv4_2,
            input_channels=512, 
            filter_height=3,
            filter_width=3, 
            num_filters=512, 
            stride_y=1, 
            stride_x=1, 
            padding='SAME', 
            name='conv4_3')
        self.pool4 = self.max_pool(
            x=self.conv4_3,
            filter_height=2,
            filter_width=2, 
            stride_y=2, 
            stride_x=2, 
            padding='SAME', 
            name='pool4')

        # Fifth stack
        self.conv5_1 = self.conv_layer(
            x=self.pool4,
            input_channels=512, 
            filter_height=3,
            filter_width=3, 
            num_filters=512, 
            stride_y=1, 
            stride_x=1, 
            padding='SAME', 
            name='conv5_1')
        self.conv5_2 = self.conv_layer(
            x=self.conv5_1,
            input_channels=512, 
            filter_height=3,
            filter_width=3, 
            num_filters=512, 
            stride_y=1, 
            stride_x=1, 
            padding='SAME', 
            name='conv5_2')
        self.conv5_3 = self.conv_layer(
            x=self.conv5_2,
            input_channels=512, 
            filter_height=3,
            filter_width=3, 
            num_filters=512, 
            stride_y=1, 
            stride_x=1, 
            padding='SAME', 
            name='conv5_3')
        self.pool5 = self.max_pool(
            x=self.conv5_3,
            filter_height=2,
            filter_width=2, 
            stride_y=2, 
            stride_x=2, 
            padding='SAME', 
            name='pool5')

        # FC layer 6
        self.flat6 = tf.reshape(
            tensor=self.pool5,
            shape=[-1, 7*7*512],
            name='flatten6')
        self.fc6 = self.fc_layer(
            x=self.flat6,
            num_input=7*7*512,
            num_output=4096, 
            name='fc6',
            is_relu=True)
        self.drop6 = self.dropout(
            x=self.fc6,
            keep_rate=self.KEEP_RATE,
            name='drop6')

        # FC layer 7
        self.fc7 = self.fc_layer(
            x=self.drop6,
            num_input=4096,
            num_output=4096, 
            name='fc7',
            is_relu=True)
        self.drop7 = self.dropout(
            x=self.fc7,
            keep_rate=self.KEEP_RATE,
            name='drop7')

        # FC layer 8, logits
        self.logits = self.fc_layer(
            x=self.drop7,
            num_input=4096,
            num_output=self.NUM_CLASSES, 
            name='fc8',
            is_relu=False)

    def load_weights(
            self,
            encoding,
            session):

        weights_dict = np.load(self.WEIGHTS_PATH, encoding=encoding).item()

        for op_name in weights_dict:
            if op_name not in self.SKIP_LAYERS:
                with tf.variable_scope(op_name, reuse=True):
                    for weights in weights_dict[op_name]:
                        if len(weights.shape) == 1:
                            if self.RETRAIN:
                                var = tf.get_variable('biases')
                            else:
                                var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(weights))
                        else:
                            if self.RETRAIN:
                                var = tf.get_variable('weights')
                            else:
                                var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(weights))
