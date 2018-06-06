import tensorflow as tf
import numpy as np

class My_AlexNet:

    def __init__(self, x, keep_rate, num_classes, skip_layers=None, weights_path=None, retrain=True):

        self.X = x
        self.KEEP_RATE = keep_rate
        self.NUM_CLASSES = num_classes
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
        
    def lrn (
        self,
        x,
        radius,
        alpha,
        beta,
        bias,
        name):
    
        return tf.nn.local_response_normalization(
            input=x,
            depth_radius=radius,
            alpha=alpha,
            beta=beta,
            bias=bias,
            name=name)
    
    def build(
        self):

        # Layer #1
        # Conv 1
        self.conv1 = self.conv_layer(
            x=self.X,
            input_channels=3, 
            filter_height=11,
            filter_width=11, 
            num_filters=96, 
            stride_y=4, 
            stride_x=4, 
            padding='VALID', 
            name='conv1')
    
        # Norm 1
        self.norm1 = self.lrn (
            x=self.conv1,
            radius=2,
            alpha=1e-05,
            beta=0.75,
            bias=1.0,
            name='norm1')
    
        # Pool1
        self.pool1 = self.max_pool(
            x=self.norm1,
            filter_height=3,
            filter_width=3, 
            stride_y=2, 
            stride_x=2, 
            padding='VALID', 
            name='pool1')
    
        # Layer #2
        # Conv 2
        self.conv2 = self.conv_layer(
            x=self.pool1,
            input_channels=96, 
            filter_height=5, 
            filter_width=5, 
            num_filters=256, 
            stride_y=1, 
            stride_x=1, 
            padding='SAME', 
            name='conv2')
    
        # Norm 2
        self.norm2 = self.lrn (
            x=self.conv2,
            radius=2,
            alpha=1e-05,
            beta=0.75,
            bias=1.0,
            name='norm2')
    
        # Pool 2
        self.pool2 = self.max_pool(
            x=self.norm2,
            filter_height=3,
            filter_width=3, 
            stride_y=2, 
            stride_x=2, 
            padding='VALID', 
            name='pool2')
    
        # Layer 3
        # Conv 3
        self.conv3 = self.conv_layer(
            x=self.pool2,
            input_channels=256, 
            filter_height=3, 
            filter_width=3, 
            num_filters=384, 
            stride_y=1, 
            stride_x=1, 
            padding='SAME', 
            name='conv3')
    
        # Layer 4
        # Conv 4
        self.conv4 = self.conv_layer(
            x=self.conv3,
            input_channels=384, 
            filter_height=3, 
            filter_width=3, 
            num_filters=384, 
            stride_y=1, 
            stride_x=1, 
            padding='SAME', 
            name='conv4')
    
        # Layer 5
        # Conv 5
        self.conv5 = self.conv_layer(
            x=self.conv4,
            input_channels=384, 
            filter_height=3, 
            filter_width=3, 
            num_filters=256, 
            stride_y=1, 
            stride_x=1, 
            padding='SAME', 
            name='conv5')
    
        # Pool 5
        self.pool5 = self.max_pool(
            x=self.conv5,
            filter_height=3,
            filter_width=3, 
            stride_y=2, 
            stride_x=2, 
            padding='VALID', 
            name='pool5')
        
        # Layer 6
        # Flatten
        self.flat6 = tf.reshape(
            tensor=self.pool5,
            shape=[-1, 6 * 6 * 256],
            name='flatten6')
    
        # Fc 6 
        self.fc6 = self.fc_layer(
            x=self.flat6,
            num_input=6 * 6 * 256,
            num_output=4096, 
            name='fc6',
            is_relu=True)
    
        # Drop 6
        self.drop6 = self.dropout(
            x=self.fc6,
            keep_rate=self.KEEP_RATE,
            name='drop6')
    
        # Layer 7
        # Fc 7 
        self.fc7 = self.fc_layer(
            x=self.drop6,
            num_input=4096,
            num_output=4096, 
            name='fc7',
            is_relu=True)
    
        # Drop 7
        self.drop7 = self.dropout(
            x=self.fc7,
            keep_rate=self.KEEP_RATE,
            name='drop7')
    
        # Logit layer 8
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
        


    
    

         
