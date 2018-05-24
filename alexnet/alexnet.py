import numpy as np
import tensorflow as tf

class My_AlexNet():

    def __init__(self, x, keep_rate, num_class, reinit_layer, weight_path, is_training):

        """
        Inputs:
        - x: Input images, tf.placeholder
        - keep_rate: Probability to keep data in dropout, tf.placeholder
        - num_class: Number of new class, int
        - reinit_layer: Names of the layers to be reinitialized, list of strings
        - weight_path: Path to the pretrained weights, string
        - is_training: If retraining the model, boolean 
        """

        pass

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
   
    def conv_layer(self):
        pass
 
    def fc_layer(self):
        pass

    def max_pool(self):
        pass

    def local_response(self):
        pass


