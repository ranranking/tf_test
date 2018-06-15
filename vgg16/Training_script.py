import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from vgg16 import *

def _image_parser (path, label):
    
    # Read Image
    image_file = tf.read_file(path)
    image_decoded = tf.image.decode_jpeg(image_file, channels=3)
    image_resized = tf.image.resize_images(image_decoded, [256, 256])

    return image_resized, label

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='../train20.txt', type=str)
parser.add_argument('--test_dir', default='../test20.txt', type=str)
parser.add_argument('--log_dir', default='./log/', type=str)
parser.add_argument('--weight_path', default='../vgg16.npy', type=str)
parser.add_argument('--model_id', default=0.001, type=float)
parser.add_argument('--model_path', default=None, type=str)
parser.add_argument('--shuffle_buffer', default=250, type=int)
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--epoch', default=10, type=int)
parser.add_argument('--val_step', default=10, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--display_step', default=1, type=int)
parser.add_argument('--num_class', default=20, type=int)
parser.add_argument('--dropout_keep_rate', default=0.75, type=float)
parser.add_argument('--retrain_all', default=True, type=bool)
parser.add_argument('--skip_layer', nargs='+',  default=['fc7', 'fc8'])


def main(args):
    # Misc
    TRAIN_DIR = args.train_dir
    TEST_DIR = args.test_dir
    LOG_DIR = args.log_dir
    MODEL_ID = args.model_id
    IMGNET_PRE_MODEL = args.model_path
    WEIGHT_DIR = args.weight_path
    
    # Training Parameters
    SHUFFLE_BUFFER= args.shuffle_buffer
    LEARNING_RATE = args.learning_rate
    MOMENTUM = args.momentum
    NUM_EPOCHS = args.epoch
    VAL_STEPS = args.val_step
    BATCH_SIZE = args.batch_size
    DISPLAY_STEP = args.display_step
    NUM_CLASSES = args.num_class
    KEEP_RATE = args.dropout_keep_rate
    RETRAIN = args.retrain_all
    SKIP_LAYER = args.skip_layer
    
    print('Parameters: ')
    print('Training dir: %s' % TRAIN_DIR)
    print('Testing dir: %s' % TEST_DIR)
    print('Log dir: %s' % LOG_DIR)
    print('Model ID: %s' % MODEL_ID)
    print('Imagenet pre model: %s' % IMGNET_PRE_MODEL)
    print('Weight dir: %s' % WEIGHT_DIR)
    print('Shuggle buffer: %s' % SHUFFLE_BUFFER)
    print('Learning rate: %s' % LEARNING_RATE)
    print('Momentum: %s' % MOMENTUM)
    print('Num of epochs: %s' % NUM_EPOCHS)
    print('Validation steps: %s' % VAL_STEPS)
    print('Batch size: %s' % BATCH_SIZE)
    print('Display step: %s' % DISPLAY_STEP)
    print('Num of classes: %s' % NUM_CLASSES)
    print('Dropout keep rate: %s' % KEEP_RATE)
    print('If retrain all weights: %s' % RETRAIN)
    print('Skip layers: %s' % SKIP_LAYER)
    print(' ')

    print('Load data')
    # Data preparation
    train = pd.read_csv(TRAIN_DIR, delimiter=' ', header=None).sample(frac=1)  
    val = train.sample(frac=0.1, random_state=10)
    train = train.drop(val.index)
    test = pd.read_csv(TEST_DIR, delimiter=' ', header=None).sample(frac=1)  
    print('Done')

    print('Build graph') 
    # Model graph
    model_graph = tf.Graph()
    with model_graph.as_default():
        
        keep_prob = tf.placeholder(tf.float32)
        rand_crop = tf.placeholder(tf.bool)
        
        # Dataset
        trn_ds = tf.data.Dataset.from_tensor_slices(
            (np.array(train[0]), np.array(train[1]))).map(_image_parser).shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)
        val_ds = tf.data.Dataset.from_tensor_slices(
            (np.array(val[0]), np.array(val[1]))).map(_image_parser).shuffle(SHUFFLE_BUFFER).repeat().batch(BATCH_SIZE)
        test_ds = tf.data.Dataset.from_tensor_slices(
            (np.array(test[0]), np.array(test[1]))).map(_image_parser).batch(BATCH_SIZE)
        
        # String Handle
        handle = tf.placeholder(tf.string, [])
        iterator = tf.data.Iterator.from_string_handle(handle, trn_ds.output_types, trn_ds.output_shapes)
        x, y = iterator.get_next()
        
        # Dataset iterators
        training_iterator = trn_ds.make_initializable_iterator()
        validation_iterator = val_ds.make_one_shot_iterator()
        testing_iterator = test_ds.make_one_shot_iterator()
        
        # Image Summary
        # tf.summary.image('input', x, 5)
        
        # Build Model
        vgg = MY_VGG16(x=x, keep_rate=keep_prob, num_classes=NUM_CLASSES, 
                       batch_size=BATCH_SIZE, 
                       skip_layers=SKIP_LAYER, weights_path=WEIGHT_DIR,
                       retrain=RETRAIN, random_crop=rand_crop)
        vgg.build()
        
        # Image Summary
        # tf.summary.image('input', vgg.final_input, 5)
        
        # Logits and Predictions
        logits = vgg.logits
        
        with tf.variable_scope('predictions'):
            pred_classes  = tf.argmax(logits, axis=1, name='prediction_label')
            softmax = tf.nn.softmax(logits, name='softmax')
            tf.summary.histogram('prediction_label', pred_classes)
            tf.summary.histogram('softmax', softmax)
    
        # Loss and optimizer
        with tf.variable_scope('cross_entropy_loss'):   
    #         cross_entropy = -tf.reduce_sum(input_tensor=y * tf.log(softmax), axis=-1, name='cross_entropy')
    #         loss_op = tf.reduce_mean(cross_entropy, name='mean_loss')
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits, name='sparse_cross_entropy')
            loss_op = tf.reduce_mean(cross_entropy, name='mean_loss')
            tf.summary.scalar('cross_entropy_loss', loss_op)
        
        with tf.variable_scope('train'):
            optimizer = tf.train.MomentumOptimizer(learning_rate=LEARNING_RATE, momentum=MOMENTUM, name='momentum_optimizer')
            train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step(), name='loss_minimization')
        
        # Evaluation
        with tf.variable_scope('accuracy'):
    #         true_labels = tf.argmax(y, 1, name='true_label')
            correct_pred = tf.equal(pred_classes, y, name='true_pred_equal')
            accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='mean_accuracy')
            tf.summary.scalar('accuracy', accuracy_op)
        
        # Global Initializer
        global_init = tf.global_variables_initializer()
        
        # Merge Summary
        summary = tf.summary.merge_all()
        
        # Global saver
        saver = tf.train.Saver()

    print('Done')

    # Runing session
    with tf.Session(graph=model_graph) as sess:
        
        # Debugger
    #     sess = tf_debug.TensorBoardDebugWrapperSession(sess, "marr:7000")
        
    #     # Run Global Initializer
        print('Start initializing.')
        sess.run(global_init)
        print('Done')
        
        # Iterator Handles
        training_handle = sess.run(training_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())
        testing_handle = sess.run(testing_iterator.string_handle())
        
        # Load ImageNet Weights
        print('Starting loading pretrained weights.')
        vgg.load_weights(session=sess, encoding='latin1')
        print('Loaded.')
        
        # Restore pretrained model
    #     print('Loading imagenet model')
    #     saver.restore(sess, IMGNET_PRE_MODEL)
    #     print('Model restored.')
    
        # Writer
        writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'model_%s'%MODEL_ID, 'model_summary'), graph=model_graph)
        
        step = 0
        
        print('Start training.')
        
        for epoch in range(NUM_EPOCHS):
            
            sess.run(training_iterator.initializer)
            while True:
                # Training
                try:
                    sess.run(train_op, feed_dict={keep_prob: KEEP_RATE, rand_crop: True, handle: training_handle})
    
                    if step % DISPLAY_STEP == 0:
    
                        loss_val, acc, s = sess.run([loss_op, accuracy_op, summary],
                                                    feed_dict={keep_prob: 1.0, rand_crop: False, handle: training_handle})
    
                        writer.add_summary(s, step)
    
                        print("Epoch " + str(epoch) + ", Step " + str(step) + ", Minibatch Loss= " + \
                              "{:.4f}".format(loss_val) + ", Training Accuracy= " + \
                              "{:.3f}".format(acc))
                        
                    step += 1
    
                except tf.errors.OutOfRangeError:
                    break
    
                # Validation
                if step % (DISPLAY_STEP * 10) == 0 and step != 0: 
    
                    val_acc = 0
                    val_loss = 0
    
                    for val_step in range(VAL_STEPS):
                        loss_val, acc = sess.run([loss_op, accuracy_op], feed_dict={keep_prob: 1.0, rand_crop: False, handle: validation_handle})
                        val_acc += acc
                        val_loss += loss_val
    
                    print("Validation" + ", Minibatch Loss= " + \
                          "{:.4f}".format(val_loss/VAL_STEPS) + ", Training Accuracy= " + \
                          "{:.3f}".format(val_acc/VAL_STEPS))
                
            # Save the model every epoch
            saver.save(sess, os.path.join(LOG_DIR, 'model_%s'%MODEL_ID, "epoch.model.ckpt"), epoch)
    
        # Save the final model
        saver.save(sess, os.path.join(LOG_DIR, 'model_%s'%MODEL_ID, "final.model.ckpt"), step)
    
        # Testing
        test_acc = 0
        test_loss = 0
        testing_iter = 0
    
        while True:        
            try:
                loss_val, acc = sess.run([loss_op, accuracy_op], feed_dict={keep_prob: 1.0, rand_crop: False, handle: testing_handle})
                test_acc += acc
                test_loss += loss_val
                testing_iter += 1
            except tf.errors.OutOfRangeError:
                break
    
        assert testing_iter != 0
        print("Testing" + ", Minibatch Loss= " + \
              "{:.4f}".format(test_loss/testing_iter) + ", Training Accuracy= " + \
              "{:.3f}".format(test_acc/testing_iter))
                
        writer.close()
        
        print('Done.')
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
