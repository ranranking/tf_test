import os
import numpy as np
import pandas as pd
import tensorflow as tf
from vgg16 import *

def _image_parser (path, label):
    
    # Convert One Hot Label
    one_hot_label = tf.one_hot(label, NUM_CLASSES)
    
    # Read Image
    image_file = tf.read_file(path)
    image_decoded = tf.image.decode_jpeg(image_file, channels=3)
    image_resized = tf.image.resize_images(image_decoded, INPUT_IMAGE_SIZE)

    return image_resized, one_hot_label

# ----------------------------
# ----------------------------
# ----------------------------

# Misc
LOG_DIR = './log/'
MODEL_ID = 1
IMGNET_PRE_MODEL = './imagenet_pretrained_20/model.ckpt'

# Training Parameters
MEAN_IMAGE = np.array([104., 117., 124.])
SHUFFLE_BUFFER= 250
LEARNING_RATE = 0.005
MOMENTUM = 0.9
NUM_EPOCHS = 1
NUM_STEPS = 5000
BATCH_SIZE = 64
DISPLAY_STEP = 1
INPUT_IMAGE_SIZE = [256, 256]
NUM_CLASSES = 20
KEEP_RATE = 0.75

# ----------------------------
# ----------------------------
# ----------------------------

# Data preparation
train = pd.read_csv('../train20.txt', delimiter=' ', header=None).sample(frac=1)
val = train.sample(frac=0.1, random_state=10)
train = train.drop(val.index)
test = pd.read_csv('../test20.txt', delimiter=' ', header=None).sample(frac=1)

# ----------------------------
# ----------------------------
# ----------------------------

# Model Graph
model_graph = tf.Graph()
with model_graph.as_default():
    
    keep_prob = tf.placeholder(tf.float32)
    rand_crop = tf.placeholder(tf.bool)
    
    # Dataset
    trn_ds = tf.data.Dataset.from_tensor_slices(
        (np.array(train[0]), np.array(train[1]))).map(_image_parser).shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)
    val_ds = tf.data.Dataset.from_tensor_slices(
        (np.array(val[0]), np.array(val[1]))).map(_image_parser).shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices(
        (np.array(test[0]), np.array(test[1]))).map(_image_parser).batch(BATCH_SIZE)
    
    # String Handle
    handle = tf.placeholder(tf.string, [])
    iterator = tf.data.Iterator.from_string_handle(handle, trn_ds.output_types, trn_ds.output_shapes)
    x, y = iterator.get_next()
    
    # Dataset iterators
    training_iterator = trn_ds.make_initializable_iterator()
    validation_iterator = val_ds.make_initializable_iterator()
    testing_iterator = test_ds.make_one_shot_iterator()
    
    # Image Summary
    # tf.summary.image('input', x, 5)
    
    # Build Model
    vgg = MY_VGG16(x=x, keep_rate=KEEP_RATE, num_classes=NUM_CLASSES, 
                   batch_size=BATCH_SIZE, mean_image=MEAN_IMAGE,
                   skip_layers=['fc8'],weights_path='../vgg16.npy',
                   retrain=True, random_crop=rand_crop)
    vgg.build()
    
    # Image Summary
    tf.summary.image('input', vgg.final_input, 5)
    
    # Logits and Predictions
    logits = vgg.logits
    
    with tf.variable_scope('predictions'):
        pred_classes  = tf.argmax(logits, axis=1, name='prediction_label')
        prob = tf.nn.softmax(logits, name='prob')
        tf.summary.histogram('prediction_label', pred_classes)
        tf.summary.histogram('prob', prob)

    # Loss and optimizer
    with tf.variable_scope('cross_entropy_loss'):
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y), name='mean_loss')
        tf.summary.scalar('cross_entropy_loss', loss_op)
    
    with tf.variable_scope('train'):
        optimizer = tf.train.MomentumOptimizer(learning_rate=LEARNING_RATE, momentum=MOMENTUM, name='momentum_optimizer')
        train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step(), name='loss_minimization')
    
    # Evaluation
    with tf.variable_scope('accuracy'):
        true_labels = tf.argmax(y, 1, name='true_label')
        correct_pred = tf.equal(pred_classes, true_labels, name='true_pred_equal')
        accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='mean_accuracy')
        tf.summary.scalar('accuracy', accuracy_op)
    
    # Global Initializer
    global_init = tf.global_variables_initializer()
    
    # Merge Summary
    summary = tf.summary.merge_all()
    
    # Global saver
    saver = tf.train.Saver()

# ----------------------------
# ----------------------------
# ----------------------------

# Running Session

with tf.Session(graph=model_graph) as sess:
    
    # Debugger
    # sess = tf_debug.TensorBoardDebugWrapperSession(sess, "marr:7000")
    
    # Run Global Initializer
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
    writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'model_%s'%MODEL_ID), graph=model_graph)
    
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
            if step % (DISPLAY_STEP * 100) == 0 and step != 0:

                sess.run(validation_iterator.initializer)

                val_acc = 0
                val_loss = 0
                validation_iter = 0

                while True:
                    try:
                        loss_val, acc = sess.run([loss_op, accuracy_op], feed_dict={keep_prob: 1.0, rand_crop: False, handle: validation_handle})
                        val_acc += acc
                        val_loss += loss_val
                        validation_iter += 1
                    except tf.errors.OutOfRangeError:
                        break

                assert validation_iter != 0
                print("Validation" + ", Minibatch Loss= " + \
                      "{:.4f}".format(val_loss/validation_iter) + ", Training Accuracy= " + \
                      "{:.3f}".format(val_acc/validation_iter))
            
        # Save the model every epoch
        saver.save(sess, os.path.join(LOG_DIR, "epoch.model.ckpt"), epoch)

    # Save the final model
    saver.save(sess, os.path.join(LOG_DIR, "final.model.ckpt"), step)

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