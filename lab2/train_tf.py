import time
import os
import math

import tensorflow.contrib.layers as layers
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import skimage as ski
import skimage.io


DATA_DIR = '/home/dinek/datasets/MNIST/'
SAVE_DIR = "/home/dinek/source/fer/tf/"

config = {}
config['max_epochs'] = 8
config['batch_size'] = 50
config['save_dir'] = SAVE_DIR
config['weight_decay'] = 1e-1
config['lr_policy'] = {'lr': 1e-4}

np.random.seed(int(time.time() * 1e6) % 2**31)
dataset = input_data.read_data_sets(DATA_DIR, one_hot=True)
train_x = dataset.train.images
train_x = train_x.reshape([-1, 28, 28, 1])
train_y = dataset.train.labels
valid_x = dataset.validation.images
valid_x = valid_x.reshape([-1, 28, 28, 1])
valid_y = dataset.validation.labels
test_x = dataset.test.images
test_x = test_x.reshape([-1, 28, 28, 1])
test_y = dataset.test.labels
train_mean = train_x.mean()
train_x -= train_mean
valid_x -= train_mean
test_x -= train_mean

weight_decay = config['weight_decay']


def build_model(inputs, labels, num_classes):
    weight_decay = 1e-3
    conv1sz = 16
    conv2sz = 32
    fc3sz = 512
    with tf.contrib.framework.arg_scope([layers.convolution2d],
        kernel_size=5, stride=1, padding='SAME', activation_fn=tf.nn.relu,
        weights_initializer=layers.variance_scaling_initializer(),
        weights_regularizer=layers.l2_regularizer(weight_decay)):

        net = layers.convolution2d(inputs, conv1sz, scope='conv1', variables_collections=['weights'])
        net = layers.max_pool2d(net, kernel_size=2, stride=2, scope='pool1')
        net = layers.convolution2d(net, conv2sz, scope='conv2')
        net = layers.max_pool2d(net, kernel_size=2, stride=2, scope='pool2')

    with tf.contrib.framework.arg_scope([layers.fully_connected],
        activation_fn=tf.nn.relu,
        weights_initializer=layers.variance_scaling_initializer(),
        weights_regularizer=layers.l2_regularizer(weight_decay)):
  
        net = layers.flatten(net)
        net = layers.fully_connected(net, fc3sz, scope='fc3')
    
    logits = layers.fully_connected(net, num_classes, activation_fn=None, scope='logits')
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels, name='loss')) + tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    return logits, loss

def draw_conv_filters(epoch, step, weights, save_dir):
    w = weights.copy()
    num_filters = w.shape[3]
    num_channels = w.shape[2]
    k = w.shape[0]
    assert w.shape[0] == w.shape[1]
    w = w.reshape(k, k, num_channels, num_filters)
    w -= w.min()
    w /= w.max()
    border = 1
    cols = 8
    rows = math.ceil(num_filters / cols)
    width = cols * k + (cols-1) * border
    height = rows * k + (rows-1) * border
    img = np.zeros([height, width, num_channels])
    for i in range(num_filters):
        r = int(i / cols) * (k + border)
        c = int(i % cols) * (k + border)
        img[r:r+k,c:c+k,:] = w[:,:,:,i]
    filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
    if num_channels == 1:
        img = img.reshape((height, width))
    ski.io.imsave(os.path.join(save_dir, filename), img)


def train(net, loss, session, inputs, y_correct, train_x, train_y, valid_x, valid_y):
    lr_policy = config['lr_policy']
    batch_size = config['batch_size']
    max_epochs = config['max_epochs']
    save_dir = config['save_dir']
    num_examples = train_x.shape[0]
    assert num_examples % batch_size == 0
    num_batches = num_examples // batch_size
    
    trainer = tf.train.AdamOptimizer(lr_policy['lr'])
    train_step = trainer.minimize(loss)
    session.run(tf.initialize_all_variables())

    for epoch in range(1, max_epochs+1):
        
        permutation_idx = np.random.permutation(num_examples)
        train_x = train_x[permutation_idx]
        train_y = train_y[permutation_idx]

        for i in range(num_batches):

            batch_x = train_x[i*batch_size:(i+1)*batch_size, :]
            batch_y = train_y[i*batch_size:(i+1)*batch_size, :]
            
            loss_val, _ = session.run([loss, train_step], feed_dict={inputs: batch_x, y_correct: batch_y})
            if i % 100 == 0:
                print("epoch {}, step {}/{}, batch loss = {:.2f}".format(epoch, i * batch_size, num_examples, loss_val))
                conv1_weights = tf.get_collection('weights')[0]
                weight_val = session.run(conv1_weights)
                draw_conv_filters(epoch, i*batch_size, weight_val, save_dir)
                
                accuracy = get_accuracy(net, batch_y, batch_x, session, inputs, y_correct)
                print('accuracy = {}'.format(accuracy))
        
        validation_accuracy = 0
        num_valid_batches = valid_x.shape[0] // batch_size
        for i in range(num_valid_batches):

            batch_x = valid_x[i*batch_size:(i+1)*batch_size, :]
            batch_y = valid_y[i*batch_size:(i+1)*batch_size, :]

            if batch_x.size and batch_y.size:
                validation_accuracy += get_accuracy(net, batch_y, batch_x, session, inputs, y_correct)
        
        print('Validation accuracy = {}'.format(validation_accuracy / float(num_valid_batches)))


def get_accuracy(net, pred_y, x, session, inputs, y_correct):
    
    net_out = session.run(net, feed_dict={inputs: x, y_correct: pred_y})
    correct = np.argmax(net_out, axis=1) == np.argmax(pred_y, axis=1)
    #import pdb; pdb.set_trace()
    return correct[correct==True].size / float(correct.size)
 

inputs = tf.placeholder(tf.float32, (None, 28, 28, 1))
y_correct = tf.placeholder(tf.float32, (None, 10))
net, loss = build_model(inputs, y_correct, 10)

session = tf.Session()


train(net, loss, session, inputs, y_correct, train_x, train_y, valid_x, valid_y)