import os
import time
import math

import pickle
import numpy as np
import tensorflow.contrib.layers as layers
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import skimage as ski
import skimage.io
import matplotlib.pyplot as plt


def shuffle_data(data_x, data_y):
    indices = np.arange(data_x.shape[0])
    np.random.shuffle(indices)
    shuffled_data_x = np.ascontiguousarray(data_x[indices])
    shuffled_data_y = np.ascontiguousarray(data_y[indices])
    return shuffled_data_x, shuffled_data_y


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict


def one_hot_encode(array, num_classes):
    one_hot_arr = np.zeros((len(array), num_classes))
    one_hot_arr[np.arange(len(array)), array] = 1
    return one_hot_arr


def plot_training_progress(save_dir, data):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))

    linewidth = 2
    legend_size = 10
    train_color = 'm'
    val_color = 'c'

    num_points = len(data['train_acc'])
    x_data = np.linspace(1, num_points, num_points)
    ax1.set_title('Cross-entropy loss')
    ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
            linewidth=linewidth, linestyle='-', label='train')
    ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
            linewidth=linewidth, linestyle='-', label='validation')
    ax1.legend(loc='upper right', fontsize=legend_size)
    ax2.set_title('Average class accuracy')
    ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
            linewidth=linewidth, linestyle='-', label='train')
    ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
            linewidth=linewidth, linestyle='-', label='validation')
    ax2.legend(loc='upper left', fontsize=legend_size)
    ax3.set_title('Learning rate')
    ax3.plot(x_data, data['lr'], marker='o', color=train_color,
            linewidth=linewidth, linestyle='-', label='learning_rate')
    ax3.legend(loc='upper left', fontsize=legend_size)

    save_path = os.path.join(save_dir, 'training_plot.pdf')
    print('Plotting in: ', save_path)
    plt.savefig(save_path)


DATA_DIR = '/home/dinek/cifar'
SAVE_DIR = "/home/dinek/source/fer/cifar/"

img_height = 32
img_width = 32
num_channels = 3
num_classes = 10


train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
train_y = []

for i in range(1, 6):
    subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
    train_x = np.vstack((train_x, subset['data']))
    train_y += subset['labels']

train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0,2,3,1)
train_y = np.array(train_y, dtype=np.int32)

subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0,2,3,1).astype(np.float32)
test_y = np.array(subset['labels'], dtype=np.int32)
test_y = one_hot_encode(test_y, num_classes)

valid_size = 5000
train_x, train_y = shuffle_data(train_x, train_y)
valid_x = train_x[:valid_size, ...]
valid_y = train_y[:valid_size, ...]
valid_y = one_hot_encode(valid_y, num_classes)
train_x = train_x[valid_size:, ...]
train_y = train_y[valid_size:, ...]
train_y = one_hot_encode(train_y, num_classes)
data_mean = train_x.mean((0,1,2))
data_std = train_x.std((0,1,2))

train_x = (train_x - data_mean) / data_std
valid_x = (valid_x - data_mean) / data_std
test_x = (test_x - data_mean) / data_std

config = {}
config['max_epochs'] = 8
config['batch_size'] = 50
config['weight_decay'] = 1e-4
config['lr_policy'] = 1e-3
config['save_dir'] = SAVE_DIR


def build_model(inputs, labels, num_classes):
    weight_decay = config['weight_decay']
    conv1sz = 16
    conv2sz = 32
    fc3sz = 256
    fc4sz = 128
    with tf.contrib.framework.arg_scope([layers.convolution2d],
        kernel_size=5, stride=1, padding='SAME', activation_fn=tf.nn.relu,
        weights_initializer=layers.variance_scaling_initializer(),
        weights_regularizer=layers.l2_regularizer(weight_decay)):

        net = layers.convolution2d(inputs, conv1sz, scope='conv1', variables_collections=['weights'])
        net = layers.max_pool2d(net, kernel_size=3, stride=2, scope='pool1')
        net = layers.convolution2d(net, conv2sz, scope='conv2')
        net = layers.max_pool2d(net, kernel_size=2, stride=2, scope='pool2')

    with tf.contrib.framework.arg_scope([layers.fully_connected],
        activation_fn=tf.nn.relu,
        weights_initializer=layers.variance_scaling_initializer(),
        weights_regularizer=layers.l2_regularizer(weight_decay)):
  
        net = layers.flatten(net)
        net = layers.fully_connected(net, fc3sz, scope='fc3')
        net = layers.fully_connected(net, fc4sz, scope='fc4')
    
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


def train(net, loss, session, inputs, y_correct, train_x, train_y, valid_x, valid_y, test_x, test_y):
    lr = config['lr_policy']
    batch_size = config['batch_size']
    max_epochs = config['max_epochs']
    save_dir = config['save_dir']
    num_examples = train_x.shape[0]
    assert num_examples % batch_size == 0
    num_batches = num_examples // batch_size
    
    trainer = tf.train.AdamOptimizer(lr)
    train_step = trainer.minimize(loss)
    session.run(tf.initialize_all_variables())

    plot_data = {}
    plot_data['train_loss'] = []
    plot_data['valid_loss'] = []
    plot_data['train_acc'] = []
    plot_data['valid_acc'] = []
    plot_data['lr'] = []

    for epoch in range(1, max_epochs+1):
        
        permutation_idx = np.random.permutation(num_examples)
        train_x = train_x[permutation_idx]
        train_y = train_y[permutation_idx]

        train_accuracy = 0
        train_loss = 0
        for i in range(num_batches):

            batch_x = train_x[i*batch_size:(i+1)*batch_size, :]
            batch_y = train_y[i*batch_size:(i+1)*batch_size, :]
            
            loss_val, _ = session.run([loss, train_step], feed_dict={inputs: batch_x, y_correct: batch_y})
            train_loss += loss_val
            if i % 100 == 0:
                print("epoch {}, step {}/{}, batch loss = {:.2f}".format(epoch, i * batch_size, num_examples, loss_val))
                conv1_weights = tf.get_collection('weights')[0]
                weight_val = session.run(conv1_weights)
                draw_conv_filters(epoch, i*batch_size, weight_val, save_dir)
                
                accuracy = get_accuracy(net, batch_y, batch_x, session, inputs, y_correct)
                train_accuracy += accuracy
                print('accuracy = {}'.format(accuracy))
                
        plot_data['train_loss'] += [train_loss / float(num_batches)]
        plot_data['train_acc'] += [train_accuracy / 11]
        validation_accuracy = 0
        num_valid_batches = valid_x.shape[0] // batch_size
        validation_loss = 0
        for i in range(num_valid_batches):

            batch_x = valid_x[i*batch_size:(i+1)*batch_size, :]
            batch_y = valid_y[i*batch_size:(i+1)*batch_size, :]
            valid_loss = session.run(loss, feed_dict={inputs: batch_x, y_correct: batch_y})
            validation_loss += valid_loss

            if batch_x.size and batch_y.size:
                validation_accuracy += get_accuracy(net, batch_y, batch_x, session, inputs, y_correct)
        
        print('Validation accuracy = {}'.format(validation_accuracy / float(num_valid_batches)))    
        
        plot_data['valid_loss'] += [validation_loss / float(num_valid_batches)]
        plot_data['valid_acc'] += [validation_accuracy / float(num_valid_batches)]
        plot_data['lr'] += [lr]
    
    test_accuracy = 0
    num_test_batches = test_x.shape[0] // batch_size
    for i in range(num_test_batches):

        batch_x = test_x[i*batch_size:(i+1)*batch_size, :]
        batch_y = test_y[i*batch_size:(i+1)*batch_size, :]

        if batch_x.size and batch_y.size:
            test_accuracy += get_accuracy(net, batch_y, batch_x, session, inputs, y_correct)
    
    print('Test accuracy = {}'.format(test_accuracy / float(num_test_batches)))

    plot_training_progress(config['save_dir'], plot_data)


def get_accuracy(net, pred_y, x, session, inputs, y_correct):
    
    net_out = session.run(net, feed_dict={inputs: x, y_correct: pred_y})
    correct = np.argmax(net_out, axis=1) == np.argmax(pred_y, axis=1)
    #import pdb; pdb.set_trace()
    return correct[correct==True].size / float(correct.size)


inputs = tf.placeholder(tf.float32, (None, train_x.shape[1], train_x.shape[2], train_x.shape[3]))
y_correct = tf.placeholder(tf.float32, (None, num_classes))
net, loss = build_model(inputs, y_correct, num_classes)

session = tf.Session()


train(net, loss, session, inputs, y_correct, train_x, train_y, valid_x, valid_y, test_x, test_y)