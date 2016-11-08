import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from data import graph_data, graph_surface, eval_perf_binary, sample_gmm_2d


class TFDeep(object):

    def __init__(self, layers, delta=0.02, l=0.001):

        self.layers = layers

        self.X = tf.placeholder(tf.float32, [None, layers[0]], name='X')
        self.Y_ = tf.placeholder(tf.float32, [None, layers[-1]], name='Y_')

        self.W = []
        self.b = []
        for i in range(len(layers) - 1):
            self.W.append(tf.Variable(tf.random_normal([layers[i], layers[i+1]], stddev=0.5), name="W_{}".format(i)))
            self.b.append(tf.Variable(tf.random_normal([1, layers[i+1]], stddev=0.5), name='b_{}'.format(i)))
        
        self.h = [self.X]
        last_w = self.W.pop()
        last_b = self.b.pop()
        for w, b_i in zip(self.W, self.b):
            self.h.append(tf.nn.relu(tf.matmul(self.h[-1], w) + b_i))
        
        self.h.append(tf.matmul(self.h[-1], last_w) + last_b)
        self.probs = tf.nn.softmax(self.h[-1])

        #self.loss = tf.nn.softmax_cross_entropy_with_logits(self.probs, self.Y_)
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.Y_ * tf.log(self.probs), 1))

        self.trainer = tf.train.GradientDescentOptimizer(delta)
        self.train_step = self.trainer.minimize(self.loss)

        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

    def train(self, X, Y_, iterations):
        for i in range(iterations):
            _, val_loss = self.session.run([self.train_step, self.loss], feed_dict={self.X: X, self.Y_: Y_})
            if i % 10 == 0:
                print i, val_loss

    
    def eval(self, X):
        P = self.session.run([self.probs], feed_dict={self.X: X})

        return P


if __name__ == '__main__':
    
    X, y = sample_gmm_2d(6, 3, 10)

    y_ = OneHotEncoder().fit_transform(y).toarray()
    
    deep = TFDeep([2, 3])
    deep.train(X, y_, 10000)

    y = y.flatten()

    probs = deep.eval(X)
    Y = np.argmax(probs[0], axis=1)

    print eval_perf_binary(Y, y)

    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(lambda x: np.argmax(deep.eval(x)[0], axis=1), bbox, offset=0.5)
    graph_data(X, Y, y)