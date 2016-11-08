import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import numpy as np

from data import sample_gmm_2d, graph_surface, eval_perf_binary, graph_data


class TFLogreg(object):
    
    def __init__(self, D, C, param_delta=0.02, l=0.001):
        self.X  = tf.placeholder(tf.float32, [None, D])
        self.Y_ = tf.placeholder(tf.float32, [None, None])
        self.W = tf.Variable(tf.random_normal([D, C], stddev=0.5))
        self.b = tf.Variable(tf.random_normal([1, C], stddev=0.5))

        self.probs = tf.nn.softmax((tf.matmul(self.X, self.W) + self.b))

        self.loss = tf.reduce_mean(-tf.reduce_sum(self.Y_ * tf.log(self.probs), 1)) + l * tf.nn.l2_loss(self.W)

        self.trainer = tf.train.GradientDescentOptimizer(param_delta)
        self.train_step = self.trainer.minimize(self.loss)

        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

    def train(self, X, Y_, param_niter):
        for i in range(param_niter):
            _, val_loss = self.session.run([self.train_step, self.loss], feed_dict={self.X: X, self.Y_: Y_})
            if i % 1000 == 0:
                print i, val_loss

    def eval(self, X):
        P = self.session.run([self.probs], feed_dict={self.X: X})

        return P


if __name__ == '__main__':
    X, y = sample_gmm_2d(6, 3, 50)
    C = len(np.lib.arraysetops.unique(y))

    y_ = OneHotEncoder().fit_transform(y).toarray()

    logreg = TFLogreg(2, C)
    logreg.train(X, y_, 10000)
    probs = logreg.eval(X)
    Y = np.argmax(probs[0], axis=1)
    
    y = y.flatten()
    print eval_perf_binary(Y, y)

    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(lambda x: np.argmax(logreg.eval(x)[0], axis=1), bbox, offset=0.5)
    graph_data(X, Y, y)