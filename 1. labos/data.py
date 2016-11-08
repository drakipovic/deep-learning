import math

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score


class Random2DGaussian(object):
    
    min_x = 0
    max_x = 10
    min_y = 0
    max_y = 10

    def __init__(self):
        self.mean = [(self.max_x-self.min_x) * np.random.random_sample() + self.min_x, 
                        (self.max_y-self.min_y) * np.random.random_sample() + self.min_y]
        
        eigval_x = (np.random.random_sample() * (self.max_x - self.min_x) / 5)**2
        eigval_y = (np.random.random_sample() * (self.max_y - self.min_y) / 5)**2
        D = [[eigval_x, 0], [0, eigval_y]]
        
        angle = 2 * math.pi * np.random.random_sample()
        R = [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]

        self.cov = np.dot(np.dot(R, D), np.transpose(R))

    def get_sample(self, n):
        return np.random.multivariate_normal(self.mean, self.cov, size=n)


def sample_gmm_2d(K, C, N):
    X, Y = [], []

    for i in range(K):
        gauss = Random2DGaussian()
        j = np.random.randint(C)

        x = gauss.get_sample(N)
        X.extend(x)

        y=[[j] for k in range(N)]
        Y.extend(y)
    
    return np.array(X), np.array(Y)


def graph_data(X, Y, Y_, special=None):
    correct_idx = (Y==Y_).T.flatten()
    wrong_idx = (Y!=Y_).T.flatten()

    s = np.ones(Y.shape) * 20
    if special is not None:
        s[special] *= 2

    plt.scatter(X[correct_idx, 0], X[correct_idx, 1], c=Y_[correct_idx], marker='o', label='correct', s=s[correct_idx])
    plt.scatter(X[wrong_idx, 0], X[wrong_idx, 1], c=Y_[wrong_idx], marker='s', label='wrong', s=s[wrong_idx])
    plt.legend(loc='upper left')
    plt.show()


def eval_perf_binary(Y, Y_):
    acc = accuracy_score(Y, Y_)
    prec = precision_score(Y, Y_, average='binary')
    rec = recall_score(Y, Y_, average='binary')
    return acc, prec, rec


def graph_surface(fun, rect, offset):
    x_min, x_max = rect[0][0], rect[1][0]
    y_min, y_max = rect[0][1], rect[1][1]

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, num=100),np.linspace(y_min, y_max, num=100))
    x = xx.flatten()
    y = yy.flatten()
    XX = np.stack((x, y), axis=1)

    Z = fun(XX)
    plt.pcolormesh(xx, yy, Z.reshape(xx.shape))
    plt.contour(xx, yy, Z.reshape(xx.shape), levels = [offset])
    