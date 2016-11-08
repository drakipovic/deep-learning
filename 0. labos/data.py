import math
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


def binlogreg_train(X, Y_, param_niter=1000, param_delta=0.001):
    w, b = np.random.randn(2, 1), 0
    for i in range(param_niter):
        scores = np.dot(X, w) + b

        probs = np.exp(scores) / (1 + np.exp(scores))
        loss = -np.sum(np.log(probs[Y_==1]) + np.log(1 - probs[Y_==0]))
        
        if i % 100 == 0:
            print "Iteration {}: loss: {}".format(i, loss)
        
        dl_dscores = probs - Y_

        grad_w = np.dot(np.transpose(X), dl_dscores)
        grad_b = np.sum(dl_dscores)

        w += -param_delta * grad_w
        b += -param_delta * grad_b
    
    return w, b


def binlogreg_classify(X, w, b):
    s = np.dot(X, w) + b
    probs = np.exp(s) / (1 + np.exp(s))

    return probs


def sample_gauss_2d(C, N):
    
    X = []
    Y = []
    
    for i in range(C):
        gauss = Random2DGaussian()
        x = gauss.get_sample(N)
        X.extend(x)

        y=[[i] for j in range(N)]
        Y.extend(y)

    return np.array(X), np.array(Y)
        


def eval_perf_binary(Y, Y_):
    acc = accuracy_score(Y, Y_)
    prec = precision_score(Y, Y_, average='binary')
    rec = recall_score(Y, Y_, average='binary')
    return acc, prec, rec



def decision_func(X):
    scores = X[:,0] + X[:,1] - 5
    return scores>0.5


def graph_data(X, Y, Y_):
    correct_idx = (Y==Y_).T.flatten()
    wrong_idx = (Y!=Y_).T.flatten()

    plt.scatter(X[correct_idx, 0], X[correct_idx, 1], c=Y_[correct_idx], marker='o', label='correct')
    plt.scatter(X[wrong_idx, 0], X[wrong_idx, 1], c=Y_[wrong_idx], marker='s', label='wrong')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    X, Y_ = sample_gauss_2d(2, 100)
    w, b = binlogreg_train(X, Y_)
    probs = binlogreg_classify(X, w, b)
    Y = probs>0.5

    print eval_perf_binary(Y, Y_)

    graph_data(X, Y, Y_)

