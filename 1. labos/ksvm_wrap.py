from sklearn.svm import SVC
import numpy as np

from data import graph_data, graph_surface, sample_gmm_2d, eval_perf_binary


class KSVMWrapper(object):

    def __init__(self, X, Y_, c=1, g='auto'):
        self.clf = SVC(C=c, gamma=g)
        self.clf.fit(X, Y_)

    def predict(self, X):
        return self.clf.predict(X)
    
    def get_scores(self, X):
        return self.clf.decision_function(X)
    
    @property
    def support(self):
        return self.clf.support_


if __name__ == '__main__':

    X, y = sample_gmm_2d(6, 2, 10)

    ksvmw = KSVMWrapper(X, y)
    y_ = ksvmw.predict(X)
    
    y = y.flatten()
    print eval_perf_binary(y_, y)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(lambda x: ksvmw.get_scores(x), bbox, offset=0)
    graph_data(X, y_, y, special=ksvmw.support)