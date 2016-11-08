import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from data import sample_gmm_2d, eval_perf_binary, graph_data, graph_surface


def forward(X, w_1, w_2, b_1, b_2):
    s_1 = np.dot(X, w_1) + b_1

    h_1 = np.maximum(s_1, np.zeros(s_1.shape))
    
    s_2 = np.dot(h_1, w_2) + b_2
    es = np.exp(s_2)
    probs = es / np.sum(es, axis=1, keepdims=True)
    return probs, h_1


def fcann_train(X, y, C, iterations=10000, delta=0.003, l=1e-3, hl_size=5):
    #tezine prvog sloja
    w_1 = np.random.randn(X.shape[1], hl_size)
    b_1 = np.random.randn(1, hl_size)

    #tezine drugog sloja
    w_2 = np.random.randn(hl_size, C)
    b_2 = np.random.randn(1, C)

    for i in range(iterations):
        probs, h_1 = forward(X, w_1, w_2, b_1, b_2)

        gs2 = probs - y
        grad_w2 = np.dot(h_1.T, gs2)
        grad_b2 = np.sum(gs2, axis=0)

        gh1 = np.dot(gs2, w_2.T)
        gs1 = gh1 * (h_1 > 0)
        
        grad_w1 = np.dot(X.T, gs1)
        grad_b1 = np.sum(gs1, axis=0)

        w_1 += -delta * grad_w1
        w_2 += -delta * grad_w2
        b_1 += -delta * grad_b1
        b_2 += -delta * grad_b2

    
    return w_1, w_2, b_1, b_2



if __name__ == "__main__":
    X, y = sample_gmm_2d(6, 4, 30)
    
   
    C = len(np.lib.arraysetops.unique(y))
    
    #X = np.array([[1, 2], [2, 3], [4, 5]])
    #y = np.array([0, 1, 1])[np.newaxis]
    
    y_ = OneHotEncoder().fit_transform(y).toarray()

    w_1, w_2, b_1, b_2 = fcann_train(X, y_, C)

    probs, _ = forward(X, w_1, w_2, b_1, b_2)

    Y = np.argmax(probs, axis=1)
    y = y.flatten()
    print eval_perf_binary(Y, y)

    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(lambda x: np.argmax(forward(x, w_1, w_2, b_1, b_2)[0], axis=1), bbox, offset=0.5)
    graph_data(X, y, Y)   