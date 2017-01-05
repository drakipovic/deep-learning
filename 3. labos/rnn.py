import os
import operator

import numpy as np

from database import Database

batch_size = 5

class RNN(object):

    def __init__(self, hidden_size, sequence_length, vocab_size, learning_rate):
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.theta = 1e-7

        self.U = np.random.normal(scale=1e-2, size=(hidden_size, vocab_size)) 
        self.W = np.random.normal(scale=1e-2, size=(hidden_size, hidden_size)) 
        self.b = np.zeros((hidden_size, 1))  
        
        self.V = np.random.normal(scale=1e-2, size=(vocab_size, hidden_size))
        self.c = np.zeros((vocab_size, 1))

        self.memory_U, self.memory_W, self.memory_V = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V)
        self.memory_b, self.memory_c = np.zeros_like(self.b), np.zeros_like(self.c)

    @staticmethod
    def rnn_step_forward(x, h_prev, U, W, b):
        h = np.dot(W, h_prev.T) + np.dot(U, x.T) + b
        h_current = np.tanh(h)

        return h_current.T, (h_current.T, h_prev.T, x)
    
    @staticmethod
    def rnn_forward(x, h0, U, W, b):
        cache = []
        h = np.zeros((h0.shape[0], x.shape[1]+1, h0.shape[1]))
        h[:,0,:] = h0
        
        for t in range(x.shape[1]):
            h_t, cache_t = RNN.rnn_step_forward(x[:,t,:], h[:,t,:], U, W, b)
            h[:,t+1,:] = h_t
            cache.append(cache_t)

        return h, cache


    def rnn_step_backward(self, grad_next, cache):
        h = cache[0]
        h_prev = cache[1]
        x = cache[2]
        dtanh = 1 - h**2

        dh_prev = np.dot(grad_next * dtanh, self.W)
        dU = np.dot((grad_next * dtanh).T, x)
        dW = np.dot((grad_next * dtanh).T, h_prev.T)
        db = np.sum(grad_next * dtanh, axis=0, keepdims=True)

        return dh_prev, dU, dW, db

    def rnn_backward(self, dh, cache):
        dU = 0
        dW = 0
        db = 0
        dh_prev = 0
        #print dh.shape

        for t in range(self.sequence_length-1, 0, -1):
            dh_t = dh[:,t,:] + dh_prev
            dh_prev, du, dw, db = self.rnn_step_backward(dh_t, cache[t])
            
            dU += np.clip(du, -5, 5)
            dW += np.clip(dw, -5, 5)
            db += np.clip(db, -5, 5)


        return dU, dW, db

    @staticmethod
    def output(h, V, c):
        o = np.zeros((h.shape[0], h.shape[1] - 1, V.shape[0]))

        for t in range(1, h.shape[1]):
            o_t = np.dot(h[:,t,:], V.T) + c.T

            o[:,t-1,:] = o_t

        return o

    @staticmethod
    def softmax(s):
        exps = np.exp(s)
        return exps / np.sum(exps, axis=2, keepdims=True)

    @staticmethod
    def loss(y, o):
        return np.log(np.sum(np.exp(o), axis=1)) - np.sum(y * o, axis=1)

    def output_loss_and_grads(self, h, V, c, y):
        o = RNN.output(h, V, c)
        yhat = RNN.softmax(o)
        loss = RNN.loss(y, o)
        dh = np.dot((yhat - o), V)

        dV = 0
        dc = 0

        do = yhat - y

        for t in range(self.sequence_length):
            dV += np.clip(np.dot(do[:,t,:].T, h[:,t+1,:]), -5, 5)
            dc += np.clip(np.sum(do[:,t,:], keepdims=True), -5, 5)

        return loss, dh, dV, dc
    
    def update(self, dU, dW, db, dV, dc):
        # print self.U.shape, dU.shape, self.memory_U.shape
        # print self.W.shape, dW.shape, self.memory_W.shape
        # print self.b.shape, db.shape, self.memory_b.shape
        # print self.V.shape, dV.shape, self.memory_V.shape
        # print self.c.shape, dc.shape, self.memory_c.shape

        self.memory_U += np.square(dU)
        self.memory_b += np.square(db.T)
        self.memory_W += np.square(dW)
        self.memory_V += np.square(dV)
        self.memory_c += np.square(dc.T)

        self.U -= self.learning_rate * dU / (np.sqrt(self.memory_U + self.theta))
        self.b -= self.learning_rate * db.T / (np.sqrt(self.memory_b + self.theta))
        self.W -= self.learning_rate * dW / (np.sqrt(self.memory_W + self.theta))
        self.V -= self.learning_rate * dV / (np.sqrt(self.memory_V + self.theta))
        self.c -= self.learning_rate * dc.T / (np.sqrt(self.memory_c + self.theta))

    def step(self, h0, x, y):
        h, cache = RNN.rnn_forward(x, h0, self.U, self.W, self.b)
        loss, dh, dV, dc = self.output_loss_and_grads(h, self.V, self.c, y)
        print 'Loss: {}'.format(np.mean(loss))
        dU, dW, db = self.rnn_backward(dh, cache)
        self.update(dU, dW, db, dV, dc)

        return loss, h[:,-1,:]

    def sample(self, seed='HAN:\nIs that good or bad?\n\n', n_sample=300):
        h0 = np.zeros((self.hidden_size, self.hidden_size))
        x = db.encode(seed)

        h, _ = RNN.rnn_forward(np.array([x]), h0, self.U, self.W, self.b)
        o = RNN.output(h, self.V, self.c)
        s = RNN.softmax(o)[0]
        s = np.argmax(s, axis=1)
        c = db.decode(s)[-1]
        sentence = [c]
        for i in range(n_sample - len(seed)):
            x = db.encode(c)
            h, _ = RNN.rnn_forward(np.array([x]), h[:,-1,:], self.U, self.W, self.b)
            o = RNN.output(h, self.V, self.c)
            s = RNN.softmax(o)[0]
            s = np.argmax(s, axis=1)
            c = db.decode(s)[0]
            sentence.append(c)

        return ''.join(sentence)


def run_language_model(dataset, max_epochs, hidden_size=100, sequence_length=30, learning_rate=1e-3, sample_every=50):
    
    vocab_size = dataset.vocab_size
    rnn = RNN(hidden_size, sequence_length, vocab_size, learning_rate)

    current_epoch = 0 
    batch = 0

    h0 = np.zeros((batch_size, hidden_size))

    print rnn.sample()
    while current_epoch < max_epochs: 
        print 'current_epoch: {}'.format(current_epoch)

        b = 0
        for x, y in dataset.batches():
            loss, h0 = rnn.step(h0, x, y)
            b += 1
            if b % sample_every == 0:
                print rnn.sample()
                

        current_epoch += 1
        h0 = np.zeros((batch_size, hidden_size))
    
   

db = Database(batch_size, 30)



if __name__ == '__main__':
    file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'selected_conversations.txt')
    db.preprocess(file_path)

    run_language_model(db, 30)