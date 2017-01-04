from database import Database


class RNN(object):

    def __init__(self, hidden_size, sequence_length, vocab_size, learning_rate):
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate

        self.U = np.random.normal(scale=1e-2, size=(hidden_size, vocab_size)) # ... input projection
        self.W = np.random.normal(scale=1e-2, size=(hidden_size, hidden_size)) # ... hidden-to-hidden projection
        self.b = np.zeros((hidden_size, 1)) # ... input bieas        
        
        self.V = np.random.normal(scale=1e-2, size=(vocab_size, hidden_size)) # ... output projection
        self.c = np.zeros((vocab_size, 1)) # ... output bias

        # memory of past gradients - rolling sum of squares for Adagrad
        self.memory_U, self.memory_W, self.memory_V = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V)
        self.memory_b, self.memory_c = np.zeros_like(self.b), np.zeros_like(self.c)

    @staticmethod
    def rnn_step_forward(x, h_prev, U, W, b):
        # A single time step forward of a recurrent neural network with a 
        # hyperbolic tangent nonlinearity.

        # x - input data (minibatch size x input dimension)
        # h_prev - previous hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)

        h = np.dot(W, h_prev.T) + np.dot(U, x.T) + b
        h_current = np.tanh(h)
        # return the new hidden state and a tuple of values needed for the backward step

        return h_current, (h_current, h_prev, x)
    
    @staticmethod
    def rnn_forward(x, h0, U, W, b):
        # Full unroll forward of the recurrent neural network with a 
        # hyperbolic tangent nonlinearity

        # x - input data for the whole time-series (minibatch size x sequence_length x input dimension)
        # h0 - initial hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)
        cache = []
        h = np.zeros(h0.shape[0], x.shape[1]+1, h0.shape[1])
        h[:,0,:] = h0
        
        for t in range(x.shape[1]):
            h_t, cache_t = RNN.rnn_step_forward(x[:,t,:], h[:,t,:], U, W, b)
            h[:,t+1,:] = h_t
            cache.append(cache_t)

    # return the hidden states for the whole time series (T+1) and a tuple of values needed for the backward step

    return h, cache


    def rnn_step_backward(self, grad_next, cache):
        # A single time step backward of a recurrent neural network with a 
        # hyperbolic tangent nonlinearity.

        # grad_next - upstream gradient of the loss with respect to the next hidden state and current output
        # cache - cached information from the forward pass
        h = cache[0]
        h_prev = cache[1]
        x = cache[2]

        dtanh = 1 - h**2

        dh_prev = np.dot(grad_next, np.dot(dtanh, W))
        dU = np.dot(grad_next, np.dot(dtanh, x))
        dW = np.dot(grad_next, np.dot(dtanh, h_prev))
        db = np.sum(np.dot(grad_next, dtanh), axis=0, keepdims=True)
        
        # compute and return gradients with respect to each parameter
        # HINT: you can use the chain rule to compute the derivative of the
        # hyperbolic tangent function and use it to compute the gradient
        # with respect to the remaining parameters

        return dh_prev, dU, dW, db

    def rnn_backward(self, dh, cache):
        # Full unroll forward of the recurrent neural network with a 
        # hyperbolic tangent nonlinearity

        dU = np.zeros((self.hidden_size, self.vocab_size))
        dW = np.zeros((self.hidden_size, self.hidden_size))
        db = np.zeros((self.hidden_size, 1))
        dh_prev = 0

        for t in range(self.sequence_length-1, 0, -1):
            dh_t = dh[:,t,:] + dh_prev
            dh_prev, du, dw, db = self.rnn_step_backward(dh_t, cache)

            dU += np.clip(du, -5, 5)
            dW += np.clip(dw, -5, 5)
            db += np.clip(db, -5, 5)
        # compute and return gradients with respect to each parameter
        # for the whole time series.
        # Why are we not computing the gradient with respect to inputs (x)?

        return dU, dW, db