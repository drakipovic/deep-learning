import os
from collections import Counter

import numpy as np


class Database(object):

    def __init__(self, batch_size, sequence_length):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
    
    def preprocess(self, input_file):
        with open(input_file, "r") as f:
            data = f.read()

        sorted_chars = [c[0] for c in Counter(data).most_common()]

        self.voc_size = len(sorted_chars)
        self.char2id = dict(zip(sorted_chars, range(len(sorted_chars))))
        self.id2char = {k:v for v,k in self.char2id.items()}

        self.x = self.encode(data)
        print self.x

    def encode(self, sequence):
        x = np.array(list(map(self.char2id.get, sequence)))

        ret = np.zeros((x.shape[0], self.voc_size))
        ret[np.arange(x.shape[0]), x] = 1
        
        return ret

    def decode(self, encoded_sequence):
        return np.array(list(map(self.id2char.get, encoded_sequence)))

    def batches(self):
        return BatchIterator(self.x, self.batch_size, self.sequence_length, self.voc_size)


class BatchIterator(object):

    def __init__(self, x, batch_size, seq_len, voc_size):
        self.x = x[:-1]
        self.y = x[1:]
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.voc_size = voc_size
        self.start = 0
        self.end = len(x)
        self.in_one_batch = seq_len * batch_size
        self.shape = (batch_size, seq_len, voc_size)

    def __iter__(self):
        return self

    def next(self):
        self.start += self.in_one_batch
        if self.start >= self.end:
            self.start = 0
            raise StopIteration
        
        return self.x[self.start-self.in_one_batch:self.start].reshape(self.shape), self.y[self.start-self.in_one_batch:self.start].reshape(self.shape)


if __name__ == '__main__':
    file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'selected_conversations.txt')

    db = Database(100, 30)

    db.preprocess(file_path)
    print '-' * 70
    batches = db.batches()
    for x, y in batches:
        print x
        print '*' * 70
        print y
        print '-' * 70