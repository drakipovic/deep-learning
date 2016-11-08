import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.preprocessing import OneHotEncoder

from tf_deep import TFDeep

tf.app.flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')
mnist = input_data.read_data_sets(tf.app.flags.FLAGS.data_dir, one_hot=True)


N=mnist.train.images.shape[0]
D=mnist.train.images.shape[1]
C=mnist.train.labels.shape[1]

layers = [D, C]
tf_deep = TFDeep(layers)

tf_deep.train(mnist.train.images, mnist.train.labels, 1000)

probs = deep.eval(mnist.train.images)
Y = np.argmax(probs[0], axis=1)

print eval_perf_binary(Y, y)