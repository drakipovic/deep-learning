import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import sympy


s_a, s_b, s_x, s_y = sympy.var('a,b,x,y')
s_l = sympy.Matrix([(s_a * s_x + s_b - s_y)**2])
dl = s_l.jacobian([s_a, s_b])
sympy.pprint(dl)


X  = tf.placeholder(tf.float32, [None])
Y_ = tf.placeholder(tf.float32, [None])
a = tf.Variable(0.0)
b = tf.Variable(0.0)

# afini regresijski model
Y = a * X + b

# kvadratni gubitak
loss = (Y-Y_)**2
dl_a = 2 * X * (a * X + b - Y_)
dl_b = 2 * a * X + 2 * b - 2 * Y_

# optimizacijski postupak: gradijentni spust
trainer = tf.train.GradientDescentOptimizer(0.001)

grads_and_vars = trainer.compute_gradients(loss, [a, b])
g = [grad[0] for grad in grads_and_vars]
apply_g = trainer.apply_gradients(grads_and_vars)

g = tf.Print(g, [g], message="Grad", first_n=5)


#train_op = trainer.minimize(loss)

## 2. inicijalizacija parametara
sess = tf.Session()
sess.run(tf.initialize_all_variables())


for i in range(1000):
  val_loss, _, val_a, val_b, val_grad, val_dl_a, val_dl_b = sess.run([loss, apply_g, a, b, g, dl_a, dl_b], feed_dict={X: [0,1,2], Y_: [1,3,5]})
  if i % 100 == 0:
    print i, val_loss, val_a, val_b, val_grad, val_dl_a, val_dl_b