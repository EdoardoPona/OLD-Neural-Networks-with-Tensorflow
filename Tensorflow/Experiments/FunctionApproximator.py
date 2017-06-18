import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random


def weight(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.2))


def bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def plot(img):
    plt.imshow(img)
    plt.show()

x = tf.placeholder(tf.float32, [1, 1])
y_ = tf.placeholder(tf.float32, [1, 1])

# weights and biases
a = weight([1, 1])
b = weight([1, 1])


y = tf.matmul(x, a) + b 

cost = tf.abs(y - y_)
train_step = tf.train.AdamOptimizer(1e-3).minimize(cost)
run_list = [cost, y, train_step]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

m = 2
p = 4

for i in xrange(int(1e4)):
    val = random.randint(0, 20)
    y = m*val + p
    cost, out, _ = sess.run(run_list, feed_dict={x: np.array([[val]]), y_: np.array([[y]])})
    if i % 500 == 0:
        print cost, y, out
