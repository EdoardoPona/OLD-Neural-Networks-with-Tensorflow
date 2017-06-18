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

x = tf.placeholder(tf.float32, [1, 1])      # positional index in the sequence
y = tf.placeholder(tf.float32, [1, 1])      # actual value in the sequence


# weights and biases
w0 = weight([1, 1])
b0 = bias([1, 1])

w1 = weight([1, 1])
b1 = bias([1, 1])

w2 = weight([1, 1])
b2 = bias([1, 1])

# model
h0 = tf.nn.relu(tf.matmul(x, w0) + b0)          # one layer is enough for linear sequences
h1 = tf.nn.relu(tf.matmul(h0, w1) + b1)
out = tf.matmul(h1, w2) + b2

# training
cost = tf.abs(out - y)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
run_list = [cost, out, train_step]

sess = tf.Session()
sess.run(tf.global_variables_initializer())


seq = [[[3]], [[6]], [[9]], [[12]], [[15]]]
n = [[[1]], [[2]], [[3]], [[4]], [[5]]]

for i in xrange(int(2e5)):
    index = random.randint(0, 4)
    cost, out, _ = sess.run(run_list, feed_dict={x: n[index], y: seq[index]})
    if i % 3000 == 0:
        print 'Step: ', i, 'Real:', seq[index], 'Predicted: ', out,  'Cost: ', cost

cost, out, _ = sess.run(run_list, feed_dict={x: [[6]], y: [[18]]})
weight = sess.run(w0)
bias = sess.run(b0)
print 'Real:', [[36]], 'Predicted: ', out,  'Cost: ', cost
print 'Weight: ', weight
print 'Bias: ', bias
