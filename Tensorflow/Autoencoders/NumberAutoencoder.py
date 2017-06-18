"""A simple autoencoder that reconstructs a list of numbers"""
import tensorflow as tf
import numpy as np


def make_data(batch_size):
    return np.random.randint(low=10, high=30, size=(batch_size, 25))

x = tf.placeholder(tf.float32, shape=[None, 25])

# weights and biases
# we don't use b_dx biases at the moment, since we are experimenting with -b_ex+1, which seems to work better
w0 = tf.Variable(tf.truncated_normal([25, 40], stddev=0.2))
b_e0 = tf.Variable(tf.constant(0.1, shape=[40]))
b_d0 = tf.Variable(tf.constant(0.1, shape=[25]))

w1 = tf.Variable(tf.truncated_normal([40, 30], stddev=0.2))
b_e1 = tf.Variable(tf.constant(0.1, shape=[30]))
b_d1 = tf.Variable(tf.constant(0.1, shape=[40]))

w2 = tf.Variable(tf.truncated_normal([30, 20], stddev=0.2))
b_e2 = tf.Variable(tf.constant(0.1, shape=[20]))
b_d2 = tf.Variable(tf.constant(0.1, shape=[30]))

w3 = tf.Variable(tf.truncated_normal([20, 15], stddev=0.2))
b_e3 = tf.Variable(tf.constant(0.1, shape=[15]))
b_d3 = tf.Variable(tf.constant(0.1, shape=[20]))

# encoder
e0 = tf.nn.relu(tf.matmul(x, w0) + b_e0)
e1 = tf.nn.relu(tf.matmul(e0, w1) + b_e1)
e2 = tf.nn.relu(tf.matmul(e1, w2) + b_e2)
latent_vector = tf.matmul(e2, w3) + b_e3

# decoder
# we could try using the same biases, but one layer before, and only use a new one for the output (experiment)
# or we could use new biases for everything

d0 = tf.nn.relu(tf.matmul(latent_vector, w3, transpose_b=True) - b_e2)
d1 = tf.nn.relu(tf.matmul(d0, w2, transpose_b=True) - b_e1)
d2 = tf.nn.relu(tf.matmul(d1, w1, transpose_b=True) - b_e0)
output = tf.matmul(d2, w0, transpose_b=True) + b_d0

cost = tf.reduce_mean(tf.abs(x - output))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cost)

run_list = [train_step, cost]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# training
for i in xrange(int(5e4)):
    data = make_data(50)
    _, cost = sess.run(run_list, feed_dict={x: data})
    if i % 500 == 0:
        print 'Iteration: {0} \n Cost: {1}'.format(i, cost)

# testing once 
array = make_data(1)
print array
output = sess.run(output, feed_dict={x: array}).astype(np.int32)
print output
