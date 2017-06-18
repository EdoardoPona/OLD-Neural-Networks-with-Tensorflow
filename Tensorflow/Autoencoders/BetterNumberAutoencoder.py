"""NumberAutoencoder.py, but coded better, look more like some final product"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def make_data(shape):
    return np.random.randint(low=10, high=30, size=shape)


def weight(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.2))


def bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

input_len = 784
batch_size = 50

input_shape = [batch_size, input_len]
x = tf.placeholder(tf.float32, shape=[batch_size, input_len])

# model_shape = [input_len, 50, 30, 20, 15]     for input_len = 25
# model_shape = [input_len, 120, 80, 50, 30]    for input_len = 50
model_shape = [input_len, 500, 400, 300, 250]
print 'Model shape: ', model_shape

weights = [weight([model_shape[i], model_shape[i+1]]) for i in xrange(len(model_shape) - 1)]
biases = [bias([model_shape[i]]) for i in xrange(len(model_shape))]

# encoder
h = x
for i in xrange(len(model_shape) - 2):
    h = tf.nn.relu(tf.matmul(h, weights[i]) + biases[i+1])

latent_vector = tf.matmul(h, weights[i+1]) + biases[i+2]      # last layer without activation, it's the latent vector

weights.reverse()
biases.reverse()

# decoder
h = latent_vector
for i in xrange(len(model_shape) - 2):
    h = tf.nn.relu(tf.matmul(h, weights[i], transpose_b=True) - biases[i+1])

output = tf.matmul(h, weights[i+1], transpose_b=True) - biases[i+2]

cost = tf.reduce_mean(tf.abs(x - output))
train_step = tf.train.AdamOptimizer(5e-4).minimize(cost)

run_list = [train_step, cost]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# training
for i in xrange(int(1e4)):
    data = mnist.train.next_batch(batch_size)[0]        # [0] are the images, [1] are the labels which we don't need
    _, cost = sess.run(run_list, feed_dict={x: data})           # performing train_step, and getting the cost
    if i % 250 == 0:
        print 'Iteration: {0} \n Cost: {1}'.format(i, cost)

# array = make_data([1, input_len])
array = mnist.train.next_batch(1)[0]
output = np.reshape(sess.run(output, feed_dict={x: array}), [28, 28])
array = np.reshape(array, [28, 28])

plt.imshow(array)
plt.show()
plt.imshow(output)
plt.show()
