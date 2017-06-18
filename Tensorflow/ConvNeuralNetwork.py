"""Convolutional Neural Network for MNIST"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# helper functions for creating weights and biases
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


# helper functions for convolution and pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def avg_pool(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, shape=[None, 784])     # placeholder for inputs (28x28 images)
y_ = tf.placeholder(tf.float32, shape=[None, 10])     # placeholder for labels , 10 classes

# creating the graph
# weights and biases
w_conv1 = weight_variable([5, 5, 1, 32])            # the first two dimensions are the patch size, then the n of inputs and outputs
b_conv1 = bias_variable([32])

w_conv2 = weight_variable([4, 4, 32, 50])
b_conv2 = bias_variable([50])

w_conv3 = weight_variable([4, 4, 50, 80])
b_conv3 = bias_variable([80])

w_fc1 = weight_variable([7*7*80, 1024])         # the first dim is 7*7*64 because the image has gone through two pooling layers,
# with k=2, so the image size is halved each time. 64 is the output of the conv layer before this
b_fc1 = bias_variable([1024])

w_fc2 = weight_variable([1024, 10])     # final weight variable, readout layer
b_fc2 = bias_variable([10])

x_image = tf.reshape(x, [-1, 28, 28, 1])        # 1 is the color channel

h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1)

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_conv3 = tf.nn.relu(conv2d(h_conv2, w_conv3) + b_conv3)
h_pool2 = max_pool(h_conv3)

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*80])        # flattening so we can feed it in the fc layer
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# dropout before the readout layer, prevents overfitting on larger models
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

y = tf.matmul(h_fc1_drop, w_fc2) + b_fc2            # prediction

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cost)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in xrange(3000):
    batch = mnist.train.next_batch(150)       # training on mnist data from tensorflow

    if i % 10 == 0:
        acc = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}, session=sess)
        print 'Step: ', i, '\t Accuracy:', acc

    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
