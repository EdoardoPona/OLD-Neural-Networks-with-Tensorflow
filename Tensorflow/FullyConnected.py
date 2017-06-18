"""A fully connected network for MNIST"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])     # placeholder for inputs (28x28 images)
y_ = tf.placeholder(tf.float32, shape=[None, 10])     # placeholder for labels , 10 classes

""" when using multiple layers, the weights should start as tf.random_normal, not tf.zeros, or accuracy will drastically drop
read:  http://stackoverflow.com/questions/41993311/adding-more-layers-to-tensorflow-mnist-tutorial-makes-accuracy-drop"""

W0 = tf.Variable(tf.random_normal([784, 256], stddev=0.2))
b0 = tf.Variable(tf.constant(0.1, shape=[256]))     # initializing the biases as 0.1 because we are using relu, to avoid dead neurons

W = tf.Variable(tf.random_normal([256, 10], stddev=0.2))       # 784 inputs and 10 outputs
b = tf.Variable(tf.constant(0.1, shape=[10]))

h0 = tf.nn.relu(tf.add(tf.matmul(x, W0), b0))       # hidden_layer
y = tf.add(tf.matmul(h0, W), b)         # prediction

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))       # loss function
# cost = tf.abs(y - y_)

train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cost)        # training step, using one of tf's optimizers

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# training
for i in xrange(1000):
    batch = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))       # list of booleans that say wether prediction was correct or not

    if i % 30 == 0 or i == 1000:
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}, session=sess))
