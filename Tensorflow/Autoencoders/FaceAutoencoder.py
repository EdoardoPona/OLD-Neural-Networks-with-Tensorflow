"""A convolutional autoencoder trained to reconstruct my face"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils
import cv2


# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# images, _, _ = utils.get_data('data.csv', flat=False, rgb=False)

im_w = 47
im_h = 70
im_ch = 1      # number of color channels

images = utils.get_image_data('me/1', size=[im_w, im_h])
print 'Finished loading data'


# helper functions for weights and biases
def weight(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.2))


def bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


# helper functions for convolution and pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def deconv2d(x, W, shape):
    return tf.nn.conv2d_transpose(x, W, strides=[1, 1, 1, 1], padding='SAME', output_shape=shape)


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def avg_pool(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def save_image(image, filename):
    cv2.imwrite(filename, image)


def plot(img):
    plt.imshow(img)
    plt.show()


batch_size = 25
x = tf.placeholder(tf.float32, shape=[None, im_h, im_w, im_ch])     

# encoder weights and biases
# w_conv1 = weight([5, 5, im_ch, 300])      for mnist
w_conv1 = weight([12, 12, im_ch, 1700])     # 2500 for face_autoencoder0.ckpt, it performs well because the 2500 is very big
b_conv1 = bias([1700])

b_deconv1 = bias([im_ch])


# encoder
h_conv1 = tf.nn.relu(conv2d(x, w_conv1) + b_conv1)


# decoder
output = tf.nn.relu(deconv2d(h_conv1, w_conv1, [tf.shape(x)[0], im_h, im_w, im_ch]) + b_deconv1)    # shouldn't really use relu here,
# but it gives better results


# cost and optimizer

cost = tf.reduce_mean(tf.abs(output - x))
# cost = tf.div(tf.reduce_mean(tf.square(tf.subtract(output, x))), 2)   # mean squared error
train_step = tf.train.AdamOptimizer(1e-3).minimize(cost)

run_list = [cost, train_step]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(write_version=tf.train.SaverDef.V1, max_to_keep=1)       # saving variables 
saver.restore(sess=sess, save_path='Checkpoints/face_autoencoder1.ckpt')


# training
def train(iter_num):
    print 'Starting Training'
    for i in xrange(iter_num):
        # data = mnist.train.next_batch(batch_size)[0]        # [0] are the images, [1] are the labels which we don't need
        data = utils.get_batch_only_imgaes(batch_size, images)
        data = np.reshape(data, [data.shape[0], data.shape[1], data.shape[2], 1])
        cost, _ = sess.run(run_list, feed_dict={x: data})           # performing train_step, getting the cost

        if i % 3 == 0:
            print 'Iteration: {0} Cost: {1}'.format(i, cost)

        if i % 15 == 0 and i > 14:
            saver.save(sess, 'Checkpoints/face_autoencoder1.ckpt')

    saver.save(sess, 'Checkpoints/face_autoencoder1.ckpt')

# train(5000)

# testing once 
input_image = np.array(cv2.resize(cv2.imread('me/1/0.jpg', 0), (47, 70)))
input_image = np.reshape(input_image, [1, 70, 47, 1])

output = sess.run(output, feed_dict={x: input_image})
output = np.reshape(output, [output.shape[1], output.shape[2]]).astype(np.float32)

save_image(output, 'me/output3.jpg')

plot(np.reshape(input_image, [input_image.shape[1], input_image.shape[2]]).astype(np.float32))
plot(output)
