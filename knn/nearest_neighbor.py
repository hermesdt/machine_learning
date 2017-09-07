import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data", one_hot=True)

train_images, train_labels = mnist.train.next_batch(5000)
test_images, test_labels = mnist.test.next_batch(200)

x_train = tf.placeholder(tf.float32, [None, 784])
x_test = tf.placeholder(tf.float32)

diff = x_train - x_test
abs = tf.abs(diff)
pred = tf.reduce_sum(abs, axis = 1)

init = tf.global_variables_initializer()

accuracy = 0.

with tf.Session() as sess:
    sess.run(init)

    for i in range(len(test_images)):
        test_image, test_label = test_images[i], test_labels[i]
        test_label = np.argmax(test_label)
        distances = sess.run(pred, {x_train: train_images, x_test: test_image})
        best_index = np.argmin(distances)

        predicted_label = np.argmax(train_labels[best_index])

        if predicted_label == test_label:
            accuracy += 1. / len(test_labels)

    print("accuracy: {}".format(accuracy))


# tf.abs(tf.)
# sum = tf.reduce_sum()
# pred = tf.reduce_sum()
