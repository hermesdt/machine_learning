import tensorflow as tf
import numpy as np

sess = tf.Session()
tf.contrib.

def convert_to_binary(vector, digit):
    # vector.

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data", one_hot=True)

train_images, train_labels = mnist.train.next_batch(500)
test_images, test_labels = mnist.test.next_batch(100)

x_train = tf.placeholder(tf.float32, [None, 784])
x_test = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32, [None, 3], name="X")
y = tf.placeholder(tf.float32, name="y")
theta = tf.Variable(tf.zeros((3, 1), tf.float32), name="theta")
one = tf.constant(1., tf.float32, name="one")

numerator = one
denominator = one + tf.exp(-tf.matmul(X, theta))
sigmoid = numerator / denominator


sess = tf.Session()

y_1 = y * tf.log(sigmoid)
y_0 = (one - y) * tf.log(one - sigmoid)

cost = tf.reduce_mean(-y_1 - y_0)
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

x_data = [[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]]
y_data = [[1], [0], [0], [1]]
x_data = train_images
y_data = train_labels

init = tf.global_variables_initializer()
sess.run(init)
for i in range(1000):
    sess.run(optimizer, {X: x_data, y: y_data})
    print("cost", sess.run(cost, {X: x_data, y: y_data}))



print("theta", sess.run(theta))
print("sigmoid", sess.run(sigmoid, {X: x_data, y: y_data}))
