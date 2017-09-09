import tensorflow as tf
import numpy as np

X = tf.placeholder(tf.float32, [None, 3], name="X")
y = tf.placeholder(tf.float32, name="y")
theta = tf.Variable(tf.zeros((3, 1), tf.float32), name="theta")
one = tf.constant(1., tf.float32, name="one")

numerator = one
denominator = one + tf.exp(-tf.matmul(X, theta))
sigmoid = numerator / denominator

# -y.*log.(Utils.sigmoid(X, θ)) .- (1 .- y).*log.(1 - Utils.sigmoid(X, θ))
sess = tf.Session()

y_1 = y * tf.log(sigmoid)
y_0 = (one - y) * tf.log(one - sigmoid)

cost = tf.reduce_mean(-y_1 - y_0)
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

x_data = [[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]]
y_data = [[1], [0], [0], [1]]

init = tf.global_variables_initializer()
sess.run(init)
for i in range(1000):
    sess.run(optimizer, {X: x_data, y: y_data})
    print("cost", sess.run(cost, {X: x_data, y: y_data}))



print("theta", sess.run(theta))
print("sigmoid", sess.run(sigmoid, {X: x_data, y: y_data}))
