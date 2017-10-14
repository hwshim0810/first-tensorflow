"""
Tensor ver
"""
import tensorflow as tf

tf.set_random_seed(747)

W = tf.Variable(5.0)
X = [1, 2, 3]
Y = [1, 2, 3]

hypothesis = X * W

# Cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

sess = tf.Session()

# Minimize => Tensor
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

for step in range(22):
    print(step, sess.run(W))
    sess.run(train)
