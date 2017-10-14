import tensorflow as tf

# Import Err tkinter
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt

tf.set_random_seed(747)

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)
hypothesis = X * W

# Cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

sess = tf.Session()

W_history = []
cost_history = []

for i in range(-22, 22):
    curr_W = i * 0.1
    curr_cost = sess.run(cost, feed_dict={W: curr_W})

    W_history.append(curr_W)
    cost_history.append(curr_cost)

# Show
plt.plot(W_history, cost_history)
plt.show()
