import tensorflow as tf

tf.set_random_seed(747)

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = X * W

# Cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(22):
    sess.run(update, feed_dict={X: x_data, Y: y_data})  # 새로 구한 W 대입
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))
