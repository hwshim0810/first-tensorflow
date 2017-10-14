import tensorflow as tf

tf.set_random_seed(222)

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Placeholder 를 이용
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

hypothesis = X * W + b

# Cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()

# Initializes global variables
sess.run(tf.global_variables_initializer())

for step in range(2222):
    cost_val, W_val, b_val, _ = sess.run(
        [cost, W, b, train],
        feed_dict={X: [1, 2, 3], Y: [1, 2, 3]}
    )

    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

# 기대값 : W[1.], b:[0.]

# Test Model

print(sess.run(hypothesis, feed_dict={X: [5]}))
print(sess.run(hypothesis, feed_dict={X: [2.2]}))
print(sess.run(hypothesis, feed_dict={X: [1.2, 2.2]}))

for step in range(2002):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, train],
                 feed_dict={X: [1, 2, 3, 4, 5],
                            Y: [2.1, 3.1, 4.1, 5.1, 6.1]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

# Test model
print(sess.run(hypothesis, feed_dict={X: [5]}))
print(sess.run(hypothesis, feed_dict={X: [2.5]}))
print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))
