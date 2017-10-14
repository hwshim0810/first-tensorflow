import tensorflow as tf

tf.set_random_seed(777)

x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# 가설 : 선형으로 추측
hypothesis = x_train * W + b

# Cost function : 가설에 의한 값과 주어진 데이터의 차를 이용
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()

# Initializes global variables
sess.run(tf.global_variables_initializer())

for step in range(2222):
    sess.run(train)

    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))

# 기대값 : W[1.], b:[0.]
