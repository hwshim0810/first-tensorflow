import tensorflow as tf

tf.set_random_seed(747)

filename_queue = tf.train.string_input_producer(
    ['data-02.csv'], shuffle=False, name='filename_queue')  # Not Shuffle

# 파일이름 Queue 생성 후 reader 를 통해 읽어서 Decoder(Csv, binary 등등 변환) 전달
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# 기본 Column 타입 및 값 정의
record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

# Collect batches of csv in
train_x_batch, train_y_batch = \
    tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Start populating the filename queue
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2002):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])  # 읽은 값
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

coord.request_stop()
coord.join(threads)

# # Shuffle Batch
# min_after_dequeue = 10000   # 클수룩 많이 섞음: 느려지고 많은 메모리 사용
#
# """
# Capacity must be larger than min_after_dequeue
# and the amount larger determines the maximum we will prefetch.
#
# Recommendation)
# min_after_dequeue + (num_threads + a small safety margin) * batch_size
# """
# capacity = min_after_dequeue + 3 * batch_size
# example_batch, label_batch = tf.train.shuffle_batch(
#     [example, label], batch_size=batch_size, capacity=capacity,
#     min_after_dequeue=min_after_dequeue
# )
