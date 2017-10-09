import tensorflow as tf

nd1 = tf.placeholder(tf.float32)
nd2 = tf.placeholder(tf.float32)

add_nd = nd1 + nd2  # provides a shortcut for tf.add(a,b)

sess = tf.Session()

print(sess.run(add_nd, feed_dict={nd1: 2, nd2: 20}))
print(sess.run(add_nd, feed_dict={nd1: [2, 20], nd2: [0.2, 2.2]}))

print('===============================')

add_triple = add_nd * 3

print(sess.run(add_triple, feed_dict={nd1: 5, nd2: 2}))
