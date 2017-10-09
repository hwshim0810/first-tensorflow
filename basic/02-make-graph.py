import tensorflow as tf

nd1 = tf.constant(3.0, tf.float32)
nd2 = tf.constant(4.0)  # also tf.float32 implicitly
nd3 = tf.add(nd1, nd2)

sess = tf.Session()

print(sess.run([nd1, nd2]))
print(sess.run(nd3))
