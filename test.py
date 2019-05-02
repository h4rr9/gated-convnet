import tensorflow as tf
import layers
import numpy as np

X = np.tile(np.arange(1800), 4).reshape(4,6,300)
A = tf.Variable(X, dtype=tf.float32)
print(A.shape)
# out = layers.conv1d(A, filters=10, k_size=1, strides=1, padding='VALID')
padding = tf.constant([[0,0],[2,0],[0,0]])

out = tf.pad(A, padding)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    pad = sess.run(out)
