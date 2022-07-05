import numpy as np
import tensorflow as tf
from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d, gconv2d_util

# Construct graph
x = tf.keras.backend.placeholder(shape=[None, 9, 9, 3])

gconv_indices, gconv_shape_info, w_shape = gconv2d_util(
    h_input='Z2', h_output='D4', in_channels=3, out_channels=64, ksize=3)
w = tf.Variable(tf.random.truncated_normal(w_shape, stddev=1.))
y = gconv2d(input=x, filter=w, strides=[1, 1, 1, 1], padding='SAME',
            gconv_indices=gconv_indices, gconv_shape_info=gconv_shape_info)

gconv_indices, gconv_shape_info, w_shape = gconv2d_util(
    h_input='D4', h_output='D4', in_channels=64, out_channels=64, ksize=3)
w = tf.Variable(tf.random.truncated_normal(w_shape, stddev=1.))
y = gconv2d(input=y, filter=w, strides=[1, 1, 1, 1], padding='SAME',
            gconv_indices=gconv_indices, gconv_shape_info=gconv_shape_info)

# Compute
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
y = sess.run(y, feed_dict={x: np.random.randn(10, 9, 9, 3)})
sess.close()

print(y.shape) # (10, 9, 9, 512)