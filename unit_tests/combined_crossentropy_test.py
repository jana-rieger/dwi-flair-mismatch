import unittest
import numpy as np
import tensorflow as tf

from tensorflow.keras.losses import binary_crossentropy, sparse_categorical_crossentropy, \
    SparseCategoricalCrossentropy, BinaryCrossentropy
from ADT.losses import CombinedCrossentropy


class TestCombinedCrossentropy(unittest.TestCase):
    def setUp(self):
        self.dich_class_threshold = 2
        self.gt = tf.convert_to_tensor(np.array([0, 1, 2, 1, 0, 2, 0, 1]), dtype=tf.float32)
        self.pred = tf.convert_to_tensor(np.array([[0.7, 0.3, 0],
                                                   [0.1, 0.3, 0.6],
                                                   [0.3, 0.6, 0.1],
                                                   [0.2, 0.5, 0.3],
                                                   [0.4, 0.25, 0.35],
                                                   [0, 0, 1],
                                                   [1, 0, 0],
                                                   [0, 1, 0]]), dtype=tf.float32)
        self.b_gt = tf.math.greater(self.gt, tf.constant([self.dich_class_threshold], dtype=tf.float32))
        self.b_pred = tf.clip_by_value(tf.reduce_sum(self.pred[:, self.dich_class_threshold + 1:], axis=1),
                                       clip_value_min=0, clip_value_max=1)

    def test_just_categorical(self):
        loss = CombinedCrossentropy(alpha=1, dich_class_threshold=self.dich_class_threshold)
        self.assertEqual(loss(self.gt, self.pred),
                         tf.reduce_mean(sparse_categorical_crossentropy(self.gt, self.pred)))

    def test_just_binary(self):
        loss = CombinedCrossentropy(alpha=0, dich_class_threshold=self.dich_class_threshold)
        self.assertEqual(loss(self.gt, self.pred),
                         tf.reduce_mean(binary_crossentropy(self.b_gt, self.b_pred)))


if __name__ == '__main__':
    unittest.main()
