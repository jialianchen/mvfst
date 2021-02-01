# train.py
import os
import sys
import numpy as np
import tensorflow as tf

if __name__ == '__main__':

    config = tf.ConfigProto(log_device_placement=False,allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    train_dir = os.path.join('demo_model/', "demo")

    x = tf.placeholder(dtype=tf.int32, shape=[None, 2], name='x')

    w = tf.Variable(tf.ones(shape=[2, 1], dtype=tf.int32), dtype=tf.int32, name='w')

    # a * w
    res = tf.matmul(x, w, name='res')

    with tf.Session(config=config) as sess:

        feed_dict = dict()
        feed_dict[x] = [[1, 2],[3, 4]]
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # get results and save model
        res = sess.run(feed_dict=feed_dict, fetches=[res])
        saver.save(sess, train_dir)

        print("result: ", res[0])
        
    