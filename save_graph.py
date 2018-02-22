# -*- coding: utf-8 -*-

import tensorflow as tf
from board import BVCNT, FEATURE_CNT
import model

with tf.get_default_graph().as_default():
    dn = model.DualNetwork()
    x = tf.placeholder("float", shape=[None, BVCNT, FEATURE_CNT], name="x")
    pv = dn.model(x, temp=0.7, dr=1.0)
    sess = dn.create_sess("pre_train/model.ckpt")
    dn.save_graph()
