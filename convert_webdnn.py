import tensorflow as tf
from webdnn.frontend.tensorflow import TensorFlowConverter
from webdnn.backend import generate_descriptor
from board import BVCNT, FEATURE_CNT
import model

with tf.get_default_graph().as_default():
    dn = model.DualNetwork()
    x = tf.placeholder("float", shape=[None, BVCNT, FEATURE_CNT], name="x")
    p, v = dn.model(x, temp=0.7, dr=1.0)
    sess = dn.create_sess("pre_train/model.ckpt")
    graph = TensorFlowConverter(sess).convert([x], [p, v])
    print("generating webgpu...")
    exec_info = generate_descriptor("webgpu", graph)
    exec_info.save("./output")
    print("done")
    print("generating webgl...")
    exec_info = generate_descriptor("webgl", graph)
    exec_info.save("./output")
