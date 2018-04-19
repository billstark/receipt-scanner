import tensorflow as tf
import os.path as ops
import numpy as np
import cv2
import argparse
try:
    from cv2 import cv2
except ImportError:
    pass

from crnn_model import crnn_model
from global_configuration import config
from local_utils import log_utils, data_utils, image_utils

SCRIPT_PATH = ops.dirname(ops.abspath(__file__))
DEFAULT_WEIGHTS_PATH = ops.join(SCRIPT_PATH, './shadownet/shadownet_2018-04-18-12-29-11.ckpt-19999')
INPUT_NAME = 'input'
NUMBER_OF_CLASSES = 61

def recognize(imgs_np, weights_path=DEFAULT_WEIGHTS_PATH):
    inputdata = tf.placeholder(dtype=tf.float32, shape=[1, image_utils.EXPECTED_HEIGHT, image_utils.EXPECTED_WIDTH, 3], name=INPUT_NAME)

    net = crnn_model.ShadowNet(phase='Test', hidden_nums=256, layers_nums=2, seq_length=25, num_classes=NUMBER_OF_CLASSES)

    with tf.variable_scope('shadow'):
        net_out = net.build_shadownet(inputdata=inputdata)

    decodes, _ = tf.nn.ctc_beam_search_decoder(inputs=net_out, sequence_length=25*np.ones(1), merge_repeated=False)

    decoder = data_utils.TextFeatureIO()

    # config tf session
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = config.cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH

    # config tf saver
    saver = tf.train.Saver()

    sess = tf.Session(config=sess_config)

    imgs_np = [image_utils.standardize_image(img) for img in imgs_np]
    imgs = [np.expand_dims(img, axis=0).astype(np.float32) for img in imgs_np]

    with sess.as_default():

        results = []

        saver.restore(sess=sess, save_path=weights_path)

        for img in imgs:
            preds = sess.run(decodes, feed_dict={inputdata: img})
            preds = decoder.writer.sparse_tensor_to_str(preds[0])
            results.append(preds[0])

        sess.close()

    return results
