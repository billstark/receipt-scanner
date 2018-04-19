import tensorflow as tf
import os.path as ops
import numpy as np
import cv2
import argparse
try:
    from cv2 import cv2
except ImportError:
    pass

from CRNNModel.crnn_model import crnn_model
from CRNNModel.global_configuration import config
from CRNNModel.local_utils import log_utils, data_utils, image_utils

SCRIPT_PATH = ops.dirname(ops.abspath(__file__))
DEFAULT_WEIGHTS_PATH = ops.join(SCRIPT_PATH, './shadownet/shadownet_2018-04-19-14-32-11.ckpt-9999')
INPUT_NAME = 'input'
NUMBER_OF_CLASSES = 71

class Classifier:

    def __init_shadownet(self):
        self.inputdata = tf.placeholder(dtype=tf.float32, shape=[1, image_utils.EXPECTED_HEIGHT, image_utils.EXPECTED_WIDTH, 3], name=INPUT_NAME)
        self.net = crnn_model.ShadowNet(phase='Test', hidden_nums=256, layers_nums=2, seq_length=25, num_classes=NUMBER_OF_CLASSES)

        with tf.variable_scope('shadow'):
            net_out = self.net.build_shadownet(inputdata=self.inputdata)

        self.decodes, _ = tf.nn.ctc_beam_search_decoder(inputs=net_out, sequence_length=25*np.ones(1), merge_repeated=False)

    def __init_decoder(self):
        self.decoder = data_utils.TextFeatureIO()

    def __init_session(self):
        # config tf session
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = config.cfg.TRAIN.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH

        # config tf saver
        self.saver = tf.train.Saver()
        self.sess = tf.Session(config=sess_config)

    def __init__(self, weights_path=DEFAULT_WEIGHTS_PATH):
        self.weights_path = weights_path
        self.__init_shadownet()
        self.__init_decoder()
        self.__init_session()

    def recognize_img(self, img_np):
        img_np = image_utils.standardize_image(img_np)
        img_np = np.expand_dims(img_np, axis=0).astype(np.float32)
        with self.sess.as_default():
            self.saver.restore(sess=self.sess, save_path=self.weights_path)
            preds = self.sess.run(self.decodes, feed_dict={self.inputdata: img_np})
            preds = self.decoder.writer.sparse_tensor_to_str(preds[0])
            return preds[0]

    def recognize_imgs(self, imgs_np):
        imgs_np = [image_utils.standardize_image(img_np) for img_np in imgs_np]
        imgs_np = [np.expand_dims(img_np, axis=0).astype(np.float32) for img_np in imgs_np]

        with self.sess.as_default():
            results = []
            self.saver.restore(sess=self.sess, save_path=self.weights_path)
            for img_np in imgs_np:
                preds = self.sess.run(self.decodes, feed_dict={self.inputdata: img_np})
                preds = self.decoder.writer.sparse_tensor_to_str(preds[0])
                results.append(preds[0])
        return results
