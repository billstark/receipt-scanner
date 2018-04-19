import numpy as np
import cv2
import argparse
import os
from CRNNModel.image_classifier import Classifier

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

def process_img(img_path):
    path = os.path.join(SCRIPT_PATH, img_path)
    assert os.path.exists(path), 'No such image exists'

    # read images here

    classifier = Classifier()

    # recognize imgs here

    # write imgs here

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, dest='receipt_img_path',
                        default=DEFAULT_LABEL_FILE,
                        help='The receipt image file path')
    args = parser.parse_args()
    process_img(args.receipt_img_path)
