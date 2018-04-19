import numpy as np
import cv2
import argparse
import os
from ReceiptGenerator.line_seg import cut_lines
from ReceiptProcessor.output_text import output_text
from CRNNModel.image_classifier import Classifier

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

def process_img(img_path):
    path = os.path.join(SCRIPT_PATH, img_path)
    assert os.path.exists(path), 'No such image exists'

    img = cv2.imread(path)

    label, words, box_list = cut_lines(img)

    cv2.imwrite('test_label.png', label)
    classifier = Classifier()

    text_list = classifier.recognize_imgs(words)

    out_text = output_text(text_list, box_list)

    print('\n\n\n\n' + out_text)
    return out_text

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, dest='receipt_img_path',
                        help='The receipt image file path')
    args = parser.parse_args()
    process_img(args.receipt_img_path)
