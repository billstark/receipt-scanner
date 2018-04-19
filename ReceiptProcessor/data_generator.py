import os
import cv2

from ReceiptGenerator.draw_receipt import create_crnn_sample

NUM_OF_TRAINING_IMAGES = 3000
NUM_OF_TEST_IMAGES = 1000

TEXT_TYPES = ['word', 'word_column', 'word_bracket', 'int', 'float', 'price_left', 'price_right', 'percentage']
# TEXT_TYPES = ['word']

with open('./ReceiptProcessor/training_images/Train/sample.txt', 'w') as input_file:
    for type in TEXT_TYPES:
        if not os.path.exists('./ReceiptProcessor/training_images/Train/{}'.format(type)):
            os.mkdir('./ReceiptProcessor/training_images/Train/{}'.format(type))
        for i in range(0, NUM_OF_TRAINING_IMAGES):
            img, label = create_crnn_sample(type)
            cv2.imwrite('./ReceiptProcessor/training_images/Train/{}/{}.jpg'.format(type, i), img)
            input_file.write('{}/{}.jpg {}\n'.format(type, i, label))

with open('./ReceiptProcessor/training_images/Test/sample.txt', 'w') as input_file:
    for type in TEXT_TYPES:
        if not os.path.exists('./ReceiptProcessor/training_images/Test/{}'.format(type)):
            os.mkdir('./ReceiptProcessor/training_images/Test/{}'.format(type))
        for i in range(0, NUM_OF_TEST_IMAGES):
            img, label = create_crnn_sample(type)
            cv2.imwrite('./ReceiptProcessor/training_images/Test/{}/{}.jpg'.format(type, i), img)
            input_file.write('{}/{}.jpg {}\n'.format(type, i, label))
