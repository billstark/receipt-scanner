import cv2

from ReceiptGenerator.draw_receipt import create_crnn_sample

NUM_OF_TRAINING_IMAGES = 300
NUM_OF_TEST_IMAGES = 100

with open('./training_images/Train/sample.txt', 'w') as input_file:
    for i in range(0, NUM_OF_TRAINING_IMAGES):
        img, label = create_crnn_sample()
        cv2.imwrite('./training_images/Train/{}.jpg'.format(i))
        input_file.wirte('{}.jpg {}'.format(i, label))
