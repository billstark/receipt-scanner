import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def rand_int_range(start, end):
    return random.randint(int(start), int(end))


def white_color_tpl():
    val = rand_int_range(230, 255)
    return (val, val, val, 255)


def black_color_tpl():
    val = rand_int_range(0, 25)
    return (val, val, val, 255)

def shadow_tpl():
    val = rand_int_range(10, 50)
    return (0, 0, 0, val)


def pick_resource(path):
    _path = 'resources/' + path
    items = [x for x in os.listdir(_path) if '.DS_Store' not in x]
    result = _path + '/' + random.choice(items)
    return result


def rand_char():
    special_chars = ['-', '_',',','/','|','(',')','&','!','#','XX']
    return random.choice([chr(x) for x in range(ord('a'), ord('z')+1)] + special_chars +[chr(x) for x in range(ord('0'), ord('9')+1)])


def rand_seq(uppercase_policy='upper', max_length=40):
    seq = ''
    while len(seq) < max_length:
        next_sub = ''.join([rand_char() for _ in range(rand_int_range(2, 10))]) + ' '
        if uppercase_policy == 'upper':
            next_sub = next_sub.upper()
        elif uppercase_policy == 'title':
            next_sub = next_sub[0].upper() + next_sub[1:]
        seq += next_sub
    seq = seq[:40]
    return seq

def rand_crnn_char():
    special_chars = ['-', '_','!','@','#','$','%','^','&','*','(',')',' ','XX','?','<',',','>','.',';','/',':','\'','{','}','[',']','=','+','|','~',' ']
    return random.choice([chr(x) for x in range(ord('a'), ord('z')+1)] + special_chars + [chr(x) for x in range(ord('A'), ord('Z')+1)]+[chr(x) for x in range(ord('0'), ord('9')+1)])


def rand_price(start=0, end=10000):
    return float("{0:.2f}".format(random.uniform(start, end)))


def price_to_str(price, currency_mark, currency_side):
    if currency_side == 'left':
        return currency_mark + str(price)
    return str(price) + currency_mark


def padding(func):
    def func_wrapper(*args, **kwargs):
        text = func(*args, **kwargs)
        n = len(text)
        lack_left = (5 - n) / 2
        lack_right = 5 - n - lack_left
        (lack_left, lack_right) = (lack_left, lack_right) if random.randint(0, 1) else (lack_right, lack_left)
        return int(lack_left) * ' ' + text + int(lack_right) * ' '
    return func_wrapper


def crnn_word():
    max_length = random.randint(1, 5)
    return ''.join([rand_crnn_char() for _ in range(max_length)])

def crnn_line():
    max_length = random.randint(15, 25)
    return ''.join([rand_crnn_char() for _ in range(max_length)])



def crnn_word_column():
    max_length = random.randint(1, 4)
    return ''.join([rand_crnn_char() for _ in range(max_length)]) + ':'


def crnn_word_bracket():
    max_length = random.randint(1, 4)
    return '(' + ''.join([rand_crnn_char() for _ in range(max_length)]) + ')'


def crnn_int():
    return str(rand_int_range(0, pow(10, random.randint(0, 5))))


def crnn_float():
    return str(rand_price())[-5:]


def crnn_price_left():
    return str(price_to_str(rand_price(0, 100), '$', 'left'))


def crnn_price_right():
    return str(price_to_str(rand_price(0, 100), '$', 'right'))


def crnn_percentage():
    return str(rand_price(0, random.randint(0, 2))) + '%'


@padding
def crnn_line_text(typ):
    if typ == 'word':
        return crnn_word()
    elif typ == 'word_column':
        return crnn_word_column()
    elif typ == 'word_bracket':
        return crnn_word_bracket()
    elif typ == 'int':
        return crnn_int()
    elif typ == 'float':
        return crnn_float()
    elif typ == 'price_left':
        return crnn_price_left()
    elif typ == 'price_right':
        return crnn_price_right()
    elif typ == 'percentage':
        return crnn_percentage()
    elif typ =='line':
        return crnn_line()


def surrounded_text(text):
    n = len(text)
    top = ''.join([rand_crnn_char() for _ in range(n+2)])
    bottom = ''.join([rand_crnn_char() for _ in range(n+2)])
    left = rand_crnn_char()
    right = rand_crnn_char()
    surrounded = top + '\n' + left + text + right + '\n' + bottom
    return surrounded


def normalized_avg(numbers):
    average = np.average(numbers)
    variance = np.var(numbers)
    if variance == 0:
        return numbers[0], numbers[0]
    n = len(numbers)
    tolerance = 4 # Higher for more tolerance over outliers
    filtered_numbers = []
    while True:
        filtered_numbers = [number for number in numbers if -tolerance <= (number - average) / np.sqrt(variance/n) <= tolerance]
        if not filtered_numbers:
            tolerance *= 1.5
        else:
            break

    return (np.max(filtered_numbers), np.average(filtered_numbers))
