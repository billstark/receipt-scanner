import os
import random
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
    special_chars = ['-', '_']
    return random.choice([chr(x) for x in range(ord('a'), ord('z')+1)] + special_chars)


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
