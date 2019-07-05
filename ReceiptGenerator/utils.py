import os
import random
import numpy as np
import datetime
import time
from PIL import Image, ImageDraw, ImageFont, ImageFilter


def strTimeProp(start, end, format, prop):
    """Get a time at a proportion of a range of two formatted times.

    start and end should be strings specifying times formated in the
    given format (strftime-style), giving an interval [start, end].
    prop specifies how a proportion of the interval to be taken after
    start.  The returned time will be in the specified format.
    """

    stime = time.mktime(time.strptime(start, format))
    etime = time.mktime(time.strptime(end, format))

    ptime = stime + prop * (etime - stime)

    return time.strftime(format, time.localtime(ptime))


def randomDate(start, end, prop):
    return strTimeProp(start, end, '%m/%d/%Y %I:%M %p', prop)

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
    max_length = random.randint(1, 10)
    return ''.join([rand_crnn_char() for _ in range(max_length)])

def crnn_line():
    max_length = random.randint(25, 40)
    return ''.join([rand_crnn_char() for _ in range(max_length)])

def crnn_date():
    return randomDate("1/1/1980 12:00 AM", "7/1/2019 11:59 PM", random.random())

def crnn_word_column():
    max_length = random.randint(1, 15)
    return ''.join([rand_crnn_char() for _ in range(max_length)]) + ':'


def crnn_word_bracket():
    max_length = random.randint(1, 10)
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

def crnn_items_prices_left():
    max_length_word = random.randint(4, 20)

    return ''.join([rand_crnn_char() for _ in range(max_length_word)]) + ' '  + str(crnn_price_left())

def crnn_items_prices_right():
    max_length_word = random.randint(4, 20)
    max_length_spaces = random.randint(10,20)
    return ''.join([rand_crnn_char() for _ in range(max_length_word)]) + ' '  + str(crnn_price_right())

def crnn_tot_left():
    tots = ['Total', 'TOTAL', 'TOT', 'DUE', 'AMOUNT', 'BALANCE', 'Due', 'Amount','Balance']
    sep = [':','-',' ']
    return tots[random.randint(0,len(tots)-1)]+sep[random.randint(0,len(sep)-1)]+str(crnn_price_left())

def crnn_tot_right():
    tots = ['Total', 'TOTAL', 'TOT', 'DUE', 'AMOUNT', 'BALANCE', 'Due', 'Amount','Balance']
    sep = [':','-',' ']
    return tots[random.randint(0,len(tots)-1)]+sep[random.randint(0,len(sep)-1)]+str(crnn_price_right())

def crnn_tax():
    tax = random.choice(['Service Tax', 'Tax', 'Sales Tax', 'Federal Tax', 'State Tax'])
    sep = random.choice([':','-',' '])
    case = random.choice(['U','L','N'])
    val = random.choice([str(crnn_price_left()), str(crnn_percentage()),str(crnn_price_right())])
    if case == 'U':
        return str(tax).upper() + str(sep) + str(val)
    elif case == 'L':
        return str(tax).lower() + str(sep) + str(val)
    else:
        return str(tax) + str(sep) + str(val)


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
    elif typ =='date':
        return crnn_date()
    elif typ =='tax':
        return crnn_tax()
    elif typ == 'totR':
        return crnn_tot_right()
    elif typ == 'totL':
        return crnn_tot_left()
    elif typ == 'priceR':
        return crnn_items_prices_right()
    elif typ == 'priceL':
        return crnn_items_prices_left()


def surrounded_text(text):
    n = len(text)
    top = ''.join([' ' for _ in range(n)])
    bottom = ''.join(['rand_crnn_char()' for _ in range(n)])
    left = rand_crnn_char()
    right = rand_crnn_char()
    surrounded = top+ '\n' + text + '\n' + bottom
    #print(surrounded)
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
