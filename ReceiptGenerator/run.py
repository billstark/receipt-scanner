import sys
from utils import *
from items_creator import create_item_list
import math
import shutil
import textwrap
import json
import datetime
import time
import Augmentor
import functools


def create_config():
    config = {
        'seperator': random.choice(['-', '_', '*', '.']),
        'num_col': random.choice(range(20, 50)),
        'item_space_count': random.choice(range(1, 4)),
        'break_long_words': random.choice([True, False, False, False]),
        'has_table_header': random.choice([True, False]),
        'distort_val': random.choice(range(2, 20)),
        'margin_hori': random.choice(range(40, 80)),
        'margin_vert': random.choice(range(40, 200)),
        'dist_name_price': random.choice(range(2, 6)),
        'item_name_uppercase_policy': random.choice(['upper', 'title']),
        'currency_mark': random.choice(['$']),
        'currency_side': random.choice(['left', 'right']),
        'price_min': random.choice(range(0, 10)),
        'price_max': random.choice(range(10, 10000)),
        'num_row_prefix': random.choice(range(3, 6)),
        'num_row_prefix_2': random.choice(range(0, 5)),
        'num_row_suffix': random.choice(range(0, 4)),
        'num_row_suffix_2': random.choice(range(1, 5)),
        'blur': random.choice(range(0, 3)),
    }
    return config


def item_to_text(item, num_col, dist_name_price, break_long_words):
    name = item['name']
    price = item['price_str']
    qty = item['qty']
    space_for_name = num_col - len(price) - dist_name_price - 4 # 4 for QTY
    wrapper = textwrap.TextWrapper(width=space_for_name, break_long_words=break_long_words, replace_whitespace=False)
    lines = wrapper.wrap(name)
    lines = ['    ' + line for line in lines]
    lines[0] = str(qty) + lines[0][len(str(qty)):]
    last_line = lines[-1]
    whitespace = (num_col - len(last_line) - len(price)) * ' '
    last_line = lines[-1] + whitespace + price
    lines[-1] = last_line
    return '\n'.join(lines)


def items_to_text(items, num_col, dist_name_price, item_space_count, break_long_words):
    return ('\n' * item_space_count).join(item_to_text(item, num_col, dist_name_price, break_long_words) for item in items)


def create_separator_line(seperator, num_col):
    return seperator * num_col


def create_decor_text_lines(num_col, num_row):
    return [rand_seq(max_length=random.choice(range((num_col)//4, num_col))) for _ in range(num_row)]

def create_centered_text(num_col, num_row):
    lines = [line.center(num_col, ' ') for line in create_decor_text_lines(num_col, num_row)]
    return '\n'.join(lines)

def create_left_aligned_text(num_col, num_row):
    return '\n'.join(create_decor_text_lines(num_col, num_row))


def draw_text(text, margin_hori, margin_vert):
    img = Image.new('RGBA', (1, 1), white_color_tpl())
    drawer = ImageDraw.Draw(img)
    fnt = ImageFont.truetype(pick_resource('fonts'), 40)
    textsize = drawer.textsize(text, font=fnt)
    width = 2*margin_hori+textsize[0]
    height = 2*margin_vert+textsize[1]
    img = Image.open(pick_resource('paper-texture'), 'r')
    img_w, img_h = img.size
    scale = max((width * 1.0 / img_w), (height * 1.0 / img_h))
    img = img.resize((int(math.ceil(scale * img_w)), int(math.ceil(scale * img_h))), Image.ANTIALIAS)
    img_w, img_h = img.size
    x = rand_int_range(0, img_w - width)
    y = rand_int_range(0, img_h - height)
    img = img.crop((x, y, x + width, y + height))
    img = img.convert('RGBA')
    drawer = ImageDraw.Draw(img)
    drawer.text((margin_hori, margin_vert), text, font=fnt, fill=black_color_tpl())
    return img, drawer


def add_img_to_canvas(path, dest):
    img = Image.open(path, 'r')
    img_w, img_h = img.size
    bg_w, bg_h = int(img_w * 1.5), int(img_h * 1.5)
    background = Image.open(pick_resource('background'), 'r')
    background = background.resize((bg_w, bg_h), Image.ANTIALIAS)
    offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
    background.paste(img, offset)
    background.crop((int(img_w * 0.15), int(img_h * 0.15), int(img_w * 1.35), int(img_h * 1.35))).save(dest + '/image_distorted.png')


def shadow_outline(shadow_part, size):
    w = size[0]
    h = size[1]
    hw = w/2
    hh = h/2

    def left():
        return rand_int_range(0, hw)
    def right():
        return rand_int_range(hw, w)
    def top():
        return rand_int_range(0, hh)
    def bottom():
        return rand_int_range(hh, h)
    def topleft():
        return (0, 0)
    def topright():
        return (w, 0)
    def bottomleft():
        return (0, h)
    def bottomright():
        return (w, h)

    if shadow_part == (1):
        return [topright(), (w, top()), (right(), 0)]
    elif shadow_part == (2):
        return [topleft(), (left(), 0), (0, top())]
    elif shadow_part == (3):
        return [bottomleft(), (0, bottom()), (left(), h)]
    elif shadow_part == (4):
        return [bottomright(), (right(), h), (w, bottom())]
    elif shadow_part == (1, 2):
        return [topleft(), topright(), (w, top()/2), (0, top()/2)]
    elif shadow_part == (1, 4):
        return [topright(), bottomright(), ((right() + hw)/2, h), ((right() + hw)/2, 0)]
    elif shadow_part == (2, 3):
        return [bottomleft(), topleft(), (left()/2, 0), (left()/2, h)]
    elif shadow_part == (3, 4):
        return [bottomright(), bottomleft(), (0, (bottom() + hh)/2), (w, (bottom() + hh)/2)]

def draw_receipt():
    config = create_config()
    item_count_min = 2
    item_count_max = 6
    seperator = config['seperator']
    num_col = config['num_col']
    item_space_count = config['item_space_count']
    break_long_words = config['break_long_words']
    has_table_header = config['has_table_header']
    margin_hori = config['margin_hori']
    margin_vert = config['margin_vert']
    dist_name_price = config['dist_name_price']
    item_name_uppercase_policy = config['item_name_uppercase_policy']
    currency_mark = config['currency_mark']
    currency_side = config['currency_side']
    price_min = config['price_min']
    price_max = config['price_max']
    distort_val = config['distort_val']
    num_row_prefix = config['num_row_prefix']
    num_row_prefix_2 = config['num_row_prefix_2']
    num_row_suffix = config['num_row_suffix']
    num_row_suffix_2 = config['num_row_suffix_2']
    blur = config['blur']
    item_list = create_item_list(start=item_count_min, end=item_count_max, uppercase_policy=item_name_uppercase_policy, currency_mark=currency_mark, currency_side=currency_side, price_min=price_min, price_max=price_max)
    items = item_list['items']
    items_text = items_to_text(items, num_col, dist_name_price, item_space_count, break_long_words)
    separator_line = create_separator_line(seperator, num_col)
    item_space = '\n' * item_space_count

    # items text
    text = items_text + item_space + separator_line + item_space + item_to_text(item_list['total'], num_col, dist_name_price, break_long_words)

    # add header if needed
    if has_table_header:
        text = 'Qty Item' + ((num_col - len('Qty Item') - len('Price')) * ' ') + 'Price' + item_space + separator_line + item_space + text

    # add decorators
    prefix = create_centered_text(num_col, num_row_prefix)
    prefix_2 = create_left_aligned_text(num_col, num_row_prefix_2)
    suffix = create_left_aligned_text(num_col, num_row_suffix)
    suffix_2 = create_centered_text(num_col, num_row_suffix_2)

    large_space = item_space * 2
    text = prefix + large_space + prefix_2 + large_space + text + large_space + suffix + large_space + suffix_2
    set_id = 'item_' + str(int(round(time.time() * 1000)))
    directory = 'results/{}'.format(set_id)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(directory + '/back'):
        os.makedirs(directory + '/back')

    # Draw
    img, drawer = draw_text(text, margin_hori, margin_vert)

    # Save original
    img.save(directory + '/image.png', 'png')

    # Blur
    img = img.filter(ImageFilter.GaussianBlur(radius=blur))

    # Save for further processing
    img.save(directory + '/image_blur.png', 'png')

    # Add background
    add_img_to_canvas(directory + '/image_blur.png', directory + '/back')

    # Distort
    pipeline = Augmentor.Pipeline(directory + '/back')
    pipeline.random_distortion(probability=1, grid_width=distort_val, grid_height=distort_val, magnitude=100/distort_val)
    pipeline.sample(1)
    distorted_dir = directory + '/back/output'
    distorted_path = distorted_dir + '/' + [x for x in os.listdir(distorted_dir) if 'back' in x][0]
    os.rename(distorted_path, directory + '/image_distorted.png')
    shutil.rmtree(directory + '/back')

    # Create text file
    text_f = open(directory + '/text.txt', 'w')
    text_f.write(text)
    text_f.close()

    # Create items json file
    item_f = open(directory + '/items.json', 'w')
    item_f.write(json.dumps(dict(items=item_list, config=config), indent=4, sort_keys=True))
    item_f.close()


def create_sample(count):
    for i in range(count):
        draw_receipt()
        print('{} more job(s) remain'.format(count - i - 1))


if len(sys.argv) > 1:
    if sys.argv[1] == 'clear':
        if 'results' in os.listdir('.'):
            shutil.rmtree('results')
    else:
        try:
            count = int(sys.argv[1])
            if count < 1:
                print('Invalid parameter')
            else:
                create_sample(count)
                os.system('open results')
        except ValueError:
            print('Invalid parameter.')
