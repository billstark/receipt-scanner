from utils import *


def create_item(uppercase_policy='upper', currency_mark='$', currency_side='left', price_min=0, price_max=10000):
    price = rand_price(start=price_min, end=price_max)
    qty = rand_int_range(start=1, end=100)
    return dict(name=rand_seq(uppercase_policy=uppercase_policy), price=price, price_str=price_to_str(price, currency_mark, currency_side), qty=qty)


def create_item_list(start=2, end=6, uppercase_policy='upper', currency_mark='$', currency_side='left', price_min=0, price_max=10000):
    count = rand_int_range(start=start, end=end)
    items = [create_item(uppercase_policy=uppercase_policy, currency_mark=currency_mark, currency_side=currency_side, price_min=price_min, price_max=price_max) for _ in range(count)]
    total_price = sum([item['price'] for item in items])
    total = dict(name='TOTAL', price=total_price, price_str=price_to_str(total_price, currency_mark, currency_side), qty=' ')
    return dict(items=items, total=total)
