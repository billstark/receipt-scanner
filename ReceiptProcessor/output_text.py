from ReceiptGenerator.utils import normalized_avg
from ReceiptGenerator.bounding_box import BoundingBox

def cluster(data, maxgap, get_compare_val=None):
    if not get_compare_val:
        get_compare_val = lambda x: x
    data = sorted(data, key=get_compare_val)
    groups = [[data[0]]]
    for x in data[1:]:
        if abs(get_compare_val(x) - get_compare_val(groups[-1][-1])) <= maxgap:
            groups[-1].append(x)
        else:
            groups.append([x])
    return groups


def output_text(text_list, box_list):
    get_compare_val = lambda x: x[1][1] + x[1][3]
    text_box_tpls = zip(text_list, box_list)
    maxgap = normalized_avg([box[3] for box in box_list])[1] * 0.4
    text_box_tpl_clusters = cluster(text_box_tpls, maxgap, get_compare_val)
    out_text = ''
    for line in text_box_tpl_clusters:
        sorted_word_tpls = sorted(line, key=lambda x: x[1][0])
        for word_tpl in sorted_word_tpls:
            out_text += word_tpl[0] + ' '

        out_text += '\n'

    return out_text
