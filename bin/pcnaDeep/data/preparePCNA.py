# -*- coding: utf-8 -*-
import os
import re
import json
import numpy as np
import detectron2.structures as st
import math


def load_PCNA_from_json(json_path, image_path, width=1200, height=1200):
    """Load PCNA training data and ground truth from json.

    Args:
        json_path (str): path to .json ground truth in VIA2 format.
        image_path (str): path to raw image.
        width (int): width of the image.
        height (int): height of the image.

    """
    cc_stageDic = {"G1/G2": 0, "S": 1, "M": 2, "E": 3}

    with open(json_path, 'r', encoding='utf8') as fp:
        ann = json.load(fp)
    outs = []
    for key in list(ann.keys()):
        ann_img = ann[key]
        fn = ann_img['filename']
        regions = ann_img['regions']
        id = re.search('(.+)\.\w*', fn).group(1)
        out = {'file_name': os.path.join(image_path, fn), 'height': height, 'width': width, 'image_id': id,
               'annotations': []}

        for r in regions:
            phase = r['region_attributes']['phase']
            shape = r['shape_attributes']
            x = shape['all_points_x']
            y = shape['all_points_y']
            bbox = [math.floor(np.min(x)), math.floor(np.min(y)), math.ceil(np.max(x)), math.ceil(np.min(y))]
            edge = [0 for i in range(len(x) + len(y))]
            edge[::2] = x
            edge[1::2] = y
            # register output
            out['annotations'].append(
                {'bbox': bbox, 'bbox_mode': st.BoxMode.XYXY_ABS, 'category_id': cc_stageDic[phase],
                 'segmentation': [edge.copy()]})

        outs.append(out)
    return outs


def load_PCNAs_json(json_paths, image_paths):
    """Load multiple training dataset.
    """
    import random
    assert len(json_paths) == len(image_paths)
    out = []
    for i in range(len(json_paths)):
        print('Loading dataset from: ' + image_paths[i])
        dic = load_PCNA_from_json(json_paths[i], image_paths[i])
        out += dic
    random.shuffle(out)
    return out
