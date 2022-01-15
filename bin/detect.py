# -*- coding: utf-8 -*-
# Modified by Yifan Gui from FAIR Detectron2 v0.4, Apache 2.0 licence.
import argparse
import json
import multiprocessing as mp
import os
import time
import gc

import numpy as np
import pandas as pd
import skimage.io as io
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from pcnaDeep.predictor import VisualizationDemo, pred2json, predictFrame
from pcnaDeep.data.utils import getDetectInput


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="pcnaDeep script for detection stage only.")
    parser.add_argument(
        "--config-file",
        default="../config/dtrnCfg.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--stack_input",
        help="Path to image stack file.",
    )
    parser.add_argument(
        "--bf",
        help="Path to bight field image file.",
    )
    parser.add_argument(
        "--pcna",
        help="Path to PCNA image file file.",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save outputs.",
    )
    parser.add_argument(
        "--prefix",
        help="Output file name. If not given, will deduce from inputs.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--is_slice",
        action="store_true",
    )
    parser.add_argument(
        "--vis_out",
        action="store_true",
    )
    parser.add_argument(
        "--sat",
        type=float,
        help="Saturated pixel when enhancing contrast. Only applies to separate channels. Default 1",
        default=1,
    )
    parser.add_argument(
        "--gamma",
        type=float,
        help="Gamma correction factor, enhance (<1) or suppress (>1) intensity non-linearly. Default 1",
        default=1,
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.stack_input is not None or args.bf is not None:
        # Input image must be uint8
        if args.stack_input is not None:
            imgs = io.imread(args.stack_input)
            if args.is_slice:
                imgs = np.expand_dims(imgs, axis=0)
            if args.prefix is None:
                args.prefix = os.path.basename(args.stack_input).split('.')[0]
        else:
            dic = io.imread(args.bf)
            mcy = io.imread(args.pcna)
            if args.prefix is None:
                args.prefix = os.path.basename(args.bf).split('.')[0].split('_')[0]
            if args.is_slice:
                dic = np.expand_dims(dic, axis=0)
                mcy = np.expand_dims(mcy, axis=0)
            imgs = getDetectInput(mcy, dic, sat=args.sat, gamma=args.gamma, torch_gpu=True)
            del dic, mcy
            gc.collect()

        print("Run on image shape: "+str(imgs.shape))
        imgs_out = []
        table_out = pd.DataFrame()
        json_out = {}
        
        for i in range(imgs.shape[0]):
            img = imgs[i,:]
            start_time = time.time()
            detected = False
            if not args.vis_out:
                # Generate json output readable by VIA2
                img_relabel, out_props = predictFrame(imgs[i, :], i, demo, size_flt=1000, edge_flt=0)
                file_name = args.prefix + '-' + "%04d" % i + '.png'
                dic_frame = pred2json(img_relabel, out_props, file_name)
                json_out[file_name] = dic_frame
                if 'regions' in dic_frame.keys():
                    detected = len(dic_frame['regions'])
            else:
                # Generate visualized output
                predictions, visualized_output = demo.run_on_image(img)
                detected = len(predictions["instances"])
                if detected == 0:
                    detected = False
                imgs_out.append(visualized_output.get_image())
            logger.info(
                "{}: {} in {:.2f}s".format(
                    'frame'+str(i),
                    "detected {} instances".format(detected) if detected else 'no instances detected.',
                    time.time() - start_time,
                )
            )
        prefix = args.prefix
        if not args.vis_out:
            with(open(os.path.join(args.output, prefix+'.json'), 'w', encoding='utf8')) as file:
                json.dump(json_out, file)
        else:
            out = np.stack(imgs_out, axis=0)
            io.imsave(os.path.join(args.output, prefix+'_vis.tif'), out)
