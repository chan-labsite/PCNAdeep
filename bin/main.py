# -*- coding: utf-8 -*-
import argparse
import multiprocessing as mp
import os
import re
import time
import yaml
import pprint
import gc
import numpy as np
import pandas as pd
import skimage.io as io
import torch
from skimage.util import img_as_ubyte
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from pcnaDeep.predictor import VisualizationDemo, predictFrame
from pcnaDeep.refiner import Refiner
from pcnaDeep.resolver import Resolver
from pcnaDeep.tracker import track
from pcnaDeep.split import split_frame, join_frame, join_table, resolve_joined_stack
from pcnaDeep.data.utils import getDetectInput
from tqdm import trange


def setup_cfg(args):
    # load Detectron2 config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.dtrn_config)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def check_PCNA_cfg(config, img_shape):
    """Check the integrity of PCNAdeep configs.
    """
    try:
        if float(config['PIX_SATURATE']) < 0 or float(config['PIX_SATURATE']) > 100:
            raise ValueError('Pixel saturation should be within range 0~100.')
        if float(config['EDGE_FLT']) < 0 or float(config['EDGE_FLT']) > np.min([img_shape[1], img_shape[2]])/2:
            raise ValueError('Edge region should not be larger than image size or negative.')
        if float(config['SIZE_FLT']) > img_shape[1] * img_shape[2]:
            raise ValueError('Object size filter should not be larger than image size.')
        if float(config['TRACKER']['DISPLACE']) >= np.min([img_shape[1], img_shape[2]]):
            raise ValueError('Tracker displacement should be smaller than image size.')
        if float(config['TRACKER']['GAP_FILL']) >= img_shape[0]:
            raise ValueError('Tracker memory should be smaller than time frame length.')
        for i in ['MAX_BG', 'MIN_S', 'MIN_M']:
            if float(config['POST_PROCESS'][i]) >= img_shape[0] or float(config['POST_PROCESS'][i]) <=0:
                raise ValueError('Cell cycle phase length should be positive and smaller than frame length.')
        for i in ['SMOOTH', 'MAX_FRAME_TRH', 'SEARCH_RANGE']:
            to_check = float(config['POST_PROCESS']['REFINER'][i])
            if to_check >= img_shape[0] or to_check <= 0:
                raise ValueError(i + ' should be smaller than frame length and positive.')
        to_check = float(config['POST_PROCESS']['RESOLVER']['MIN_LINEAGE'])
        if to_check < 0 or to_check > img_shape[0]:
            raise ValueError('Minimum track length should not be negative or longer then frame length.')
        to_check = float(config['POST_PROCESS']['RESOLVER']['G2_TRH'])
        if to_check <= 0:
            raise ValueError('G2 intensity threshold should be positive.')
    except KeyError as e:
        raise KeyError('Field not found in config file: ' + str(e))
    return


def get_parser():
    parser = argparse.ArgumentParser(description="pcnaDeep configs.")
    parser.add_argument(
        "--dtrn-config",
        default="../config/dtrnCfg.yaml",
        metavar="FILE",
        help="path to detectron2 model config file",
    )
    parser.add_argument(
        "--pcna-config",
        default="../config/pcnaCfg.yaml",
        metavar="FILE",
        help="path to pcnaDeep tracker/refiner/resolver config file",
    )
    parser.add_argument(
        "--pcna",
        default=None,
        help="Path to PCNA channel time series image.",
    )
    parser.add_argument(
        "--bf",
        default=None,
        help="Path to bright field channel time series image.",
    )
    parser.add_argument(
        "--stack-input",
        default=None,
        help="Path to composite image stack file. Will overwrite pcna or dic input, not recommended.",
    )
    parser.add_argument(
        "--output",
        help="Output directory",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify pcnaDeep config options using the command-line 'KEY VALUE' pairs. For pcnaDeep config, "
             "begin with pcna., e.g., pcna.TRACKER.DISPLACE 100. For detectron2 config, follow detectron2 docs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def main(stack, config, output, prefix, logger):
    check_PCNA_cfg(config, stack.shape)

    logger.info("Run on image shape: " + str(stack.shape))
    table_out = pd.DataFrame()
    mask_out = []
    spl = int(config['SPLIT']['GRID'])
    edge_raw = config['EDGE_FLT']  # not filter edge objects when resolving separate tiles.
    if spl:
        config['EDGE_FLT'] = 0
        new_imgs = []
        for i in range(stack.shape[0]):
            splited = split_frame(stack[i,:].copy(), n=spl)
            for j in range(splited.shape[0]):
                new_imgs.append(splited[j,:])
        stack = np.stack(new_imgs, axis=0)
        del new_imgs

    edge = config['EDGE_FLT']
    size_flt = config['SIZE_FLT']
    instances_frame = []
    start_time = time.time()
    with trange(stack.shape[0], unit='img') as trg:
        for i in trg:
            img_relabel, out_props = predictFrame(stack[i,:], i, demo, edge_flt=edge, size_flt=size_flt)
            table_out = table_out.append(out_props)
            img_relabel = torch.from_numpy(img_relabel.astype('int16'))  # new
            mask_out.append(img_relabel)
            trg.set_description('Frame %i' % i)
            trg.set_postfix(instances=str(out_props.shape[0]))
            instances_frame.append(out_props.shape[0])

    logger.info(
        "{}: {} in {:.2f}s".format(
            'Total frame '+str(stack.shape[0]),
            "Mean detected instances: {}".format(np.mean(instances_frame)),
            time.time() - start_time,
        )
    )
    
    tw = stack.shape[1]
    del stack
    gc.collect()
    mask_out = torch.stack(mask_out, axis=0)
    mask_out = mask_out.numpy()

    if spl:
        mask_out = join_frame(mask_out.copy(), n=spl)
        table_out = join_table(table_out.copy(), n=spl, tile_width=tw)
        mask_out, table_out = resolve_joined_stack(mask_out, table_out, n=spl, 
                                                   boundary_width=config['SPLIT']['EDGE_SPLIT'],
                                                   dilate_time=config['SPLIT']['DILATE_ROUND'],
                                                   filter_edge_width=edge_raw)
    
    logger.info('Tracking...')
    track_out = track(df=table_out, displace=int(config['TRACKER']['DISPLACE']),
                        gap_fill=int(config['TRACKER']['GAP_FILL']))
    track_out.to_csv(os.path.join(output, prefix + '_tracks.csv'), index=False)

    if np.max(mask_out) < 255:
        mask_out = img_as_ubyte(mask_out)
    io.imsave(os.path.join(output, prefix + '_mask.tif'), mask_out)

    logger.info('Refining and Resolving...')
    post_cfg = config['POST_PROCESS']
    refiner_cfg = post_cfg['REFINER']
    if not bool(refiner_cfg['MASK_CONSTRAINT']['ENABLED']):
        logger.info('Mask constraint disabled')
        mask_out = None
        df = None
    else:
        logger.info('Mask constraint enabled.')
        df = float(refiner_cfg['MASK_CONSTRAINT']['DILATE_FACTOR'])
    myRefiner = Refiner(track_out, threshold_mt_F=int(refiner_cfg['MAX_DIST_TRH']),
                        threshold_mt_T=int(refiner_cfg['MAX_FRAME_TRH']), smooth=int(refiner_cfg['SMOOTH']),
                        maxBG=float(post_cfg['MAX_BG']),
                        minM=float(post_cfg['MIN_M']), search_range=int(refiner_cfg['SEARCH_RANGE']),
                        sample_freq=float(refiner_cfg['SAMPLE_FREQ']),
                        model_train=refiner_cfg['SVM_TRAIN_DATA'], svm_c=int(refiner_cfg['C']),
                        mode=refiner_cfg['MODE'], mask=mask_out, dilate_factor=df, 
                        aso_trh=float(refiner_cfg['ASO_TRH']), dist_weight=float(refiner_cfg['DIST_WEIGHT']))
    ann, track_rfd, mt_dic, imprecise = myRefiner.doTrackRefine()
    del mask_out
    gc.collect()
    
    ann.to_csv(os.path.join(output, prefix + '_tracks_ann.csv'), index=0)
    logger.debug(pprint.pformat(mt_dic, indent=4))

    myResolver = Resolver(track_rfd, ann, mt_dic, maxBG=float(post_cfg['MAX_BG']), minS=float(post_cfg['MIN_S']),
                          minM=float(post_cfg['MIN_M']),
                          minLineage=int(post_cfg['RESOLVER']['MIN_LINEAGE']), impreciseExit=imprecise,
                          G2_trh=int(post_cfg['RESOLVER']['G2_TRH']))
    track_rsd, phase = myResolver.doResolve()
    track_rsd.to_csv(os.path.join(output, prefix + '_tracks_refined.csv'), index=0)
    phase.to_csv(os.path.join(output, prefix + '_phase.csv'), index=0)

    logger.info(prefix + ' Finished: ' + time.strftime("%Y/%m/%d %H:%M:%S", time.localtime()))
    logger.info('='*50)

    return


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger(name='pcna', abbrev_name='pcna', 
                          output=os.path.join(args.output, 'log.txt'))
    logger.info("Arguments: " + str(args))
    # resolve pcnaDeep Config
    with open(args.pcna_config, 'rb') as f:
        date = yaml.safe_load_all(f)
        pcna_cfg_dict = list(date)[0]
    dtrn_opts = []
    i = 0
    while i < len(args.opts)/2:
        o = args.opts[2*i]
        value = args.opts[2*i+1]
        l = o.split('.')
        if l[0] == 'pcna' or l[0] == 'PCNA':
            if len(l) == 2:
                pcna_cfg_dict[l[1]] = value
            elif len(l) >= 3:
                cur_ref = pcna_cfg_dict[l[1]]
                for j in range(2, len(l)-1):
                    cur_ref = cur_ref[l[j]]
                cur_ref[l[-1]] = value
        else:
            dtrn_opts.append(o)
            dtrn_opts.append(value)
        i += 1
    args.opts = dtrn_opts
    cfg = setup_cfg(args)
    logger.info("Finished setup.")
    demo = VisualizationDemo(cfg)

    logger.info("Start inferring.")
    ipt = args.stack_input

    if bool(pcna_cfg_dict['BATCH']):
        if args.pcna is not None and args.bf is not None:
            if os.path.isdir(args.pcna) and os.path.isdir(args.bf) and os.path.isdir(args.output):
                pcna_imgs = os.listdir(args.pcna)
                dic_imgs = os.listdir(args.bf)
                pairs = []
                for pi in pcna_imgs:
                    prefix = os.path.basename(pi)
                    mat_obj = re.match('(.+)pcna\.tif|(.+)PCNA\.tif',prefix)
                    if mat_obj is None:
                        raise ValueError('PCNA file ' + pi + ' must ends with \"pcna\" and in .tif format')
                    prefix = mat_obj.group(1)
                    if prefix is None:
                        prefix = mat_obj.group(2)
                        pcna_fp = prefix + 'PCNA.tif'
                    else:
                        pcna_fp = prefix + 'pcna.tif'

                    if prefix + 'bf.tif' in dic_imgs:
                        dic_fp = prefix + 'bf.tif'
                    elif prefix + 'BF.tif' in dic_imgs:
                        dic_fp = prefix + 'BF.tif'
                    else:
                        raise ValueError('Bright field file ' + prefix + 'bf.tif does not exit.')
                    prefix = prefix[:-1] if prefix[-1] in ['_','-'] else prefix
                    pairs.append((prefix, pcna_fp, dic_fp))
                
                for si in pairs:
                    md = os.path.join(args.output, si[0])
                    if os.path.exists(md):
                        logger.warning('Directory ' + md + ' exists, will override files inside.')
                    else:
                        os.mkdir(md)
                    imgs = getDetectInput(io.imread(os.path.join(args.pcna, si[1])), 
                                          io.imread(os.path.join(args.bf, si[2])),
                                          sat=float(pcna_cfg_dict['PIX_SATURATE']),
                                          gamma=float(pcna_cfg_dict['GAMMA']), torch_gpu=True)
    
                    main(stack=imgs, config=pcna_cfg_dict, output=os.path.join(args.output, si[0]), 
                         prefix=si[0], logger=logger)
                    del imgs
                    gc.collect()
            else:
                raise ValueError('Must input directory in batch mode, not single file.')
        
        elif ipt is not None:
            if os.path.isdir(ipt):
                stack_imgs = os.listdir(ipt)
                for si in stack_imgs:
                    prefix = re.match('(.+)\.\w+',si).group(1)
                    prefix = prefix[:-1] if prefix[-1] in ['_','-'] else prefix
                    md = os.path.join(args.output, prefix)
                    if os.path.exists(md):
                        logger.warning('Directory ' + md + ' exists, will override files inside.')
                    else:
                        os.mkdir(md)
                    imgs = io.imread(os.path.join(ipt, si))

                    main(stack=imgs, config=pcna_cfg_dict, output=os.path.join(args.output, prefix), 
                         prefix=prefix, logger=logger)
                    del imgs
                    gc.collect()
            else:
                raise ValueError('Must input directory in batch mode, not single file.')

    if (ipt is not None or (args.pcna is not None and args.bf is not None)) and not bool(pcna_cfg_dict['BATCH']):
        flag = True
        if ipt is not None:
            prefix = os.path.basename(ipt)
            prefix = re.match('(.+)\.\w+',prefix).group(1)
            # Input image must be uint8
            imgs = io.imread(ipt)
        else:
            prefix = os.path.basename(args.pcna)
            prefix = re.match('(.+)\.\w+', prefix).group(1)
            pcna = io.imread(args.pcna)
            dic = io.imread(args.bf)
            logger.info("Generating composite...")
            imgs = getDetectInput(pcna, dic, sat=float(pcna_cfg_dict['PIX_SATURATE']),
                                  gamma=float(pcna_cfg_dict['GAMMA']), torch_gpu = True)
            del pcna
            del dic
            gc.collect()
        
        if prefix.split('-')[-1] in ['DIC', 'dic', 'mCy', 'mcy', 'pcna', 'PCNA']:
            prefix = '-'.join(prefix.split('-')[:-1])
        elif prefix.split('_')[-1] in ['DIC', 'dic', 'mCy', 'mcy', 'pcna', 'PCNA']:
            prefix = '_'.join(prefix.split('_')[:-1])

        main(stack=imgs, config=pcna_cfg_dict, output=args.output, prefix=prefix, logger=logger)
