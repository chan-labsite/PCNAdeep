# -*- coding: utf-8 -*-
import os
import logging
import subprocess
from pcnaDeep.data.preparePCNA import load_PCNAs_json
from pcnaDeep.data.annotate import relabel_trackID, label_by_track, get_lineage_txt, load_trks, lineage_dic2txt, \
    break_track, save_seq, generate_calibanTrk
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, DatasetEvaluators, print_csv_format
from detectron2.data import build_detection_test_loader
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2 import model_zoo


class pcna_detectronEvaluator:

    def __init__(self, cfg_path, dataset_ann_path, dataset_path, out_dir, class_name=["G1/G2", "S", "M", "E"], 
                 confidence_threshold=0.5):
        """Evaluate Detectron2 performance using COCO matrix

        Args:
            cfg_path (str): path to config file.
            dataset_ann_path (str): path to the testing dataset annotation `json` file.
            dataset_path (str): path to testing dataset, must in format of pcnaDeep: separate dic and mcy folders.
            out_dir (str): output directory.
            class_name (list): classification name, should be the same as training config.
            confidence_threshold (float): confidence threshold, default 0.5.
        """
        self.CLASS_NAMES = class_name
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.merge_from_file(cfg_path)
        self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        self.cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold

        self.out = out_dir
        DatasetCatalog.register("pcna_test", lambda: load_PCNAs_json(list(dataset_ann_path), list(dataset_path)))
        MetadataCatalog.get("pcna_test").set(thing_classes=self.CLASS_NAMES, evaluator_type='coco')

        self.logger = logging.getLogger('detectron2')
        self.logger.setLevel(logging.DEBUG)
        handler1 = logging.StreamHandler()
        handler1.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        handler1.setFormatter(formatter)
        self.logger.addHandler(handler1)

    def run_evaluate(self, model_path):
        """Run evaluation

        Args:
            model_path (str): path to pre-trained model.
        """
        evaluator = COCOEvaluator(dataset_name="pcna_test", output_dir=self.out, 
                                  tasks=('segm',), distributed=True)
        
        self.cfg.MODEL.WEIGHTS = model_path
        val_loader = build_detection_test_loader(self.cfg, "pcna_test")
        model = build_model(self.cfg)
        result = inference_on_dataset(model, val_loader, evaluator)
        #print(model)
        #print(val_loader)
        #print(evaluator)
        print(result)
        print_csv_format(result)


class pcna_ctcEvaluator:

    def __init__(self, root, dt_id, digit_num=3, t_base=0, path_ctc_software=None, init_dir=True):
        """Evaluation of tracking output
        """
        self.dt_id = dt_id
        self.digit_num = digit_num
        self.t_base = t_base
        self.root = root
        self.path_ctc_software = path_ctc_software
        if init_dir:
            self.init_ctc_dir()
        self.trk_path = None

    def set_evSoft(self, path_ctc_software):
        """Set evaluation software path

        Args:
            path_ctc_software (str): path to CTC evaluation software
        """
        self.path_ctc_software = path_ctc_software

    def generate_raw(self, stack):
        """Save raw images by slice

        Args:
            stack (numpy.ndarray): raw image
        """
        fm = ("%0" + str(self.digit_num) + "d") % self.dt_id
        save_seq(stack, os.path.join(self.root, fm), 't', dig_num=self.digit_num, base=self.t_base)
        return

    def generate_ctc(self, mask, track, mode='RES'):
        """Generate standard format for Cell Tracking Challenge Evaluation, for RES or GT.

        Args:
            mask (numpy.ndarray): mask output, no need to have cell cycle labeled
            track (pandas.DataFrame): tracked object table, can have gaped tracks
            mode (str): either "RES" or "GT".
        """
        track_new = relabel_trackID(track.copy())
        track_new = break_track(track_new.copy())
        tracked_mask = label_by_track(mask.copy(), track_new.copy())
        txt = get_lineage_txt(track_new)
        fm = ("%0" + str(self.digit_num) + "d") % self.dt_id

        if mode == 'RES':
            # write out processed files for RES folder
            save_seq(tracked_mask, os.path.join(self.root, fm + '_RES'), 'mask', dig_num=self.digit_num, base=self.t_base, sep='')
            txt.to_csv(os.path.join(self.root, fm + '_RES', 'res_track.txt'), sep=' ', index=0, header=False)
        elif mode == 'GT':
            fm = os.path.join(self.root, fm + '_GT')
            self.__saveGT(fm, txt, mask)
        else:
            raise ValueError('Can only generate CTC format files as RES or GT, not: ' + mode)
        return

    def __saveGT(self, fm, txt, mask):
        """Save ground truth in Cell Tracking Challenge format.
        """
        txt.to_csv(os.path.join(fm, 'TRA', 'man_track.txt'), index=0, sep=' ', header=False)
        save_seq(mask, os.path.join(fm, 'SEG'), 'man_seg', dig_num=self.digit_num, base=self.t_base, sep='')
        save_seq(mask, os.path.join(fm, 'TRA'), 'man_track', dig_num=self.digit_num, base=self.t_base, sep='')
        return

    def caliban2ctcGT(self, trk_path):
        """Convert caliban ground truth to Cell Tracking Challenge ground truth.

        Args:
            trk_path (str): path to deepcell-label .trk file.
        """
        t = load_trks(trk_path)
        self.trk_path = trk_path
        lin = t['lineages']
        mask = t['y']
        txt = lineage_dic2txt(lin)

        fm = ("%0" + str(self.digit_num) + "d") % self.dt_id
        fm = os.path.join(self.root, fm + '_GT')
        self.__saveGT(fm, txt, mask)
        return

    def init_ctc_dir(self):
        """Initialize Cell Tracking Challenge directory

        Directory example  
            >-----0001----------  
               >-----0001_RES---  
               >-----0001_GT----  
                   >----SEG------  
                   >----TRA------ 
        
        """
        root = self.root
        fm = ("%0" + str(self.digit_num) + "d") % self.dt_id
        if not os.path.isdir(os.path.join(root, fm)) and not os.path.isdir(os.path.join(root, fm + '_RES')) and \
                not os.path.isdir(os.path.join(root, fm + '_GT')):
            os.mkdir(os.path.join(root, fm))
            os.mkdir(os.path.join(root, fm + '_RES'))
            os.mkdir(os.path.join(root, fm + '_GT'))
            os.mkdir(os.path.join(root, fm + '_GT', 'SEG'))
            os.mkdir(os.path.join(root, fm + '_GT', 'TRA'))
        else:
            raise IOError('Directory already existed.')
        return

    def generate_Trk(self, raw, mask, displace=100, gap_fill=5, out_dir=None, track=None, render_phase=False):
        """Generate deepcell caliban readable trk file

        Args:
            out_dir (str): output directory of the file (optional, default root)
            raw (numpy.ndarray): raw image
            mask (numpy.ndarray): image mask
            displace (int): for tracking, maximum displace of two linked objects between frame (default: 100)
            gap_fill (int): for tracking, memory track for some frames is disappeared (default: 5)
            track (pandas.DataFrame): tracked object table (optional to suppress tracking on mask)
            render_phase (bool): if true, will resolve labels on mask into cell cycle phase, the label should
            follow {10: 'G1/G2', 50: 'S', 100: 'M', 200: 'G1/G2'}
        """
        if out_dir is None:
            out_dir = self.root
        tracked = generate_calibanTrk(raw=raw, mask=mask, out_dir=out_dir, dt_id=self.dt_id, digit_num=self.digit_num,
                            track=track, displace=displace, gap_fill=gap_fill, render_phase=render_phase)
        return tracked

    def evaluate(self):
        """Call CTC evaluation software to run ((Unix) Linux/Mac only)
        """
        fm = ("%0" + str(self.digit_num) + "d") % self.dt_id
        if self.path_ctc_software is None:
            raise ValueError('CTC evaluation software path not set yet. Call through pcna_ctcEvaluator.set_evSoft()')
        wrap_root = "\"" + self.root + "\""
        wrap_tra = "\"" + os.path.join(self.path_ctc_software, 'TRAMeasure') + "\""
        wrap_seg = "\"" + os.path.join(self.path_ctc_software, 'SEGMeasure') + "\""
        subprocess.run(wrap_tra + ' ' + wrap_root + ' ' + fm + ' ' + str(
            self.digit_num), shell=True)
        subprocess.run(wrap_seg + ' ' + wrap_root + ' ' + fm + ' ' + str(
            self.digit_num), shell=True)
        return
