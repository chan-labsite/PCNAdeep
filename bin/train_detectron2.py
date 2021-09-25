# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Yifan Gui from FAIR Detectron2, Apache 2.0 licence.
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_train_loader, DatasetCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from pcnaDeep.data.preparePCNA import load_PCNAs_json
from detectron2 import model_zoo
import detectron2.data.transforms as T


def build_sem_seg_train_aug(cfg):
    
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )
    ]
    
    if cfg.INPUT.CROP.ENABLED:
        augs.append(
            T.RandomCrop(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
            )
        )
    augs.append(T.RandomFlip())
    augs.append(T.RandomSaturation(0.5, 1.5))
    augs.append(T.RandomRotation([0,90,270], sample_style='choice'))
    return augs


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(
                COCOEvaluator(dataset_name=dataset_name, output_dir=output_folder, tasks=('segm',), distributed=True))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                    torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                    torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=build_sem_seg_train_aug(cfg))
        return build_detection_train_loader(cfg, mapper=mapper)


def register_train(cfg):
    DatasetCatalog.register("pcna", lambda: load_PCNAs_json(cfg.TRAIN_ANN_PATH, cfg.TRAIN_PATH))
    MetadataCatalog.get("pcna").set(thing_classes=cfg.CLASS_NAMES, evaluator_type='coco')


def register_test(cfg):
    DatasetCatalog.register("pcna_test", lambda: load_PCNAs_json(cfg.TEST_ANN_PATH, cfg.TEST_PATH))
    MetadataCatalog.get("pcna_test").set(thing_classes=cfg.CLASS_NAMES, evaluator_type='coco')


def setup(args):
    """
    Create configs and perform basic setups.
    """

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    #cfg.MODEL.WEIGHTS = "../output/mulscale_sat_rot/model_final.pth"
    cfg.DATASETS.TRAIN = ("pcna",)
    cfg.DATASETS.TEST = ("pcna_test",)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.SOLVER.BASE_LR = 0.1
    cfg.SOLVER.WARMUP_ITERS = 400
    cfg.SOLVER.WARMUP_FACTOR = 1/cfg.SOLVER.WARMUP_ITERS
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.STEPS = (800, 1400, 1800, 2200)
    cfg.SOLVER.CHECKPOINT_PERIOD = 2000
    cfg.TEST.EVAL_PERIOD = 600  # 600

    cfg.SOLVER.MAX_ITER = 2800  #  2800
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # change according to class number
    cfg.TEST.DETECTIONS_PER_IMAGE = 1024

    # Avoid overlapping
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5  #  default 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05  #  default 0.05
    #cfg.MODEL.PANOPTIC_FPN.COMBINE.OVERLAP_THRESH = 0.9  #  default 0.5
    cfg.MODEL.RPN.NMS_THRESH = 0.7  #  default 0.7

    # Augmentation
    cfg.INPUT.MIN_SIZE_TRAIN = 1200
    cfg.INPUT.MAX_SIZE_TRAIN = 2048
    cfg.INPUT.MIN_SIZE_TEST = 0  # no resize when test
    cfg.INPUT.MAX_SIZE_TEST = 2048
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = 'choice'
    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.CROP.TYPE = 'relative'
    cfg.INPUT.CROP.SIZE = [0.8, 0.8]
    cfg.TEST.AUG.ENABLED = False

    ### Data
    # Class metadata
    cfg.CLASS_NAMES = ["G1/G2", "S", "M", "E"]
    
    # Dataset metadata
    ###========= USER: Specify metadata below ====================
    DATASET_ROOT = ''  # USER: Specify the root directory to the datasets.
    TRAIN_PREFIX = ['d1', 'd2']  # USER: Specify the folder name of the training datasets.
    TEST_PREFIX = ['t1']  # USER: Specify the folder name of the testing datasets.
    ###============================================================

    TRAIN_PATH = []
    TRAIN_ANN_PATH = []
    for p in TRAIN_PREFIX:
        TRAIN_PATH.append(os.path.join(DATASET_ROOT, p))
        TRAIN_ANN_PATH.append(os.path.join(DATASET_ROOT, p+'.json'))

    cfg.TRAIN_PATH = TRAIN_PATH
    cfg.TRAIN_ANN_PATH = TRAIN_ANN_PATH

    TEST_PATH = []
    TEST_ANN_PATH = []
    for p in TEST_PREFIX:
        TEST_PATH.append(os.path.join(DATASET_ROOT, p))
        TEST_ANN_PATH.append(os.path.join(DATASET_ROOT, p+'.json'))
    cfg.TEST_PATH = TEST_PATH
    cfg.TEST_ANN_PATH = TEST_ANN_PATH

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    register_train(cfg)
    register_test(cfg)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    #cfg = setup(args)
    #register_test(cfg)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,)
    )
