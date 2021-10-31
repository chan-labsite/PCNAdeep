# -*- coding: utf-8 -*-
# Modified by Yifan Gui from FAIR Detectron2, Apache 2.0 licence.
import atexit
import bisect
import multiprocessing as mp
import torch
import numpy as np
import skimage.measure as measure
import skimage.morphology as morphology
import copy
import pandas as pd

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from pcnaDeep.data.utils import filter_edge, expand_bbox


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Copied from Facebook Detectron2 Demo. Apache 2.0 Licence.

        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        ).set(thing_classes=['G1/G2', 'S', 'M', 'E'])
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image, vis=True):
        """
        Adapted from Facebook Detectron2 Demo. Apache 2.0 Licence.

        Args:
            image (numpy.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.

            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        if vis == False:
            return predictions
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output


class AsyncPredictor:
    """
    Copied from Facebook Detectron2 Demo. Apache 2.0 Licence.

    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
        
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5


def pred2json(mask, label_table, fp):
    """Transform detectron2 prediction to VIA2 (VGG Image Annotator) json format.

    Args:
        mask (numpy.ndarray): binary mask with all instance labeled with unique label.
        label_table (pandas.DataFrame): metadata of the mask; must contain `continuous_label`, `predicted_class` and
            `emerging` columns.
        fp (str): file name for this frame.

    Returns:
        dict: json format readable by VIA2 annotator.
    """

    region_tmp = {"shape_attributes": {"name": "polygon", "all_points_x": [], "all_points_y": []},
                  "region_attributes": {"phase": ''}}

    if np.sum(mask) == 0:
        return {}

    tmp = {"filename": fp, "size": mask.astype('bool').size, "regions": [], "file_attributes": {}}
    for region in measure.regionprops(label_image=mask, intensity_image=None):
        if region.image.shape[0] < 2 or region.image.shape[1] < 2:
            continue
        # register regions
        cur_tmp = copy.deepcopy(region_tmp)
        bbox = list(region.bbox)
        bbox[0], bbox[1] = bbox[1], bbox[0]  # swap x and y
        bbox[2], bbox[3] = bbox[3], bbox[2]
        ct = measure.find_contours(region.image, 0.5)
        if len(ct) < 1:
            continue
        ct = ct[0]
        if ct[0][0] != ct[-1][0] or ct[0][1] != ct[-1][1]:
            # non connected
            ct_image = np.zeros((bbox[3] - bbox[1] + 2, bbox[2] - bbox[0] + 2))
            ct_image[1:-1, 1:-1] = region.image.copy()
            ct = measure.find_contours(ct_image, 0.5)[0]
            # edge = measure.approximate_polygon(ct, tolerance=0.001)
            edge = ct
            for k in range(len(edge)):  # swap x and y
                x = edge[k][0] - 1
                if x < 0:
                    x = 0
                elif x > region.image.shape[0] - 1:
                    x = region.image.shape[0] - 1
                y = edge[k][1] - 1
                if y < 0:
                    y = 0
                elif y > region.image.shape[1] - 1:
                    y = region.image.shape[1] - 1
                edge[k] = [y, x]
            edge = edge.tolist()
            elements = list(map(lambda x: tuple(x), edge))
            edge = list(set(elements))
            edge.sort(key=elements.index)
            edge = np.array(edge)
            edge[:, 0] += bbox[0]
            edge[:, 1] += bbox[1]
            edge = list(edge.ravel())
            edge += edge[0:2]
        else:
            edge = ct
            for k in range(len(edge)):  # swap x and y
                edge[k] = [edge[k][1], edge[k][0]]
            edge[:, 0] += bbox[0]
            edge[:, 1] += bbox[1]
            edge = list(edge.ravel())
        cur_tmp['shape_attributes']['all_points_x'] = edge[::2]
        cur_tmp['shape_attributes']['all_points_y'] = edge[1::2]
        sub_label = label_table[label_table['continuous_label'] == region.label]
        cur_tmp['region_attributes']['phase'] = sub_label['phase'].iloc[0]
        if sub_label['emerging'].iloc[0] == 1:
            cur_tmp['region_attributes']['phase'] = 'E'
        tmp['regions'].append(cur_tmp)

    return tmp


def predictFrame(img, frame_id, demonstrator, is_gray=False, size_flt=1000, edge_flt=50):
    """Predict single frame and deduce meta information.
    
    Args:
        img (numpy.ndarray): must be `uint8` image slice.
        frame_id (int): index of the slice, start from 0.
        demonstrator (VisualizationDemo): an detectron2 demonstrator object.
        size_flt (int): size filter, in pixel^2.
        is_gray (bool): whether the slice is gray. If true, will convert to 3 channels at first.
        edge_flt (int): filter objects at the edge, whose classification may be imprecise, in pixel.

    Returns:
        tuple: labeled mask and corresponding table.
    """

    if is_gray:
        img = np.stack([img, img, img], axis=2)  # convert gray to 3 channels
    # Generate mask or visualized output
    predictions = demonstrator.run_on_image(img, vis=False)

    # Generate mask
    mask = predictions['instances'].pred_masks
    mask = mask.char().cpu().numpy()
    mask_slice = np.zeros((mask.shape[1], mask.shape[2])).astype('uint16')  # uint16 locks object detection within 65536

    # For visualising class prediction
    # 0: G1/G2, 1: S, 2: M, 3: E-early G1
    cls = predictions['instances'].pred_classes
    conf = predictions['instances'].scores_all.cpu().numpy()
    factor = {0: 'G1/G2', 1: 'S', 2: 'M', 3: 'E'}
    ovl_count = 0
    for s in range(mask.shape[0]):
        if np.sum(mask[s, :, :]) < size_flt:
            continue
        sc = np.max(conf[s])
        ori = np.max(mask_slice[mask[s, :, :] != 0])
        if ori != 0:
            ovl_count += 1
            if sc <= np.max(conf[ori - 1]):
                mask[s, mask_slice == ori] = 0
        mask_slice[mask[s, :, :] != 0] = s + 1

    img_relabel = measure.label(mask_slice, connectivity=1) 
    # original segmentation may have separated region, flood and re-label it
    img_relabel = morphology.remove_small_objects(img_relabel, min_size=size_flt)
    img_relabel = measure.label(img_relabel, connectivity=1)
    props_relabel = measure.regionprops_table(img_relabel,intensity_image=img[:,:,0],  properties=(
                    'label', 'bbox', 'centroid', 'mean_intensity', 'major_axis_length', 'minor_axis_length'))
    props_relabel = pd.DataFrame(props_relabel)
    props_relabel.columns = ['continuous_label','bbox-0', 'bbox-1', 'bbox-2', 'bbox-3',
                             'Center_of_the_object_0', 'Center_of_the_object_1', 'mean_intensity', 
                             'major_axis', 'minor_axis']

    original_labels = []
    for i in range(props_relabel.shape[0]):
        original_labels.append(int(mask_slice[int(props_relabel['Center_of_the_object_0'].iloc[i]), 
                                              int(props_relabel['Center_of_the_object_1'].iloc[i])]))
    out_props = props_relabel.copy()
    out_props['label'] = original_labels
    out_props['frame'] = frame_id

    phase = []
    g_confid = []
    s_confid = []
    m_confid = []
    e = []
    background = []
    dic_mean = []
    dic_std = []
    for row in range(out_props.shape[0]):
        lb_image = int(out_props.iloc[row]['continuous_label'])
        lb_ori = int(out_props.iloc[row]['label'])

        # get background intensity
        b1,b3,b2,b4 = expand_bbox((out_props.iloc[row]['bbox-0'], out_props.iloc[row]['bbox-1'],
                                   out_props.iloc[row]['bbox-2'], out_props.iloc[row]['bbox-3']),
                                  2, img_relabel.shape)
        obj_region = img_relabel[b1:b2, b3:b4].copy()
        its_region = img[b1:b2, b3:b4, 0].copy()
        dic_region = img[b1:b2, b3:b4, 2].copy()
        background.append(np.mean(its_region[obj_region == 0]))
        dic_mean.append(np.mean(dic_region[obj_region == lb_image]))
        dic_std.append(np.std(dic_region[obj_region == lb_image]))
        
        # get confidence score and emerging status
        p = factor[cls[lb_ori - 1].item()]
        if p == 'E':
            p = 'G1/G2'
            e.append(1)
        else:
            e.append(0)
        confid = conf[lb_ori - 1]
        phase.append(p)
        s_confid.append(confid[1])
        g_confid.append(confid[0] + confid[3])
        m_confid.append(confid[2])

    out_props['phase'] = phase
    out_props['Probability of G1/G2'] = g_confid
    out_props['Probability of S'] = s_confid
    out_props['Probability of M'] = m_confid
    out_props['emerging'] = e
    out_props['background_mean'] = background
    out_props['BF_mean'] = dic_mean
    out_props['BF_std'] = dic_std
    
    del out_props['label']
    return filter_edge(img_relabel, out_props, edge_flt)
