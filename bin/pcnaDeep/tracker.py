# -*- coding: utf-8 -*-
import gc
import trackpy as tp
import skimage.measure as measure
import skimage.io as io
from skimage.util import img_as_uint
from skimage.morphology import remove_small_objects
import pandas as pd
import numpy as np
from pcnaDeep.data.utils import json2mask, expand_bbox, getDetectInput


def track(df, displace=40, gap_fill=5):
    """Track and relabel mask with trackID.

    Args:
        df (pandas.DataFrame): Data frame with fields:
            - Center_of_the_object_0: x location of each object.
            - Center_of_the_object_1: y location of each object.
            - frame: time location.
            - BF_mean: mean intensity of bright field image.
            - BF_std: standard deviation of bright field image.
            - (other optional columns)

        displace (int): maximum distance an object can move between frames.
        gap_fill (int): temporal filling fo tracks.
    
    Return:
        (pandas.DataFrame): tracked object table.
    """
    TRACK_WITH_DIC = True

    f = df[['Center_of_the_object_0', 'Center_of_the_object_1', 'BF_mean', 'BF_std', 'frame']]
    f.columns = ['x', 'y', 'BF_mean', 'BF_std', 'frame']
    if TRACK_WITH_DIC:
        pc = f.columns[:-1]
    else:
        pc = ['x','y']
    t = tp.link(f, search_range=displace, memory=gap_fill, adaptive_stop=0.4 * displace, pos_columns=pc)
    t.columns = ['Center_of_the_object_0', 'Center_of_the_object_1', 'BF_mean', 'BF_std', 'frame', 'trackId']
    out = pd.merge(df, t, on=['Center_of_the_object_0', 'Center_of_the_object_1', 'BF_mean', 'BF_std', 'frame'])
    #  change format for downstream
    out['trackId'] += 1
    out['lineageId'] = out['trackId']
    out['parentTrackId'] = 0
    out = out[
        ['frame', 'trackId', 'lineageId', 'parentTrackId', 'Center_of_the_object_0', 'Center_of_the_object_1', 'phase',
         'Probability of G1/G2', 'Probability of S', 'Probability of M', 'continuous_label', 'major_axis', 'minor_axis',
         'mean_intensity', 'emerging', 'background_mean', 'BF_mean', 'BF_std']]
    names = list(out.columns)
    names[4] = 'Center_of_the_object_1'
    names[5] = 'Center_of_the_object_0'
    names[6] = 'predicted_class'
    out.columns = names
    out = out.sort_values(by=['trackId', 'frame'])

    return out


def track_mask(mask, displace=40, gap_fill=5, render_phase=False, size_min=100, PCNA_intensity=None, BF_intensity=None):
    """Track binary mask objects.

    Args:
        mask (numpy.ndarray): cell mask, can either be binary or labeled with cell cycle phases.
        displace (int): distance restriction, see `track()`.
        gap_fill (int): time restriction, see `track()`.
        render_phase (bool): whether to deduce cell cycle phase from the labeled mask.
        size_min (int): remove object smaller then some size, in case the mask labeling is not precise.
        PCNA_intensity (numpy.ndarray): optional, if supplied, will extract fore/backgound PCNA intensity,
        BF_intensity (numpy.ndarray): optional, if supplied, will extract bright field intensity & std for tracking.
            First three channels must have same length as the mask.

    Returns:
        (pandas.DataFrame): tracked object table.
        (mask_lbd): mask with each frame labeled with object IDs.
    """
    BBOX_FACTOR = 2  # dilate the bounding box when calculating the background intensity.
    PHASE_DIC = {10: 'G1/G2', 50: 'S', 100: 'M', 200: 'G1/G2'}
    p = pd.DataFrame()
    mask_lbd = np.zeros(mask.shape)
    h = mask.shape[1]
    w = mask.shape[2]
    
    for i in range(mask.shape[0]):
        # remove small objects
        mask_lbd[i, :, :] = measure.label(mask[i, :, :], connectivity=1).astype('uint16')

    if np.max(mask_lbd) <= 255:
        mask_lbd = mask_lbd.astype('uint8')
    else:
        mask_lbd = img_as_uint(mask_lbd)

    mask_lbd = remove_small_objects(mask_lbd, min_size=size_min, connectivity=1)
    mask[mask_lbd == 0] = 0

    if PCNA_intensity is None or BF_intensity is None:
        PCNA_intensity = mask.copy()
        BF_intensity = mask.copy()

    for i in range(mask.shape[0]):
        props = measure.regionprops_table(mask_lbd[i, :, :], intensity_image=mask[i, :, :],
                                          properties=('bbox', 'centroid', 'label', 'max_intensity',
                                                      'major_axis_length', 'minor_axis_length'))
        props = pd.DataFrame(props)
        props.columns = ['bbox-0', 'bbox-1', 'bbox-2', 'bbox-3', 'Center_of_the_object_0', 'Center_of_the_object_1',
                         'continuous_label', 'max_intensity', 'major_axis', 'minor_axis']
        l = props['max_intensity']
        phase = []
        probG = []
        probS = []
        probM = []
        e = []
        background = []
        its = []
        dic_mean = []
        dic_std = []

        for k in range(props.shape[0]):
            if render_phase:
                # render phase
                ps = PHASE_DIC[int(l[k])]
                if int(l[k]) == 200:
                    e.append(1)
                else:
                    e.append(0)
                phase.append(ps)
                if ps == 'G1/G2':
                    probG.append(1)
                    probS.append(0)
                    probM.append(0)
                elif ps == 'S':
                    probG.append(0)
                    probS.append(1)
                    probM.append(0)
                else:
                    probG.append(0)
                    probS.append(0)
                    probM.append(1)
            else:
                probG.append(0)
                probS.append(0)
                probM.append(0)
                e.append(0)
                phase.append(0)
            # extract intensity
            b1, b3, b2, b4 = expand_bbox((props.iloc[k][0], props.iloc[k][1],
                                          props.iloc[k][2], props.iloc[k][3]), BBOX_FACTOR, (h,w))
            lbd = int(props.iloc[k][6])
            obj_region = mask_lbd[i, b1:b2, b3:b4].copy()
            its_region = PCNA_intensity[i, b1:b2, b3:b4].copy()
            dic_region = BF_intensity[i, b1:b2, b3:b4].copy()
            if 0 not in obj_region:
                background.append(0)
            else:
                background.append(np.mean(its_region[obj_region == 0]))
            cal = obj_region == lbd
            its.append(np.mean(its_region[cal]))
            dic_mean.append(np.mean(dic_region[cal]))
            dic_std.append(np.std(dic_region[cal]))

        props['Probability of G1/G2'] = probG
        props['Probability of S'] = probS
        props['Probability of M'] = probM
        props['emerging'] = e
        props['phase'] = phase
        props['frame'] = i
        props['mean_intensity'] = its
        props['background_mean'] = background
        props['BF_mean'] = dic_mean
        props['BF_std'] = dic_std
        del props['max_intensity'], props['bbox-0'], props['bbox-1'], props['bbox-2'], props['bbox-3']
        p = p.append(props)

    track_out = track(p, displace=displace, gap_fill=gap_fill)
    return track_out, mask_lbd


def track_GT_json(fp_json, height=1200, width=1200, displace=40, gap_fill=5, size_min=100,
                  fp_intensity_image=None, fp_pcna=None, fp_bf=None,
                  sat=None, gamma=None):
    """Track ground truth VIA json file. Wrapper of `track_mask()`

    Args:
        fp_json (str): file path to the json file.
        height (int): pixel height of the mask corresponding to GT json.
        width (int): pixel width of the mask corresponding to GT json.
        displace (int): distance restriction, see `track()`.
        gap_fill (int): time restriction, see `track()`.
        size_min (int): remove object smaller then some size, in case the mask labeling is not precise.
        fp_intensity_image (str): optional image file path, if supplied, will extract fore/backgound PCNA intensity, and
            bright field intensity/std for tracking.
            Must has the same shape as mask, so will override height and width.
        fp_pcna (str): optional file path to PCNA channel image stack.
        fp_bf (str): optional file path to bright field image stack.
        sat (float): saturated pixel percentage when rescaling intensity image. If `None`, no rescaling will be done.
        gamma (float): gamma-correction factor. If `None`, will not perform.

    Returns:
        (pandas.DataFrame): tracked object table.
        (mask_lbd): mask with each frame labeled with object IDs.

    Note:
        - If supplied with `fp_intensity_image` (composite image stack), will omit `fp_pcna` or `fp_bf`.
        - `fp_pcna` and `fp_bf` must be supplied at the same time.
    """
    if fp_intensity_image:
        intensity_image = io.imread(fp_intensity_image)
        if len(intensity_image.shape) < 4:
            raise ValueError('Not enough intensity image dimensions! Both PCNA and bright field channels required.')
        height = intensity_image.shape[1]
        width = intensity_image.shape[2]
        PCNA_intensity = intensity_image[:,:,:,0]
        BF_intensity = intensity_image[:,:,:,-1]
    elif fp_pcna is not None and fp_bf is not None:
        PCNA_intensity = io.imread(fp_pcna)
        BF_intensity = io.imread(fp_bf)
    elif fp_pcna is None or fp_bf is None:
        raise ValueError('PCNA channel image stack must be supplied with bright field together.')
    else:
        PCNA_intensity = None
        BF_intensity = None

    if sat and PCNA_intensity is not None:
        if gamma is None:
            gamma = 1
        comp = getDetectInput(PCNA_intensity, BF_intensity, sat, gamma)
        PCNA_intensity = comp[:, :, :, 0].copy()
        BF_intensity = comp[:, :, :, -1].copy()
        del comp
        gc.collect()

    mask = json2mask(fp_json, out='', height=height, width=width, label_phase=True, mask_only=True)
    return track_mask(mask, displace=displace, gap_fill=gap_fill, size_min=size_min, PCNA_intensity=PCNA_intensity,
                      BF_intensity=BF_intensity, render_phase=True)
