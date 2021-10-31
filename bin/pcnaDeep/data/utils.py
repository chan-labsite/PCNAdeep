# -*- coding: utf-8 -*-
import copy
import json
import os
import re
import torch
import numpy as np
import pandas as pd
import skimage.exposure as exposure
import skimage.io as io
import skimage.measure as measure
from PIL import Image, ImageDraw
from skimage.util import img_as_ubyte
import warnings


def json2mask(ip, height, width, out=None, label_phase=False, mask_only=False):
    """Draw mask according to VIA2 annotation and summarize information

    Args:
        ip (str): input directory of the json file.
        out (str): optonal, output directory of the image and summary table.
        height (int): image height.
        width (int): image width.
        label_phase (bool): whether to label the mask with values corresponding to cell cycle classification or not. 
            If true, will label as the following values: 'G1/G2':10, 'S':50, 'M':100;
            If false, will output binary masks.
        mask_only (bool): whether to suppress file output and return mask only.

    Outputs:
        `png` files of object masks.
    """

    OUT_PHASE = label_phase
    PHASE_DIS = {"G1/G2": 10, "S": 50, "M": 100, "E": 200}
    stack = []
    with open(ip, 'r', encoding='utf8')as fp:
        j = json.load(fp)
        if '_via_img_metadata' in list(j.keys()):
            j = j['_via_img_metadata']
        for key in list(j.keys()):
            img = Image.new('L', (height, width))
            dic = j[key]
            objs = dic['regions']  # containing all object areas
            draw = ImageDraw.Draw(img)
            for o in objs:
                x = o['shape_attributes']['all_points_x']
                y = o['shape_attributes']['all_points_y']
                xys = [0 for i in range(len(x) + len(y))]
                xys[::2] = x
                xys[1::2] = y
                phase = o['region_attributes']['phase']
                draw.polygon(xys, fill=PHASE_DIS[phase], outline=0)
            img = np.array(img)

            if not OUT_PHASE:
                img = img_as_ubyte(img.astype('bool'))
            if mask_only:
                stack.append(img)
            else:
                if out is None:
                    out = '.'
                io.imsave(os.path.join(out, dic['filename']), img)
        if mask_only:
            return np.stack(stack, axis=0)

    return


def mask2json(in_dir, out_dir, phase_labeled=False, phase_dic={10: "G1/G2", 50: "S", 100: "M", 200: 'E'},
              prefix='object_info'):
    """Generate VIA2-readable json file from masks

    Args:
        in_dir (str): input directory of mask slices in .png format. Stack input is not implemented.
        out_dir (str): output directory for .json output
        phase_labeled (bool): whether cell cycle phase has already been labeled. 
            If true, a phase_dic variable should be supplied to resolve phase information.
        phase_dic (dic): lookup dictionary of cell cycle phase labeling on the mask.
        prefix (str): prefix of .json output.
    
    Outputs:
        prefix.json in VIA2 format. Note the output is not a VIA2 project, so default image directory
            must be set for the first time of labeling.
    """
    out = {}
    region_tmp = {"shape_attributes": {"name": "polygon", "all_points_x": [], "all_points_y": []},
                  "region_attributes": {"phase": "G1/G2"}}

    imgs = os.listdir(in_dir)
    for i in imgs:
        if re.search('.png', i):

            img = io.imread(os.path.join(in_dir, i))
            # img = binary_erosion(binary_erosion(img.astype('bool')))
            img = img.astype('bool')
            tmp = {"filename": os.path.join(i), "size": img.size, "regions": [], "file_attributes": {}}
            regions = measure.regionprops(measure.label(img, connectivity=1), img)
            for region in regions:
                if region.image.shape[0] < 2 or region.image.shape[1] < 2:
                    continue
                # register regions
                cur_tmp = copy.deepcopy(region_tmp)
                if phase_labeled:
                    cur_tmp['region_attributes']['phase'] = phase_dic[int(region.mean_intensity)]
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
                    # edge = measure.approximate_polygon(ct, tolerance=0.4)
                    edge = ct
                    for k in range(len(edge)):  # swap x and y
                        edge[k] = [edge[k][1], edge[k][0]]
                    edge[:, 0] += bbox[0]
                    edge[:, 1] += bbox[1]
                    edge = list(edge.ravel())
                cur_tmp['shape_attributes']['all_points_x'] = edge[::2]
                cur_tmp['shape_attributes']['all_points_y'] = edge[1::2]
                tmp['regions'].append(cur_tmp)
            out[i] = tmp

    with(open(os.path.join(out_dir, prefix + '.json'), 'w', encoding='utf8')) as fp:
        json.dump(out, fp)
    return


def getDetectInput(pcna, dic, gamma=1, sat=1, torch_gpu=False):
    """Generate pcna-mScarlet and DIC channel to RGB format for detectron2 model prediction

    Args:
        pcna (numpy.ndarray): uint16 PCNA-mScarlet image stack (T*H*W).
        dic (numpy.ndarray): uint16 DIC or phase contrast image stack.
        gamma (float): gamma adjustment, >0, default 0.8.
        sat (float): percent saturation, 0~100, default 0.
        torch_gpu (bool): use torch to speed up calculation.

    Returns:
        (numpy.ndarray): uint8 composite image (T*H*W*C)
    """
    stack = pcna
    dic_img = dic
    if stack.dtype != np.dtype('uint16') or dic_img.dtype != np.dtype('uint16'):
        raise ValueError('Input image must be in uint16 format.')
    if sat < 0 or sat > 100:
        raise ValueError('Saturated pixel should not be negative or exceeds 100')

    print("Saturation: " + str(sat) + ", Gamma " + str(gamma))
    if len(stack.shape) < 3:
        stack = np.expand_dims(stack, axis=0)
        dic_img = np.expand_dims(dic_img, axis=0)

    outs = []
    rg = (sat, 100-sat)
    for f in range(stack.shape[0]):
        # rescale mCherry intensity
        fme = exposure.adjust_gamma(stack[f, :, :], gamma)
        fme = exposure.rescale_intensity(fme, in_range=tuple(np.percentile(fme, rg)))
        dic_img[f, :, :] = exposure.rescale_intensity(dic_img[f, :, :],
                                                      in_range=tuple(np.percentile(dic_img[f, :, :], rg)))

        # save two-channel image for downstream
        fme = img_as_ubyte(fme)
        dic_slice = img_as_ubyte(dic_img[f, :, :])
        slice_list = [fme, fme, dic_slice]

        s = np.stack(slice_list, axis=2)
        if torch_gpu:
            s = torch.from_numpy(s)
        outs.append(s)
    
    if torch_gpu:
        final_out = torch.stack(outs, axis=0).numpy()
    else:
        final_out = np.stack(outs, axis=0)
    print("Shape: ", final_out.shape)
    return final_out


def retrieve(table, mask, image, rp_fields=[], funcs=[]):
    """Retrieve extra skimage.measure.regionprops fields of every object;
        Or apply customized functions to extract features form the masked object.
        
    Args:
        table (pandas.DataFrame): object table tracked or untracked, 
            should have 2 fields:
            1. frame: time location; 
            2. continuous label: region label on mask
        mask (numpy.ndarray): labeled mask corresponding to table
        image (numpy.ndarray): intensity image, only the first channel allowed
        rp_fields (list(str)): skimage.measure.regionpprops allowed fields
        funcs (list(function)): customized function that outputs one value from
            an array input
            
    Returns:
        labeled object table with additional columns
    """
    track = table
    if rp_fields == [] and funcs == []:
        return track

    new_track = pd.DataFrame()
    track = track.sort_values(by=['frame', 'continuous_label'])
    for f in np.unique(track['frame']).tolist():
        sl = mask[f, :, :]
        img = image[f, :, :]
        sub = track[track['frame'] == f]

        if rp_fields:
            if 'label' not in rp_fields:
                rp_fields.append('label')
            props = pd.DataFrame(measure.regionprops_table(sl, img, properties=tuple(rp_fields)))
            new = pd.merge(sub, props, left_on='continuous_label', right_on='label')
            new = new.drop(['label'], axis=1)

        if funcs:
            p = measure.regionprops(sl, img)
            out = {'label': []}
            for fn in funcs:
                out[fn.__name__] = []
            for i in p:
                out['label'].append(i.label)
                i_img = img.copy()
                i_img[sl != i.label] = 0
                for fn in funcs:
                    out[fn.__name__].append(fn(i_img))
            new2 = pd.DataFrame(out)
            if rp_fields:
                new2 = pd.merge(new, new2, left_on='continuous_label', right_on='label')
                new2 = new2.drop(['label'], axis=1)
            else:
                new2 = pd.merge(sub, new2, left_on='continuous_label', right_on='label')

        if rp_fields:
            if funcs:
                new_track = new_track.append(new2)
            else:
                new_track = new_track.append(new)
        elif funcs:
            new_track = new_track.append(new2)

    return new_track


def mt_dic2mt_lookup(mt_dic):
    """Convert mt_dic to mitosis lookup
    
    Args:
        mt_dic (dict): standard mitosis info dictionary in pcnaDeep
    
    Returns:
        mt_lookup (pd.DataFrame): mitosis lookup table with 3 columns:
            trackA (int) | trackB (int) | Mitosis? (int, 0/1)
    """
    out = {'par': [], 'daug': [], 'mitosis': []}
    for i in list(mt_dic.keys()):
        for j in list(mt_dic[i]['daug'].keys()):
            out['par'].append(i)
            out['daug'].append(j)
            out['mitosis'].append(1)
    return pd.DataFrame(out)


def get_outlier(array, col_ids=None):
    """Get outlier index in an array, specify target column
    
    Args:
        array (numpy.ndarray): original array
        col_ids ([int]): target columns to remove outliers. Default all
        
    Returns:
        index of row containing at least one outlier
    """
    
    if col_ids is None:
        col_ids = list(range(array.shape[1]))
    
    idx = []
    for c in col_ids:
        col = array[:,c]
        idx.extend(list(np.where(np.abs(col - np.mean(col)) > 3 * np.std(col))[0]))
    
    idx = list(set(idx))
    idx.sort()
    return idx


def deduce_transition(l, tar, confidence, min_tar, max_res, escape=0, casual_end=True):
    """ Deduce mitosis exit and entry based on adaptive searching

        Args:
            l (list): list of the target cell cycle phase
            tar (str): target cell cycle phase
            min_tar (int): minimum duration of an entire target phase
            confidence (numpy.ndarray): matrix of confidence
            max_res (int): maximum accumulative duration of unwanted phase
            escape (int): do not consider the first n instances
            casual_end (bool): at the end of the track, whether loosen criteria of a match

        Returns:
            tuple: two indices of the classification list corresponding to entry and exit
    """
    mp = {'G1/G2': 0, 'S': 1, 'M': 2}
    confid_cls = list(map(lambda x: confidence[x, mp[l[x]]], range(confidence.shape[0])))
    idx = np.array([i for i in range(len(l)) if l[i] == tar])
    idx = idx[idx >= escape].tolist()
    if len(idx) == 0:
        return None
    if len(idx) == 1:
        return idx[0], idx[0]
    found = False
    i = 0
    g_panelty = 0
    acc_m = confid_cls[idx[0]]
    cur_m_entry = idx[i]
    m_exit = None
    while i < len(idx) - 1:
        acc_m += confid_cls[idx[i + 1]]
        g_panelty += np.sum(confid_cls[idx[i] + 1:idx[i + 1]])
        if acc_m >= min_tar:
            found = True
            if g_panelty < max_res:
                g_panelty = 0
                acc_m = 0
        if g_panelty >= max_res:
            if found:
                m_exit = idx[i]
                break
            else:
                g_panelty = 0
                acc_m = 0
                cur_m_entry = idx[i + 1]
        i += 1
    if i == (len(idx) - 1) and found:
        m_exit = idx[-1]
    elif g_panelty < max_res and (found or 
                                  casual_end and idx[i] - cur_m_entry + 1 >= min_tar and cur_m_entry != idx[-1]):
        found = True
        m_exit = idx[-1]
       
    '''
    elif casual_end and i == (len(idx) - 1) and g_panelty < max_res and not found and cur_m_entry != idx[-1]:
        found = True
        m_exit = idx[-1]
        if m_exit - cur_m_entry + 1 < min_tar:
            return None
    '''

    if found and m_exit is not None:
        return cur_m_entry, m_exit
    else:
        return None


def find_daugs(track, track_id):
    """Return list of daughters according to certain parent track ID.

    Args:
        track (pandas.DataFrame): tracked object table.
        track_id (int): track ID.
    """
    rt = list(np.unique(track.loc[track['parentTrackId'] == track_id, 'trackId']))
    if not rt:
        return []
    else:
        to_rt = rt.copy()
        for trk in rt:
            to_rt.extend(find_daugs(track, trk))
        return to_rt


def filter_edge(img, props, edge_flt):
    """Filter objects at the edge

    Args:
        img (numpy.ndarray): mask image with object labels.
        props (pandas.DataFrame): part of the object table.
        edge_flt (int): pixel width of the edge area.
    """
    ebd = np.zeros((img.shape[0] - 2 * edge_flt, img.shape[1] - 2 * edge_flt))
    ebd = np.pad(ebd, ((edge_flt, edge_flt), (edge_flt, edge_flt)), mode='constant', constant_values=(1, 1))
    for i in props.index:
        if ebd[int(props['Center_of_the_object_0'].loc[i]), int(props['Center_of_the_object_1'].loc[i])] == 1:
            img[img == props['continuous_label'].loc[i]] = 0
            props = props.drop(index=i)

    return img, props


def expand_bbox(bbox, factor, limit):
    """Expand bounding box by factor times.

    Args:
        bbox (tuple): (x1, y1, x2, y2).
        factor (float): positive value, expand height and width by multiplying the factor.
            Round if result is not integer.
            The output shape will be (factor + 1) ** 2 times of the original size.
        limit (tuple): (x_max, y_max), limit values to avoid boundary crush.

    Returns:
        (tuple): new bounding box (x1, y1, x2, y2).
    """
    if factor < 0:
        raise ValueError('Must expand bounding box with a positive factor.')

    h = bbox[2] - bbox[0]
    w = bbox[3] - bbox[1]
    factor = factor / 2
    x1, y1, x2, y2 = bbox
    x1 -= factor * h
    y1 -= factor * w
    x2 += factor * h
    y2 += factor * w

    new_bbox = [x1,y1,x2,y2]
    for i in range(len(new_bbox)):
        new_bbox[i] = int(np.round(new_bbox[i]))
    if new_bbox[0] < 0:
        new_bbox[0] = 0
    if new_bbox[1] < 0:
        new_bbox[1] = 0
    if new_bbox[2] >= limit[0]:
        new_bbox[2] = limit[0] - 1
    if new_bbox[3] >= limit[1]:
        new_bbox[3] = limit[1] - 1

    return tuple(new_bbox)


def align_table_and_mask(table, mask, align_morph=False, align_int=False, image=None, pcna=None, bf=None):
    """For every object in the mask, check if is consistent with the table. If no, remove the object in the mask.

    Args:
        table (pandas.DataFrame): (tracked) object table.
        mask (numpy.ndarray): labeled object mask, object label should be corresponding to `continuous_label` column in the table.
    """
    BBOX_FACTOR = 2  # dilate the bounding box when calculating the background intensity.
    if not (pcna is not None and bf is not None):
        if align_int and (image is None or len(image.shape)<4):
            raise ValueError('Must supply intensity image with dimension txyc if align_int is True.')
        if (not align_morph) and align_int:
            raise ValueError('Must set align_morph=True if align_int is True.')
    
    if image:
        pcna = image[:,:,:,0]
        bf = image[:,:,:,-1]
    count = 0
    count_up = 0
    h, w = mask.shape[1], mask.shape[2]
    if align_morph:
        new = pd.DataFrame()
    for i in range(mask.shape[0]):
        sub = table[table['frame'] == i].copy()
        sls = mask[i,:,:].copy()
        lbs = sorted(list(np.unique(sls)))
        if lbs[0] == 0:
            del lbs[0]
        registered = list(sub['continuous_label'])
        rmd = list(set(lbs) - set(registered))
        if rmd:
            for j in rmd:
                sls[sls == j] = 0
                count += 1
            mask[i,:,:] = sls
        if align_morph:
            props = measure.regionprops(mask[i,:,:])
            for p in props:
                lb = p.label
                obj = sub[sub['continuous_label'] == lb]
                if obj.shape[0]<1:
                    raise ValueError('Object in the mask not registered in the table!')
                y,x = p.centroid
                if np.round(obj['Center_of_the_object_0'].iloc[0],3) == np.round(x,3) and np.round(obj['Center_of_the_object_1'].iloc[0],3) == np.round(y,3):
                    # The object is unchanged if coordinate matches
                    continue
                else:
                    print('Update object ' + str(lb) + ' at frame ' + str(i))
                    count_up += 1
                    # Update morphology
                    sub.loc[obj.index, 'Center_of_the_object_0'] = x
                    sub.loc[obj.index, 'Center_of_the_object_1'] = y
                    min_axis, maj_axis = p.minor_axis_length, p.major_axis_length
                    sub.loc[obj.index, 'minor_axis'] = min_axis
                    sub.loc[obj.index, 'major_axis'] = maj_axis
                    if align_int:
                        # Update inensity
                        b1, b3, b2, b4 = expand_bbox(p.bbox, BBOX_FACTOR, (h,w))
                        obj_region = mask[i, b1:b2, b3:b4].copy()
                        its_region = pcna[i, b1:b2, b3:b4].copy()
                        dic_region = bf[i, b1:b2, b3:b4].copy()
                        if 0 not in obj_region:
                            sub.loc[obj.index, 'background_mean'] = 0
                        else:
                            sub.loc[obj.index, 'background_mean'] = np.mean(its_region[obj_region == 0])
                        cal = obj_region == lb
                        sub.loc[obj.index, 'mean_intensity'] = np.mean(its_region[cal])
                        sub.loc[obj.index, 'BF_mean'] = np.mean(dic_region[cal])
                        sub.loc[obj.index, 'BF_std'] = np.std(dic_region[cal])
            new = new.append(sub.copy())
    
    print('Removed ' + str(count) + ' objects.')

    if align_morph:
        print('Updated ' + str(count_up) + ' objects.')
        return mask, new
    else:
        return mask


def merge_obj_tables(a, b, col, mode='label'):
    """Merge two object tables according to shared frame and continuous label / location identity.

    Args:
        a (pandas.DataFrame): donor table. Record not found in acceptor will be ignored.
        b (pandas.DataFrame): acceptor table. Record cannot be matched with donor will results in NA and warned.
        col (str): key in both a and b that aimed to merge. Only allow one key and a time 
        mode (str): either 'label' or 'loc'.

    Note:
        Both a and b tables must have the keys:
            - Center_of_the_object_0
            - Center_of_the_object_1
            - continuous label
            - frame
            - (key to merge)
        
        In loc mode, location will be rounded to 3 digits before matching.
    """

    rs = []
    if col not in a.columns:
        raise ValueError(col + ' not found in donor table.')

    if mode == 'loc':
        a['Center_of_the_object_0'] = np.ceil(a['Center_of_the_object_0'])
        a['Center_of_the_object_1'] = np.ceil(a['Center_of_the_object_1'])

    for i in range(b.shape[0]):
        fme = b['frame'].iloc[i]

        if mode == 'label':
            lb = b['continuous_label'].iloc[i]
            cd = a[(b['continuous_label'] == lb) & (a['frame'] == fme)]
        else:
            x = np.ceil(b['Center_of_the_object_0'].iloc[i])
            y = np.ceil(b['Center_of_the_object_1'].iloc[i])
            cd = a[(a['Center_of_the_object_0'] == x) & (a['Center_of_the_object_1'] == y) & (a['frame'] == fme)]

        if cd.shape[0] == 0:
            warnings.warn('Not found in donor for record: ' + str(list(b.iloc[i])))
            rs.append(np.nan)
        else:
            rs.append(cd[col].iloc[0])
    b[col] = rs
    
    return b
