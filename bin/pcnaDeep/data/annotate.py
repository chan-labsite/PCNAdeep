# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import skimage.io as io
from skimage.util import img_as_uint
from skimage.util import img_as_ubyte


def relabel_trackID(label_table):
    """Relabel trackID in tracking table, starting from 1.

    Args:
        label_table (pandas.DataFrame): tracked object table.

    Returns:
        pandas.DataFrame: tracked object table with relabeled trackID.
    """

    dic = {}
    ori = list(np.unique(label_table['trackId']))
    for i in range(1, len(ori) + 1):
        dic[ori[i - 1]] = i
    dic[0] = 0
    for i in range(label_table.shape[0]):
        label_table.loc[i, 'trackId'] = dic[label_table['trackId'][i]]
        label_table.loc[i, 'parentTrackId'] = dic[label_table['parentTrackId'][i]]
        label_table.loc[i, 'lineageId'] = dic[label_table['lineageId'][i]]

    return label_table


def label_by_track(mask, label_table):
    """Label objects in mask with track ID

    Args:
        mask (numpy.ndarray): uint8 np array, output from main model.
        label_table (pandas.DataFrame): track table.
    
    Returns:
        numpy.ndarray: uint8/16 dtype based on track count.
    """

    assert mask.shape[0] == np.max(label_table['frame'] + 1)

    if np.max(label_table['trackId']) * 2 > 254:
        mask = mask.astype('uint16')

    for i in np.unique(label_table['frame']):
        sub_table = label_table[label_table['frame'] == i]
        sl = mask[i, :, :].copy()
        lbs = np.unique(sl).tolist()

        '''
        if lbs[-1] + 1 != len(lbs):
            raise ValueError('Mask is not continuously or wrongly labeled.')
        '''

        ori_labels = set(lbs) - {0}
        untracked = list(ori_labels - set(list(sub_table['continuous_label'])))
        #  remove untracked
        for j in untracked:
            sl[mask[i, :, :] == j] = 0
        #  update tracked
        for j in sub_table.index:
            sl[mask[i, :, :] == sub_table.loc[j, 'continuous_label']] = sub_table.loc[j, 'trackId']
        mask[i, :, :] = sl.copy()
    return mask


def get_lineage_txt(label_table):
    """Generate txt table in Cell Tracking Challenge (CTC) format.

    Args:
        label_table (pandas.DataFrame): table processed, should not has gaped tracks.

    Returns:
        pandas.DataFrame: lineage table in .txt format that fits CTC.
    """

    dic = {'id': [], 'appear': [], 'disappear': [], 'parent': []}
    for i in np.unique(label_table['trackId']):
        sub = label_table[label_table['trackId'] == i]
        begin = np.min(sub['frame'])
        end = np.max(sub['frame'])
        parent = np.unique(sub['parentTrackId'])

        dic['id'].append(i)
        dic['appear'].append(int(begin))
        dic['disappear'].append(int(end))
        dic['parent'].append(int(parent))

    return pd.DataFrame(dic)


def break_track(label_table):
    """Break tracks in a lineage table into single tracks, where
    NO gaped tracks allowed. All gaps will be transferred into parent-daughter
    relationship.

    Args:
        label_table (pandas.DataFrame): tracked object table to process.

    Algorithm:
        1. Rename raw parentTrackId to mtParTrk.
        2. Initiate new parentTrackId column with 0.
        3. Separate all tracks individually.

    Notes:
        In original lineage table, single track can be gaped, lineage only associates
        mitosis tracks, not gaped tracks.
    
    Returns:
        pandas.DataFrame: processed tracked object table.
    """

    # For parent track that has one daughter extrude into the parent frame,
    #       e.g. parent: t1-10; daughter1: t8-20; daughter2: t11-20.
    # re-organize the track by trimming parent and add to daughter,
    #       i.e. parent: t1-7; daughter1: t8-20; daughter2: t8-10, t11-20
    # If both daughter extrude, e.g. daughter2: t9-20, then trim parent directly
    # to t1-8. Since this indicates faulty track, warning shown
    # *** this should NOT usually happen

    for l in np.unique(label_table['trackId']):
        daugs = np.unique(label_table[label_table['parentTrackId'] == l]['trackId'])
        if len(daugs) == 2:
            daug1 = label_table[label_table['trackId'] == daugs[0]]['frame'].iloc[0]
            daug2 = label_table[label_table['trackId'] == daugs[1]]['frame'].iloc[0]
            par = label_table[label_table['trackId'] == l]
            par_frame = par['frame'].iloc[-1]
            if par_frame >= daug1 and par_frame >= daug2:
                label_table.drop(par[(par['frame'] >= daug1) | (par['frame'] >= daug2)].index, inplace=True)
                raise UserWarning('Faluty mitosis, check parent: ' + str(l) +
                                  ', daughters: ' + str(daugs[0]) + '/' + str(daugs[1]))
            elif par_frame >= daug1:
                # migrate par to daug2
                label_table.loc[par[par['frame'] >= daug1].index, 'trackId'] = daugs[1]
                label_table.loc[par[par['frame'] >= daug1].index, 'parentTrackId'] = l
            elif par_frame >= daug2:
                # migrate par to daug1
                label_table.loc[par[par['frame'] >= daug2].index, 'trackId'] = daugs[0]
                label_table.loc[par[par['frame'] >= daug2].index, 'parentTrackId'] = l

    label_table = label_table.sort_values(by=['trackId', 'frame'])

    # break tracks individually
    max_trackId = np.max(label_table['trackId'])
    label_table['mtParTrk'] = label_table['parentTrackId']
    label_table['parentTrackId'] = 0
    label_table['ori_trackId'] = label_table['trackId']
    new_table = pd.DataFrame()

    for l in np.unique(label_table['trackId']):
        tr = label_table[label_table['trackId'] == l].copy()

        if np.max(tr['frame']) - np.min(tr['frame']) + 1 != tr.shape[0]:
            sep, max_trackId = separate(list(tr['frame']).copy(), list(tr['mtParTrk']).copy(), l, base=max_trackId)
            tr.loc[:, 'frame'] = sep['frame']
            tr.loc[:, 'trackId'] = sep['trackId']
            tr.loc[:, 'parentTrackId'] = sep['parentTrackId']
            tr.loc[:, 'mtParTrk'] = sep['mtParTrk']

        new_table = new_table.append(tr)

        # For tracks that have mitosis parents, find new ID of their parents
    for l in np.unique(new_table['trackId']):
        tr = new_table[new_table['trackId'] == l].copy()

        ori_par = list(tr['mtParTrk'])[0]
        if ori_par != 0:
            app = np.min(tr['frame'])
            search = new_table[new_table['ori_trackId'] == ori_par]
            new_par = search.iloc[np.argmin(abs(search['frame'] - app))]['trackId']
            new_table.loc[tr.index, 'mtParTrk'] = new_par

    for i in range(new_table.shape[0]):
        new_table.loc[i, 'parentTrackId'] = np.max(
            ((new_table['parentTrackId'][i]), new_table['mtParTrk'][i]))  # merge mitosis information in to parent

    return new_table


def separate(frame_list, mtPar_list, ori_id, base):
    """For single gaped track, separate it into all complete tracks.
    
    Args:
        frame_list (list): frames list, length equals to label table.
        mtPar_list (list): mitosis parent list, for solving mitosis relationship.
        ori_id (int): original track ID.
        base (int): base track ID, will assign new track ID sequentially from base + 1.

    Returns:
        dict: Dictionary of having following keys: frame, trackId, parentTrackId, mtParTrk.
    """

    trackId = [ori_id for _ in range(len(frame_list))]
    parentTrackId = [0 for _ in range(len(frame_list))]
    for i in range(1, len(frame_list)):
        if frame_list[i] - frame_list[i - 1] != 1:
            trackId[i:] = [base + 1 for s in range(i, len(trackId))]
            parentTrackId[i:] = [trackId[i - 1] for s in range(i, len(trackId))]
            mtPar_list[i:] = [0 for s in range(i, len(trackId))]
            base += 1
    rt = {'frame': frame_list, 'trackId': trackId, 'parentTrackId': parentTrackId, 'mtParTrk': mtPar_list}
    return rt, base


def save_seq(stack, out_dir, prefix, dig_num=3, dtype='uint16', base=0, img_format='.tif', keep_chn=True, sep='-'):
    """Save image stack and label sequentially.
    
    Args:
        stack (numpy array) : image stack in THW format (Time, Height, Width).
        out_dir (str) : output directory.
        prefix (str) : prefix of single slice, output will be prefix-000x.tif/png.
            (see sep below for separator).
        dig_num (int) : digit number (3 -> 00x) for labeling image sequentially.
        dtype (numpy.dtype) : data type to save, either 'uint8' or 'uint16'.
        base (int) : base number of the label (starting from).
        img_format (str): image format, '.tif' or '.png', remind the dot.
        keep_chn (bool): whether to keep full channel or not.
        sep (str): separator between file name and id, default '-'.
    """
    if len(stack.shape) == 4 and not keep_chn:
        stack = stack[:, :, :, 0]

    for i in range(stack.shape[0]):
        fm = ("%0" + str(dig_num) + "d") % (i + base)
        name = os.path.join(out_dir, prefix + sep + fm + img_format)
        if dtype == 'uint16':
            img = img_as_uint(stack[i, :])
        elif dtype == 'uint8':
            img = img_as_ubyte(stack[i, :])
        else:
            raise ValueError("Seq save only accepts uint8 or uint16 format.")
        io.imsave(name, img)

    return


def findM(gt_cls, direction='begin'):
    """Find M exit/entry from ground truth classification.

    The method assumes that all mitosis classification is continuous, therefore only suitable.
    for processing classification ground truth. For processing prediction, use `pcnaDeep.refiner.deduce_transition`.

    Args:
        gt_cls (list): list of classifications.
        direction (str): begin/end, search M from which terminal of the classification list.

    Returns:
        int: index of the mitosis entry/exit.
    """
    # possible for parent end with 'G', but daughter must begin with 'M'
    i = 0
    if direction == 'begin':
        if gt_cls[0] != 'M':
            return None
        while gt_cls[i] == 'M':
            i += 1
            if i == len(gt_cls):
                break
        return i - 1
    else:
        gt_cls = gt_cls[::-1]
        if 'M' not in gt_cls:
            return None
        i = gt_cls.index('M')
        while gt_cls[i] == 'M':
            i += 1
            if i == len(gt_cls):
                break
        return -i


def check_continuous_track(table):
    """Check if every track is continuous (no gap). Returns trackID list that is gaped.
    """
    out = []
    for i in list(np.unique(table['trackId'])):
        f = table[table['trackId'] == i]['frame'].tolist()
        if f[-1] - f[0] != len(f) - 1:
            out.append(i)
    return out
