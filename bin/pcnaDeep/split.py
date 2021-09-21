# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import skimage.measure as measure
import skimage.morphology as morphology
from pcnaDeep.data.utils import filter_edge


def split_frame(frame, n=4):
    """Split frame into several quadrants.

    Args:
        frame (numpy.ndarray): single frame slice to split, shape HWC, if HW, will expand C.
        n (int): split count, either 4 or 9.

    Returns:
        numpy.ndarray: stack of split slice, order by row.
    """
    if n not in [4, 9]:
        raise ValueError('Split number should be 4 or 9.')

    if frame.shape[0] != frame.shape[1]:
        raise ValueError('Frame should be square.')

    if len(frame.shape) < 3:
        frame = np.expand_dims(frame, axis=2)
    if frame.shape[0] / n != int(frame.shape[0] / n):
        pd_out = (frame.shape[0] // n + 1) * n - frame.shape[0]
        frame = np.pad(frame, ((0, pd_out), (0, pd_out), (0, 0)), 'constant', constant_values=(0,))

    row = np.split(frame, np.sqrt(n), axis=0)
    tile = []
    for r in row:
        tile.extend(np.split(r, np.sqrt(n), axis=1))

    return np.stack(tile, axis=0)


def join_frame(stack, n=4, crop_size=None):
    """For each n frame in the stack, join into one complete frame (by row).

    Args:
        stack (numpy.ndarray): tiles to join.
        n (int): each n tiles to join, should be either 4 or 9.
        crop_size (int): crop the square image into certain size (lower-right), default no crop.

    Returns:
        numpy.ndarray: stack of joined frames.
    """

    if n not in [4, 9]:
        raise ValueError('Join tile number should either be 4 or 9.')

    if stack.shape[0] < n or stack.shape[0] % n != 0:
        raise ValueError('Stack length is not multiple of tile count n.')

    p = int(np.sqrt(n))
    out_stack = []
    stack = stack.astype('uint16')
    for i in range(int(stack.shape[0] / n)):
        count = 1
        frame = []
        for j in range(p):
            row = []
            for k in range(p):
                new_stack, count_add = relabel_seq(stack[int(j * p + k + i * n), :], base=count)
                count += count_add
                row.append(new_stack)
            row = np.concatenate(np.array(row), axis=1)
            frame.append(row)
        frame = np.concatenate(np.array(frame), axis=0)
        out_stack.append(frame)

    out_stack = np.stack(out_stack, axis=0)

    if crop_size is not None:
        out_stack = out_stack[:, :crop_size, :crop_size, :]

    if np.max(stack) <= 255:
        out_stack = out_stack.astype('uint8')
    else:
        out_stack = out_stack.astype('uint16')

    return out_stack


def join_table(table, n=4, tile_width=1200):
    """Join object table according to tiled frames.

    Args:
        table (pandas.DataFrame): object table to join,
            essential columns: frame, Center_of_the_object_0 (x), Center_of_the_object_1 (y).
            The method will join frames by row.
        n (int): each n frames form a tiled slice, either 4 or 9.
        tile_width (int): width of each tile.

    Returns:
        pandas.DataFrame: object table for further processing (tracking, resolving)
    """
    NINE_DICT = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (1, 0), 4: (1, 1),
                 5: (1, 2), 6: (2, 0), 7: (2, 1), 8: (2, 2)}
    FOUR_DICT = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
    if n not in [4, 9]:
        raise ValueError('Join tile number should either be 4 or 9.')

    if (np.max(table['frame']) + 1) < n or (np.max(table['frame']) + 1) % n != 0:
        raise ValueError('Stack length is not multiple of tile count n.')

    out = pd.DataFrame(columns=table.columns)
    n = int(np.sqrt(n))
    for frame in range(int((np.max(table['frame']) + 1) / (n * n))):
        for i in range(n):
            sub_table = table[
                (table['frame'] < ((i + 1) * n + frame * n * n)) & (table['frame'] >= (i * n + frame * n * n))].copy()
            mod_x = []
            mod_y = []
            mod_b0 = []
            mod_b1 = []
            mod_b2 = []
            mod_b3 = []
            for j in range(n):
                sub_tile_table = sub_table[sub_table['frame'] == (i * n + j + frame * n * n)].copy()
                for k in range(sub_tile_table.shape[0]):
                    x = sub_tile_table['Center_of_the_object_0'].iloc[k]
                    y = sub_tile_table['Center_of_the_object_1'].iloc[k]
                    b0 = sub_tile_table['bbox-0'].iloc[k]
                    b1 = sub_tile_table['bbox-1'].iloc[k]
                    b2 = sub_tile_table['bbox-2'].iloc[k]
                    b3 = sub_tile_table['bbox-3'].iloc[k]
                    if n * n == 4:
                        time_x = FOUR_DICT[i * n + j][0] * tile_width
                        time_y = FOUR_DICT[i * n + j][1] * tile_width
                    else:
                        time_x = NINE_DICT[i * n + j][0] * tile_width
                        time_y = NINE_DICT[i * n + j][1] * tile_width
                    x += time_x
                    b0 += time_x
                    b2 += time_x
                    y += time_y
                    b1 += time_y
                    b3 += time_y
                    mod_x.append(x)
                    mod_y.append(y)
                    mod_b0.append(b0)
                    mod_b1.append(b1)
                    mod_b2.append(b2)
                    mod_b3.append(b3)

            sub_table.loc[:, 'Center_of_the_object_0'] = mod_x
            sub_table.loc[:, 'Center_of_the_object_1'] = mod_y
            sub_table.loc[:, 'bbox-0'] = mod_b0
            sub_table.loc[:, 'bbox-1'] = mod_b1
            sub_table.loc[:, 'bbox-2'] = mod_b2
            sub_table.loc[:, 'bbox-3'] = mod_b3
            sub_table.loc[:, 'frame'] = frame
            out = out.append(sub_table)

    return out


def relabel_seq(frame, base=1):
    """Relabel single frame sequentially.
    """
    out = np.zeros(frame.shape)
    props = measure.regionprops(frame)
    for i in range(len(props)):
        lb = props[i].label
        out[frame == lb] = i + base

    return out, len(props)


def register_label_to_table(frame, table):
    """Register labels to the table according to centroid localization.
        WARNING: will round location to 2 decimal.
    """
    tb = measure.regionprops_table(frame, properties=('centroid', 'label'))
    tb = pd.DataFrame(tb)
    tb.columns = ['Center_of_the_object_0', 'Center_of_the_object_1', 'label']
    tb['Center_of_the_object_1'] = np.round(tb['Center_of_the_object_1'], 2)
    tb['Center_of_the_object_0'] = np.round(tb['Center_of_the_object_0'], 2)
    table['Center_of_the_object_0'] = np.round(table['Center_of_the_object_0'], 2)
    table['Center_of_the_object_1'] = np.round(table['Center_of_the_object_1'], 2)

    out = pd.merge(table, tb, on=['Center_of_the_object_0', 'Center_of_the_object_1'])
    out['continuous_label'] = out['label']
    del out['label']
    missing = table.shape[0] - out.shape[0]
    return out, missing


def resolve_joined_stack(stack, table, n=4, boundary_width=5, dilate_time=3, filter_edge_width=50):
    """Wrapper of `resolved_joined_frame()` which resolves merged tiles by each frame.
        Filter imprecise objects at the edge.
    """
    out_table = pd.DataFrame(columns=table.columns)
    for i in range(stack.shape[0]):
        sub = table[table['frame'] == i].copy()
        new_frame, new_table = resolve_joined_frame(stack[i, :].copy(), sub,
                                                    n=n,
                                                    boundary_width=boundary_width,
                                                    dilate_time=dilate_time,
                                                    filter_edge_width=filter_edge_width)
        stack[i, :] = new_frame.astype(stack.dtype)
        out_table = out_table.append(new_table)

    return stack, out_table


def resolve_joined_frame(frame, table, n=4, boundary_width=5, dilate_time=3, filter_edge_width=50):
    """Resolve joined frame and table of single slice.

    Args:
        frame (numpy.ndarray): joined image slice.
        table (pandas.DataFrame): object table with coordinate adjusted from joining.
        n (int): tile count.
        boundary_width (int): maximum pixel value for sealing objects at the boundary.
        dilate_time (int): round of dilation on boundary objects to seal them.
        filter_edge_width (int): filter objects at the edge.

    Returns:
        numpy.ndarray: relabeled slice with objects at the edge joined.
        pandas.DataFrame: resolved object with object labels updated.

    Note:
        Objects at the edge will be deleted, new objects due to joining will be registered.

        Cell cycle phase information (prediction class and confidence) is drawn from the object
        of larger size.

    """
    frame = frame.copy()
    table = table.copy()

    if n not in [4, 9]:
        raise ValueError('Join tile number should either be 4 or 9.')

    if frame.shape[0] % n != 0:
        raise ValueError('Stack size is not multiple of tile count n.')

    table = register_label_to_table(frame, table)[0]

    n = int(np.sqrt(n))
    width = int(frame.shape[0] / n)

    new_table = pd.DataFrame()
    new_frame = np.zeros(frame.shape)
    obj_count = 1

    cdds = []  # candidate objects to seal
    for i in range(n):
        for j in range(n):

            col_low = j * width
            row_low = i * width
            col_bd = (j + 1) * width - 1
            row_bd = (i + 1) * width - 1
            props = measure.regionprops(frame)

            for p in props:
                x, y = np.round(p.centroid, 2)
                if row_bd > x >= row_low and col_bd > y >= col_low:
                    row = table[(table['Center_of_the_object_1'] == y) & (table['Center_of_the_object_0'] == x)]
                    if row.shape[0] == 0:
                        continue
                    ct = 0

                    for lc in p.bbox:
                        if lc in range(col_bd - boundary_width, col_bd + boundary_width) or \
                                lc in range(row_bd - boundary_width, row_bd + boundary_width):
                            ct += 1

                    if ct >= 1:
                        # boundary object found
                        # binary dilation to seal
                        sl = np.zeros(frame.shape)
                        sl[frame == p.label] = 1
                        for r in range(dilate_time):
                            sl = morphology.dilation(sl)
                        frame[sl > 0] = p.label

                        # update info in table
                        new_p = measure.regionprops(measure.label(sl))[0]
                        new_x, new_y = np.round(new_p.centroid, 2)
                        table.loc[table['continuous_label'] == p.label, 'Center_of_the_object_1'] = new_y
                        table.loc[table['continuous_label'] == p.label, 'Center_of_the_object_0'] = new_x
                        table.loc[table['continuous_label'] == p.label, 'bbox-0'] = new_p.bbox[0]
                        table.loc[table['continuous_label'] == p.label, 'bbox-1'] = new_p.bbox[1]
                        table.loc[table['continuous_label'] == p.label, 'bbox-2'] = new_p.bbox[2]
                        table.loc[table['continuous_label'] == p.label, 'bbox-3'] = new_p.bbox[3]
                        table.loc[table['continuous_label'] == p.label, 'major_axis'] = new_p.major_axis_length
                        table.loc[table['continuous_label'] == p.label, 'minor_axis'] = new_p.minor_axis_length

                        cdds.append(p.label)

    flooded = measure.label(frame.astype('bool'))
    props = measure.regionprops(frame)

    for p in props:
        if p.label not in frame: continue
        x, y = np.round(p.centroid, 2)
        if p.label in cdds:

            fld_label = int(flooded[int(p.centroid[0]), int(p.centroid[1])])
            new_frame[flooded == fld_label] = obj_count
            old = np.unique(frame[flooded == fld_label]).tolist()
            if 0 in old:
                old.remove(0)
            if len(old) == 1:
                row = table[(table['Center_of_the_object_1'] == y) & (table['Center_of_the_object_0'] == x)]
                row.loc[:]['continuous_label'] = obj_count
                new_table = new_table.append(row)
            else:
                # deduce object information
                temp_mask = np.zeros(frame.shape)
                temp_mask[flooded == fld_label] = 1
                tbl = measure.regionprops(measure.label(temp_mask))[0]
                sub = pd.DataFrame()
                for rm in old:
                    sub = sub.append(table[table['continuous_label'] == rm])
                confid = [np.mean(sub['Probability of G1/G2']),
                          np.mean(sub['Probability of S']),
                          np.mean(sub['Probability of M'])]
                mi = np.mean(sub['mean_intensity'])
                pred_cls = ['G1/G2', 'S', 'M'][int(np.argmax(confid))]
                row = sub.iloc[0]
                # register information
                row.loc[:]['bbox-0'] = tbl.bbox[0]
                row.loc[:]['bbox-1'] = tbl.bbox[1]
                row.loc[:]['bbox-2'] = tbl.bbox[2]
                row.loc[:]['bbox-3'] = tbl.bbox[3]
                row.loc[:]['Center_of_the_object_1'] = y
                row.loc[:]['Center_of_the_object_0'] = x
                row.loc[:]['major_axis'] = tbl.major_axis_length
                row.loc[:]['minor_axis'] = tbl.minor_axis_length
                row.loc[:]['continuous_label'] = obj_count
                row.loc[:]['phase'] = pred_cls
                row.loc[:]['Probability of G1/G2'] = confid[0]
                row.loc[:]['Probability of S'] = confid[1]
                row.loc[:]['Probability of M'] = confid[2]
                row.loc[:]['mean_intensity'] = mi
                new_table = new_table.append(row)

                # remove sealed objects to block further recording
                frame[flooded == fld_label] = 0

            obj_count += 1
        else:
            row = table[(table['Center_of_the_object_1'] == y) & (table['Center_of_the_object_0'] == x)]
            if row.shape[0] > 0:
                if row['continuous_label'].values[0] in frame:
                    row.loc[:]['continuous_label'] = obj_count
                    new_table = new_table.append(row)
                    new_frame[frame == p.label] = obj_count
                    obj_count += 1

    return filter_edge(new_frame, new_table, filter_edge_width)
