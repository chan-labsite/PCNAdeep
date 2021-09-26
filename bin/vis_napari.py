import napari
import skimage.io as io
import pandas as pd
import argparse
import re
import pprint
import numpy as np
from pcnaDeep.data.utils import find_daugs, align_table_and_mask, expand_bbox
import skimage.measure as measure


class Trk_viewer(napari.Viewer):

    def __init__(self, track_path, image_path, mask_path, frame_base=1, title='pcnaDeep', ndisplay=2, order=(),
                 axis_labels=(), show=True):
        """
        To correct track ID, mitosis relationship, cell cycle classifications.

        Args:
            track_path (str): path to tracked object table.
            image_path (str): intensity image.
            mask_path (str): segmentatio mask.
            frame_base (int): base of counting frames, default 1.
        """

        super().__init__(title=title, ndisplay=ndisplay, order=order, axis_labels=axis_labels, show=show)
        self.track_path = track_path
        self.mask_path = mask_path
        self.track = pd.read_csv(track_path)
        self.track = self.track.sort_values(by=['trackId','frame'])
        self.saved = self.track.copy()
        self.frame_base = frame_base
        self.parser = argparse.ArgumentParser()
        self.__construct_parser()
        self.track_count = int(np.max(self.track['trackId']))
        self.mask = io.imread(self.mask_path)

        self.add_image(io.imread(image_path), name='raw')
        self.add_labels(self.mask.copy(), name='segm')
        track_data = self.track.loc[:][['trackId', 'frame', 'Center_of_the_object_1', 'Center_of_the_object_0']]
        label_data = self.track.loc[:][['frame', 'Center_of_the_object_1', 'Center_of_the_object_0']]
        label_data = label_data.to_numpy()
        track_data = track_data.to_numpy()
        self.text_config = {'text':'name', 'size': 8, 'color': 'yellow'}
        self.add_tracks(track_data, name='tracks')
        self.add_points(label_data, name='name', size=0,
                        properties={'name':self.track.loc[:]['name'].to_numpy()}, text=self.text_config)

        return

    def __setattr__(self, key, value):
        if key not in ['track_path', 'track', 'saved', 'original', 'frame_base', 'parser', 'track_count',
                       'text_config', 'mask', 'mask_path']:
            return super().__setattr__(key, value)
        else:
            self.__dict__[key] = value

    def __construct_parser(self):
        self.parser.add_argument("-t", help="Track ID.")
        self.parser.add_argument("-t1", help="Old track ID to replace.")
        self.parser.add_argument("-t2", help="New track ID to replace with.")
        self.parser.add_argument("-f", help="Time frame.")
        self.parser.add_argument("-p", help="Parent track ID.")
        self.parser.add_argument("-d", help="Daughter track ID.")
        self.parser.add_argument("-l", help="Correct classification to assign.")
        self.parser.add_argument("-s", help="Correct classification on single slice.", action='store_true')
        self.parser.add_argument("-e", help="Correct classification until the specified frame.")
        self.parser.add_argument("-ds", help="Daughter list, comma separated.")
        self.parser.add_argument("-o", help="Object ID on the mask.")

        return

    def create_or_replace(self, old_id, frame, new_id=None):
        """Create a new track ID or replace with some track ID
        after certain frame. If the old track has daughters, new track ID will be the parent.

        Args:
            old_id (int): old track ID.
            frame (int): frame to begin with new ID.
            new_id (int): new track ID, only required when replacing track identity.
        """
        if old_id not in self.track['trackId']:
            raise ValueError('Selected track is not in the table.')
        if frame not in list(self.track[self.track['trackId'] == old_id]['frame']):
            raise ValueError('Selected frame is not in the original track.')

        dir_daugs = list(np.unique(self.track.loc[self.track['parentTrackId'] == old_id, 'trackId']))
        for dd in dir_daugs:
            self.del_parent(dd)

        if new_id is None:
            self.track_count += 1
            new = self.track_count
            new_lin = new
            new_par = 0
        else:
            if new_id not in self.track['trackId']:
                raise ValueError('Selected new ID not in the table.')
            old_frame = list(self.track[self.track['trackId'] == new_id]['frame'])
            new_frame = list(self.track.loc[(self.track['trackId'] == old_id) &
                                            (self.track['frame'] >= frame), 'frame'])
            if len(old_frame + new_frame) != len(set(old_frame + new_frame)):
                raise ValueError('Selected new ID track overlaps with old one.')

            new = new_id
            new_lin = self.track[self.track['trackId'] == new_id]['lineageId'].values[0]
            new_par = self.track[self.track['trackId'] == new_id]['parentTrackId'].values[0]
        self.track.loc[(self.track['trackId'] == old_id) & (self.track['frame'] >= frame), 'trackId'] = new
        self.track.loc[self.track['trackId'] == new, 'lineageId'] = new_lin
        self.track.loc[self.track['trackId'] == new, 'parentTrackId'] = new_par
        # daughters of the new track, change lineage
        daugs = find_daugs(self.track, new)
        if daugs:
            self.track.loc[self.track['trackId'].isin(daugs), 'lineageId'] = new_lin
        print('Replaced/Created track ' + str(old_id) + ' from ' + str(frame + self.frame_base) +
              ' with new ID ' + str(new) + '.')

        for dd in dir_daugs:
            if dd != new:
                self.create_parent(new, dd)

        return

    def create_parent(self, par, daug):
        """Create parent-daughter relationship.

        Args:
            par (int): parent track ID.
            daug (int): daughter track ID.
        """
        if par not in self.track['trackId']:
            raise ValueError('Selected parent is not in the table.')
        if daug not in self.track['trackId']:
            raise ValueError('Selected daughter is not in the table.')

        ori_par = self.track[self.track['trackId'] == daug]['parentTrackId'].iloc[0]
        if ori_par != 0:
            raise ValueError('One daughter cannot have more than one parent, disassociate ' + str(ori_par) + '-'
                             + str(daug) + ' first.')

        par_lin = self.track[self.track['trackId'] == par]['lineageId'].iloc[0]
        # daughter itself
        self.track.loc[self.track['trackId'] == daug, 'lineageId'] = par_lin
        self.track.loc[self.track['trackId'] == daug, 'parentTrackId'] = par
        # daughter of the daughter
        daugs_of_daug = find_daugs(self.track, daug)
        if daugs_of_daug:
            self.track.loc[self.track['trackId'].isin(daugs_of_daug), 'lineageId'] = par_lin
        print('Parent ' + str(par) + ' associated with daughter ' + str(daug) + '.')

        return

    def del_parent(self, daug):
        """Remove parent-daughter relationship, for a daughter.

        Args:
            daug (int): daughter track ID.
        """
        if daug not in self.track['trackId']:
            raise ValueError('Selected daughter is not in the table.')
        if self.track[self.track['trackId'] == daug]['parentTrackId'].iloc[0] == 0:
            raise ValueError('Selected daughter does not have a parent.')

        # daughter itself
        self.track.loc[self.track['trackId'] == daug, 'lineageId'] = daug
        self.track.loc[self.track['trackId'] == daug, 'parentTrackId'] = 0
        # daughters of the daughter, change lineage
        daugs = find_daugs(self.track, daug)
        if daugs:
            self.track.loc[self.track['trackId'].isin(daugs), 'lineageId'] = daug

        return

    def correct_cls(self, trk_id, frame, cls, mode='to_next', end_frame=None):
        """Correct cell cycle classification, will also influence confidence score.

        Args:
            trk_id (int): track ID to correct.
            frame (int): frame to correct or begin with correction.
            cls (str): new classification to assign.
            mode (str): either 'to_next', 'single', or 'range'
            end_frame (int): optional, in 'range' mode, stop correction at this frame.
        """
        if trk_id not in self.track['trackId']:
            raise ValueError('Selected track is not in the table.')
        if cls not in ['G1', 'G2', 'M', 'S', 'G1/G2', 'E']:
            raise ValueError('cell cycle phase can only be G1, G2, G1/G2, S, M or E.')

        clss = list(self.track[self.track['trackId'] == trk_id]['predicted_class'])
        frames = list(self.track[self.track['trackId'] == trk_id]['frame'])
        if frame not in frames:
            raise ValueError('Selected frame is not in the original track.')
        fm_id = frames.index(frame)
        idx = self.track[self.track['trackId'] == trk_id].index
        if mode == 'single':
            rg = [fm_id]
        elif mode == 'range':
            if end_frame not in frames:
                raise ValueError('Selected end frame is not in the original track.')
            rg = [i for i in range(fm_id, frames.index(end_frame + 1))]
        elif mode == 'to_next':
            cur_cls = clss[fm_id]
            j = fm_id + 1
            while j < len(clss):
                if clss[j] == cur_cls:
                    j += 1
                else:
                    break
            rg = [i for i in range(fm_id, j)]
        else:
            raise ValueError('Mode can only be single, to_next or range, not ' + mode)

        for r in rg:
            if cls == 'E':
                cls_resolved = 'G1'
                cls_predicted = 'G1/G2'
                self.track.loc[idx[r], 'emerging'] = 1
            elif cls in ['G1', 'G2']:
                cls_resolved = cls
                cls_predicted = 'G1/G2'
            else:
                cls_resolved = cls
                cls_predicted = cls
            self.track.loc[idx[r], 'resolved_class'] = cls_resolved
            self.track.loc[idx[r], 'predicted_class'] = cls_predicted
            if cls in ['G1', 'G2', 'G1/G2']:
                self.track.loc[idx[r], 'Probability of G1/G2'] = 1
                self.track.loc[idx[r], 'Probability of S'] = 0
                self.track.loc[idx[r], 'Probability of M'] = 0
            elif cls == 'S':
                self.track.loc[idx[r], 'Probability of G1/G2'] = 0
                self.track.loc[idx[r], 'Probability of S'] = 1
                self.track.loc[idx[r], 'Probability of M'] = 0
            else:
                self.track.loc[idx[r], 'Probability of G1/G2'] = 0
                self.track.loc[idx[r], 'Probability of S'] = 0
                self.track.loc[idx[r], 'Probability of M'] = 1
        print('Classification for track ' + str(trk_id) + ' corrected as ' + str(cls) + ' from ' +
              str(frames[rg[0]] + self.frame_base) + ' to ' + str(frames[rg[-1]] + self.frame_base) + '.')

        return

    def delete_track(self, trk_id, frame=None):
        """Delete entire track. If frame supplied, only delete object at specified frame.

        Args:
            trk_id (int): track ID.
            frame (int): time frame.
        """
        if trk_id not in self.track['trackId']:
            raise ValueError('Selected track is not in the table.')

        mask = self.layers[1].data
        del_trk = self.track[self.track['trackId'] == trk_id]
        if frame is None:
            # For all direct daughter of the track to delete, first remove association
            dir_daugs = list(np.unique(self.track.loc[self.track['parentTrackId'] == trk_id, 'trackId']))
            for dd in dir_daugs:
                self.del_parent(dd)

            # Delete entire track
            for i in range(del_trk.shape[0]):
                fme = del_trk['frame'].iloc[i]
                lb = del_trk['continuous_label'].iloc[i]
                msk_slice = mask[fme, :, :]
                msk_slice[msk_slice == lb] = 0
                mask[fme, :, :] = msk_slice
            self.track = self.track.drop(index=del_trk.index)
        else:
            del_trk = del_trk[del_trk['frame'] == frame]
            lb = del_trk['continuous_label'].iloc[0]
            msk_slice = mask[frame, :, :]
            msk_slice[msk_slice == lb] = 0
            mask[frame, :, :] = msk_slice
            self.track = self.track.drop(index=self.track[(self.track['trackId'] == trk_id) &
                                                          (self.track['frame'] == frame)].index)
        self.layers[1].data = mask
        return

    def save(self, mask_flag=True):
        """Save current table.
        """
        mask = self.layers[1].data
        track = self.track.copy()
        if mask_flag:
            mask, track = align_table_and_mask(track, mask, align_morph=True, align_int=True, image=self.layers[0].data)
            if int(np.max(mask)) <= 255:
                io.imsave(self.mask_path, mask.astype('uint8'))
            else:
                io.imsave(self.mask_path, mask)
        self.mask = mask.copy()
        self.getAnn()
        track.to_csv(self.track_path, index=None)
        self.saved = track.copy()
        self.track = track.copy()
        print('Saved.')
        return

    def revert(self):
        """Revert to last saved version.
        """
        self.layers[1].data = self.mask.copy()
        self.track = self.saved.copy()
        return

    def getAnn(self):
        """Add an annotation column to tracked object table
        The annotation format is track ID - (parentTrackId, optional) - resolved_class
        """
        ann = []
        cls_col = 'resolved_class'
        if cls_col not in self.track.columns:
            print('Phase not resolved yet. Using predicted phase classifications.')
            cls_col = 'predicted_class'
        track_id = list(self.track['trackId'])
        parent_id = list(self.track['parentTrackId'])
        cls_lb = list(self.track[cls_col])
        for i in range(self.track.shape[0]):
            inform = [str(track_id[i]), str(parent_id[i]), cls_lb[i]]
            if inform[1] == '0':
                del inform[1]
            ann.append('-'.join(inform))
        self.track['name'] = ann
        return

    def edit_div(self, par, daugs, new_frame):
        """Change division time of parent and daughter to a new time location

        Args:
            par (int): parent track ID
            daugs (list): daughter tracks IDs
            new_frame (int):
        """
        if par not in self.track['trackId']:
            raise ValueError('Selected parent track is not in the table.')
        for d in daugs:
            if d not in self.track['trackId']:
                raise ValueError('Selected daughter track is not in the table.')
            if self.track[self.track['trackId'] == d]['parentTrackId'].iloc[0] != par:
                raise ValueError('Selected daughter track does not corresponding to the input parent.')

        new_frame -= 1
        sub_par = self.track[self.track['trackId'] == par]
        time_daugs = []
        sub_daugs = pd.DataFrame()
        for i in daugs:
            sub_daugs = sub_daugs.append(self.track[self.track['trackId'] == i])
        time_daugs.extend(list(sub_daugs['frame']))
        if new_frame not in list(sub_par['frame']) and new_frame not in time_daugs:
            raise ValueError('Selected new time frame not in either parent or daughter track.')

        if new_frame not in list(sub_par['frame']):
            # push division later
            edit = sub_daugs[sub_daugs['frame'] <= new_frame]
            if len(np.unique(edit['trackId'])) > 1:
                raise ValueError('Multiple daughters at selected new division, should only have one')

            # get and assign edit index
            par_id = sub_par['trackId'].iloc[0]
            par_lin = sub_par['lineageId'].iloc[0]
            par_par = sub_par['parentTrackId'].iloc[0]
            self.track.loc[edit.index, 'trackId'] = par_id
            self.track.loc[edit.index, 'lineageId'] = par_lin
            self.track.loc[edit.index, 'parentTrackId'] = par_par
        else:
            new_frame += 1
            # draw division earlier
            edit = sub_par[sub_par['frame'] >= new_frame]
            # pick a daughter that appears earlier and assign tracks to that daughter
            f_min = np.argmin(sub_daugs['frame'])
            if len(np.unique(sub_daugs[sub_daugs['frame'] == f_min]['trackId'])) > 1:
                raise ValueError('Multiple daughters exist at frame of mitosis, should only be one. '
                                 'Or break mitotic track first.')
            trk = list(sub_daugs['trackId'])[int(f_min)]
            sel_daugs = sub_daugs[sub_daugs['trackId'] == trk]

            # get and assign edit index
            daug_id = sel_daugs['trackId'].iloc[0]
            daug_par = sel_daugs['parentTrackId'].iloc[0]
            daug_lin = sel_daugs['lineageId'].iloc[0]
            self.track.loc[edit.index, 'trackId'] = daug_id
            self.track.loc[edit.index, 'parentTrackId'] = daug_par
            self.track.loc[edit.index, 'lineageId'] = daug_lin

        return

    def register_obj(self, obj_id, frame, trk_id, cls):
        """Register a new object that has been drawn on the mask

        Args:
            obj_id (int): Object ID of the drawn mask.
            frame (int): Frame location.
            trk_id (int): Track ID.
            cls (str): Cell cycle classification.
        """
        BBOX_FACTOR = 2  # dilate the bounding box when calculating the background intensity.
        mask = self.layers[1].data
        image = self.layers[0].data
        h, w = mask.shape[1], mask.shape[2]
        msk_slice = mask[frame, :, :]
        trk_slice = self.track[self.track['frame'] == frame]
        if obj_id in list(trk_slice['continuous_label']):
            raise ValueError('Object label has been occupied, draw with a bigger label. Current max label: ' +
                             str(np.max(trk_slice['continuous_label'])))
        if trk_id in list(trk_slice['trackId']):
            raise ValueError('Track ID already exists in the selected frame.')
        if cls not in ['G1', 'G2', 'M', 'S', 'G1/G2', 'E']:
            raise ValueError('Cell cycle phase can only be G1, G2, G1/G2, S, M or E.')
        if obj_id not in msk_slice:
            raise ValueError('Object ID is not in the given frame of mask. Draw again. Current max label: ' +
                             str(np.max(trk_slice['continuous_label'])))

        # Register identity of the object
        if cls in ['G1', 'G2', 'G1/G2', 'E']:
            prd_cls = 'G1/G2'
            prob_G = 1
            prob_S = 0
            prob_M = 0
        else:
            prd_cls = cls
            prob_G = 0
            if cls == 'M':
                prob_M = 1
                prob_S = 0
            else:
                prob_M = 0
                prob_S = 1
        if cls == 'E':
            emerging = 1
        else:
            emerging = 0

        new_row = {'frame': frame, 'trackId': trk_id, 'predicted_class': prd_cls, 'resolved_class': cls,
                   'emerging': emerging, 'Probability of G1/G2': prob_G, 'Probability of S': prob_S,
                   'Probability of M': prob_M, 'continuous_label': obj_id, 'mean_intensity': np.nan,
                   'background_mean': np.nan, 'BF_mean': np.nan, 'BF_std': np.nan}
        if trk_id in list(self.track['trackId']):
            # Old track
            old_trk = self.track[self.track['trackId'] == trk_id]
            old_lin = old_trk['lineageId'].iloc[0]
            old_par = old_trk['parentTrackId'].iloc[0]
            new_row['lineageId'] = old_lin
            new_row['parentTrackId'] = old_par
            if old_par != 0:
                nm = '-'.join([str(trk_id), str(old_par), cls])
            else:
                nm = '-'.join([str(trk_id), cls])
        else:
            # New track
            self.track_count = int(np.max((self.track_count, trk_id)))
            new_row['lineageId'] = trk_id
            new_row['parentTrackId'] = 0
            nm = '-'.join([str(trk_id), cls])
        new_row['name'] = nm

        # Register measurements of the object morphology.
        misc = np.zeros(msk_slice.shape, dtype='uint8')
        misc[msk_slice == obj_id] = 1
        p = measure.regionprops(label_image=misc)[0]
        y, x = p.centroid
        min_axis, maj_axis = p.minor_axis_length, p.major_axis_length
        new_row['Center_of_the_object_0'] = x
        new_row['Center_of_the_object_1'] = y
        new_row['minor_axis'] = min_axis
        new_row['major_axis'] = maj_axis
        # Register object inensity
        b1, b3, b2, b4 = expand_bbox(p.bbox, BBOX_FACTOR, (h,w))
        obj_region = misc[b1:b2, b3:b4].copy()
        its_region = image[frame, b1:b2, b3:b4, 0].copy()
        dic_region = image[frame, b1:b2, b3:b4, 2].copy()
        if 0 not in obj_region:
            new_row['background_mean'] = 0
        else:
            new_row['background_mean'] = np.mean(its_region[obj_region == 0])
        cal = obj_region == 1
        new_row['mean_intensity'] = np.mean(its_region[cal])
        new_row['BF_mean'] = np.mean(dic_region[cal])
        new_row['BF_std'] = np.std(dic_region[cal])
        # For extra fields
        for i in set(list(self.track.columns)) - set(list(new_row.keys())):
            new_row[i] = np.nan

        self.track = self.track.append(new_row, ignore_index=True)
        self.track = self.track.sort_values(by=['trackId', 'frame'])

    def get_mx(self, frame):
        mask = self.layers[1].data
        return int(np.max(mask[frame,:,:]))

    def doCorrect(self):
        """Iteration for user command input.
        """
        while True:
            ipt = input("@ Correct > ")
            ipt_list = re.split('\s+', ipt)

            cmd = ipt_list[0]
            #  print(ipt_list[1:])
            args = self.parser.parse_args(ipt_list[1:])
            if args.f:
                args.f = int(args.f) - self.frame_base
                if args.f > int(np.max(self.track['frame'])):
                    print('Frame value larger than the image!')
                    continue
            if args.e:
                args.e = int(args.e) - self.frame_base
                if args.e > int(np.max(self.track['frame'])):
                    print('Frame value larger than the image!')
                    continue

            try:
                if cmd == 'cls':
                    md = 'to_next'
                    if args.s:
                        md = 'single'
                    elif args.e:
                        md = 'range'
                        args.e = int(args.e)
                    assert args.t is not None
                    assert args.f is not None
                    assert args.l is not None
                    self.correct_cls(int(args.t), args.f, str(args.l), md, end_frame=args.e)
                elif cmd == 'r':
                    assert args.t1 is not None
                    assert args.t2 is not None
                    assert args.f is not None
                    self.create_or_replace(int(args.t1), args.f, int(args.t2))
                elif cmd == 'c':
                    assert args.f is not None
                    assert args.t is not None
                    self.create_or_replace(int(args.t), args.f)
                elif cmd == 'cp':
                    assert args.p is not None
                    assert args.d is not None
                    self.create_parent(int(args.p), int(args.d))
                elif cmd == 'dp':
                    assert args.d is not None
                    self.del_parent(int(args.d))
                elif cmd == 'del':
                    assert args.t is not None
                    if args.f is None:
                        self.delete_track(int(args.t))
                    else:
                        self.delete_track(int(args.t), args.f)
                elif cmd == 'div':
                    ds = args.ds
                    assert args.f is not None
                    assert args.p is not None
                    if ds is None:
                        raise ValueError('Specify daughters in comma-separated format.')
                    self.edit_div(int(args.p), list(map(lambda x: int(x), args.ds.split(','))), int(args.f))
                elif cmd == 'b':
                    assert args.o is not None
                    assert args.f is not None
                    assert args.l is not None
                    assert args.t is not None
                    self.register_obj(int(args.o), int(args.f), int(args.t), args.l)
                elif cmd == 's':
                    self.save()
                elif cmd == 'q':
                    break
                elif cmd == 'wq':
                    self.save()
                    break
                elif cmd == 'revert':
                    self.revert()
                elif cmd == 'mx':
                    vl = self.get_mx(int(args.f))
                    print('Maximum object label at frame ' + str(args.f) + ': ' + str(vl))
                else:
                    print("Wrong command argument!")
                    print("=================== Available Commands ===================\n")
                    pprint.pprint(
                        {'cls -t  -f  -l (-s/-e)': 'Correct cell cycle classification for track (t) from frame'
                                                   ' (f) with class label (l). Default correct until next '
                                                   'phase transition; specify end frame with (e) or '
                                                   'only correct single slice (s)',
                         'r   -t1 -t2 -f        ': 'Replace track ID (t1) with a new one (t2) from frame (f).',
                         'c   -t  -f            ': 'Create new track ID for track (t) from frame (f)',
                         'cp  -p  -d            ': 'Create parent (p) - daughter (d) relationship',
                         'dp  -d                ': 'Delete parent - daughter (d) relationship',
                         'del -t                ': 'Delete entire track',
                         'del -t -f             ': 'Delete object of track (t) at frame (f)',
                         'div -p -ds -f         ': 'Set division time of one mitosis event involving '
                                                   'parent (p) and daughters (ds, comma separated) '
                                                   'at frame (f)',
                         'mx -f                 ': 'Check the maximum object label of frame (f).',
                         'b -o -f -t -l         ': 'Register new object on the mask, with the object ID (o), at frame (f), '
                                                   'its track ID (t) and cell cycle classification (l).',
                         'q                     ': 'Quit the interface',
                         's                     ': 'Save the current table',
                         'wq                    ': 'Save and quit the interface',
                         'revert                ': 'Revert to last saved version'})

                    print("\n=================== Parameter Usages ====================\n")
                    self.parser.print_help()

                # Update visualisation
                self.getAnn()
                track_data = self.track.loc[:][['trackId', 'frame', 'Center_of_the_object_1', 'Center_of_the_object_0']]
                track_data = track_data.to_numpy()
                self.layers[2].data = track_data
                label_data = self.track.loc[:][['frame', 'Center_of_the_object_1', 'Center_of_the_object_0']]
                label_data = label_data.to_numpy()
                self.layers[3].properties.clear()
                self.layers[3].data = label_data
                self.layers[3].properties['name'] = self.track.loc[:]['name'].to_numpy()
                self.layers[3].refresh_text()
                
            except ValueError as v:
                print(repr(v))
                continue
            except AssertionError:
                print('Essential parameter not supplied!')
                continue

        return


def get_parser():
    parser = argparse.ArgumentParser(description="pcnaDeep script for visualisation and correction.")
    parser.add_argument(
        "--track",
        metavar="FILE",
        help="path to tracked object table csv file",
    )
    parser.add_argument(
        "--mask",
        metavar="FILE",
        help="path to mask file",
    )
    parser.add_argument(
        "--image",
        metavar="FILE",
        help="path to image file",
    )
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    viewer = Trk_viewer(track_path=args.track,
                   image_path=args.image,
                   mask_path=args.mask,
                   frame_base=0)
    napari.run()
