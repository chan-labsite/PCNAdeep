# -*- coding: utf-8 -*-
import os
import subprocess
from pcnaDeep.data.annotate import relabel_trackID, label_by_track, get_lineage_txt, break_track, save_seq


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
        tracked_mask = tracked_mask.astype('uint16')

        if mode == 'RES':
            # write out processed files for RES folder
            save_seq(tracked_mask, os.path.join(self.root, fm + '_RES'), 'mask', dig_num=self.digit_num, base=self.t_base, sep='')
            txt.to_csv(os.path.join(self.root, fm + '_RES', 'res_track.txt'), sep=' ', index=0, header=False)
        elif mode == 'GT':
            fm = os.path.join(self.root, fm + '_GT')
            self.__saveGT(fm, txt, tracked_mask)
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

    def init_ctc_dir(self):
        """Initialize Cell Tracking Challenge directory

        An example directory ::

            |-- 0001
                |-- 0001_RES
                |-- 0001_GT
                    |-- SEG
                    |-- TRA
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
