# -*- coding: utf-8 -*-
import logging
import pandas as pd
import numpy as np
import pprint
from pcnaDeep.data.utils import deduce_transition, find_daugs
from pcnaDeep.data.annotate import findM
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


def list_dist(a, b):
    """Count difference between elements of two lists.

    Args:
        a (list): classifications with method A
        b (list): classifications with method B
    """

    count = 0
    assert len(a) == len(b)
    for i in range(len(a)):
        if a[i] != b[i]:
            count += 1
        if a[i] == 'G1/G2' and (b[i] in ['G1', 'G2', 'G1*', 'G2*']):
            count -= 1

    return count


def get_rsv_input_gt(track, gt_name='predicted_class', G2_trh=200, no_cls_GT=False):
    """Deduce essential input of resolver from a ground truth.
    """

    logger = logging.getLogger('pcna.Resolver.resolveGroundTruth')
    track['lineageId'] = track['trackId']

    if 'emerging' not in track.columns:
        track['emerging'] = 0
    if 'BF_mean' not in track.columns:
        track['BF_mean'] = 0
    if 'BF_std' not in track.columns:
        track['BF_std'] = 0

    # resolve classification, if supplied
    if not no_cls_GT:
        track['Probability of G1/G2'] = 0
        track['Probability of S'] = 0
        track['Probability of M'] = 0
        track.loc[track[gt_name].str.contains('G'), 'Probability of G1/G2'] = 1
        track.loc[track[gt_name] == 'E', 'Probability of G1/G2'] = 1
        track.loc[track[gt_name] == 'S', 'Probability of S'] = 1
        track.loc[track[gt_name] == 'M', 'Probability of M'] = 1
        track.loc[track[gt_name] == 'S', 'predicted_class'] = 'S'
        track.loc[track[gt_name].str.contains('G'), 'predicted_class'] = 'G1/G2'
        track.loc[track[gt_name] == 'E', 'predicted_class'] = 'E'
        track.loc[track[gt_name] == 'M', 'predicted_class'] = 'M'
        track.loc[track[gt_name] == 'E', 'emerging'] = 1

    ann = {'track': [], 'mitosis_parent': [], 'm_entry': [], 'm_exit': []}
    mt_dic = {}
    imprecise_m = []
    for i in np.unique(track['trackId']):
        sub = track[track['trackId'] == i]
        ann['track'].append(i)
        par = list(sub['parentTrackId'])[0]
        if par:
            par_lin = list(track.loc[track['trackId'] == par, 'lineageId'])[0]
            track.loc[track['trackId'] == i, 'lineageId'] = par_lin
            track.loc[track['parentTrackId'] == i, 'lineageId'] = par_lin
            m_exit = findM(sub[gt_name].tolist(), 'begin')
            if m_exit is None:
                m_exit = sub['frame'].iloc[0]
                logger.warning('Mitosis exit not found for daughter: ' + str(i))
                imprecise_m.append(i)
            else:
                m_exit = sub['frame'].iloc[m_exit]
            if par not in mt_dic.keys():
                mt_dic[par] = {'daug': {i: {'dist': 0, 'm_exit': m_exit}}, 'div': -1}
            else:
                mt_dic[par]['daug'][i] = {'dist': 0, 'm_exit': m_exit}
            ann['mitosis_parent'].append(int(par))
            ann['m_entry'].append(None)
            ann['m_exit'].append(int(m_exit))
        else:
            ann['mitosis_parent'].append(None)
            ann['m_entry'].append(None)
            ann['m_exit'].append(None)
    ann = pd.DataFrame(ann, dtype=int)
    for i in mt_dic.keys():
        par_sub = track[track['trackId'] == i]
        m_entry = findM(par_sub[gt_name].tolist(), 'end')
        if m_entry is None:
            logger.warning('Mitosis entry not found for parent: ' + str(i) + '. Will not resolve this M phase.')
        else:
            m_entry = par_sub['frame'].iloc[m_entry]
            mt_dic[i]['div'] = m_entry
            ann.loc[ann['track'] == i, 'm_entry'] = m_entry

    # Note resolver only takes predicted class G1/G2, no G1 or G2;
    # Resolver classifies G1 or G2 based on intensity, so first mask intensity and background intensity,
    # then recover from table joining
    track_masked = track.copy()
    if 'mean_intensity' not in track_masked.columns:
        track_masked['mean_intensity'] = 0
        track_masked.loc[track_masked[gt_name] == 'G2', 'mean_intensity'] = G2_trh + 1
    if 'background_mean' not in track_masked.columns:
        track_masked['background_mean'] = 0
    track_masked.loc[track_masked[gt_name].str.contains('G'), gt_name] = 'G1/G2'
    track_masked['predicted_class'] = track_masked[gt_name]

    logger.debug(pprint.pformat(mt_dic))

    return track_masked, ann, mt_dic, imprecise_m


def resolve_from_gt(track, gt_name='predicted_class', extra_gt=None, G2_trh=None, no_cls_GT=False,
                    minG=1, minS=1, minM=1, minLineage=0):
    """Resolve cell cycle phase from the ground truth. Wrapper of `get_rsv_input_gt()`.

    Args:
        track (pandas.DataFrame): data frame of each object each row, must have following columns:
            - trackId, frame, parentTrackId, <ground truth classification column>
        gt_name (str): refers to the column in track that corresponds to ground truth classification.
        extra_gt (str): refers to the column in track that has G2 ground truth if `gt_name` does not. See notes below.
        G2_trh (int): intensity threshold for classifying G2 phase (for arrest tracks only).
        no_cls_GT (bool): Set to `true` if no classification ground truth is provided.
            Will resolve based on current classifications.
        minG (int): minimum G phase frame length (default 1).
        minS (int): minimum S phase frame length (default 1).
        minM (int): minimum M phase frame length (default 1).
        minLineage (int): minimum lineage frame length to resolve (default 0, resolve all tracks).

    Note:
        - If do not want G2 to be classified based on thresholding, rather, based on ground truth classification.
            Simply leave `G2_trh=None` and the threshold will be calculated as the smallest average intensity of G2 phase
            in labeled tracks (outlier smaller than mena - 3*sd excluded).
        - If the ground truth column does not contain `G2` instances, tell the program to look at
            an extra partially G2 ground truth column like `resolved_class` to extract information. This may be useful when
            `predicted_class` has been corrected from the Correction Interface which only contains G1/G2 but not G2. In this
            case, you can also assign `resolved_class` as the ground truth classification column. Both will work.  
        - If `mean_intensity` or `background_mean` column is not in the table, will set the threshold to 100.  
        - Use at own risk if the input classification in not reliable.
    """
    if G2_trh is None:
        if 'mean_intensity' in track.columns:
            if 'G2' not in list(track[gt_name]):
                assert 'G2' in list(track[extra_gt]), 'G2 not found in either gt_name or extra_gt columns.'
                G2_tracks = track.loc[track[extra_gt] == 'G2']
            else:
                G2_tracks = track.loc[track[gt_name] == 'G2']
            avgs = []
            for i in np.unique(G2_tracks['trackId']):
                sub = G2_tracks[G2_tracks['trackId'] == i]
                avgs.append(np.mean(sub['mean_intensity'] - sub['background_mean']))
            avgs = np.array(avgs)
            avgs = avgs[avgs >= (np.mean(avgs) - np.std(avgs) * 3)]
            G2_trh = int(np.floor(np.min(avgs))) - 1
        elif 'mean_intensity' not in track.columns or 'background_mean' not in track.columns:
            G2_trh = 100  # dummy threshold
        else:
            raise ValueError('Must provide a G2 intensity threshold or provide G2 ground truth classification.')
    print('Using G2 intensity threshold: ' + str(G2_trh))
    track_masked, ann, mt_dic, imprecise_m = get_rsv_input_gt(track, gt_name, G2_trh=G2_trh, no_cls_GT=no_cls_GT)
    r = Resolver(track_masked, ann, mt_dic, maxBG=minG, minS=minS, minM=minM, minLineage=minLineage,
                 impreciseExit=imprecise_m, G2_trh=G2_trh)
    rsTrack, phase = r.doResolve()
    rsTrack = rsTrack[['trackId', 'frame', 'resolved_class', 'name']]
    if 'resolved_class' in track.columns:
        del track['resolved_class']
    if 'name' in track.columns:
        del track['name']
    rsTrack = track.merge(rsTrack, on=['trackId','frame'])

    return rsTrack, phase


class Resolver:

    def __init__(self, track, ann, mt_dic, maxBG=25, minS=20, minM=10, minLineage=10, impreciseExit=None, G2_trh=100):
        """Resolve cell cycle duration, identity G1 or G2.

        Args:
            - pcnaDeep.tracker outputs:
                track (pandas.DataFrame): tracked object table;
                ann (pandas.DataFrame): track annotation table;
                mt_dic (dict): mitosis information lookup dictionary;
                impreciseExit (list): list of tracks which M-G1 transition not clearly labeled.
            - GPR algorithm parameters for searching S/M phase:
                maxBG (float): maximum background class appearance allowed within target phase;
                minS (float): minimum target S phase length;
                minM (float): minimum target M phase length.
            - Options:
                minLineage (int): minimum lineage length to record in the output phase table.
                G2_trh (int): G2 intensity threshold for classifying arrested G1/G2 tracks. Background subtracted.
        """
        if impreciseExit is None:
            impreciseExit = []
        self.logger = logging.getLogger('pcna.Resolver')
        self.impreciseExit = impreciseExit
        self.track = track
        self.ann = ann
        self.maxBG = maxBG
        self.minS = minS
        self.mt_dic = mt_dic
        self.minM = minM
        self.rsTrack = None
        self.minLineage = minLineage
        self.unresolved = []
        self.mt_unresolved = []
        self.arrest = {}  # trackId : arrest phase
        self.G2_trh = G2_trh
        self.phase = pd.DataFrame(columns=['track', 'type', 'G1', 'S', 'M', 'G2', 'parent'])

    def doResolve(self):
        """Main function of class resolver.
        
        Returns:
            pandas.DataFrame: tracked object table with additional column 'resolved_class'.
            pandas.DataFrame: phase table with cell cycle durations.
        """

        self.logger.info('Resolving cell cycle phase...')
        track = self.track.copy()
        rt = pd.DataFrame()
        for i in np.unique(track['lineageId']):
            d = track[track['lineageId'] == i]
            t = self.resolveLineage(d, i)
            rt = rt.append(t)
        rt = rt.sort_values(by=['trackId', 'frame'])
        self.rsTrack = rt.copy()
        self.check_trans_integrity()
        self.mt_unresolved = list(np.unique(self.mt_unresolved))
        if self.mt_unresolved:
            self.logger.warning('Sequential mitosis without S phase; Ignore tracks: ' + str(self.mt_unresolved)[1:-1])
        if self.unresolved:
            self.logger.warning('Numerous classification change after resolving, check: ' + str(self.unresolved)[1:-1])

        self.resolveArrest(self.G2_trh)
        phase = self.doResolvePhase()
        self.getAnn()
        return self.rsTrack, phase

    def check_trans_integrity(self):
        """Check track transition integrity. If transition other than G1->S; S->G2, G2->M, M->G1 found, do not resolve.
        """
        for t in np.unique(self.rsTrack['trackId']):
            if t not in self.mt_unresolved and t not in self.unresolved:
                sub = self.rsTrack[self.rsTrack['trackId'] == t]
                rcls = list(sub['resolved_class'])
                for i in range(1,sub.shape[0]):
                    if rcls[i-1] != rcls[i]:
                        trs = '-'.join([rcls[i-1], rcls[i]])
                        if trs not in ['G1-S','S-G2','G2-M','M-G1']:
                            self.logger.warning('Wrong transition ' + trs + ' in track: ' + str(t))
        return

    def getAnn(self):
        """Add an annotation column to tracked object table
        The annotation format is track ID - (parentTrackId, optional) - resolved_class
        """
        ann = []
        cls_col = 'resolved_class'
        if cls_col not in self.rsTrack.columns:
            print('Phase not resolved yet. Using predicted phase classifications.')
            cls_col = 'predicted_class'
        track_id = list(self.rsTrack['trackId'])
        parent_id = list(self.rsTrack['parentTrackId'])
        cls_lb = list(self.rsTrack[cls_col])
        for i in range(self.rsTrack.shape[0]):
            inform = [str(track_id[i]), str(parent_id[i]), cls_lb[i]]
            if inform[1] == '0':
                del inform[1]
            ann.append('-'.join(inform))
        self.rsTrack['name'] = ann
        return

    def resolveArrest(self, G2_trh=None):
        """Determine G1/G2 arrest tracks.
            - If `G2_trh` is supplied, determine G2 based on background-subtracted mean of the track
                (averaged across frames).
            - If `G2_trh` is not supplied, assign G1 or G2 classification according to 2-center K-mean.

        Args:
            G2_trh (int): int between 1-255, above the threshold will be classified as G2.
        """

        trk = self.rsTrack[self.rsTrack['trackId'].isin(list(self.arrest.keys()))].copy()
        intensity = []
        ids = []
        for i in self.arrest.keys():
            sub = trk[trk['trackId'] == i]
            if self.arrest[i] == 'S':
                continue
            corrected_mean = np.mean(sub['mean_intensity'] - sub['background_mean'])
            intensity.append(corrected_mean)
            ids.append(i)

        if G2_trh is None:
            self.logger.warning('No G2 threshold provided, using KMean clustering to distinguish arrested G1/G2 track.')
            X = np.expand_dims(np.array(intensity), axis=1)
            X = MinMaxScaler().fit_transform(X)
            y = list(KMeans(2).fit_predict(X))
        else:
            if G2_trh < 1 or G2_trh > 255:
                raise ValueError('G2 threshold must be within the interval: 1~255.')
            y = []
            for i in range(len(ids)):
                if intensity[i] > G2_trh:
                    y.append(1)
                else:
                    y.append(0)

        for i in range(len(ids)):
            if y[i] == 0:
                self.arrest[ids[i]] = 'G1'
                self.rsTrack.loc[self.rsTrack['trackId'] == ids[i], 'resolved_class'] = 'G1*'
            else:
                self.arrest[ids[i]] = 'G2'
                self.rsTrack.loc[self.rsTrack['trackId'] == ids[i], 'resolved_class'] = 'G2*'
        return

    def resolveLineage(self, lineage, main):
        """Resolve all tracks in a lineage recursively
        main (int): the parent track ID of current search
        """

        info = self.ann.loc[self.ann['track'] == main]
        m_entry = info['m_entry'].values[0]
        m_exit = info['m_exit'].values[0]

        if len(np.unique(lineage['trackId'])) == 1:
            rsd = self.resolveTrack(lineage.copy(), m_entry=m_entry, m_exit=m_exit)
            return rsd
        else:
            out = pd.DataFrame()
            lg = lineage[lineage['trackId'] == main]
            out = out.append(self.resolveTrack(lg.copy(), m_entry=m_entry, m_exit=m_exit))
            daugs = self.mt_dic[main]['daug']
            for i in list(daugs.keys()):
                out = out.append(
                    self.resolveLineage(lineage[lineage['trackId'].isin(find_daugs(lineage, i) + [i])].copy(), i))
            return out

    def resolveTrack(self, trk, m_entry=None, m_exit=None):
        """Resolve single track.
        
        Args:
            trk (pandas.DataFrame): track table
            m_entry (int): time of mitosis entry corresponding to 'frame' column in table
            m_exit (int): time of mitosis exit corresponding to 'frame' column in table
            
                If no m time supplied, only treat as G1/G2/S track.
                Arrested track not resolved, return full G1/G2 list.
            
        Returns:
            pandas.DataFrame table with addition column of resolved class
        """

        UNRESOLVED_FRACTION = 0.2  # after resolving the class, if more than x% class has been corrected, label with
        resolved_class = ['G1/G2' for _ in range(trk.shape[0])]
        if trk.shape[0] == 0:
            raise ValueError('Track not found!')
            #return None

        track_id = trk['trackId'].tolist()[0]
        cls = list(trk['predicted_class'])
        if list(np.unique(cls)) == ['S']:
            trk['resolved_class'] = 'S'
            self.arrest[track_id] = 'S'
            return trk

        confid = np.array(trk[['Probability of G1/G2', 'Probability of S', 'Probability of M']])
        out = deduce_transition(l=cls, tar='S', confidence=confid, min_tar=self.minS,
                                max_res=self.maxBG, casual_end=False)

        flag = False
        if not (out is None or out[0] == out[1]):
            flag = True
            a = (out[0], np.min((out[1] + 1, len(resolved_class) - 1)))
            resolved_class[a[0]:a[1] + 1] = ['S' for _ in range(a[0], a[1] + 1)]

            if a[0] > 0:
                resolved_class[:a[0]] = ['G1' for _ in range(a[0])]
            if a[1] < len(resolved_class) - 1:
                resolved_class[a[1]:] = ['G2' for _ in range(len(resolved_class) - a[1])]

        frame = trk['frame'].tolist()
        if m_exit is not None:
            emerging = trk['emerging'].tolist()
            if 1 in emerging:
                exit_idx = int(np.min((frame.index(m_exit), emerging.index(1))))  # Emerging classification refers to G1
            else:
                exit_idx = frame.index(m_exit)
            resolved_class[:exit_idx + 1] = ['M' for _ in range(exit_idx + 1)]
            i = exit_idx + 1
            while i < len(resolved_class):
                if resolved_class[i] == 'G1/G2':
                    resolved_class[i] = 'G1'
                else:
                    break
                i += 1
        if m_entry is not None:
            resolved_class[frame.index(m_entry):] = ['M' for _ in range(len(resolved_class) - frame.index(m_entry))]
            i = frame.index(m_entry) - 1
            while i >= 0:
                if resolved_class[i] == 'G1/G2':
                    resolved_class[i] = 'G2'
                else:
                    break
                i -= 1

        if not flag and m_exit is not None and m_entry is not None:
            resolved_class = cls.copy()
            self.mt_unresolved.extend([track_id] + find_daugs(self.track, track_id))

        if m_exit is None and m_entry is None:
            # some tracks begin/end with mitosis and not associated during refinement. In this case, override any
            # classification at terminal. Only track ends/begin with M will be kept, otherwise override with G1/G2.
            # WARNING: this can leads to false negative
            mt_out_begin = deduce_transition(l=cls, tar='M', confidence=confid, min_tar=1,
                                             max_res=self.maxBG)
            mt_out_end = deduce_transition(l=cls[::-1], tar='M', confidence=confid[::-1, :], min_tar=1,
                                           max_res=self.maxBG)
            if mt_out_end is not None:
                #  check if out and end interval overlaps
                compare = (len(cls)-mt_out_end[1]-1, len(cls)-mt_out_end[0]-1)
                if mt_out_begin is not None:
                    if compare[0] == mt_out_begin[0] and compare[1] == mt_out_begin[1]:
                        #  if overlap, assign larger index one to None
                        if mt_out_end[0] < mt_out_begin[0]:
                            mt_out_begin = None
                        else:
                            mt_out_end = None

            if mt_out_begin is not None and mt_out_end is None:
                if mt_out_begin[0] == 0:
                    resolved_class[mt_out_begin[0]: mt_out_begin[1] + 1] = ['M' for _ in
                                                                            range(mt_out_begin[0], mt_out_begin[1] + 1)]
                    # if followed with G1/G2 only, change to G1
                    if np.unique(resolved_class[mt_out_begin[1] + 1:]).tolist() == ['G1/G2']:
                        resolved_class = ['G1' if i == 'G1/G2' else i for i in resolved_class]

            if mt_out_end is not None and mt_out_begin is None:
                if mt_out_end[0] == 0:
                    resolved_class = resolved_class[::-1]
                    resolved_class[mt_out_end[0]: mt_out_end[1] + 1] = ['M' for _ in
                                                                        range(mt_out_end[0], mt_out_end[1] + 1)]
                    if np.unique(resolved_class[mt_out_end[1] + 1:]).tolist() == ['G1/G2']:
                        resolved_class = ['G2' if i == 'G1/G2' else i for i in resolved_class]
                    resolved_class = resolved_class[::-1]

        trk['resolved_class'] = resolved_class
        if np.unique(resolved_class).tolist() == ['G1/G2']:
            self.arrest[track_id] = 'G1'
        if list_dist(cls, resolved_class) > UNRESOLVED_FRACTION * len(resolved_class):
            self.unresolved.append(track_id)
        return trk

    def doResolvePhase(self):
        """Resolve phase durations
        """
        out = {'track': [], 'type': [], 'length': [], 'lin_length':[], 'arrest': [], 
               'G1': [], 'S': [], 'M': [], 'G2': [], 'parent': []}

        # register tracks
        for i in range(self.ann.shape[0]):
            info = self.ann.loc[i, :]
            if info['track'] in self.unresolved or info['track'] in self.mt_unresolved:
                continue
            sub = self.rsTrack[self.rsTrack['trackId'] == info['track']]
            lin = list(sub['lineageId'])[0]
            sub_lin = self.rsTrack[self.rsTrack['lineageId'] == lin]
            lin_length = int(np.max(sub_lin['frame']) - np.min(sub_lin['frame'])) + 1
            length = int(np.max(sub['frame']) - np.min(sub['frame']) + 1)
            par = info['mitosis_parent']
            if par is None or par in self.mt_unresolved:
                par = 0
            out['track'].append(info['track'])
            out['length'].append(length)
            out['parent'].append(par)
            out['lin_length'].append(lin_length)
            out['M'].append(np.nan)  # resolve later

            if list(np.unique(sub['resolved_class'])) == ['G1*']:
                out['type'].append('arrest' + '-G1')
                out['arrest'].append(length)
                out['G1'].append(np.nan)
                out['S'].append(np.nan)
                out['G2'].append(np.nan)
            elif list(np.unique(sub['resolved_class'])) == ['G2*']:
                out['type'].append('arrest' + '-G2')
                out['arrest'].append(length)
                out['G1'].append(np.nan)
                out['S'].append(np.nan)
                out['G2'].append(np.nan)
            elif list(np.unique(sub['resolved_class'])) == ['S']:
                out['type'].append('arrest' + '-S')
                out['arrest'].append(length)
                out['G1'].append(np.nan)
                out['S'].append(np.nan)
                out['G2'].append(np.nan)
            elif list(np.unique(sub['resolved_class'])) == ['M']:
                out['type'].append('arrest' + '-M')
                out['arrest'].append(length)
                out['G1'].append(np.nan)
                out['S'].append(np.nan)
                out['G2'].append(np.nan)
            else:
                out['type'].append('normal')
                out['arrest'].append(np.nan)
                cls = list(np.unique(sub['resolved_class']))
                remain = ['G1', 'G2', 'S']
                for c in cls:
                    if c == 'M' or c == 'G1/G2':
                        continue
                    fme = list(sub[sub['resolved_class'] == c]['frame'])
                    lgt = fme[-1] - fme[0] + 1
                    if sub['resolved_class'].tolist()[0] == c:
                        lgt = '>' + str(lgt)
                    elif sub['resolved_class'].tolist()[-1] == c:
                        lgt = '>' + str(lgt)
                    out[c].append(lgt)
                    remain.remove(c)
                for u in remain:
                    out[u].append(np.nan)
        out = pd.DataFrame(out)

        # register mitosis, mitosis time only registered in daughter 'M' column
        # **Only registered mitosis in mt_dic will be recorded
        for i in self.mt_dic.keys():
            if i in self.mt_unresolved:
                continue
            for j in self.mt_dic[i]['daug'].keys():
                m = self.mt_dic[i]['daug'][j]['m_exit'] - self.mt_dic[i]['div'] + 1
                out.loc[out['track'] == j, 'M'] = int(m)

        # filter length
        out = out[out['lin_length'] >= self.minLineage]

        # imprecise M (daughter associated with parent, but daughter no M classification) exit labeled
        out['imprecise_exit'] = 0
        for i in out['track'].tolist():
            if i in self.impreciseExit:
                out.loc[out['track'] == i, 'imprecise_exit'] = 1

        return out
