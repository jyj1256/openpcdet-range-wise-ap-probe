"""
WaymoNpzDataset — 기존 전처리 .npz(points N×5, gt_boxes M×7, gt_classes)를
OpenPCDet 파이프라인에 직접 연결하는 어댑터. 재변환 없이 dataset_infos.json 재활용.

npz 포맷: points(N,5)=x,y,z,intensity,elongation, gt_boxes(M,7)=x,y,z,dx,dy,dz,heading,
          gt_classes(M,) int 0=소형차,1=대형차,2=이륜차
"""
import copy
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch

from ..dataset import DatasetTemplate

# npz의 정수 클래스 → OpenPCDet ASCII 클래스명 (yaml의 CLASS_NAMES와 일치해야 함)
CLASS_IDX_TO_NAME = {0: 'Vehicle', 1: 'Large_Vehicle', 2: 'Cyclist'}
# OpenPCDet split 모드 → dataset_infos.json의 split 값
SPLIT_MAP = {'train': 'training', 'test': 'validation', 'val': 'validation'}


class WaymoNpzDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training,
            root_path=root_path, logger=logger
        )
        # 정수 클래스 → 클래스명: yaml의 CLASS_NAMES 순서로 생성(3클래스/4클래스 자동 대응)
        # 3클래스=[Vehicle,Large_Vehicle,Cyclist], 4클래스=[...,Motorcycle,Bicycle]
        self.class_idx_to_name = {i: n for i, n in enumerate(class_names)}
        self.CROP_RADIUS = self.dataset_cfg.get('CROP_RADIUS', 0)  # 근거리 원형 크롭 반경(m), 0=미사용
        self.DROP_BOX_MIN_PTS = self.dataset_cfg.get('DROP_BOX_MIN_PTS', 0)  # 학습 시 점<N 박스=박스+내부점 통째삭제(0=미사용)
        self.PAINT_DIR = self.dataset_cfg.get('PAINT_DIR', None)  # 설정시 점을 이 폴더의 painted npz(예 frames_v2_paint, 10채널)로 로드
        # Waymo LEVEL_1 조회표(원본 parquet의 difficulty/num_lidar_points_in_box). 설정 시 평가 기준이
        # 공식 정의로 바뀐다: 점>5 & not hard = 채점 / 점1~5 or hard = 무시 / 점0 = 삭제 (2026-09-11 지시)
        self.WAYMO_L1_LOOKUP = self.dataset_cfg.get('WAYMO_L1_LOOKUP', None)
        self._l1_lut = None
        # 데이터 루트 (info['file']이 이 경로 기준 상대경로). 기본 C:/waymo_detection
        self.data_root = Path(self.dataset_cfg.get('DATA_ROOT', self.root_path))
        infos_json = self.data_root / self.dataset_cfg.get('INFOS_JSON', 'dataset_infos.json')
        with open(infos_json, encoding='utf-8') as f:
            all_infos = json.load(f)

        split = self.dataset_cfg.DATA_SPLIT[self.mode]
        target = SPLIT_MAP.get(split, split)
        self.infos = [i for i in all_infos if i['split'] == target]

        interval = self.dataset_cfg.get('SAMPLED_INTERVAL', {}).get(self.mode, 1)
        if interval and interval > 1:
            self.infos = self.infos[::interval]

        # 디버그/게이트용: 프레임 수 제한
        limit = self.dataset_cfg.get('LIMIT_FRAMES', {}).get(self.mode, None)
        if limit:
            self.infos = self.infos[:limit]

        if self.logger is not None:
            self.logger.info('WaymoNpz %s(%s): %d frames' % (self.mode, target, len(self.infos)))

    def __len__(self):
        return len(self.infos)

    def _load_npz(self, info):
        fp = info['file']
        if getattr(self, 'PAINT_DIR', None):
            fp = self.PAINT_DIR + '/' + fp.split('/')[-1]  # frames_v2/xxx.npz -> frames_v2_paint/xxx.npz (10채널)
        data = np.load(self.data_root / fp)
        points = data['points'].astype(np.float32)
        # intensity(0~22400, 극단 outlier)를 tanh로 정규화 — 미정규화 시 박스회귀 학습 실패
        points[:, 3] = np.tanh(points[:, 3])
        if 'gt_override' in info:
            # 라벨 정제 ablation용: infos json이 npz의 gt를 대체 (points는 npz 그대로)
            gt_boxes = np.array(info['gt_override']['boxes'], np.float32).reshape(-1, 7)
            gt_classes = np.array(info['gt_override']['classes'], np.int64).reshape(-1)
        else:
            gt_boxes = data['gt_boxes'].astype(np.float32).reshape(-1, 7)
            gt_classes = data['gt_classes'].reshape(-1).astype(np.int64)
        # 근거리 원형 크롭(config CROP_RADIUS): 원 밖 점 제거 + 원에 '걸치는' 박스는 박스+내부점 통째 삭제
        crop_r = float(getattr(self, 'CROP_RADIUS', 0) or 0)
        if crop_r > 0:
            keep_pt = np.hypot(points[:, 0], points[:, 1]) <= crop_r
            if len(gt_boxes):
                cx, cy, dx, dy, hd = (gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 6])
                cs, sn = np.cos(hd), np.sin(hd)
                cmax = np.zeros(len(gt_boxes), np.float32)
                for sx in (0.5, -0.5):
                    for sy in (0.5, -0.5):
                        px = cx + sx*dx*cs - sy*dy*sn
                        py = cy + sx*dx*sn + sy*dy*cs
                        cmax = np.maximum(cmax, np.hypot(px, py))
                fully_in = cmax <= crop_r          # 4코너 다 원 안 = 완전 포함만 유지
                removed = ~fully_in
                if removed.any():
                    from ...ops.roiaware_pool3d import roiaware_pool3d_utils
                    idx = roiaware_pool3d_utils.points_in_boxes_cpu(
                        torch.from_numpy(points[:, :3]), torch.from_numpy(gt_boxes[removed])).numpy()
                    keep_pt &= idx.sum(0) == 0      # 삭제 박스(걸침/밖) 내부 점도 제거
                gt_boxes = gt_boxes[fully_in]; gt_classes = gt_classes[fully_in]
            points = points[keep_pt]
        # 이상치 박스 필터 (라벨 노이즈 제거 — 박스회귀가 노이즈에 민감)
        if getattr(self, 'FILTER_OUTLIERS', True) and len(gt_boxes):
            B = {0: ([2.0, 8.0], [1.0, 3.5], [1.0, 3.0]),
                 1: ([3.0, 25.0], [1.5, 4.0], [1.5, 5.5]),
                 2: ([0.5, 3.5], [0.3, 1.6], [0.8, 2.5]),
                 3: ([0.5, 3.5], [0.3, 1.6], [0.8, 2.5])}  # 4클래스(자전거)용; 3클래스엔 무영향
            d = gt_boxes[:, 3:6]
            keep = np.ones(len(gt_boxes), dtype=bool)
            for c, (bx, by, bz) in B.items():
                m = gt_classes == c
                keep &= ~m | ((d[:, 0] >= bx[0]) & (d[:, 0] <= bx[1]) &
                              (d[:, 1] >= by[0]) & (d[:, 1] <= by[1]) &
                              (d[:, 2] >= bz[0]) & (d[:, 2] <= bz[1]))
            gt_boxes = gt_boxes[keep]; gt_classes = gt_classes[keep]
        # 점<N 박스 통째 제거 (박스 + 내부 점 삭제)
        #  DROP_BOX_MIN_PTS      : 학습 시에만 적용(기존 pt20 실험용)
        #  DROP_BOX_MIN_PTS_EVAL : val/test에도 적용 (2026-09-07 추가)
        #    — infos의 min_pts 필터는 box_semstats 기준이라 frames_v2 실측과 어긋나,
        #      점 0~2개짜리 검출 불가능 GT가 평가에 남아 있었다. 이걸 점까지 같이 지운다.
        drop_min = float(getattr(self, 'DROP_BOX_MIN_PTS', 0) or 0)
        if not getattr(self, 'training', False):
            drop_min = float(getattr(self, 'DROP_BOX_MIN_PTS_EVAL', 0) or 0)
        if drop_min > 0 and len(gt_boxes):
            from ...ops.roiaware_pool3d import roiaware_pool3d_utils
            idxmat = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, :3]), torch.from_numpy(gt_boxes)).numpy()  # (nbox, npts)
            drop = idxmat.sum(1) < drop_min
            if drop.any():
                pts_in_drop = idxmat[drop].sum(0) > 0        # 삭제 박스 내부 점
                points = points[~pts_in_drop]                # 점 제거
                gt_boxes = gt_boxes[~drop]; gt_classes = gt_classes[~drop]  # 박스 제거
        # 점 개수 필터 (Waymo LEVEL_1 기준) — TOP LiDAR만 써서 77%가 점<5개라 학습 불가 객체 제거
        min_pts = getattr(self, 'MIN_POINTS', 5)
        if min_pts > 0 and len(gt_boxes):
            from ...ops.roiaware_pool3d import roiaware_pool3d_utils
            npts = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, :3]), torch.from_numpy(gt_boxes)).numpy().sum(1)
            keep = npts >= min_pts
            gt_boxes = gt_boxes[keep]; gt_classes = gt_classes[keep]
        return points, gt_boxes, gt_classes

    def __getitem__(self, index):
        info = self.infos[index]
        points, gt_boxes, gt_classes = self._load_npz(info)
        gt_names = np.array([self.class_idx_to_name[int(c)] for c in gt_classes])

        input_dict = {
            'frame_id': Path(info['file']).stem,
            'points': points,
            'gt_names': gt_names,
            'gt_boxes': gt_boxes,
        }
        return self.prepare_data(data_dict=input_dict)

    # ── Waymo LEVEL_1 판정 (원본 parquet 라벨을 중심 최근접으로 붙임) ──
    def _waymo_l1_flags(self, info, gt_boxes, tol=0.3):
        """1=채점(LEVEL_1) / 0=무시(점1~5 또는 hard) / -1=삭제(점0, 조회 실패 포함)"""
        if self._l1_lut is None:
            p = str(self.WAYMO_L1_LOOKUP)
            p = p if os.path.isabs(p) else str(self.data_root / p)
            import time as _t
            for _i in range(200):        # 백신(V3)이 새 파일 스캔 중이면 OSError — 잠깐 기다렸다 재시도
                try:
                    with open(p, 'rb') as f:
                        self._l1_lut = pickle.load(f)
                    break
                except OSError:
                    _t.sleep(0.1)
            if self._l1_lut is None:
                raise OSError('WAYMO_L1_LOOKUP 로드 실패: %s' % p)
        b = os.path.basename(info['file'])[:-4]
        if b[:3] in ('va_', 'tr_'): b = b[3:]
        parts = b.split('_')
        tab = self._l1_lut.get(('_'.join(parts[:-1]), int(parts[-1])))
        if tab is None:
            return None                       # 조회표에 없는 프레임 → 호출부가 기존 기준으로 처리
        flags = np.full(len(gt_boxes), -1, np.int8)
        if not len(gt_boxes):
            return flags
        for k in range(len(gt_boxes)):
            d = np.hypot(tab[:, 0] - gt_boxes[k, 0], tab[:, 1] - gt_boxes[k, 1])
            j = int(d.argmin())
            if d[j] > tol:
                continue                      # 원본에서 못 찾음 → 삭제 취급
            npts, hard = tab[j, 3], (tab[j, 2] == 2)
            if npts > 5 and not hard: flags[k] = 1
            elif npts >= 1:           flags[k] = 0
        return flags

    # ── 평가: gt를 npz에서 직접 로드, 정확한 경기대 지표(BEV IoU, 일괄 0.7) 계산 ──
    def evaluation(self, det_annos, class_names, **kwargs):
        from ...ops.iou3d_nms import iou3d_nms_utils  # rotated BEV IoU (CUDA)
        import torch

        iou_thresh = float(kwargs.get('iou_thresh', 0.7))  # 기본 IoU 0.7 (2026-09-11 지시, 0.5 폐기)
        # 클래스별 IoU 임계값 (Waymo 공식: 차량 0.7, 이륜차 0.5). per_class_iou로 override
        thr_by = {c: iou_thresh for c in class_names}
        thr_by.update(kwargs.get('per_class_iou', {}))
        # LEVEL_1: 점 min_points개 이상 GT만 평가 (Waymo 공식 기준)
        min_points = int(kwargs.get('eval_min_points', 0))
        # 근거리 크롭 평가.
        #  - explicit max_range(probe): GT 중심거리 기준(간이)
        #  - CROP_RADIUS(near20 등): 학습이 코너기준 '걸침=삭제'라 GT도 코너 기준 완전포함만 평가(일치)
        explicit_mr = float(kwargs.get('max_range', 0) or 0)
        crop_r = float(getattr(self, 'CROP_RADIUS', 0) or 0)
        max_range = explicit_mr or crop_r        # 예측 crop(중심기준) 및 공통
        gt_corner = (explicit_mr == 0 and crop_r > 0)  # near20: GT를 코너기준 완전포함으로
        from ...ops.roiaware_pool3d import roiaware_pool3d_utils
        gt_annos = []
        l1_miss = 0          # 조회표에 없던 프레임 수
        for info in self.infos:
            data = np.load(self.data_root / info['file'])
            pts = data['points'].astype(np.float32)
            # gt_override 있으면 그걸로 평가(4클래스 세분 라벨/정제 라벨); 없으면 npz 원본
            if 'gt_override' in info:
                gt_boxes = np.array(info['gt_override']['boxes'], np.float32).reshape(-1, 7)
                gt_classes = np.array(info['gt_override']['classes'], np.int64).reshape(-1)
            else:
                gt_boxes = data['gt_boxes'].astype(np.float32).reshape(-1, 7)
                gt_classes = data['gt_classes'].reshape(-1).astype(np.int64)
            # 점 부족 GT는 '제거'가 아니라 Waymo식 '무시(ignore)' 처리 (2026-09-07)
            #  - GT 집계에서 빼되, 그 위치에 나온 검출은 오탐으로도 세지 않는다
            #  - 점군은 손대지 않는다(센서 관측값 불변 = Waymo 벤치마크 원칙)
            ig_boxes = np.zeros((0, 7), np.float32); ig_classes = np.zeros(0, np.int64)
            mp = min_points
            # WAYMO_L1_LOOKUP이 있으면 공식 LEVEL_1 정의가 실측 점수 기준을 대체한다
            l1_on = bool(kwargs.get('waymo_l1', True)) and getattr(self, 'WAYMO_L1_LOOKUP', None)
            if l1_on and len(gt_boxes):
                fl = self._waymo_l1_flags(info, gt_boxes)
                if fl is None:
                    l1_miss += 1                      # 조회 실패 프레임은 기존(실측 점) 기준으로
                else:
                    # ignore_low_pts=False(probe --no_ignore)면 비대상 GT를 무시가 아니라 '완전 제외'로
                    # 처리한다 — 그 자리 검출은 오탐(FP)이 된다. 두 잣대 비교용(lbl 실험).
                    if kwargs.get('ignore_low_pts', True):
                        ig_boxes = gt_boxes[fl == 0]; ig_classes = gt_classes[fl == 0]
                    gt_boxes = gt_boxes[fl == 1]; gt_classes = gt_classes[fl == 1]
                    mp = 0                            # 점 기준은 조회표가 담당
            if mp > 0 and len(gt_boxes):
                npts = roiaware_pool3d_utils.points_in_boxes_cpu(
                    torch.from_numpy(pts[:, :3]), torch.from_numpy(gt_boxes)).numpy().sum(1)
                keep = npts >= mp
                # ignore_low_pts=False면 무시하지 않고 완전 제외(그 자리 검출은 오탐으로 셈)
                if kwargs.get('ignore_low_pts', True):
                    ig_boxes = gt_boxes[~keep]; ig_classes = gt_classes[~keep]
                gt_boxes = gt_boxes[keep]; gt_classes = gt_classes[keep]
            if max_range > 0 and len(ig_boxes):
                ig_keep = np.hypot(ig_boxes[:, 0], ig_boxes[:, 1]) <= max_range
                ig_boxes = ig_boxes[ig_keep]; ig_classes = ig_classes[ig_keep]
            if max_range > 0 and len(gt_boxes):
                if gt_corner:
                    # 코너 기준 완전포함만 유지(학습의 걸침=삭제와 일치)
                    cx, cy, dx, dy, hd = (gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 6])
                    cs, sn = np.cos(hd), np.sin(hd); cmx = np.zeros(len(gt_boxes), np.float32)
                    for sx in (0.5, -0.5):
                        for sy in (0.5, -0.5):
                            cmx = np.maximum(cmx, np.hypot(cx+sx*dx*cs-sy*dy*sn, cy+sx*dx*sn+sy*dy*cs))
                    keep = cmx <= max_range
                else:
                    keep = np.hypot(gt_boxes[:, 0], gt_boxes[:, 1]) <= max_range
                gt_boxes = gt_boxes[keep]; gt_classes = gt_classes[keep]
            names = np.array([self.class_idx_to_name[int(c)] for c in gt_classes])
            ig_names = np.array([self.class_idx_to_name[int(c)] for c in ig_classes])
            gt_annos.append({'boxes': gt_boxes, 'names': names,
                             'ig_boxes': ig_boxes, 'ig_names': ig_names})

        # 클래스별 AP(11-point), 검출률, 분류정확도 집계
        n_cls = len(class_names)
        tp_by_cls = {c: [] for c in class_names}
        score_by_cls = {c: [] for c in class_names}
        ngt_by_cls = {c: 0 for c in class_names}
        total_gt = 0; detected = 0; correct_cls = 0

        for det, gt in zip(det_annos, gt_annos):
            gb = gt['boxes']; gn = gt['names']
            total_gt += len(gb)
            pb = det['boxes_lidar']; pn = det['name']; ps = det['score']
            if max_range > 0 and len(pb):
                pm = np.hypot(pb[:, 0], pb[:, 1]) <= max_range
                pb = pb[pm]; pn = pn[pm]; ps = ps[pm]
            if len(gb) == 0:
                for c in class_names:
                    tp_by_cls and None
                continue
            # 검출/분류용 매칭(클래스 무관 최근접 IoU)
            if len(pb) > 0:
                iou = iou3d_nms_utils.boxes_iou_bev(
                    torch.from_numpy(pb[:, :7]).float().cuda(),
                    torch.from_numpy(gb[:, :7]).float().cuda()).cpu().numpy()  # (P,G)
            else:
                iou = np.zeros((0, len(gb)))
            matched_pred = np.zeros(len(pb), dtype=bool)
            order = np.argsort(-ps) if len(ps) else []
            for g in range(len(gb)):
                col = iou[:, g].copy() if len(pb) else np.zeros(0)
                col[matched_pred] = -1
                gthr = thr_by.get(gn[g], iou_thresh)  # GT 클래스별 임계값
                if len(col) and col.max() >= gthr:
                    p = int(col.argmax()); matched_pred[p] = True
                    detected += 1
                    if pn[p] == gn[g]:
                        correct_cls += 1
            # 클래스별 AP용 TP/FP — 표준 1:1 그리디 매칭
            #  (2026-09-06 수정) 이전에는 예측마다 독립적으로 max IoU>=임계만 봐서
            #  같은 GT에 붙은 중복 박스가 전부 TP로 세어졌고, 재현율이 100%를 넘어
            #  11-point AP의 recall=1.0 지점을 인위적으로 채워 AP가 부풀었다.
            #  이제 점수 내림차순으로 GT를 하나씩만 점유하고, 나머지는 FP로 센다.
            for c in class_names:
                ngt_by_cls[c] += int((gn == c).sum())
            igb = gt.get('ig_boxes', np.zeros((0, 7), np.float32))
            ign = gt.get('ig_names', np.zeros(0, dtype=object))
            iou_ig = None
            if len(pb) and len(igb):
                iou_ig = iou3d_nms_utils.boxes_iou_bev(
                    torch.from_numpy(pb[:, :7]).float().cuda(),
                    torch.from_numpy(igb[:, :7]).float().cuda()).cpu().numpy()  # (P, I)
            for c in class_names:
                pmask = (pn == c) if len(pb) else np.zeros(0, dtype=bool)
                if not pmask.any():
                    continue
                pidx = np.where(pmask)[0]
                pidx = pidx[np.argsort(-ps[pidx])]          # 점수 높은 순
                gi = np.where(gn == c)[0]
                ii = np.where(ign == c)[0] if len(igb) else np.zeros(0, dtype=int)
                thr_c = thr_by.get(c, iou_thresh)
                used = np.zeros(len(gi), dtype=bool)
                for pi in pidx:
                    hit = 0
                    if len(gi):
                        col = iou[pi, gi].copy(); col[used] = -1
                        if col.size and col.max() >= thr_c:
                            used[int(col.argmax())] = True; hit = 1
                    if hit == 0 and len(ii) and iou_ig is not None:
                        # 무시 대상(점 부족 GT)에 맞은 검출은 오탐으로 세지 않고 버린다
                        if iou_ig[pi, ii].max() >= thr_c:
                            continue
                    tp_by_cls[c].append(hit); score_by_cls[c].append(ps[pi])

        def ap_area(tp, score, n_gt):
            """면적(all-point) AP. COCO·Waymo 현행 표준. 구식 11-point의 recall=1.0 계단 없음."""
            if n_gt == 0 or len(tp) == 0:
                return 0.0
            o = np.argsort(-np.array(score))
            tp = np.array(tp)[o]; fp = 1 - tp
            tpc = np.cumsum(tp); fpc = np.cumsum(fp)
            rec = tpc / max(n_gt, 1); prec = tpc / np.maximum(tpc + fpc, 1e-9)
            mrec = np.concatenate(([0.0], rec, [rec[-1]]))
            mpre = np.concatenate(([0.0], prec, [0.0]))
            for i in range(len(mpre) - 2, -1, -1):
                mpre[i] = max(mpre[i], mpre[i + 1])
            idx = np.where(mrec[1:] != mrec[:-1])[0]
            return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

        aps = {c: ap_area(tp_by_cls[c], score_by_cls[c], ngt_by_cls[c]) for c in class_names}
        mAP = float(np.mean(list(aps.values()))) if aps else 0.0
        det_acc = detected / max(total_gt, 1)
        cls_acc = correct_cls / max(detected, 1)

        result = {'mAP': mAP, 'det_acc': det_acc, 'cls_acc': cls_acc, **{f'AP_{c}': v for c, v in aps.items()}}
        s = ('경기대 기준(IoU %.2f 일괄)\n  mAP=%.4f  차량인식=%.4f  차량분류=%.4f\n  ' % (iou_thresh, mAP, det_acc, cls_acc)
             + '  '.join('%s=%.4f' % (c, aps[c]) for c in class_names))
        return s, result

    # ── gt_sampling용 DB 생성 (npz에서 직접) ──
    def create_groundtruth_database(self, split='train', tag=None):
        import torch
        from ...ops.roiaware_pool3d import roiaware_pool3d_utils

        tag = tag or split  # 출력 파일명 태그(기본=split). 4클래스는 tag='subtype_train'로 분리
        db_save = self.data_root / ('pcdet_gt_database_%s' % tag)
        db_info_path = self.data_root / ('pcdet_dbinfos_%s.pkl' % tag)
        db_save.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        target = SPLIT_MAP.get(split, split)
        infos = [i for i in self.infos if i['split'] == target] or self.infos
        for k, info in enumerate(infos):
            if k % 1000 == 0:
                print('gt_database: %d/%d' % (k, len(infos)), flush=True)
            points, gt_boxes, gt_classes = self._load_npz(info)
            names = np.array([self.class_idx_to_name[int(c)] for c in gt_classes])
            if len(gt_boxes) == 0:
                continue
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, :3]), torch.from_numpy(gt_boxes)
            ).numpy()
            stem = Path(info['file']).stem
            for i in range(len(gt_boxes)):
                gt_points = points[point_indices[i] > 0].copy()
                if gt_points.shape[0] == 0:
                    continue
                gt_points[:, :3] -= gt_boxes[i, :3]
                fname = '%s_%s_%d.bin' % (stem, names[i], i)
                fpath = db_save / fname
                gt_points.tofile(str(fpath))
                db_info = {'name': names[i], 'path': str(fpath.relative_to(self.data_root)),
                           'gt_idx': i, 'box3d_lidar': gt_boxes[i],
                           'num_points_in_gt': gt_points.shape[0]}
                all_db_infos.setdefault(names[i], []).append(db_info)

        for c, v in all_db_infos.items():
            print('Database %s: %d' % (c, len(v)))
        with open(db_info_path, 'wb') as f:
            pickle.dump(all_db_infos, f)
        print('saved dbinfos -> %s' % db_info_path)
