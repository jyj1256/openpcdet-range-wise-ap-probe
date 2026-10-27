# -*- coding: utf-8 -*-
"""waymo_level1_drop_object_ids.csv를 우리 infos(gt_override)에 적용한다.

CSV는 TFRecord의 `laser_labels.id` 기준 삭제 목록인데, 우리 파이프라인은 TFRecord가 아니라
parquet(H:)에서 만든 frames_v2 + gt_override를 쓰고 **object_id를 보존하지 않는다.**
그래서 원본 `lidar_box` parquet에서 (세그먼트, 타임스탬프)별 박스 중심 ↔ object_id 표를 만들고,
gt_override 박스를 중심 최근접으로 매칭해 삭제 대상을 걸러낸다.
(save_cyc_fail_valid_all.py에서 검증된 방식 — 우리 박스는 parquet 박스를 그대로 옮긴 것이라 정확히 일치한다.)

키: CSV의 frame_index 대신 **timestamp_micros**를 쓴다. README는 TFRecord를 순서대로 읽을 때
frame_index를 쓰라는 것이고, 우리는 파일명에 타임스탬프가 들어 있어 (세그먼트, 타임스탬프,
object_id)로 유일하게 지목된다.

기본 적용 범위: **학습 데이터(train'/valid')만.** CSV 1절이 "학습에서 제외할 객체의 목록"이라고
명시하고, test(=Waymo validation 폴더)를 건드리면 기존 4종 모델 비교표와 잣대가 어긋나기 때문.
--apply_test를 주면 test에도 적용한다(그 경우 과거 수치와 직접 비교 불가).
"""
import os, sys, io, csv, json, time, pickle, argparse, collections
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
import pandas as pd

R = r'C:\waymo_detection'
RAW = {'tr': r"H:/내 드라이브/waymo/training", 'va': r"H:/내 드라이브/waymo/validation"}
BX = lambda a: '[LiDARBoxComponent].box.%s' % a
TS = 'key.frame_timestamp_micros'
OID = 'key.laser_object_id'

ap = argparse.ArgumentParser()
ap.add_argument('--csv', default='waymo_level1_drop_object_ids.csv')
ap.add_argument('--match', type=float, default=0.5, help='박스 중심 매칭 허용 거리(m)')
ap.add_argument('--apply_test', action='store_true', help='test(Waymo validation)에도 적용')
ap.add_argument('--test_only', action='store_true',
                help='test만 생성(이미 만든 tune을 건드리지 않는다 — 학습이 읽는 중일 때 안전)')
ap.add_argument('--suffix', default='drop')
ap.add_argument('--skip_classes', default='',
                help='이 클래스는 삭제하지 않는다(쉼표, 0=소형 1=대형 2=이륜). '
                     'drop/drop2에서 이륜 삭제가 각각 -0.8/-4.1pp 손해였다')
A = ap.parse_args()
SKIP = set(int(x) for x in A.skip_classes.split(',') if x.strip())

# ── 1) CSV → (세그먼트, 타임스탬프) -> {object_id}
drop = collections.defaultdict(set)
n_row = 0
with open(os.path.join(R, A.csv), newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        seg = os.path.basename(row['tfrecord_file'])
        if seg.startswith('segment-'): seg = seg[len('segment-'):]
        for suf in ('_with_camera_labels.tfrecord', '.tfrecord'):
            if seg.endswith(suf): seg = seg[:-len(suf)]; break
        drop[(seg, int(row['timestamp_micros']))].add(row['object_id'])
        n_row += 1
print('CSV %s행 → (세그,타임스탬프) 키 %s개' % (format(n_row, ','), format(len(drop), ',')))

# ── 2) 적용
def read_pq(path, tries=3, **kw):
    for t in range(tries):
        try:
            return pd.read_parquet(path, **kw)
        except Exception as e:
            if t == tries - 1: raise
            print('  재시도 %d (%s): %s' % (t+1, os.path.basename(path), e)); time.sleep(5)


JOBS = [] if A.test_only else [
    ('dataset_infos_v2_rt9_hv82_tune.json', 'dataset_infos_v2_rt9_hv82_%s_tune.json' % A.suffix,
     {'training', 'validation'})]
if A.apply_test or A.test_only:
    JOBS.append(('dataset_infos_v2_rt9_hv82_test.json', 'dataset_infos_v2_rt9_hv82_%s_test.json' % A.suffix,
                 {'validation'}))

NM = {0: '소형', 1: '대형', 2: '이륜'}
for infile, outfile, splits in JOBS:
    d = json.load(open(os.path.join(R, infile), encoding='utf-8'))
    bysn = collections.defaultdict(list)
    for o in d:
        if o['split'] not in splits: continue
        stem = os.path.basename(o['file'])[:-4]
        pf, rest = stem[:2], stem[3:]
        bysn[(pf, rest[:rest.rfind('_')])].append(o)
    print('\n[%s] 세그먼트 %d개 처리' % (infile, len(bysn)))
    before = collections.Counter(); after = collections.Counter()
    ndrop = collections.Counter(); nomatch = 0; noseg = 0
    for i, ((pf, sn), items) in enumerate(sorted(bysn.items()), 1):
        need = {(sn, int(os.path.basename(o['file'])[:-4].rsplit('_', 1)[1])) for o in items}
        if not any(k in drop for k in need):          # 이 세그먼트에 삭제 대상 없음
            for o in items:
                for c in o['gt_override']['classes']: before[c] += 1; after[c] += 1
            continue
        try:
            lb = read_pq(os.path.join(RAW[pf], 'lidar_box', sn + '.parquet'),
                         columns=[TS, OID, BX('center.x'), BX('center.y')])
        except Exception as e:
            print('  %s: lidar_box 실패 %s — 이 세그먼트는 원본 유지' % (sn[:22], e)); noseg += 1
            for o in items:
                for c in o['gt_override']['classes']: before[c] += 1; after[c] += 1
            continue
        g = {t: sub for t, sub in lb.groupby(TS)}
        for o in items:
            ts = int(os.path.basename(o['file'])[:-4].rsplit('_', 1)[1])
            ids = drop.get((sn, ts))
            bxs = np.array(o['gt_override']['boxes'], np.float32).reshape(-1, 7)
            cls = list(o['gt_override']['classes'])
            for c in cls: before[c] += 1
            if not ids or ts not in g:
                for c in cls: after[c] += 1
                continue
            sub = g[ts]
            cxs = sub[BX('center.x')].values; cys = sub[BX('center.y')].values
            oids = sub[OID].astype(str).values
            keep_b = []; keep_c = []
            for k in range(len(bxs)):
                dd = np.hypot(cxs - bxs[k, 0], cys - bxs[k, 1])
                j = int(dd.argmin())
                if dd[j] > A.match:
                    nomatch += 1; keep_b.append(bxs[k].tolist()); keep_c.append(cls[k]); continue
                if oids[j] in ids and cls[k] not in SKIP:
                    ndrop[cls[k]] += 1; continue          # 삭제
                keep_b.append(bxs[k].tolist()); keep_c.append(cls[k])
            o['gt_override']['boxes'] = keep_b
            o['gt_override']['classes'] = keep_c
            o['num_boxes'] = len(keep_c)
            for c in keep_c: after[c] += 1
        if i % 100 == 0: print('  %d/%d' % (i, len(bysn)))
    json.dump(d, open(os.path.join(R, outfile), 'w', encoding='utf-8'), ensure_ascii=False)
    tot_b = sum(before.values()); tot_a = sum(after.values())
    print('  저장 -> %s' % outfile)
    print('  박스 %s → %s (삭제 %s, %.2f%%)'
          % (format(tot_b, ','), format(tot_a, ','), format(tot_b-tot_a, ','),
             100.0*(tot_b-tot_a)/max(tot_b, 1)))
    for c in sorted(before):
        print('    %-4s %8s → %8s  (삭제 %s, %.2f%%)'
              % (NM.get(c, c), format(before[c], ','), format(after[c], ','),
                 format(ndrop[c], ','), 100.0*ndrop[c]/max(before[c], 1)))
    if nomatch: print('  ⚠ parquet에서 못 찾은 박스 %s개(원본 유지)' % format(nomatch, ','))
    if noseg: print('  ⚠ parquet 읽기 실패 세그먼트 %d개(원본 유지)' % noseg)
print('\n=== DROP INFOS DONE ===')
