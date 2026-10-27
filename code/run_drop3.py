# -*- coding: utf-8 -*-
"""drop3 = V0(CSV 2기준) 중 **이륜은 제외**하고 소형·대형에만 적용. 학습 → TEST → 8·11방향까지 한 큐로.

이번엔 8방향을 처음부터 포함한다(이전 drop 실험에서 빠뜨려 나중에 따로 돌렸다).
TEST가 `--save_tag drop3`로 예측을 남기므로 8방향은 추론 없이 CPU 분석만 돈다.

비교 축(전부 원본 test 5,923f · D기준 · IoU 0.7 · 20m):
  ① voxel_rcnn_lbl   원본 라벨              82.4 (소 96.9 / 대 90.8 / 이 59.3)
  ② voxel_rcnn_drop  V0 전 클래스           82.8 (소 97.1 / 대 92.7 / 이 58.5)
  ③ voxel_rcnn_drop2 V0+점0 등             82.1 (소 97.1 / 대 93.9 / 이 55.2)
  ④ voxel_rcnn_drop3 V0 −이륜(3,055개)      ?  ← 이 실험
근거: 삭제는 대형에 이롭고(+1.9 → +3.1) 이륜에 해롭다(-0.8 → -4.1). 이륜만 빼면 둘 다 취한다.
  ② voxel_rcnn_drop  CSV 2기준(-1.04%)      82.8 (소 97.1 / 대 92.7 / 이 58.5)
  ③ voxel_rcnn_drop3 +자체 3기준            ?
모델·LR·배치·스케줄·평가가 ①②와 완전히 동일해 차이가 라벨 하나로 좁혀진다.
"""
import os, sys, io, time, subprocess, datetime
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
PY = sys.executable; VX = r'C:\waymo_voxelnext'; DET = r'C:\waymo_detection'
TRAIN = os.path.join(VX, 'pcdet_train_earlystop.py')
TOP3 = os.path.join(VX, 'run_probe_top3.py')
SMOKE = os.path.join(VX, 'smoke_twostage.py')
GTDB = os.path.join(VX, 'pcdet_gen_gtdb_arg.py')
DIAG = os.path.join(VX, 'diag_boxadjust_l1.py')
WLOG = os.path.join(DET, 'drop3_watch.log')
CFG = 'cfgs/waymonpz_models/voxel_rcnn_drop3.yaml'
OUT = os.path.join(DET, 'pcdet_output', 'voxel_rcnn_drop3')
TAG = 'rt9_hv82_drop3_pt6_train'
TESTJ = 'dataset_infos_v2_rt9_hv82_test.json'      # 원본 test(①② 와 같은 잣대)


def log(m):
    l = '%s  %s' % (datetime.datetime.now().strftime('%m-%d %H:%M:%S'), m)
    open(WLOG, 'a', encoding='utf-8').write(l + '\n'); print(l)


def gpu():
    try:
        return int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                           encoding='utf-8').strip().split('\n')[0])
    except Exception:
        return 99999


def run(step, cmd, lf, append=False):
    log('%s 시작' % step)
    with open(lf, 'a' if append else 'w', encoding='utf-8') as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=r'C:\OpenPCDet\tools').returncode
    log('%s 종료 rc=%d' % (step, rc)); return rc


os.makedirs(OUT, exist_ok=True)
dl = time.time() + 900 * 3600
s = 0
while time.time() < dl and s < 3:
    s = s + 1 if gpu() < 2500 else 0
    time.sleep(20)
log('GPU 확보')

if run('gt_db(%s)' % TAG, [PY, GTDB, '--cfg', CFG, '--tag', TAG],
       os.path.join(DET, 'drop3_gtdb.log')) != 0:
    log('gt_db 실패 — 중단'); sys.exit(1)
if run('스모크', [PY, SMOKE, '--cfg', CFG], os.path.join(DET, 'drop3_smoke.txt')) != 0:
    log('스모크 실패 — 중단'); sys.exit(1)
if run('학습(강증강, val 20m@IoU0.7, 조기중단 min15/p5, top-5)',
       [PY, TRAIN, '--cfg', CFG, '--out', OUT, '--patience', '5', '--min_epochs', '15',
        '--metric', 'mAP', '--val_max_range', '20', '--val_iou', '0.7',
        '--val_min_points', '6', '--topn', '5', '--topn_early', '0'],
       os.path.join(DET, 'drop3_train.err.txt'), append=True) != 0:
    log('학습 실패 — 중단'); sys.exit(1)
run('top-5 TEST (원본 test, D 기준, IoU 0.7, 전체/30/20/10m)',
    [PY, TOP3, '--cfg', CFG, '--out', OUT, '--infos', TESTJ,
     '--ranges', '0,30,20,10', '--ious', '0.7', '--eval_min_points', '6',
     '--save_tag', 'drop3', '--score_thresh', '0.01', '--eval_score_thresh', '0.1'],
    os.path.join(DET, 'drop3_probe_TOP3.txt'))

# ── 8·11방향 (CPU, 추론 재사용)
res = os.path.join(DET, 'drop3_boxadjust_result.txt')
for ck in ('best1', 'best2'):
    pk = 'preds_drop3_%s.pkl' % ck
    if not os.path.exists(os.path.join(DET, pk)):
        log('%s 없음 — 건너뜀' % pk); continue
    for w in ('8', '11'):
        for rng in ('30', '20', '10'):
            log('[%s] %s방향 %sm' % (ck, w, rng))
            with open(res, 'a', encoding='utf-8') as f:
                subprocess.run([PY, DIAG, '--pred', pk, '--infos', TESTJ,
                                '--thr', '0.7', '--range', rng, '--ways', w,
                                '--tag', 'drop3 %s %sway' % (ck, w)],
                               stdout=f, stderr=subprocess.STDOUT, cwd=r'C:\OpenPCDet\tools')
log('=== DROP3 DONE ===')
