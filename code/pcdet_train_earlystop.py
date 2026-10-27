"""OpenPCDet 풀 학습 + 조기중단(early stopping) - 파라미터화.
   사용: python pcdet_train_earlystop.py --cfg <yaml> --out <OUT_DIR> [--patience 5] [--metric mAP]
   pcdet_train_exp.py 기반. 매 에폭 1000-val 평가 후 지표(기본 mAP)의 best 갱신이
   patience 에폭 연속 없으면 중단. best.pth(최고 지표)와 latest.pth(재개용) 모두 저장.
   NUM_EPOCHS는 상한(LR 스케줄 길이) 역할 — 조기중단이 먼저 걸리면 그 전에 멈춤.
   재시작 가능(latest.pth에 best 추적 상태 포함).
"""
import sys, io, os, gc, time, importlib, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', required=True, help='tools 기준 상대경로 yaml')
ap.add_argument('--out', required=True, help='출력 디렉토리 (절대경로)')
ap.add_argument('--patience', type=int, default=5, help='best 갱신 없이 견디는 에폭 수')
ap.add_argument('--metric', default='mAP', help="감시 지표 key (result dict): mAP / AP_Cyclist 등")
ap.add_argument('--min_delta', type=float, default=0.0, help='best 갱신으로 인정할 최소 향상폭')
ap.add_argument('--min_epochs', type=int, default=0, help='이 에폭 수 전에는 조기중단 안 함(최소 학습 보장)')
ap.add_argument('--val_max_range', type=float, default=0, help='val 평가 근거리 크롭 반경(0=전체). 조기중단 지표가 이 범위 mAP가 됨')
ap.add_argument('--val_iou', type=float, default=0.7, help='val 평가·조기중단 IoU 임계(기본 0.7 — 2026-09-11 지시로 0.5 폐기)')
ap.add_argument('--val_waymo_l1', type=int, default=0,
                help='학습 중 val에도 Waymo LEVEL_1 조회표를 쓸지(기본 0=끔). 최종 TEST(probe)는 항상 D 기준이고, best 선택 기준은 실험군끼리 통일하려고 기본 끔')
ap.add_argument('--val_min_points', type=int, default=6,
                help='val 평가 GT 최소 점 개수(실측). P2+ 표준=6 — infos의 min6(box_semstats 기준)와 실측 불일치 보정')
ap.add_argument('--val_no_ignore', action='store_true',
                help='val 평가에서 점 부족 GT를 무시(ignore)가 아니라 완전 제외 — 그 자리 검출을 오탐으로 셈')
ap.add_argument('--topn', type=int, default=3, help='저장·test할 best 체크포인트 수(best1..bestN)')
ap.add_argument('--topn_early', type=int, default=3,
                help='조기(ep<10) 에폭 별도 저장 수(best_e1..). 0이면 조기 저장 안 하고 전체 top-N만 사용')
ap.add_argument('--max_epochs', type=int, default=0, help='>0이면 루프를 이 에폭에서 멈춤(LR 스케줄은 config NUM_EPOCHS 유지)')
args = ap.parse_args()
VAL_MR = float(args.val_max_range)

def imp_retry(name, tries=600, delay=0.1):
    for a in range(tries):
        try:
            return importlib.import_module(name)
        except OSError:
            for k in [m for m in sys.modules if m.startswith('pcdet')]:
                del sys.modules[k]
            time.sleep(delay)
    raise RuntimeError(f'{name} import 실패')

cfg_mod = imp_retry('pcdet.config')
imp_retry('pcdet.models'); imp_retry('pcdet.datasets'); imp_retry('pcdet.utils.common_utils')
imp_retry('pcdet.ops.iou3d_nms.iou3d_nms_utils'); imp_retry('pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils')
cfg, cfg_from_yaml_file = cfg_mod.cfg, cfg_mod.cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

os.chdir(r'C:\OpenPCDet\tools')
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
CKPT = OUT / 'latest.pth'
BEST = OUT / 'best.pth'
logger = common_utils.create_logger(log_file=str(OUT / 'train_log.txt'))
cfg_from_yaml_file(args.cfg, cfg)
logger.info(f'CFG={args.cfg}  OUT={OUT}')

SCHED_EPOCHS = cfg.OPTIMIZATION.NUM_EPOCHS   # LR 스케줄 길이(원본 유지)
EPOCHS = args.max_epochs if args.max_epochs > 0 else SCHED_EPOCHS   # 루프 상한
BS = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
EVAL_EVERY = 1
VAL_LIMIT = 2000
PATIENCE = args.patience
METRIC = args.metric
MIN_DELTA = args.min_delta
MIN_EPOCHS = args.min_epochs
torch.manual_seed(0); np.random.seed(0)

train_set, train_loader, _ = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=BS, dist=False,
    workers=0, logger=logger, training=True)
logger.info(f'train batches/epoch: {len(train_loader)}')

ACCUM = 4
model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.OPTIMIZATION.LR, weight_decay=cfg.OPTIMIZATION.WEIGHT_DECAY)
steps_per_epoch = len(train_loader) // ACCUM
total_steps = SCHED_EPOCHS * steps_per_epoch   # LR 스케줄은 원본 길이 기준
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=cfg.OPTIMIZATION.LR, total_steps=total_steps,
    pct_start=cfg.OPTIMIZATION.PCT_START, div_factor=cfg.OPTIMIZATION.DIV_FACTOR)
logger.info(f'ACCUM={ACCUM} 유효배치={BS*ACCUM} optimizer_steps/epoch={steps_per_epoch} total={total_steps}')
logger.info(f'조기중단: metric={METRIC} patience={PATIENCE} min_delta={MIN_DELTA} min_epochs={MIN_EPOCHS} 상한={EPOCHS}ep')

start_epoch = 0; gstep = 0
best_val = -1.0; best_epoch = -1; no_improve = 0
topk = []  # test는 val mAP 상위 3개로: [{'mAP','epoch'}] 유지, cand_ep{e}.pth 파일 보관
TOPN = args.topn
topk_early = []  # 조기(ep<EARLY_MAX) 상위 2개: cande_ep{e}.pth → best_e1/e2.pth
EARLY_MAX = 10; TOPN_E = args.topn_early   # 조기 에폭 별도 저장 수(0이면 비활성, 전체 top-N만)
if CKPT.exists():
    ck = torch.load(CKPT, map_location='cuda', weights_only=False)
    model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['opt'])
    scheduler.load_state_dict(ck['sched']); start_epoch = ck['epoch']; gstep = ck.get('gstep', 0)
    best_val = ck.get('best_val', -1.0); best_epoch = ck.get('best_epoch', -1); no_improve = ck.get('no_improve', 0)
    topk = ck.get('topk', []); topk_early = ck.get('topk_early', [])
    logger.info(f'재개: epoch {start_epoch}부터 (best {METRIC}={best_val:.4f}@ep{best_epoch}, no_improve={no_improve})')


def fmt(res):
    # 보고 방침: mAP + 차종별 AP만 (차량인식/차량분류/정확도 제외).
    # 클래스는 res의 AP_* 키를 자동 표시 → 3클래스/4클래스(오토바이·자전거) 모두 대응.
    aps = ' '.join('%s=%.4f' % (k[3:], v) for k, v in res.items() if k.startswith('AP_'))
    return 'mAP=%.4f  %s' % (res.get('mAP', 0.0), aps)


def run_eval(limit):
    cfg.DATA_CONFIG['LIMIT_FRAMES'] = {'train': 0, 'test': limit} if limit else {}
    val_set, val_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=BS, dist=False,
        workers=0, logger=logger, training=False)
    cfg.DATA_CONFIG['LIMIT_FRAMES'] = {}
    model.eval(); det_annos = []
    with torch.no_grad():
        for batch in val_loader:
            load_data_to_gpu(batch)
            pred_dicts, _ = model(batch)
            det_annos += val_set.generate_prediction_dicts(batch, pred_dicts, cfg.CLASS_NAMES)
    _, res = val_set.evaluation(det_annos, cfg.CLASS_NAMES, iou_thresh=args.val_iou, max_range=VAL_MR,
                                eval_min_points=args.val_min_points,
                                waymo_l1=bool(args.val_waymo_l1),
                                ignore_low_pts=not args.val_no_ignore)
    try:
        _, res2 = val_set.evaluation(det_annos, cfg.CLASS_NAMES, iou_thresh=0.5, max_range=VAL_MR,
                                     per_class_iou={'Cyclist': 0.5},
                                     eval_min_points=args.val_min_points,
                                     waymo_l1=bool(args.val_waymo_l1),
                                     ignore_low_pts=not args.val_no_ignore)
    except Exception:
        res2 = {}
    model.train()
    # 검증용 예약분을 반납한다. 안 비우면 학습 예약분 위에 얹힌 채 남아,
    # 2스테이지(Voxel R-CNN 등)에서 VRAM이 차고 Windows WDDM이 시스템 RAM으로
    # 스필하며 에폭 시간이 13배로 늘었다(2026-09-21: 40분 → 529분).
    del val_loader, val_set, det_annos
    gc.collect(); torch.cuda.empty_cache()
    return res, res2


logger.info(f'=== 학습 시작(조기중단): 상한 {EPOCHS}ep, {len(train_loader)} batch/ep, total {total_steps} steps ===')
stopped_early = False
for epoch in range(start_epoch, EPOCHS):
    model.train(); t0 = time.time(); run_loss = 0.0
    gc.collect(); torch.cuda.empty_cache()   # 에폭 시작은 항상 깨끗한 할당자로
    optimizer.zero_grad()
    for it, batch in enumerate(train_loader):
        load_data_to_gpu(batch)
        ret, tb, _ = model(batch)
        loss = ret['loss']
        (loss / ACCUM).backward()
        run_loss += float(loss)
        if (it + 1) % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.OPTIMIZATION.GRAD_NORM_CLIP)
            optimizer.step(); optimizer.zero_grad()
            if gstep < total_steps - 1:
                scheduler.step()
            gstep += 1
        if it % 200 == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f'ep{epoch} it{it}/{len(train_loader)} loss={float(loss):.3f} '
                        f'hm={float(tb.get("hm_loss_head_0",0)):.2f} loc={float(tb.get("loc_loss_head_0",0)):.2f} lr={lr:.2e}')
    logger.info(f'=== epoch {epoch} 완료 avg_loss={run_loss/len(train_loader):.3f} ({(time.time()-t0)/60:.1f}분) '
                f'| VRAM 예약 {torch.cuda.max_memory_reserved()/2**30:.1f}GB ===')
    torch.cuda.reset_peak_memory_stats()

    # 평가 + 조기중단 판정
    cur = None
    try:
        res, _ = run_eval(VAL_LIMIT)
        logger.info(f'[val@ep{epoch}] {fmt(res)}')
        cur = float(res.get(METRIC, res.get('mAP', 0.0)))
    except Exception as e:
        logger.info(f'eval 실패: {e}')

    if cur is not None and cur > best_val + MIN_DELTA:
        best_val = cur; best_epoch = epoch; no_improve = 0
        torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(),
                    'sched': scheduler.state_dict(), 'epoch': epoch + 1, 'gstep': gstep,
                    'best_val': best_val, 'best_epoch': best_epoch, 'no_improve': no_improve}, BEST)
        logger.info(f'  >> best 갱신 {METRIC}={best_val:.4f} @ep{epoch} (best.pth 저장)')
    elif cur is not None:
        no_improve += 1
        logger.info(f'  >> 개선 없음 {no_improve}/{PATIENCE} (현재 {METRIC}={cur:.4f}, best={best_val:.4f}@ep{best_epoch})')

    # val mAP 상위 3개 체크포인트 유지(test를 top-3로 돌리기 위함)
    if cur is not None and (len(topk) < TOPN or cur > min(t['mAP'] for t in topk)):
        torch.save({'model': model.state_dict(), 'epoch': epoch + 1, 'mAP': cur}, OUT / f'cand_ep{epoch}.pth')
        topk.append({'mAP': cur, 'epoch': epoch})
        topk.sort(key=lambda t: -t['mAP'])
        for t in topk[TOPN:]:  # 순위 밖 파일 제거
            fp = OUT / f"cand_ep{t['epoch']}.pth"
            if fp.exists():
                try: os.remove(fp)
                except Exception: pass
        topk = topk[:TOPN]

    # 조기(ep<EARLY_MAX) val mAP 상위 2개 유지(조기 vs 중반 test 비교용)
    if TOPN_E > 0 and cur is not None and epoch < EARLY_MAX and (len(topk_early) < TOPN_E or cur > min(t['mAP'] for t in topk_early)):
        torch.save({'model': model.state_dict(), 'epoch': epoch + 1, 'mAP': cur}, OUT / f'cande_ep{epoch}.pth')
        topk_early.append({'mAP': cur, 'epoch': epoch})
        topk_early.sort(key=lambda t: -t['mAP'])
        for t in topk_early[TOPN_E:]:
            fp = OUT / f"cande_ep{t['epoch']}.pth"
            if fp.exists():
                try: os.remove(fp)
                except Exception: pass
        topk_early = topk_early[:TOPN_E]

    # latest 저장(재개용, best 추적 상태 포함)
    torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(),
                'sched': scheduler.state_dict(), 'epoch': epoch + 1, 'gstep': gstep,
                'best_val': best_val, 'best_epoch': best_epoch, 'no_improve': no_improve,
                'topk': topk, 'topk_early': topk_early}, CKPT)

    if no_improve >= PATIENCE:
        if (epoch + 1) < MIN_EPOCHS:
            logger.info(f'  >> (개선 {no_improve}회지만 min_epochs {MIN_EPOCHS} 전이라 계속)')
        else:
            logger.info(f'=== 조기중단: {METRIC}가 {PATIENCE}에폭 연속 개선 없음(≥min_epochs {MIN_EPOCHS}). best={best_val:.4f}@ep{best_epoch} ===')
            stopped_early = True
            break

if not stopped_early:
    logger.info(f'=== 상한 {EPOCHS}ep 도달 종료. best={best_val:.4f}@ep{best_epoch} ===')

# val mAP 상위 3개 → best1/best2/best3.pth 확정 (test는 이 3개로 돌림)
logger.info('=== top-3 체크포인트 확정 ===')
for rank, t in enumerate(topk, 1):
    src = OUT / f"cand_ep{t['epoch']}.pth"; dst = OUT / f'best{rank}.pth'
    if src.exists():
        if dst.exists(): os.remove(dst)
        os.replace(src, dst)
        logger.info(f'  top{rank}: {METRIC}={t["mAP"]:.4f} @ep{t["epoch"]} -> best{rank}.pth')
# 조기(ep<EARLY_MAX) 상위 2개 → best_e1/best_e2.pth 확정
for rank, t in enumerate(topk_early, 1):
    src = OUT / f"cande_ep{t['epoch']}.pth"; dst = OUT / f'best_e{rank}.pth'
    if src.exists():
        if dst.exists(): os.remove(dst)
        os.replace(src, dst)
        logger.info(f'  early{rank}: {METRIC}={t["mAP"]:.4f} @ep{t["epoch"]} -> best_e{rank}.pth')

# best 체크포인트로 전체 val 최종 평가
logger.info('=== best.pth 전체 val 평가 ===')
if BEST.exists():
    ck = torch.load(BEST, map_location='cuda', weights_only=False)
    model.load_state_dict(ck['model'])
res, res2 = run_eval(None)
logger.info(f'[best@ep{best_epoch} 기존잣대 IoU0.7 전체] {fmt(res)}')
logger.info(f'[best@ep{best_epoch} 공식잣대 Cyc0.5 min5] {fmt(res2)}')
logger.info('=== TRAIN DONE ===')
