# -*- coding: utf-8 -*-
"""박스 조정(8방향) 분석 — Waymo LEVEL_1(D) 기준판.

diag_final_combo.py(누적) + diag_single_best.py(단독)를 합치고, 채점 대상 판정을
원본 parquet 기반 LEVEL_1 조회표로 바꾼 것. 그래야 probe가 내는 수치와 같은 잣대가 된다.
  채점 : 원본 점>5 AND not hard      → TP/FN
  무시 : 점1~5 또는 hard             → 매칭돼도 FP로 안 셈(채점에서 빠짐)
  삭제 : 점0 / 조회 실패             → 그 자리 예측은 FP
출력: 조정 없음 · 단독 최적(전 예측 동일 배율) · 8방향 누적, 각각 score 순위와 IoU 순위.
누적·IoU 순위는 GT를 참조하는 오라클이므로 단독값과 반드시 병기한다(교수 지시).
"""
import os, sys, io, json, pickle, time, argparse
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

R = r'C:\waymo_detection'
ap_ = argparse.ArgumentParser()
ap_.add_argument('--pred', required=True)
ap_.add_argument('--infos', default='dataset_infos_v2_rt9_hv82_test.json')
ap_.add_argument('--l1', default='waymo_l1_lookup.pkl')
ap_.add_argument('--thr', type=float, default=0.7)
ap_.add_argument('--range', type=float, default=20.0)
ap_.add_argument('--tag', default='')
ap_.add_argument('--extra_ignore', default='',
                 help='우리 기준(LiDAR 저품질/카메라 가림)을 추가로 **무시**로 돌린다. 쉼표로 CSV 지정: lidar_audit_ours_full.csv,camera_occlusion_ours.csv. ※채점 잣대를 바꾸는 것이므로 보고용 주 수치로 쓰지 말 것')
ap_.add_argument('--ways', type=int, default=8, choices=[3, 8, 11],
                 help='8=한쪽 늘리기(중심 이동) / 3=중심 고정(양쪽 대칭) / 11=둘 다')
ap_.add_argument('--scales', default='-0.20,-0.10,0.10,0.20',
                 help='방향별 배율(0=기준은 자동 포함). 예: 30%%까지 보려면 '
                      '-0.30,-0.20,-0.10,0.10,0.20,0.30')
A = ap_.parse_args()
P_ = lambda p: p if os.path.isabs(p) else os.path.join(R, p)
THR, MR = A.thr, A.range
CLS = [('Vehicle', 0, '소형'), ('Large_Vehicle', 1, '대형'), ('Cyclist', 2, '이륜')]
SIDE8 = ['x+', 'x-', 'y+', 'y-', 'd++', 'd+-', 'd-+', 'd--']
CENTER3 = ['cx', 'cy', 'cd']            # 중심 고정(양쪽 대칭) — 8방향과 합치면 11방향
KOR8 = {'x+': '길이앞', 'x-': '길이뒤', 'y+': '너비좌', 'y-': '너비우',
        'd++': '대각앞좌', 'd+-': '대각앞우', 'd-+': '대각뒤좌', 'd--': '대각뒤우',
        'cx': '중심길이', 'cy': '중심너비', 'cd': '중심대각'}
if A.ways == 11:
    SIDE8 = SIDE8 + CENTER3
elif A.ways == 3:
    SIDE8 = CENTER3
SC = np.array(sorted(set([0.0] + [float(x) for x in A.scales.split(',') if x.strip()])))   # 0%=기준 포함


def corners_batch(B):
    cx, cy, dx, dy, hd = B[:, 0], B[:, 1], B[:, 2], B[:, 3], B[:, 4]
    c, s = np.cos(hd), np.sin(hd)
    lx = np.stack([dx/2, -dx/2, -dx/2, dx/2], 1)
    ly = np.stack([dy/2,  dy/2, -dy/2, -dy/2], 1)
    return np.stack([cx[:, None] + lx*c[:, None] - ly*s[:, None],
                     cy[:, None] + lx*s[:, None] + ly*c[:, None]], -1)


def poly_area(p):
    x, y = p[..., 0], p[..., 1]
    return 0.5*np.abs(np.sum(x*np.roll(y, -1, -1) - y*np.roll(x, -1, -1), -1))


EPS = 1e-12


def iou_batch(g, P):
    N = len(P)
    A_ = corners_batch(P); Bc = corners_batch(g[None])[0]
    ba = Bc; e = np.roll(Bc, -1, 0) - Bc
    d = A_[:, :, None, :] - ba[None, None, :, :]
    inA = np.all(e[None, None, :, 0]*d[..., 1] - e[None, None, :, 1]*d[..., 0] >= -1e-10, 2)
    aa = A_; ea = np.roll(A_, -1, 1) - A_
    d2 = Bc[None, :, None, :] - aa[:, None, :, :]
    inB = np.all(ea[:, None, :, 0]*d2[..., 1] - ea[:, None, :, 1]*d2[..., 0] >= -1e-10, 2)
    den = ea[:, :, None, 0]*e[None, None, :, 1] - ea[:, :, None, 1]*e[None, None, :, 0]
    w = ba[None, None, :, :] - aa[:, :, None, :]
    sd = np.where(np.abs(den) < EPS, 1.0, den)
    t = (w[..., 0]*e[None, None, :, 1] - w[..., 1]*e[None, None, :, 0])/sd
    u = (w[..., 0]*ea[:, :, None, 1] - w[..., 1]*ea[:, :, None, 0])/sd
    okX = (np.abs(den) >= EPS) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    XP = aa[:, :, None, :] + t[..., None]*ea[:, :, None, :]
    pts = np.concatenate([A_, np.broadcast_to(Bc[None], (N, 4, 2)), XP.reshape(N, 16, 2)], 1)
    val = np.concatenate([inA, inB, okX.reshape(N, 16)], 1)
    cnt = val.sum(1)
    cen = np.where(val[..., None], pts, 0.0).sum(1)/np.maximum(cnt, 1)[:, None]
    ang = np.where(val, np.arctan2(pts[..., 1]-cen[:, None, 1], pts[..., 0]-cen[:, None, 0]), 1e9)
    sp = np.take_along_axis(pts, np.argsort(ang, 1)[..., None], 1)
    idx = np.minimum(np.arange(pts.shape[1])[None, :], np.maximum(cnt-1, 0)[:, None])
    sp = np.take_along_axis(sp, idx[..., None], 1)
    inter = np.where(cnt >= 3, poly_area(sp), 0.0)
    ar = P[:, 2]*P[:, 3] + g[2]*g[3] - inter
    return np.where(ar > EPS, inter/np.maximum(ar, EPS), 0.0)


def warp_batch(P, mode, S):
    M, K = len(P), len(S); one = np.ones((M, K))
    cx = P[:, 0:1]*one; cy = P[:, 1:2]*one; dx = P[:, 2:3]*one
    dy = P[:, 3:4]*one; hd = P[:, 4:5]*one
    ux, uy = np.cos(hd), np.sin(hd); vx, vy = -np.sin(hd), np.cos(hd)
    s = S[None, :]*one
    if mode == 'x+':  return np.stack([cx+ux*s*dx/2, cy+uy*s*dx/2, dx*(1+s), dy, hd], -1)
    if mode == 'x-':  return np.stack([cx-ux*s*dx/2, cy-uy*s*dx/2, dx*(1+s), dy, hd], -1)
    if mode == 'y+':  return np.stack([cx+vx*s*dy/2, cy+vy*s*dy/2, dx, dy*(1+s), hd], -1)
    if mode == 'y-':  return np.stack([cx-vx*s*dy/2, cy-vy*s*dy/2, dx, dy*(1+s), hd], -1)
    # 중심 고정 3종(2026-09-21 추가): 중심은 그대로 두고 크기만 키움 = 양쪽으로 s/2씩.
    # 위 8방향은 한쪽으로만 늘려 중심이 s/2만큼 이동하는 방식이라 서로 다른 조작이다.
    if mode == 'cx':  return np.stack([cx, cy, dx*(1+s), dy, hd], -1)          # 길이(가로)만
    if mode == 'cy':  return np.stack([cx, cy, dx, dy*(1+s), hd], -1)          # 너비(세로)만
    if mode == 'cd':  return np.stack([cx, cy, dx*(1+s), dy*(1+s), hd], -1)    # 대각(비율 유지)
    sx = 1.0 if mode[1] == '+' else -1.0; sy = 1.0 if mode[2] == '+' else -1.0
    return np.stack([cx + sx*ux*s*dx/2 + sy*vx*s*dy/2,
                     cy + sx*uy*s*dx/2 + sy*vy*s*dy/2, dx*(1+s), dy*(1+s), hd], -1)


def ap_area(tp, keep, sc, ng):
    tp = np.asarray(tp)[np.asarray(keep)]; sc = np.asarray(sc, float)[np.asarray(keep)]
    if ng == 0 or len(tp) == 0: return 0.0
    o = np.argsort(-sc); tp = tp[o]
    tpc = np.cumsum(tp); fpc = np.cumsum(1-tp)
    rec = tpc/ng; prec = tpc/np.maximum(tpc+fpc, 1e-9)
    mrec = np.concatenate(([0.], rec, [rec[-1]])); mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(len(mpre)-2, -1, -1): mpre[i] = max(mpre[i], mpre[i+1])
    k = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[k+1]-mrec[k])*mpre[k+1]))*100


infos = json.load(open(P_(A.infos), encoding='utf-8'))
va = [o for o in infos if o['split'] == 'validation']
by_id = {q['frame_id']: q for q in pickle.load(open(P_(A.pred), 'rb'))}
LUT = pickle.load(open(P_(A.l1), 'rb'))
b5 = lambda B: np.array([[b[0], b[1], b[3], b[4], b[6]] for b in B], np.float64).reshape(-1, 5)

# ── 추가 무시 집합: (segment, timestamp, cls, round(dist,1)) → 우리 기준으로 걸린 객체
EXTRA = set()
for _f in [x.strip() for x in A.extra_ignore.split(',') if x.strip()]:
    import csv as _csv
    _n0 = len(EXTRA)
    with open(P_(_f), newline='', encoding='utf-8-sig') as _fh:
        for _r in _csv.DictReader(_fh):
            _hit = (_r.get('lidar_low_v1') == '1'
                    or _r.get('decision') == 'HEAVY_OCCLUSION_DROP_CANDIDATE')
            if _hit:
                EXTRA.add((_r['segment'], int(_r['timestamp']), int(_r['cls']),
                           round(float(_r['dist']), 1)))
    print('  추가 무시 +%s개  (%s)' % (format(len(EXTRA) - _n0, ','), _f))
if EXTRA: print('  추가 무시 총 %s개 — ⚠채점 잣대 변경' % format(len(EXTRA), ','))


def l1_flags(fn, G):
    b = os.path.basename(fn)[:-4]
    if b[:3] in ('va_', 'tr_'): b = b[3:]
    p = b.split('_')
    tab = LUT.get(('_'.join(p[:-1]), int(p[-1])))
    fl = np.full(len(G), -1, np.int8)
    if tab is None or not len(G): return fl
    for k in range(len(G)):
        d = np.hypot(tab[:, 0]-G[k, 0], tab[:, 1]-G[k, 1])
        j = int(d.argmin())
        if d[j] > 0.3: continue
        if tab[j, 3] > 5 and tab[j, 2] != 2: fl[k] = 1
        elif tab[j, 3] >= 1: fl[k] = 0
    return fl


COMBOS = [('기준(조정 없음)', None, 0.0)] + [('%s %+d%%' % (KOR8[m], round(s*100)), m, s)
                                        for m in SIDE8 for s in SC if abs(s) > 1e-9]
NCMB = len(COMBOS)
print('예측: %s%s | IoU %.1f · %gm · Waymo LEVEL_1 기준'
      % (os.path.basename(A.pred), (' [%s]' % A.tag if A.tag else ''), THR, MR))

RES = {}; NG = {}
t0 = time.time()
for cname, ci, kor in CLS:
    hits = [[] for _ in range(NCMB)]; ok = [0]*NCMB
    cum_hit, cum_mio, keeps, scos = [], [], [], []
    ng = 0
    for o in va:
        gb = np.array(o['gt_override']['boxes'], np.float32).reshape(-1, 7)
        gc = np.array(o['gt_override']['classes'], np.int64).reshape(-1)
        sel = (gc == ci) & (np.hypot(gb[:, 0], gb[:, 1]) <= MR)
        G = gb[sel]
        fl = l1_flags(o['file'], G)
        if EXTRA:
            _b = os.path.basename(o['file'])[:-4]
            _pp = (_b[3:] if _b[:3] in ('va_', 'tr_') else _b).split('_')
            _sn = '_'.join(_pp[:-1]); _ts = int(_pp[-1])
            for _k in range(len(G)):
                if fl[_k] == 1 and (_sn, _ts, ci, round(float(np.hypot(G[_k, 0], G[_k, 1])), 1)) in EXTRA:
                    fl[_k] = 0                    # 채점 → 무시로 강등
        GS = G[fl == 1]; GI = G[fl == 0]          # 채점 / 무시
        ng += len(GS)
        q = by_id.get(os.path.basename(o['file'])[:-4])
        if q is None or not len(q['boxes_lidar']): continue
        m = (np.asarray(q['name']) == cname)
        pb = np.asarray(q['boxes_lidar'], np.float32)[m]; ps = np.asarray(q['score'], np.float64)[m]
        if len(pb):
            s2 = np.hypot(pb[:, 0], pb[:, 1]) <= MR; pb = pb[s2]; ps = ps[s2]
        M = len(pb)
        if M == 0: continue
        scos += list(ps)
        P5 = b5(pb); S5 = b5(GS); I5 = b5(GI)
        order = np.argsort(-ps)
        # 조합별 IoU (프레임당 한 번 계산)
        W = np.concatenate([P5[:, None, :]] +
                           [warp_batch(P5, md, SC[SC != 0]) for md in SIDE8], 1)   # (M, NCMB, 5)
        Wf = W.reshape(-1, 5)
        AS = np.zeros((M, max(len(S5), 1), NCMB))
        if len(S5):
            AS = np.zeros((M, len(S5), NCMB))
            for j in range(len(S5)):
                AS[:, j, :] = iou_batch(S5[j], Wf).reshape(M, NCMB)
        AI = np.zeros((M, len(I5)))
        for j in range(len(I5)):
            AI[:, j] = iou_batch(I5[j], P5)        # 무시 GT는 원본 박스로 매칭
        # 무시 GT에 붙는 예측은 채점에서 제외
        keep = np.ones(M, bool)
        usedI = np.zeros(len(I5), bool)
        base_best = AS[:, :, 0].max(1) if len(S5) else np.zeros(M)
        for pi in order:
            if len(S5) and base_best[pi] >= THR: continue
            if len(I5):
                col = AI[pi].copy(); col[usedI] = -1
                if col.size and col.max() >= THR:
                    usedI[int(col.argmax())] = True; keep[pi] = False
        keeps += list(keep)
        # 조합별 매칭
        for c in range(NCMB):
            h = np.zeros(M, int)
            if len(S5):
                used = np.zeros(len(S5), bool)
                Ac = AS[:, :, c]
                for pi in order:
                    col = Ac[pi].copy(); col[used] = -1
                    if col.size and col.max() >= THR:
                        used[int(col.argmax())] = True; h[pi] = 1
                ok[c] += int(used.sum())
            hits[c] += list(h)
        # 8방향 누적 (박스별 최적 조합)
        h = np.zeros(M, int); mio = np.zeros(M)
        if len(S5):
            Amax = AS.max(2)                       # (M, NGT) 조합 중 최대 IoU
            used = np.zeros(len(S5), bool)
            for pi in order:
                col = Amax[pi].copy(); col[used] = -1
                best = col.max() if col.size else -1
                mio[pi] = max(best, 0.0)
                if best >= THR:
                    used[int(col.argmax())] = True; h[pi] = 1
        cum_hit += list(h); cum_mio += list(mio)
    NG[cname] = ng
    RES[cname] = dict(
        combos=[(COMBOS[c][0], ap_area(hits[c], keeps, scos, ng), ok[c]) for c in range(NCMB)],
        cum_score=ap_area(cum_hit, keeps, scos, ng),
        cum_iou=ap_area(cum_hit, keeps, cum_mio, ng),
        base_iou=ap_area(hits[0], keeps, [max(v, 0) for v in np.asarray(cum_mio)*0 +
                                          np.asarray(cum_mio)], ng))
    print('  %s 완료 (%.0f초) · 채점 GT %s개' % (kor, time.time()-t0, format(ng, ',')))

print('\n%-26s | %8s %8s %8s | %6s' % ('방식', '소형', '대형', '이륜', 'mAP'))
print('-'*70)
base = [RES[c]['combos'][0][1] for c, _, _ in CLS]
print('%-26s | %8.1f %8.1f %8.1f | %6.1f' % ('조정 없음 (score 순위)', base[0], base[1], base[2], np.mean(base)))
# 단독: 클래스 공통 최적 / 클래스별 최적
per_cls_best, per_cls_lbl = [], []
map_by_combo = []
for c in range(1, NCMB):
    aps = [RES[cn]['combos'][c][1] for cn, _, _ in CLS]
    map_by_combo.append((float(np.mean(aps)), COMBOS[c][0], aps))
map_by_combo.sort(reverse=True)
bm, blbl, baps = map_by_combo[0]
print('%-26s | %8.1f %8.1f %8.1f | %6.1f' % ('단독 최적(공통: %s)' % blbl, baps[0], baps[1], baps[2], bm))
for cn, _, kor in CLS:
    arr = RES[cn]['combos']
    bi = int(np.argmax([a[1] for a in arr]))
    per_cls_best.append(arr[bi][1]); per_cls_lbl.append('%s=%s' % (kor, arr[bi][0]))
print('%-26s | %8.1f %8.1f %8.1f | %6.1f' % ('단독 최적(클래스별)', per_cls_best[0], per_cls_best[1],
                                          per_cls_best[2], np.mean(per_cls_best)))
cs = [RES[c]['cum_score'] for c, _, _ in CLS]
ci_ = [RES[c]['cum_iou'] for c, _, _ in CLS]
print('%-26s | %8.1f %8.1f %8.1f | %6.1f' % ('%d방향 누적 (score 순위)' % len(SIDE8), cs[0], cs[1], cs[2], np.mean(cs)))
print('%-26s | %8.1f %8.1f %8.1f | %6.1f' % ('%d방향 누적 + IoU 순위' % len(SIDE8), ci_[0], ci_[1], ci_[2], np.mean(ci_)))
print('-'*70)
print('클래스별 단독 최적: %s' % ' · '.join(per_cls_lbl))
print('\n조합별 mAP 상위 5')
for m, lbl, aps in map_by_combo[:5]:
    print('  %-14s mAP %.1f  (소형 %.1f 대형 %.1f 이륜 %.1f)' % (lbl, m, aps[0], aps[1], aps[2]))
print('\n※ 누적과 IoU 순위는 GT를 알아야 가능한 오라클(상한선). 실제 적용 가능한 값은 단독 최적.')
print('=== BOXADJUST L1 DONE ===')
