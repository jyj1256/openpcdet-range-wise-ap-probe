# -*- coding: utf-8 -*-
"""OpenPCDet inference and near-range AP evaluation."""
import argparse, copy, importlib, io, os, pickle, sys, time
from pathlib import Path

def arguments(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--cfg',required=True); p.add_argument('--ckpt',required=True)
    p.add_argument('--infos',default=''); p.add_argument('--ranges',default='0,30,20,10')
    p.add_argument('--ious',default='0.7'); p.add_argument('--eval_min_points',type=int,default=6)
    p.add_argument('--no_ignore',action='store_true')
    p.add_argument('--score_thresh',type=float,default=0.01)
    p.add_argument('--eval_score_thresh',type=float,default=0.1)
    p.add_argument('--save_preds',default='')
    p.add_argument('--result_file',default='',help='evaluation table output path')
    return p.parse_args(argv)

def numbers(value, name):
    try: out=[float(x.strip()) for x in value.split(',') if x.strip()]
    except ValueError as e: raise ValueError('%s: 쉼표 구분 숫자가 필요합니다'%name) from e
    if not out: raise ValueError('%s 값이 비었습니다'%name)
    return out

def imp(name, tries=400):
    last=None
    for _ in range(tries):
        try: return importlib.import_module(name)
        except OSError as e:
            last=e
            for key in [k for k in sys.modules if k.startswith('pcdet')]: del sys.modules[key]
            time.sleep(.1)
    raise RuntimeError('%s import 실패'%name) from last

def load_checkpoint(model, path, logger):
    """Accept stock OpenPCDet and bundled early-stop checkpoints."""
    import torch
    ck=torch.load(path,map_location='cpu',weights_only=False)
    state=ck.get('model_state',ck.get('model')) if isinstance(ck,dict) else None
    if state is None:
        model.load_params_from_file(filename=path,logger=logger,to_cpu=True); return
    state={k[7:] if k.startswith('module.') else k:v for k,v in state.items()}
    bad=model.load_state_dict(state,strict=False)
    if bad.missing_keys: logger.warning('checkpoint 누락 key %d개',len(bad.missing_keys))
    if bad.unexpected_keys: logger.warning('checkpoint 초과 key %d개',len(bad.unexpected_keys))

def filtered_copy(annos, threshold):
    """Filter an evaluation copy; keep saved predictions raw."""
    import numpy as np
    out=copy.deepcopy(annos); before=after=0
    for anno in out:
        score=np.asarray(anno.get('score',[])); keep=score>=threshold
        before+=len(score); after+=int(keep.sum())
        for key,value in list(anno.items()):
            if isinstance(value,np.ndarray) and value.ndim and len(value)==len(score):
                anno[key]=value[keep]
    return out,before,after

def main(argv=None):
    a=arguments(argv); ranges=numbers(a.ranges,'--ranges'); ious=numbers(a.ious,'--ious')
    if any(x<0 for x in ranges): raise ValueError('--ranges는 0 이상이어야 합니다')
    if any(not 0<x<=1 for x in ious): raise ValueError('--ious는 0 초과 1 이하여야 합니다')
    if not 0<=a.score_thresh<=1 or not 0<=a.eval_score_thresh<=1:
        raise ValueError('score threshold는 0 이상 1 이하여야 합니다')
    import torch
    cm=imp('pcdet.config'); imp('pcdet.models'); imp('pcdet.datasets')
    common=imp('pcdet.utils.common_utils')
    from pcdet.datasets import build_dataloader
    from pcdet.models import build_network,load_data_to_gpu
    cfg=cm.cfg; cm.cfg_from_yaml_file(a.cfg,cfg); logger=common.create_logger()
    if a.infos:
        cfg.DATA_CONFIG.INFOS_JSON=a.infos; logger.info('INFOS_JSON 오버라이드 → %s',a.infos)
    cfg.MODEL.POST_PROCESSING.SCORE_THRESH=a.score_thresh
    head=getattr(cfg.MODEL,'DENSE_HEAD',None)
    if head is not None and getattr(head,'POST_PROCESSING',None) is not None:
        head.POST_PROCESSING.SCORE_THRESH=a.score_thresh
    logger.info('SCORE_THRESH 덮어씀 -> %g (채점은 --eval_score_thresh로 되돌림)',a.score_thresh)
    dataset,loader,_=build_dataloader(dataset_cfg=cfg.DATA_CONFIG,class_names=cfg.CLASS_NAMES,
        batch_size=1,dist=False,workers=0,logger=logger,training=False)
    model=build_network(model_cfg=cfg.MODEL,num_class=len(cfg.CLASS_NAMES),dataset=dataset)
    load_checkpoint(model,a.ckpt,logger); model.cuda().eval()
    print('예측 생성 중...',flush=True); raw=[]
    with torch.no_grad():
        for batch in loader:
            load_data_to_gpu(batch); pred,_=model(batch)
            raw.extend(dataset.generate_prediction_dicts(batch,pred,cfg.CLASS_NAMES))
    print('예측 완료: %d 프레임'%len(raw),flush=True)
    if a.save_preds:
        dest=Path(a.save_preds); dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('wb') as f: pickle.dump(raw,f,protocol=pickle.HIGHEST_PROTOCOL)
        print('예측 저장 %d프레임 -> %s'%(len(raw),dest),flush=True)
    evaluated,before,after=filtered_copy(raw,a.eval_score_thresh)
    print('채점 문턱 %g 적용: 예측 %d -> %d'%(a.eval_score_thresh,before,after),flush=True)
    rows=[]
    for radius in ranges:
        for iou in ious:
            _,result=dataset.evaluation(evaluated,cfg.CLASS_NAMES,iou_thresh=iou,
                max_range=radius,eval_min_points=a.eval_min_points,
                ignore_low_pts=not a.no_ignore)
            rows.append((radius,iou,result))
    header='반경(m) | IoU | mAP | '+' | '.join(cfg.CLASS_NAMES)
    lines=['',header,'-'*70]
    for radius,iou,result in rows:
        values=[100*float(result.get('mAP',0))]
        values += [100*float(result.get('AP_'+name,0)) for name in cfg.CLASS_NAMES]
        label='전체' if radius==0 else '%gm'%radius
        lines.append('%6s | %.1f | %5.1f | %s'%(label,iou,values[0],
            ' | '.join('%5.1f'%v for v in values[1:])))
    report='\n'.join(lines)+'\n'; print(report,end='',flush=True)
    default_result=(Path(a.save_preds).parent if a.save_preds else Path(a.ckpt).parent)/'nearrange_probe_result.txt'
    dest=Path(a.result_file or os.environ.get('PCDET_PROBE_RESULT','') or default_result)
    dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text(report,encoding='utf-8')
    print('\n저장: %s'%dest); print('=== PROBE DONE ===')

if __name__=='__main__':
    if hasattr(sys.stdout,'buffer'):
        sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',
            errors='replace',line_buffering=True)
    main()
