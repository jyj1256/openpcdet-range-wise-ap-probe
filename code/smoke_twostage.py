# -*- coding: utf-8 -*-
"""Two-stage 스모크: 모델 빌드 + train 1배치 forward/backward. RoI head CUDA op 동작 확인."""
import sys,io,os,time,importlib,argparse
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace',line_buffering=True)
def imp_retry(name,tries=600,delay=0.1):
    for _ in range(tries):
        try: return importlib.import_module(name)
        except OSError:
            for k in [m for m in sys.modules if m.startswith('pcdet')]: del sys.modules[k]
            time.sleep(delay)
    raise RuntimeError(name)
ap=argparse.ArgumentParser(); ap.add_argument('--cfg',required=True); a=ap.parse_args()
cfg_mod=imp_retry('pcdet.config'); imp_retry('pcdet.models'); imp_retry('pcdet.datasets'); imp_retry('pcdet.utils.common_utils')
cfg,cfg_from_yaml_file=cfg_mod.cfg,cfg_mod.cfg_from_yaml_file
import torch
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
os.chdir(r'C:\OpenPCDet\tools')
logger=common_utils.create_logger()
cfg_from_yaml_file(a.cfg,cfg)
CLS=list(cfg.CLASS_NAMES)
ds,dl,_=build_dataloader(dataset_cfg=cfg.DATA_CONFIG,class_names=CLS,batch_size=2,dist=False,workers=0,logger=logger,training=True)
model=build_network(model_cfg=cfg.MODEL,num_class=len(CLS),dataset=ds).cuda()
model.train()
b=next(iter(dl)); load_data_to_gpu(b)
ret,tb,disp=model(b)
loss=ret['loss'] if isinstance(ret,dict) else ret
loss.backward()
print('빌드+forward+backward OK. loss=%.3f'%float(loss))
print('tb keys:',list(tb.keys())[:8] if hasattr(tb,'keys') else '-')
print('=== TWOSTAGE SMOKE OK ===')
