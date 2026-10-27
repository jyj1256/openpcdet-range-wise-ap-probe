"""파라미터화 gt_database 생성: --cfg <model yaml> --tag <db 태그>.
   출력: pcdet_gt_database_<tag>\\ + pcdet_dbinfos_<tag>.pkl (기존 DB 안 건드림).
"""
import sys, io, os, time, importlib, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
from pathlib import Path
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', required=True)
ap.add_argument('--tag', required=True)
args = ap.parse_args()

def imp_retry(name, tries=400, delay=0.1):
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
imp_retry('pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils')
cfg, cfg_from_yaml_file = cfg_mod.cfg, cfg_mod.cfg_from_yaml_file
from pcdet.datasets import WaymoNpzDataset
from pcdet.utils import common_utils

os.chdir(r'C:\OpenPCDet\tools')
logger = common_utils.create_logger()
cfg_from_yaml_file(args.cfg, cfg)
cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST = []  # 생성 시 증강 불필요(chicken-egg 회피)
cfg.DATA_CONFIG['SAMPLED_INTERVAL'] = {'train': 1, 'test': 1}

dataset = WaymoNpzDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                          training=True, root_path=Path(cfg.DATA_CONFIG.DATA_PATH), logger=logger)
print('학습 프레임 수:', len(dataset), flush=True)
t0 = time.time()
dataset.create_groundtruth_database(split='train', tag=args.tag)
print('gt_database 생성 완료, 소요 %.1f분' % ((time.time() - t0) / 60), flush=True)
print('=== GTDB DONE ===')
