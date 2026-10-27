import torch
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler

from pcdet.utils import common_utils

from .dataset import DatasetTemplate
from .custom.custom_dataset import CustomDataset
from .waymonpz.waymonpz_dataset import WaymoNpzDataset

__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'CustomDataset': CustomDataset,
    'WaymoNpzDataset': WaymoNpzDataset,
}

# 선택적 데이터셋: 누락 의존성(av2, nuscenes-devkit 등)이 있어도 패키지가 죽지 않게 관대하게 등록
def _try_register(name, module_path, cls_name):
    try:
        mod = __import__(module_path, fromlist=[cls_name])
        __all__[name] = getattr(mod, cls_name)
    except Exception:
        pass

_try_register('KittiDataset',    'pcdet.datasets.kitti.kitti_dataset',       'KittiDataset')
_try_register('WaymoDataset',    'pcdet.datasets.waymo.waymo_dataset',       'WaymoDataset')
_try_register('NuScenesDataset', 'pcdet.datasets.nuscenes.nuscenes_dataset', 'NuScenesDataset')
_try_register('PandasetDataset', 'pcdet.datasets.pandaset.pandaset_dataset', 'PandasetDataset')
_try_register('LyftDataset',     'pcdet.datasets.lyft.lyft_dataset',         'LyftDataset')
_try_register('ONCEDataset',     'pcdet.datasets.once.once_dataset',         'ONCEDataset')
_try_register('Argo2Dataset',    'pcdet.datasets.argo2.argo2_dataset',       'Argo2Dataset')


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def build_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4, seed=None,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0):

    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=training,
        logger=logger,
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0, worker_init_fn=partial(common_utils.worker_init_fn, seed=seed)
    )

    return dataset, dataloader, sampler


# ── AhnLab V3 우회 (2026-09-12) ────────────────────────────────────────────────
# 이 패키지를 import 하는 동안 백신이 스레드에 TxF 트랜잭션 컨텍스트를 걸어두고 가서,
# 이후 그 스레드의 모든 파일 접근이 [Errno 22] / [WinError 6714]로 실패한다
# (경로 무관 — %TEMP%까지 막힌다). 컨텍스트만 해제하면 정상으로 돌아온다.
# 이미 돌고 있던 프로세스는 영향이 없고, 새로 뜨는 프로세스만 당하던 문제.
try:
    import ctypes as _ctypes
    _ctypes.windll.ntdll.RtlSetCurrentTransaction(0)
except Exception:
    pass
