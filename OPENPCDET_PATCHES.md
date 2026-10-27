# OpenPCDet Windows 빌드 메모 (Linux 환경이면 무시해도 된다)

`C:\OpenPCDet` (pcdet 0.6.0+233f849) 를 VS BuildTools 18 / MSVC 19.50 / CUDA 12.6 / torch 2.11 /
Python 3.10(`data_env`) 에서 빌드하며 필요했던 패치 모음. 툴체인이 최신이라 생긴 문제들이다.

## 빌드 명령 (PowerShell에서 cmd 경유)

```powershell
$vc="C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
$py="C:\Users\DM-MINT\miniconda3\envs\data_env\python.exe"
cmd /c "`"$vc`" && set DISTUTILS_USE_SDK=1 && set NVCC_PREPEND_FLAGS=-allow-unsupported-compiler && cd /d C:\OpenPCDet && `"$py`" setup.py develop"
```

## 적용한 패치

1. `torch/utils/cpp_extension.py:46` — `SUBPROCESS_DECODE_ARGS` 를 `('oem',)` → `('utf-8','replace')`.
   한국어 OEM(cp949)이 MSVC UTF-8 배너를 디코딩 못 해 터진다. **site-packages라 torch 재설치 시 재적용 필요.**
2. `iou3d_nms.cpp` — VLA `unsigned long long remv_cpu[col_blocks]` → `std::vector`,
   `keep.data<long>()` → `data_ptr<int64_t>()` (Windows long=32bit vs torch int64).
3. `iou3d_nms_kernel.cu` — `const float EPS` → `#define EPS 1e-8` (device 코드에서 미정의),
   `check_rect_cross` 에 `inline` 추가 (LNK2005).
4. `ingroup_inds.cpp` / `ingroup_inds_kernel.cu` — `long` → `int64_t` 전체 통일.
5. `setup.py` — `install_requires` 에서 `SharedArray` 제거 (Linux 전용, mman.h 없음).
6. eager import 우회 — `common_utils.py` / `database_sampler.py` / `waymo_dataset.py` 의
   `import SharedArray` 를 try/except, `mppnet_utils.py` 의 `from os import getgrouplist` 삭제,
   `datasets/__init__.py` 를 선택적 데이터셋 try/except 등록으로 변경 (이 묶음에 포함된 버전).
7. `pcdet/config.py` — yaml `open()` 2곳에 `encoding='utf-8'` (한국어 cp949 디코드 에러 방지).

## 백신(AhnLab V3) 때문에 생기는 import 실패

갓 빌드된 pcdet `.pyd` 를 V3가 실시간 스캔하면서 import 중 `OSError WinError 6714`(TxF) 가 난다.
pcdet을 import하는 프로세스만 영향받는다. 우회법:

- 스크립트를 `C:\OpenPCDet` **밖**(예: `C:\waymo_voxelnext`)에 두고 중립 CWD에서 실행
- `import torch` 를 **먼저** 하고, 그 다음 pcdet import를 OSError 시 재시도
  (`sys.modules`의 pcdet 항목 purge 후 0.1초 sleep, 최대 400회 — 이 묶음의 스크립트들에 이미 들어있다)
- 근본 해결은 V3 검사 제외에 `C:\OpenPCDet` 와 conda env `data_env` 추가
