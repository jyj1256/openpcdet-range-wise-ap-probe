# OpenPCDet Range-wise AP Probe

OpenPCDet 체크포인트를 한 번 추론한 뒤 같은 예측으로 거리 구간과 IoU 기준을 바꿔
클래스별 AP와 mAP를 비교하는 연구용 평가 도구입니다.

## 기능

- OpenPCDet YAML 설정 및 체크포인트 로딩
- 별도 infos 파일로 평가 데이터 교체
- 전체/30 m/20 m/10 m 등 복수 거리 평가
- 복수 IoU, 최소 GT 포인트 수, ignore 정책 지정
- 추론 문턱과 평가 문턱 분리
- 낮은 추론 문턱의 원시 예측 PKL 저장
- best1.pth ... bestN.pth, best_e*.pth, best.pth 폴백 지원

## 재현 범위

이 저장소는 평가와 예측 저장 부분을 공유합니다. 학습 코드, 데이터셋, 체크포인트는
포함하지 않습니다. GPU 수치를 재현하려면 동일한 OpenPCDet 커밋, 모델 YAML,
전처리 데이터와 체크포인트가 필요합니다.

기준 실험 환경은 OpenPCDet 0.6.0+233f849, Python 3.10, PyTorch 2.11,
CUDA 12.6, spconv 2.3.8입니다. 다른 조합의 동작을 보장한다는 의미는 아닙니다.

## 데이터셋 평가 계약

이 도구는 일반적인 OpenPCDet 예측 인터페이스 외에 다음 평가 계약을 사용합니다.

~~~python
dataset.evaluation(
    det_annos,
    class_names,
    iou_thresh=0.7,
    max_range=20,
    eval_min_points=6,
    ignore_low_pts=True,
)
~~~

데이터셋의 evaluation 메서드는 위 키워드를 처리하고 결과 사전에 mAP와
AP_<class name>을 반환해야 합니다. 기본 OpenPCDet 데이터셋을 수정 없이 연결하면
거리별 평가와 eval_min_points가 적용되지 않을 수 있습니다.

## 설치

1. 사용할 OpenPCDet을 설치합니다.
2. code 디렉터리의 두 Python 파일을 원하는 위치에 복사합니다.
3. 두 파일은 반드시 같은 디렉터리에 둡니다.
4. 모델 설정, infos, 체크포인트를 준비합니다.

직접 의존성은 numpy, torch, OpenPCDet입니다. CUDA/PyTorch/spconv 조합은
사용하는 OpenPCDet 환경에 맞춰야 하므로 별도 requirements.txt는 제공하지 않습니다.

## 단일 체크포인트

~~~bash
cd /path/to/OpenPCDet/tools
python /path/to/code/pcdet_nearrange_probe.py \
  --cfg cfgs/waymonpz_models/model.yaml \
  --ckpt /data/checkpoints/best1.pth \
  --infos /data/infos/test.json \
  --ranges 0,30,20,10 --ious 0.7 \
  --eval_min_points 6 \
  --score_thresh 0.01 --eval_score_thresh 0.1 \
  --save_preds /data/results/preds_experiment_best1.pkl \
  --result_file /data/results/near_range_best1.txt
~~~

## Top-N 체크포인트

~~~bash
python code/run_probe_top3.py \
  --cfg /path/to/OpenPCDet/tools/cfgs/waymonpz_models/model.yaml \
  --out /data/experiment/checkpoints \
  --infos /data/infos/test.json \
  --ranges 0,30,20,10 --ious 0.7 \
  --eval_min_points 6 --topn 5 \
  --save_tag experiment \
  --pred_dir /data/experiment/predictions \
  --score_thresh 0.01 --eval_score_thresh 0.1 \
  --openpcdet_tools /path/to/OpenPCDet/tools
~~~

| CLI 인자 | 환경변수 | 생략 시 |
|---|---|---|
| --openpcdet_tools | OPENPCDET_TOOLS | 현재 디렉터리 |
| --pred_dir | WAYMO_DETECTION_ROOT | 체크포인트 디렉터리 |
| --result_file | PCDET_PROBE_RESULT | 예측 PKL 또는 체크포인트 디렉터리의 결과 파일 |

기존 run_probe_top3.py 인자는 유지했으며 새 경로 인자는 모두 선택 사항입니다.

## 점수 문턱

score_thresh는 모델이 생성하고 PKL에 저장할 예측의 최소 점수입니다.
eval_score_thresh는 원시 예측의 복사본에만 적용됩니다. 예를 들어 0.01로 저장하고
0.1로 채점하면 후속 분석은 낮은 점수 예측까지 재사용하면서 기준 채점은 0.1로
유지할 수 있습니다.

## 출력 예시

~~~text
반경(m) | IoU | mAP | Vehicle | Large_Vehicle | Cyclist
----------------------------------------------------------------------
   전체 | 0.7 |  62.4 |  83.8 |  58.0 |  45.4
    30m | 0.7 |  79.6 |  95.3 |  84.7 |  58.7
    20m | 0.7 |  82.4 |  96.9 |  91.4 |  58.8
    10m | 0.7 |  81.0 |  97.1 |  94.3 |  51.7
~~~

숫자는 출력 형식을 보여주는 기존 실험 예시이며 범용 벤치마크 결과가 아닙니다.

## no_ignore

기본값은 최소 포인트 조건에 미달한 GT 위치의 검출을 FP로 세지 않는 ignore
방식입니다. no_ignore를 지정하면 해당 GT를 완전히 제외하므로 그 위치의 검출이
FP가 될 수 있습니다. 두 설정의 결과를 혼용하지 마세요.

## 검증 및 라이선스

배포본은 Python 구문, CLI 인자 계약, 옵션 전달, 출력 계약을 정적으로 검사했습니다.
데이터와 체크포인트가 없어 GPU 수치 회귀 테스트는 실행하지 않았습니다.
자체 코드는 MIT License이며 OpenPCDet과 데이터셋에는 각각의 라이선스가 적용됩니다.
