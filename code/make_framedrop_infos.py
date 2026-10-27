# -*- coding: utf-8 -*-
"""88.9 채점에 쓴 test 집합(이륜 프레임 드랍 5,800f) infos를 만든다.

원본 test(dataset_infos_v2_rt9_hv82_test.json)의 validation 5,923프레임 중
`eval_frames/dropped_frames_cyc_123.txt`에 적힌 123프레임(2.1%)을 뺀다.
빼는 기준 = CSV(waymo_level1_drop_object_ids.csv) 삭제 대상 '이륜' 박스가 하나라도 있는 프레임.
train' 7,983프레임은 건드리지 않는다(원본 그대로).

출력 = dataset_infos_v2_rt9_hv82_cycframedrop.json
"""
import os, sys, io, json, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

ap = argparse.ArgumentParser()
ap.add_argument('--root', default=r'C:\waymo_detection')
ap.add_argument('--infos', default='dataset_infos_v2_rt9_hv82_test.json')
ap.add_argument('--droplist', default='dropped_frames_cyc_123.txt')
ap.add_argument('--out', default='dataset_infos_v2_rt9_hv82_cycframedrop.json')
A = ap.parse_args()

d = json.load(open(os.path.join(A.root, A.infos), encoding='utf-8'))
drop = set(l.strip() for l in open(A.droplist, encoding='utf-8') if l.strip())
keep = [x for x in d if not (x['split'] == 'validation' and x['file'] in drop)]

n_tr = sum(1 for x in keep if x['split'] == 'training')
n_va = sum(1 for x in keep if x['split'] == 'validation')
print('train %d / validation %d (원본 validation %d, 드랍 %d)'
      % (n_tr, n_va, sum(1 for x in d if x['split'] == 'validation'), len(drop)))
json.dump(keep, open(os.path.join(A.root, A.out), 'w', encoding='utf-8'), ensure_ascii=False)
print('저장 -> %s' % A.out)
