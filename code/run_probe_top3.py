# -*- coding: utf-8 -*-
"""Evaluate best1..bestN (and optional early) OpenPCDet checkpoints."""
import argparse
import io
import os
from pathlib import Path
import subprocess
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--out', required=True, help='directory containing best*.pth')
    parser.add_argument('--infos', default='')
    parser.add_argument('--ranges', default='0,20')
    parser.add_argument('--ious', default='0.7')
    parser.add_argument('--eval_min_points', default='6')
    parser.add_argument('--no_ignore', action='store_true')
    parser.add_argument('--topn', type=int, default=5)
    parser.add_argument('--save_tag', default='')
    parser.add_argument('--score_thresh', default='0.01')
    parser.add_argument('--eval_score_thresh', default='0.1')
    parser.add_argument(
        '--pred_dir', default=os.environ.get('WAYMO_DETECTION_ROOT', ''),
        help='raw PKL directory (default: checkpoint directory; env WAYMO_DETECTION_ROOT)')
    parser.add_argument(
        '--openpcdet_tools', default=os.environ.get('OPENPCDET_TOOLS', ''),
        help='OpenPCDet tools directory (default: current directory; env OPENPCDET_TOOLS)')
    parser.add_argument('--result_file', default='', help='probe table output file')
    return parser.parse_args(argv)


def make_command(args, probe, checkpoint, prediction_dir):
    command = [
        sys.executable, str(probe), '--cfg', args.cfg, '--ckpt', str(checkpoint),
        '--ranges', args.ranges, '--ious', args.ious,
        '--eval_min_points', args.eval_min_points,
        '--score_thresh', args.score_thresh,
        '--eval_score_thresh', args.eval_score_thresh,
    ]
    if args.no_ignore:
        command.append('--no_ignore')
    if args.infos:
        command.extend(['--infos', args.infos])
    if args.save_tag:
        prediction = prediction_dir / ('preds_%s_%s.pkl' % (args.save_tag, checkpoint.stem))
        command.extend(['--save_preds', str(prediction)])
    if args.result_file:
        command.extend(['--result_file', args.result_file])
    return command


def main(argv=None):
    args = parse_args(argv)
    if args.topn < 1:
        raise ValueError('--topn은 1 이상이어야 합니다')
    probe = Path(__file__).resolve().with_name('pcdet_nearrange_probe.py')
    if not probe.is_file():
        raise FileNotFoundError('probe 스크립트 없음: %s' % probe)

    output_dir = Path(args.out).expanduser()
    prediction_dir = Path(args.pred_dir).expanduser() if args.pred_dir else output_dir
    prediction_dir.mkdir(parents=True, exist_ok=True)
    tools_dir = Path(args.openpcdet_tools).expanduser() if args.openpcdet_tools else Path.cwd()
    if not tools_dir.is_dir():
        raise NotADirectoryError('OpenPCDet tools 디렉터리 없음: %s' % tools_dir)

    slots = [('TOP%d' % rank, 'best%d.pth' % rank) for rank in range(1, args.topn + 1)]
    slots += [('EARLY%d' % rank, 'best_e%d.pth' % rank) for rank in (1, 2, 3)]
    found = False
    for label, filename in slots:
        checkpoint = output_dir / filename
        if not checkpoint.exists():
            print('[%s] %s 없음 - 스킵' % (label, checkpoint))
            continue
        found = True
        print('\n########## %s: %s ##########' % (label, filename), flush=True)
        subprocess.run(
            make_command(args, probe, checkpoint, prediction_dir),
            cwd=str(tools_dir), check=True)

    if not found:
        checkpoint = output_dir / 'best.pth'
        if checkpoint.exists():
            print('\n########## best.pth (top-N 없음, 단일) ##########', flush=True)
            subprocess.run(
                make_command(args, probe, checkpoint, prediction_dir),
                cwd=str(tools_dir), check=True)
        else:
            print('[BEST] %s 없음 - 스킵' % checkpoint)
    print('\n=== PROBE TOP3 DONE ===')


if __name__ == '__main__':
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    main()
