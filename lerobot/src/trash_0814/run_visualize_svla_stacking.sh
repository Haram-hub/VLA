#!/bin/bash

# svla_so100_stacking 데이터 시각화 실행 스크립트
# 기존 visualize_dataset.py 스크립트를 사용

echo "=== svla_so100_stacking 데이터 시각화 시작 ==="

# 기본 실행 (첫 번째 에피소드 시각화)
echo "1. 기본 실행 - 첫 번째 에피소드 시각화"
python -m lerobot.scripts.visualize_dataset \
    --repo-id lerobot/svla_so100_stacking \
    --episode-index 0 \
    --batch-size 16 \
    --num-workers 0

echo ""
echo "=== 사용 예시 ==="
echo ""
echo "다른 에피소드 시각화:"
echo "python -m lerobot.scripts.visualize_dataset --repo-id lerobot/svla_so100_stacking --episode-index 1"
echo ""
echo "시각화 데이터를 .rrd 파일로 저장:"
echo "python -m lerobot.scripts.visualize_dataset --repo-id lerobot/svla_so100_stacking --episode-index 0 --save 1 --output-dir ./visualization_output"
echo ""
echo "원격 시각화 (웹소켓 서버):"
echo "python -m lerobot.scripts.visualize_dataset --repo-id lerobot/svla_so100_stacking --episode-index 0 --mode distant --ws-port 9087"
echo ""
echo "저장된 .rrd 파일 시각화:"
echo "rerun lerobot_svla_so100_stacking_episode_0.rrd"
