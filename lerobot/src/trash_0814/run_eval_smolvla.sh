#!/bin/bash

# SmolVLA svla_so100_stacking 평가 실행 스크립트

echo "=== SmolVLA svla_so100_stacking 평가 시작 ==="

# 기본 실행
echo "1. 기본 평가 실행"
python eval_smolvla_svla_stacking.py

echo ""
echo "=== 사용 예시 ==="
echo ""
echo "다른 모델로 평가:"
echo "python eval_smolvla_svla_stacking.py --policy.path=outputs/train/smolvla/checkpoints/005000/pretrained_model"
echo ""
echo "CPU 모드로 평가:"
echo "python eval_smolvla_svla_stacking.py --policy.device=cpu"
echo ""
echo "AMP 활성화:"
echo "python eval_smolvla_svla_stacking.py --policy.use_amp=true"
echo ""
echo "결과 확인:"
echo "cat outputs/eval_smolvla_svla_stacking/eval_results.json"
