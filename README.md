## ❕서울 Ai Fellow 주관 포디아이비전 협업 프로젝트

## 목표
- LIBERO 데이터셋을 SmolVLA에 학습시키기
- 학습 후 로봇 행동 결과를 오프스크린으로 시뮬레이팅 및 영상화해 보기

### 파일 위치:

- SmolVLA모델이 CUDA에 잘 올라갔는지 확인하는 파일
: lerobot/src/inference_trained_model.py

- SmolVLA 파인튜닝하는 쉘스크립트(with svla_so100_stacking data)
: lerobot/src/0814_train_script.sh

- SmolVLA로 추론하는 파일(with svla_so100_stacking data)
: lerobot/src/inference_trained_model.py

- LIBERO 데이터 시각화 및 기본 policy 모델로 행동 추론 및 결과 시각화하는 파일
: LIBERO/0815_quick_guide_algo copy.ipynb
