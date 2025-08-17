python lerobot/scripts/train.py \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/svla_so100_stacking \
  --dataset.video_backend=pyav \
  --batch_size=64 \
  --steps=20  # 10% of training budget