#!/usr/bin/env python3
"""
SmolVLA 모델을 svla_so100_stacking 데이터셋으로 평가하는 스크립트

사용법:
python eval_smolvla_svla_stacking.py --policy.path=lerobot/smolvla_base --eval.n_episodes=5 --eval.batch_size=2
"""

import json
import logging
import time
from pathlib import Path
from pprint import pformat

import torch
from termcolor import colored
from tqdm import trange

from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.datasets import DatasetConfig
from lerobot.policies.factory import make_policy, make_policy_config
from lerobot.datasets.factory import make_dataset
from lerobot.utils.utils import init_logging, get_safe_torch_device
from lerobot.utils.random_utils import set_seed


def create_eval_config():
    """평가용 설정 생성"""
    
    # 데이터셋 설정
    dataset_config = DatasetConfig(
        repo_id="lerobot/svla_so100_stacking",
        episodes=None,  # 모든 에피소드 사용
        download_videos=True,
        video_backend="pyav"
    )
    
    # SmolVLA 정책 설정
    policy_config = make_policy_config(
        policy_type="smolvla",
        device="cuda",
        use_amp=False,
        push_to_hub=False
    )
    
    # 훈련 파이프라인 설정 (평가용)
    train_config = TrainPipelineConfig(
        dataset=dataset_config,
        policy=policy_config,
        batch_size=1,
        steps=1,  # 평가용으로 1 step만
        eval_freq=0,  # 평가 비활성화
        log_freq=1,
        save_checkpoint=False,
        output_dir=Path("outputs/eval_smolvla_svla_stacking")
    )
    
    return train_config


def evaluate_smolvla_on_dataset():
    """SmolVLA 모델을 svla_so100_stacking 데이터셋으로 평가"""
    print("=== SmolVLA svla_so100_stacking 평가 시작 ===\n")
    
    # 1. 설정 생성
    print("1. 설정 생성 중...")
    train_config = create_eval_config()
    
    print("평가 설정:")
    print(pformat(train_config.to_dict()))
    
    # 2. 디바이스 확인
    print("\n2. 디바이스 확인 중...")
    device = get_safe_torch_device(train_config.policy.device, log=True)
    print(f"사용 디바이스: {device}")
    
    # 3. 데이터셋 생성
    print("\n3. 데이터셋 생성 중...")
    try:
        dataset = make_dataset(train_config)
        print(f"✅ 데이터셋 생성 성공!")
        print(f"Episode 개수: {dataset.num_episodes}")
        print(f"총 프레임 수: {dataset.num_frames}")
        print(f"특성 키: {list(dataset.features.keys())}")
    except Exception as e:
        print(f"❌ 데이터셋 생성 실패: {e}")
        return
    
    # 4. 정책 생성
    print("\n4. SmolVLA 정책 생성 중...")
    try:
        policy = make_policy(
            cfg=train_config.policy,
            ds_meta=dataset.meta
        )
        print(f"✅ SmolVLA 정책 생성 성공!")
        print(f"모델 타입: {type(policy)}")
    except Exception as e:
        print(f"❌ 정책 생성 실패: {e}")
        return
    
    # 5. 평가 설정
    n_episodes = min(5, dataset.num_episodes)  # 최대 5개 에피소드 평가
    batch_size = 1
    print(f"\n5. 평가 설정: {n_episodes}개 에피소드, 배치 크기 {batch_size}")
    
    # 6. 데이터로더 생성
    print("\n6. 데이터로더 생성 중...")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    
    # 7. 평가 실행
    print(f"\n7. 평가 실행 시작 (총 {n_episodes}개 에피소드)...")
    policy.eval()
    
    total_loss = 0.0
    total_batches = 0
    episode_losses = []
    current_episode = -1
    episode_batch_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # 디바이스로 이동
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")
            
            # 에피소드 추적
            episode_indices = batch.get("episode_index", torch.tensor([0]))
            if isinstance(episode_indices, torch.Tensor):
                episode_idx = episode_indices[0].item()
            else:
                episode_idx = episode_indices[0] if episode_indices else 0
            
            # 새 에피소드 시작
            if episode_idx != current_episode:
                if current_episode >= 0 and episode_batch_count > 0:
                    avg_episode_loss = total_loss / episode_batch_count
                    episode_losses.append(avg_episode_loss)
                    print(f"  Episode {current_episode}: 평균 Loss = {avg_episode_loss:.6f}")
                
                current_episode = episode_idx
                episode_batch_count = 0
                total_loss = 0.0
                
                if len(episode_losses) >= n_episodes:
                    break
            
            # 추론 실행
            try:
                loss, output_dict = policy.forward(batch)
                total_loss += loss.item()
                episode_batch_count += 1
                total_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"  배치 {batch_idx}: Loss = {loss.item():.6f}")
                
            except Exception as e:
                print(f"❌ 배치 {batch_idx} 추론 실패: {e}")
                continue
    
    # 마지막 에피소드 처리
    if current_episode >= 0 and episode_batch_count > 0:
        avg_episode_loss = total_loss / episode_batch_count
        episode_losses.append(avg_episode_loss)
        print(f"  Episode {current_episode}: 평균 Loss = {avg_episode_loss:.6f}")
    
    # 8. 결과 요약
    print(f"\n=== 평가 결과 요약 ===")
    print(f"평가된 에피소드 수: {len(episode_losses)}")
    print(f"총 배치 수: {total_batches}")
    
    if episode_losses:
        avg_loss = sum(episode_losses) / len(episode_losses)
        min_loss = min(episode_losses)
        max_loss = max(episode_losses)
        
        print(f"평균 Loss: {avg_loss:.6f}")
        print(f"최소 Loss: {min_loss:.6f}")
        print(f"최대 Loss: {max_loss:.6f}")
        
        # 결과 저장
        results = {
            "evaluated_episodes": len(episode_losses),
            "total_batches": total_batches,
            "average_loss": avg_loss,
            "min_loss": min_loss,
            "max_loss": max_loss,
            "episode_losses": episode_losses,
            "model_path": train_config.policy.pretrained_path,
            "dataset": train_config.dataset.repo_id,
            "device": str(device),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        output_dir = train_config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "eval_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n결과가 저장되었습니다: {output_dir / 'eval_results.json'}")
    
    print(f"\n=== 평가 완료! ===")


def main():
    """메인 함수"""
    init_logging()
    set_seed(42)  # 재현성을 위한 시드 설정
    
    evaluate_smolvla_on_dataset()


if __name__ == "__main__":
    main()
