#!/usr/bin/env python3
"""
SmolVLA 모델에 svla_so100_stacking 데이터 1 episode를 넣어서 결과값을 출력하는 코드 (v2)

train.py를 참고하여 올바른 방식으로 구현
"""

import torch
import logging
from pathlib import Path
from pprint import pformat

from lerobot.configs import parser
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy
from lerobot.utils.utils import init_logging, get_safe_torch_device
from lerobot.utils.random_utils import set_seed

def create_test_config():
    """테스트용 설정 생성"""
    
    # SmolVLA 정책 설정
    policy_config = SmolVLAConfig(
        device="cuda",
        use_amp=False
    )
    
    return policy_config

def test_smolvla_inference():
    """SmolVLA 모델 추론 테스트"""
    print("=== SmolVLA svla_so100_stacking 테스트 시작 (v2) ===\n")
    
    # 1. 설정 생성
    print("1. 설정 생성 중...")
    policy_config = create_test_config()
    
    print("정책 설정:")
    print(pformat(policy_config.to_dict()))
    
    # 2. 디바이스 확인
    print("\n2. 디바이스 확인 중...")
    device = get_safe_torch_device(policy_config.device, log=True)
    print(f"사용 디바이스: {device}")
    
    # 3. 데이터셋 생성
    print("\n3. 데이터셋 생성 중...")
    try:
        dataset = LeRobotDataset(
            repo_id="lerobot/svla_so100_stacking",
            episodes=[0],  # 첫 번째 episode만
            download_videos=True,
            video_backend="pyav"
        )
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
        policy = make_policy(cfg=policy_config, ds_meta=dataset.meta)
        print(f"✅ SmolVLA 정책 생성 성공!")
        print(f"모델 타입: {type(policy)}")
    except Exception as e:
        print(f"❌ 정책 생성 실패: {e}")
        return
    
    # 5. 데이터로더 생성
    print("\n5. 데이터로더 생성 중...")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # 단일 배치
        shuffle=False,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    
    # 6. 추론 테스트
    print("\n6. 추론 테스트 시작...")
    policy.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            print(f"\n--- 배치 {batch_idx + 1} ---")
            
            # 디바이스로 이동
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")
            
            print(f"배치 키: {list(batch.keys())}")
            
            # 배치 크기 확인
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"{key}: {value.shape}")
                elif isinstance(value, list):
                    print(f"{key}: {value}")
            
            # 추론 실행
            try:
                loss, output_dict = policy.forward(batch)
                print(f"✅ 추론 성공!")
                print(f"Loss: {loss.item():.6f}")
                print(f"출력 키: {list(output_dict.keys()) if output_dict else 'None'}")
                
                # 출력 상세 정보
                if output_dict:
                    for key, value in output_dict.items():
                        if isinstance(value, torch.Tensor):
                            print(f"  {key}: {value.shape}, mean={value.mean().item():.6f}, std={value.std().item():.6f}")
                        else:
                            print(f"  {key}: {value}")
                
            except Exception as e:
                print(f"❌ 추론 실패: {e}")
                import traceback
                traceback.print_exc()
            
            # 첫 번째 배치만 테스트
            break
    
    # 7. 사용자 정의 자연어 지시 테스트
    print("\n7. 사용자 정의 자연어 지시 테스트...")
    custom_instructions = [
        "Pick up the red block and place it on the table",
        "Stack the blue cube on top of the green cube",
        "Move the robot arm to the center position",
        "Close the gripper and hold the object"
    ]
    
    for instruction in custom_instructions:
        print(f"\n--- 지시: '{instruction}' ---")
        
        # 배치의 task를 변경
        modified_batch = {key: value.clone() if isinstance(value, torch.Tensor) else value.copy() if isinstance(value, list) else value 
                         for key, value in batch.items()}
        modified_batch["task"] = [instruction]
        
        try:
            loss, output_dict = policy.forward(modified_batch)
            print(f"✅ 추론 성공! Loss: {loss.item():.6f}")
        except Exception as e:
            print(f"❌ 추론 실패: {e}")
    
    print(f"\n=== 테스트 완료! ===")
    print("자연어 지시 방법:")
    print("1. 데이터셋의 task 필드에서 자동으로 가져오기")
    print("2. batch['task'] 리스트에 원하는 지시를 문자열로 입력")
    print("3. 예시: batch['task'] = ['Pick up the red block']")

def main():
    """메인 함수"""
    init_logging()
    set_seed(42)  # 재현성을 위한 시드 설정
    
    test_smolvla_inference()

if __name__ == "__main__":
    main()
