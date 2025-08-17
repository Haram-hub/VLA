#!/usr/bin/env python3
"""
SmolVLA 모델에 svla_so100_stacking 데이터 1 episode를 넣어서 결과값을 출력하는 코드

자연어 지시 방법:
1. 데이터셋에서 task 정보를 가져와서 자연어 지시로 사용
2. 또는 직접 자연어 지시를 입력하여 테스트
"""

import torch
import numpy as np
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

def load_svla_stacking_dataset():
    """svla_so100_stacking 데이터셋을 로드하고 1 episode를 반환"""
    print("=== svla_so100_stacking 데이터셋 로드 중 ===")
    
    try:
        # 데이터셋 로드 (1 episode만)
        dataset = LeRobotDataset(
            repo_id="lerobot/svla_so100_stacking",
            episodes=[0],  # 첫 번째 episode만 로드
            download_videos=True,
            video_backend="pyav"  # torchcodec 대신 pyav 사용
        )
        
        print(f"데이터셋 로드 완료!")
        print(f"Episode 개수: {dataset.num_episodes}")
        print(f"총 프레임 수: {dataset.num_frames}")
        print(f"특성 키: {list(dataset.features.keys())}")
        
        return dataset
        
    except Exception as e:
        print(f"데이터셋 로드 실패: {e}")
        return None

def get_episode_data(dataset, episode_idx=0):
    """특정 episode의 모든 데이터를 수집"""
    print(f"\n=== Episode {episode_idx} 데이터 수집 중 ===")
    
    # Episode 정보 가져오기
    episode_data = dataset.meta.episodes[episode_idx]
    episode_length = episode_data["length"]
    
    print(f"Episode 길이: {episode_length} 프레임")
    print(f"Episode 정보: {episode_data}")
    
    # Episode의 모든 프레임 데이터 수집
    episode_frames = []
    for i in range(len(dataset)):
        frame_data = dataset[i]
        if frame_data["episode_index"].item() == episode_idx:
            episode_frames.append(frame_data)
    
    print(f"수집된 프레임 수: {len(episode_frames)}")
    
    # Task 정보 가져오기
    task_idx = episode_frames[0]["task_index"].item()
    task = dataset.meta.tasks[task_idx]
    print(f"Task Index: {task_idx}")
    print(f"Task: {task}")
    
    return episode_frames, task

def prepare_batch_for_smolvla(episode_frames, task):
    """SmolVLA 모델 입력을 위한 배치 데이터 준비"""
    print(f"\n=== SmolVLA 입력 데이터 준비 중 ===")
    
    # 첫 번째 프레임만 사용 (단일 timestep 테스트)
    frame = episode_frames[0]
    
    # 이미지 데이터 준비 (top 카메라 사용)
    if "observation.images.top" in frame:
        image = frame["observation.images.top"]
        print(f"Top 이미지 형태: {image.shape}")
    else:
        print("Top 이미지 데이터를 찾을 수 없습니다.")
        return None
    
    # Wrist 이미지도 확인
    if "observation.images.wrist" in frame:
        wrist_image = frame["observation.images.wrist"]
        print(f"Wrist 이미지 형태: {wrist_image.shape}")
    
    # 상태 데이터 준비
    if "observation.state" in frame:
        state = frame["observation.state"]
        print(f"상태 형태: {state.shape}")
        print(f"상태 값: {state}")
    else:
        print("상태 데이터를 찾을 수 없습니다.")
        return None
    
    # 액션 데이터 준비
    if "action" in frame:
        action = frame["action"]
        print(f"액션 형태: {action.shape}")
        print(f"액션 값: {action}")
    else:
        print("액션 데이터를 찾을 수 없습니다.")
        return None
    
    # 배치 형태로 변환 (top 카메라 사용)
    batch = {
        "observation.images.top": image.unsqueeze(0),  # (1, H, W, C)
        "observation.state": state.unsqueeze(0),  # (1, state_dim)
        "action": action.unsqueeze(0),  # (1, action_dim)
        "task": [task]  # 자연어 지시
    }
    
    print(f"배치 준비 완료!")
    print(f"자연어 지시: {task}")
    print(f"사용 가능한 키: {list(frame.keys())}")
    
    return batch

def test_smolvla_inference(policy, batch):
    """SmolVLA 모델로 추론 실행"""
    print(f"\n=== SmolVLA 추론 실행 중 ===")
    
    try:
        # 모델을 평가 모드로 설정
        policy.eval()
        
        with torch.no_grad():
            # 추론 실행
            output = policy(batch)
            
            print(f"추론 완료!")
            print(f"출력 형태: {output.shape}")
            print(f"출력 값: {output}")
            
            # 출력 통계
            print(f"출력 평균: {output.mean().item():.6f}")
            print(f"출력 표준편차: {output.std().item():.6f}")
            print(f"출력 최솟값: {output.min().item():.6f}")
            print(f"출력 최댓값: {output.max().item():.6f}")
            
            return output
            
    except Exception as e:
        print(f"추론 실패: {e}")
        return None

def test_custom_instruction(policy, batch, custom_instruction):
    """사용자 정의 자연어 지시로 테스트"""
    print(f"\n=== 사용자 정의 지시 테스트: '{custom_instruction}' ===")
    
    # 배치의 task를 사용자 정의 지시로 변경
    batch["task"] = [custom_instruction]
    
    return test_smolvla_inference(policy, batch)

def main():
    """메인 함수"""
    print("=== SmolVLA svla_so100_stacking 테스트 시작 ===\n")
    
    # 1. SmolVLA 모델 로드
    print("1. SmolVLA 모델 로드 중...")
    try:
        policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        print("✅ SmolVLA 모델 로드 성공!")
    except Exception as e:
        print(f"❌ SmolVLA 모델 로드 실패: {e}")
        return
    
    # 2. 데이터셋 로드
    print("\n2. svla_so100_stacking 데이터셋 로드 중...")
    dataset = load_svla_stacking_dataset()
    if dataset is None:
        print("❌ 데이터셋 로드 실패")
        return
    
    # 3. Episode 데이터 수집
    print("\n3. Episode 데이터 수집 중...")
    episode_frames, task = get_episode_data(dataset, episode_idx=0)
    
    # 4. 배치 데이터 준비
    print("\n4. 배치 데이터 준비 중...")
    batch = prepare_batch_for_smolvla(episode_frames, task)
    if batch is None:
        print("❌ 배치 데이터 준비 실패")
        return
    
    # 5. 기본 추론 테스트
    print("\n5. 기본 추론 테스트...")
    output = test_smolvla_inference(policy, batch)
    
    # 6. 사용자 정의 지시 테스트
    print("\n6. 사용자 정의 지시 테스트...")
    custom_instructions = [
        "Pick up the red block and place it on the table",
        "Stack the blue cube on top of the green cube",
        "Move the robot arm to the center position",
        "Close the gripper and hold the object"
    ]
    
    for instruction in custom_instructions:
        test_custom_instruction(policy, batch, instruction)
    
    print(f"\n=== 테스트 완료! ===")
    print("자연어 지시 방법:")
    print("1. 데이터셋의 task 필드에서 자동으로 가져오기")
    print("2. batch['task'] 리스트에 원하는 지시를 문자열로 입력")
    print("3. 예시: batch['task'] = ['Pick up the red block']")

if __name__ == "__main__":
    main()
