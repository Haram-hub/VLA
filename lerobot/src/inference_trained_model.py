#!/usr/bin/env python3
"""
훈련된 SmolVLA 모델로 inference를 실행하고 결과를 시각화하는 스크립트
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
import json
import os
from datetime import datetime

def find_latest_checkpoint():
    """가장 최근의 체크포인트를 찾기"""
    outputs_dir = Path("outputs/train")
    if not outputs_dir.exists():
        raise FileNotFoundError("훈련 출력 디렉토리를 찾을 수 없습니다.")
    
    # 가장 최근 날짜 폴더 찾기
    date_dirs = [d for d in outputs_dir.iterdir() if d.is_dir()]
    if not date_dirs:
        raise FileNotFoundError("훈련 출력이 없습니다.")
    
    latest_date = max(date_dirs, key=lambda x: x.name)
    
    # 해당 날짜의 가장 최근 시간 폴더 찾기
    time_dirs = [d for d in latest_date.iterdir() if d.is_dir() and "smolvla" in d.name]
    if not time_dirs:
        raise FileNotFoundError("SmolVLA 모델 폴더를 찾을 수 없습니다.")
    
    latest_checkpoint_dir = max(time_dirs, key=lambda x: x.name)
    
    print(f"체크포인트 디렉토리: {latest_checkpoint_dir}")
    
    # 체크포인트 파일을 다양한 위치에서 찾기
    possible_locations = [
        latest_checkpoint_dir / "checkpoints",
        latest_checkpoint_dir,
        latest_checkpoint_dir / "final"
    ]
    
    checkpoint_files = []
    for location in possible_locations:
        if location.exists():
            # .safetensors, .pth, .bin 파일 찾기
            for pattern in ["*.safetensors", "*.pth", "*.bin"]:
                checkpoint_files.extend(list(location.glob(pattern)))
    
    if not checkpoint_files:
        # 전체 디렉토리에서 재귀적으로 찾기
        checkpoint_files = list(latest_checkpoint_dir.rglob("*.safetensors"))
        checkpoint_files.extend(list(latest_checkpoint_dir.rglob("*.pth")))
        checkpoint_files.extend(list(latest_checkpoint_dir.rglob("*.bin")))
    
    if not checkpoint_files:
        # 디렉토리 자체를 모델 경로로 사용 (config.json이 있는 경우)
        if (latest_checkpoint_dir / "config.json").exists():
            print(f"config.json 발견: {latest_checkpoint_dir}")
            return latest_checkpoint_dir, None
        else:
            raise FileNotFoundError(f"체크포인트 파일이 없습니다: {latest_checkpoint_dir}")
    
    latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
    
    print(f"최신 체크포인트 발견: {latest_checkpoint}")
    return latest_checkpoint_dir, latest_checkpoint

def load_trained_model(checkpoint_dir, dataset, device="cuda"):
    """훈련된 모델 로드 (데이터셋 통계 포함)"""
    print("=== 훈련된 SmolVLA 모델 로드 중 ===")
    
    try:
        # 방법 1: from_pretrained로 시도
        try:
            policy = SmolVLAPolicy.from_pretrained(str(checkpoint_dir))
            policy = policy.to(device)
            policy.eval()
            print(f"from_pretrained로 모델 로드 성공: {device}")
            return policy
        except Exception as e1:
            print(f"from_pretrained 실패: {e1}")
            
        # 방법 2: 훈련 방식과 동일하게 데이터셋으로부터 모델 생성
        try:
            print("훈련 방식과 동일하게 데이터셋으로부터 모델을 생성합니다...")
            
            # 올바른 import
            from lerobot.policies.factory import make_policy, make_policy_config
            from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
            
            # SmolVLA 설정 생성
            policy_cfg = SmolVLAConfig()
            
            # 데이터셋 메타데이터로 정책 생성 #정채
            policy = make_policy(
                cfg=policy_cfg,
                ds_meta=dataset.meta
            )
            policy = policy.to(device)
            
            print("데이터셋 메타데이터와 함께 모델 생성 완료")
            
            # 체크포인트 파일 찾기
            checkpoint_files = []
            for pattern in ["*.safetensors", "*.pth", "*.bin"]:
                checkpoint_files.extend(list(checkpoint_dir.rglob(pattern)))
            
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
                print(f"체크포인트 로드 중: {latest_checkpoint}")
                
                # 체크포인트 로드
                if latest_checkpoint.suffix == ".safetensors":
                    from safetensors.torch import load_file
                    checkpoint = load_file(latest_checkpoint)
                else:
                    checkpoint = torch.load(latest_checkpoint, map_location=device)
                
                # 모델에 가중치 적용
                if "model" in checkpoint:
                    policy.load_state_dict(checkpoint["model"], strict=False)
                elif "state_dict" in checkpoint:
                    policy.load_state_dict(checkpoint["state_dict"], strict=False)
                else:
                    policy.load_state_dict(checkpoint, strict=False)
                
                print("체크포인트 적용 완료")
            else:
                print("체크포인트 파일이 없어서 기본 모델을 사용합니다.")
            
            policy.eval()
            print(f"모델이 {device}에 로드되었습니다.")
            return policy
            
        except Exception as e2:
            print(f"데이터셋 통계를 사용한 모델 로드 실패: {e2}")
            
        # 방법 3: 기본 모델 (통계 없이)
        print("기본 pre-trained 모델을 사용합니다...")
        try:
            policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
            policy = policy.to(device)
            policy.eval()
            print(f"기본 모델이 {device}에 로드되었습니다. (훈련된 가중치 없음)")
            return policy
        except Exception as e3:
            print(f"기본 모델 로드도 실패: {e3}")
        
    except Exception as e:
        print(f"모든 모델 로드 방법 실패: {e}")
        return None

def load_test_dataset():
    """테스트용 데이터셋 로드"""
    print("=== 테스트 데이터셋 로드 중 ===")
    
    try:
        # 테스트용으로 몇 개 episode 로드
        dataset = LeRobotDataset(
            repo_id="lerobot/svla_so100_stacking",
            episodes=[0, 1, 2],  # 처음 3개 episode
            download_videos=True,
            video_backend="pyav"
        )
        
        print(f"테스트 데이터셋 로드 완료!")
        print(f"Episode 개수: {dataset.num_episodes}")
        print(f"총 프레임 수: {dataset.num_frames}")
        print(f"특성 키: {list(dataset.features.keys())}")
        
        return dataset
        
    except Exception as e:
        print(f"데이터셋 로드 실패: {e}")
        return None

def run_inference_on_episode(policy, dataset, episode_idx=0, num_steps=10):
    """특정 episode에서 inference 실행"""
    print(f"\n=== Episode {episode_idx}에서 Inference 실행 ===")
    
    # Episode 정보
    episode_data = dataset.meta.episodes[episode_idx]
    episode_length = episode_data["length"]
    task_info = episode_data.get("tasks", "No task info")
    
    print(f"Episode 길이: {episode_length} 프레임")
    print(f"Task 정보: {task_info}")
    
    # Episode 시작 인덱스 찾기
    episode_start_idx = dataset.episode_data_index["from"][episode_idx].item()
    episode_end_idx = dataset.episode_data_index["to"][episode_idx].item()
    
    print(f"Episode 인덱스 범위: {episode_start_idx} ~ {episode_end_idx}")
    
    # 선택한 스텝 수만큼 inference 실행
    inference_steps = min(num_steps, episode_length)
    results = []
    
    device = next(policy.parameters()).device
    
    for i in range(inference_steps):
        step_idx = episode_start_idx + i
        
        # 데이터 가져오기
        data = dataset[step_idx]
        
        # 배치 차원 추가 및 디바이스로 이동
        batch_data = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                batch_data[key] = value.unsqueeze(0).to(device)
            else:
                batch_data[key] = value
        
        try:
            # Inference 실행
            with torch.no_grad():
                prediction = policy.select_action(batch_data)
                
            # Ground truth action
            gt_action = data.get("action", None)
            
            result = {
                "step": i,
                "global_idx": step_idx,
                "predicted_action": prediction.cpu().numpy() if isinstance(prediction, torch.Tensor) else prediction,
                "ground_truth_action": gt_action.numpy() if isinstance(gt_action, torch.Tensor) else gt_action,
                "observation_state": data.get("observation.state", None),
            }
            
            results.append(result)
            
            if i % 5 == 0:
                print(f"Step {i}/{inference_steps} 완료")
                
        except Exception as e:
            print(f"Step {i}에서 오류 발생: {e}")
            continue
    
    print(f"Inference 완료: {len(results)}개 스텝 처리됨")
    return results, task_info

def visualize_results(results, task_info, episode_idx, output_dir="inference_results"):
    """Inference 결과 시각화"""
    print(f"\n=== 결과 시각화 중 ===")
    
    if not results:
        print("시각화할 결과가 없습니다.")
        return
    
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 데이터 준비
    steps = [r["step"] for r in results]
    pred_actions = np.array([r["predicted_action"] for r in results if r["predicted_action"] is not None])
    gt_actions = np.array([r["ground_truth_action"] for r in results if r["ground_truth_action"] is not None])
    
    if len(pred_actions) == 0 or len(gt_actions) == 0:
        print("액션 데이터가 없어서 시각화를 건너뜁니다.")
        return
    
    # 차원 확인 및 안전한 처리
    print(f"예측 액션 형태: {pred_actions.shape}")
    print(f"실제 액션 형태: {gt_actions.shape}")
    
    # 2차원으로 변환 (필요한 경우)
    if len(pred_actions.shape) == 1:
        pred_actions = pred_actions.reshape(-1, 1)
    if len(gt_actions.shape) == 1:
        gt_actions = gt_actions.reshape(-1, 1)
    
    # 액션 차원 수
    action_dim = min(pred_actions.shape[1], gt_actions.shape[1])
    print(f"액션 차원: {action_dim}")
    
    # 그래프 설정
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'SmolVLA Inference Results - Episode {episode_idx}\nTask: {task_info}', fontsize=16)
    
    # 1. 액션 비교 (시간에 따른)
    ax1 = axes[0, 0]
    for i in range(min(action_dim, 6)):  # 최대 6개 차원만 표시
        ax1.plot(steps[:len(pred_actions)], pred_actions[:, i], 
                label=f'Predicted Action {i}', linestyle='-', alpha=0.8)
        ax1.plot(steps[:len(gt_actions)], gt_actions[:, i], 
                label=f'Ground Truth Action {i}', linestyle='--', alpha=0.8)
    
    ax1.set_title('Action Comparison Over Time')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Action Value')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. 액션 오차 (MAE)
    ax2 = axes[0, 1]
    action_errors = np.abs(pred_actions - gt_actions[:len(pred_actions)])
    mean_errors = np.mean(action_errors, axis=0)
    
    # numpy 배열을 안전하게 1차원으로 변환
    mean_errors_flat = np.atleast_1d(mean_errors).flatten()
    mean_errors_float = [float(x) for x in mean_errors_flat]
    
    ax2.bar(range(len(mean_errors_float)), mean_errors_float)
    ax2.set_title('Mean Absolute Error by Action Dimension')
    ax2.set_xlabel('Action Dimension')
    ax2.set_ylabel('MAE')
    ax2.grid(True, alpha=0.3)
    
    # 3. 전체 오차 히스토그램
    ax3 = axes[1, 0]
    all_errors = action_errors.flatten()
    ax3.hist(all_errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_title('Distribution of Action Errors')
    ax3.set_xlabel('Absolute Error')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    # 4. 오차 히트맵
    ax4 = axes[1, 1]
    if len(action_errors) > 0 and action_errors.shape[1] > 1:
        # 3차원 배열인 경우 2차원으로 평탄화
        if len(action_errors.shape) == 3:
            # (steps, action_dim, extra_dim) -> (steps, action_dim)
            action_errors_2d = action_errors.reshape(action_errors.shape[0], -1)
        else:
            action_errors_2d = action_errors
            
        try:
            sns.heatmap(action_errors_2d.T, ax=ax4, cmap='YlOrRd', cbar=True)
            ax4.set_title('Action Errors Heatmap')
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Action Dimension')
        except Exception as e:
            print(f"히트맵 생성 실패: {e}")
            ax4.text(0.5, 0.5, f'Heatmap error:\n{str(e)[:50]}...', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Action Errors Heatmap (Error)')
    else:
        ax4.text(0.5, 0.5, 'Not enough data\nfor heatmap', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Action Errors Heatmap')
    
    plt.tight_layout()
    
    # 그래프 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = output_path / f"inference_results_ep{episode_idx}_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"시각화 결과 저장됨: {plot_filename}")
    
    # 결과 통계 출력
    print(f"\n=== 결과 통계 ===")
    print(f"총 스텝 수: {len(results)}")
    print(f"평균 절대 오차 (MAE): {float(np.mean(all_errors)):.4f}")
    print(f"표준편차: {float(np.std(all_errors)):.4f}")
    print(f"최대 오차: {float(np.max(all_errors)):.4f}")
    print(f"최소 오차: {float(np.min(all_errors)):.4f}")
    
    # 결과 데이터 저장
    results_filename = output_path / f"inference_data_ep{episode_idx}_{timestamp}.json"
    save_data = {
        "episode_idx": episode_idx,
        "task_info": str(task_info),
        "num_steps": len(results),
        "statistics": {
            "mean_absolute_error": float(np.mean(all_errors)),
            "std_error": float(np.std(all_errors)),
            "max_error": float(np.max(all_errors)),
            "min_error": float(np.min(all_errors))
        },
        "results": results
    }
    
    with open(results_filename, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"결과 데이터 저장됨: {results_filename}")
    
    plt.show()

def main():
    """메인 함수"""
    print("=== 훈련된 SmolVLA 모델 Inference 및 시각화 ===")
    
    # CUDA 사용 가능 여부 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 디바이스: {device}")
    
    try:
        # 1. 최신 체크포인트 찾기
        checkpoint_dir, checkpoint_file = find_latest_checkpoint()
        
        # 2. 테스트 데이터셋 로드 (모델 로드 전에)
        dataset = load_test_dataset()
        if dataset is None:
            return
        
        # 3. 훈련된 모델 로드 (데이터셋 통계 포함)
        policy = load_trained_model(checkpoint_dir, dataset, device)
        if policy is None:
            return
        
        # 4. 여러 episode에서 inference 실행
        episodes_to_test = [0, 1, 2]  # 첫 3개 episode 테스트
        num_steps_per_episode = 20    # 각 episode에서 20 스텝
        
        for episode_idx in episodes_to_test:
            if episode_idx < dataset.num_episodes:
                print(f"\n{'='*50}")
                print(f"Episode {episode_idx} 처리 중...")
                
                # Inference 실행
                results, task_info = run_inference_on_episode(
                    policy, dataset, episode_idx, num_steps_per_episode
                )
                
                # 결과 시각화
                if results:
                    visualize_results(results, task_info, episode_idx)
                else:
                    print(f"Episode {episode_idx}에서 유효한 결과가 없습니다.")
            else:
                print(f"Episode {episode_idx}는 데이터셋 범위를 벗어났습니다.")
        
        print(f"\n{'='*50}")
        print("모든 inference 완료!")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
