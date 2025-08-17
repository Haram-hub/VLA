#!/usr/bin/env python3
"""
SmolVLA CUDA 호환성 테스트 스크립트
"""

import torch
import os
import sys

def test_cuda_environment():
    """CUDA 환경 테스트"""
    print("=== CUDA 환경 테스트 ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        print(f"Device capability: {torch.cuda.get_device_capability()}")
        
        # 기본 CUDA 연산 테스트
        try:
            x = torch.randn(10, 10).cuda()
            y = torch.randn(10, 10).cuda()
            z = torch.matmul(x, y)
            print("✅ 기본 CUDA 연산 성공")
        except Exception as e:
            print(f"❌ 기본 CUDA 연산 실패: {e}")
            return False
    
    return True

def test_smolvla_loading():
    """SmolVLA 모델 로드 테스트"""
    print("\n=== SmolVLA 모델 로드 테스트 ===")
    
    # CUDA 설정 최적화
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    
    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        
        print("SmolVLA 모델을 로드하는 중...")
        
        # 방법 1: 기본 GPU 로드
        policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        
        print("✅ SmolVLA 모델이 GPU에서 성공적으로 로드되었습니다!")
        print(f"모델 타입: {type(policy)}")
        return policy
        
    except Exception as e:
        print(f"❌ 기본 GPU 로드 실패: {e}")
        
        # 방법 2: auto device mapping 시도
        try:
            print("\nAuto device mapping으로 재시도...")
            policy = SmolVLAPolicy.from_pretrained(
                "lerobot/smolvla_base",
                device_map="auto",
                torch_dtype=torch.float16
            )
            print("✅ Auto device mapping으로 성공!")
            return policy
            
        except Exception as e2:
            print(f"❌ Auto device mapping 실패: {e2}")
            
            # 방법 3: CPU 모드 시도
            try:
                print("\nCPU 모드로 재시도...")
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                policy = SmolVLAPolicy.from_pretrained(
                    "lerobot/smolvla_base", 
                    device_map="cpu"
                )
                print("✅ CPU 모드로 성공!")
                return policy
                
            except Exception as e3:
                print(f"❌ CPU 모드 실패: {e3}")
                return None

def main():
    """메인 함수"""
    print("SmolVLA CUDA 호환성 테스트 시작\n")
    
    # CUDA 환경 테스트
    if not test_cuda_environment():
        print("CUDA 환경에 문제가 있습니다.")
        return
    
    # SmolVLA 모델 로드 테스트
    policy = test_smolvla_loading()
    
    if policy:
        print(f"\n🎉 성공! 모델이 로드되었습니다.")
        print(f"모델 타입: {type(policy)}")
    else:
        print(f"\n❌ 모든 방법이 실패했습니다.")
        print("PyTorch 버전이나 CUDA 설정을 다시 확인해주세요.")

if __name__ == "__main__":
    main()
