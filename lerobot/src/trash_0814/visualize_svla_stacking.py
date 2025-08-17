#!/usr/bin/env python3
"""
svla_so100_stacking 데이터의 첫 번째 에피소드를 시각화하는 코드

visualize_dataset.py를 참고하여 작성
"""

import argparse
import gc
import logging
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import rerun as rr
import torch
import torch.utils.data
import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self) -> Iterator:
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    """Convert CHW float32 torch tensor to HWC uint8 numpy array"""
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"expect channel first images, but instead {chw_float32_torch.shape}"
    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy


def visualize_svla_stacking_episode(
    dataset: LeRobotDataset,
    episode_index: int = 0,
    batch_size: int = 32,
    num_workers: int = 0,
    mode: str = "local",
    web_port: int = 9090,
    ws_port: int = 9087,
    save: bool = False,
    output_dir: Path | None = None,
) -> Path | None:
    """
    svla_so100_stacking 데이터의 특정 에피소드를 시각화
    
    Args:
        dataset: LeRobotDataset 인스턴스
        episode_index: 시각화할 에피소드 인덱스 (기본값: 0)
        batch_size: 데이터로더 배치 크기
        num_workers: 데이터로더 워커 수
        mode: 시각화 모드 ('local' 또는 'distant')
        web_port: 웹 포트 (distant 모드용)
        ws_port: 웹소켓 포트 (distant 모드용)
        save: .rrd 파일 저장 여부
        output_dir: 출력 디렉토리 (save=True일 때 필요)
    """
    
    if save:
        assert output_dir is not None, (
            "Set an output directory where to write .rrd files with `--output-dir path/to/directory`."
        )

    repo_id = dataset.repo_id

    print(f"=== svla_so100_stacking Episode {episode_index} 시각화 시작 ===")
    print(f"데이터셋: {repo_id}")
    print(f"에피소드 인덱스: {episode_index}")
    print(f"총 에피소드 수: {dataset.num_episodes}")
    print(f"총 프레임 수: {dataset.num_frames}")
    print(f"카메라 키: {dataset.meta.camera_keys}")
    print(f"특성 키: {list(dataset.features.keys())}")

    logging.info("Loading dataloader")
    episode_sampler = EpisodeSampler(dataset, episode_index)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=episode_sampler,
    )

    logging.info("Starting Rerun")

    if mode not in ["local", "distant"]:
        raise ValueError(mode)

    spawn_local_viewer = mode == "local" and not save
    rr.init(f"{repo_id}/episode_{episode_index}", spawn=spawn_local_viewer)

    # Manually call python garbage collector after `rr.init` to avoid hanging in a blocking flush
    # when iterating on a dataloader with `num_workers` > 0
    gc.collect()

    if mode == "distant":
        rr.serve(open_browser=False, web_port=web_port, ws_port=ws_port)

    logging.info("Logging to Rerun")

    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        # iterate over the batch
        for i in range(len(batch["index"])):
            rr.set_time_sequence("frame_index", batch["frame_index"][i].item())
            rr.set_time_seconds("timestamp", batch["timestamp"][i].item())

            # display each camera image
            for key in dataset.meta.camera_keys:
                if key in batch:
                    rr.log(key, rr.Image(to_hwc_uint8_numpy(batch[key][i])))

            # display each dimension of action space (e.g. actuators command)
            if "action" in batch:
                for dim_idx, val in enumerate(batch["action"][i]):
                    rr.log(f"action/{dim_idx}", rr.Scalar(val.item()))

            # display each dimension of observed state space (e.g. agent position in joint space)
            if "observation.state" in batch:
                for dim_idx, val in enumerate(batch["observation.state"][i]):
                    rr.log(f"state/{dim_idx}", rr.Scalar(val.item()))

            # display task information
            if "task" in batch:
                rr.log("task", rr.TextDocument(batch["task"][i]))

            if "next.done" in batch:
                rr.log("next.done", rr.Scalar(batch["next.done"][i].item()))

            if "next.reward" in batch:
                rr.log("next.reward", rr.Scalar(batch["next.reward"][i].item()))

            if "next.success" in batch:
                rr.log("next.success", rr.Scalar(batch["next.success"][i].item()))

    if mode == "local" and save:
        # save .rrd locally
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        repo_id_str = repo_id.replace("/", "_")
        rrd_path = output_dir / f"{repo_id_str}_episode_{episode_index}.rrd"
        rr.save(rrd_path)
        print(f"시각화 데이터가 저장되었습니다: {rrd_path}")
        return rrd_path

    elif mode == "distant":
        # stop the process from exiting since it is serving the websocket connection
        try:
            print(f"웹소켓 서버가 실행 중입니다. ws://localhost:{ws_port}로 연결하세요.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Ctrl-C received. Exiting.")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="svla_so100_stacking 데이터 시각화")

    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="시각화할 에피소드 인덱스 (기본값: 0)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="로컬에 저장된 데이터셋의 루트 디렉토리 (기본값: Hugging Face 캐시 폴더에서 로드)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="`--save 1`이 설정되었을 때 .rrd 파일을 저장할 디렉토리 경로",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="DataLoader가 로드하는 배치 크기",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="데이터 로딩을 위한 DataLoader의 프로세스 수",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        help=(
            "'local' 또는 'distant' 중 시각화 모드 선택. "
            "'local'은 로컬 머신에 데이터가 있어야 하며 로컬 뷰어를 생성합니다. "
            "'distant'는 데이터가 저장된 원격 머신에 서버를 생성합니다. "
            "로컬 머신에서 `rerun ws://localhost:PORT`로 서버에 연결하여 데이터를 시각화하세요."
        ),
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=9090,
        help="`--mode distant`가 설정되었을 때 rerun.io의 웹 포트",
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=9087,
        help="`--mode distant`가 설정되었을 때 rerun.io의 웹소켓 포트",
    )
    parser.add_argument(
        "--save",
        type=int,
        default=0,
        help=(
            "`--output-dir`로 제공된 디렉토리에 .rrd 파일을 저장합니다. "
            "뷰어 생성도 비활성화됩니다. "
            "로컬 머신에서 `rerun path/to/file.rrd`를 실행하여 데이터를 시각화하세요."
        ),
    )
    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help=(
            "데이터 타임스탬프가 데이터셋 fps 값을 준수하는지 확인하는 데 사용되는 허용 오차(초) "
            "이는 LeRobotDataset 생성자에 전달되는 인수이며 tolerance_s 생성자 인수에 매핑됩니다 "
            "지정하지 않으면 기본값 1e-4가 사용됩니다."
        ),
    )

    args = parser.parse_args()

    print("=== svla_so100_stacking 데이터셋 로드 중 ===")
    
    # 데이터셋 로드
    dataset = LeRobotDataset(
        repo_id="lerobot/svla_so100_stacking",
        root=args.root,
        tolerance_s=args.tolerance_s,
        episodes=[args.episode_index]  # 특정 에피소드만 로드
    )

    print(f"데이터셋 로드 완료!")
    print(f"선택된 에피소드: {args.episode_index}")
    print(f"에피소드 길이: {dataset.meta.episodes[args.episode_index]['length']} 프레임")
    print(f"Task: {dataset.meta.episodes[args.episode_index]['tasks']}")

    # 시각화 실행
    visualize_svla_stacking_episode(
        dataset=dataset,
        episode_index=args.episode_index,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mode=args.mode,
        web_port=args.web_port,
        ws_port=args.ws_port,
        save=bool(args.save),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

