"""
    将 Realman 数据集 转为 lerobot 数据集格式

    1.使用方法：
            python mydataset2lerobot_2angle_all_task.py --tasks_dir /home/jdtech/Documents/dataset_convert/dataset --output_dir /home/jdtech/Documents/dataset_convert/dataset_lerobot  --fps 20
    
    2.
    (1)input数据集格式：
        dataset
            grasp loppy
                episode0.hdf5
                episdoe1.hdf5
                ...
            grasp cup
                episode0.hdf5
                episode1.hdf5
               ...
            task2
                ...
    (2)output数据集格式：
        dataset_lerobot
            
                data
                    chunk000
                        episode_00000.parquet
                        episode_00001.parquet
                        ...
                meta
                    meta.json
                    stats.json
                    tasks.jsonl
                    episodes.jsonl
                video
                    chunk000
                        ovbservation.images.cam_head_left
                            episode_00000.mp4
                            episode_00001.mp4
                            ...
                        ovbservation.images.cam_head_right
                            episode_00000.mp4
                            episode_00001.mp4
                           ...
                      
    3.使用信息：
    (1) Realman：
            Group: action
            Dataset: action/arm_left
            Dataset: action/arm_right
            Dataset: action/hand_left
            Dataset: action/hand_right
            
            Group: observations
            Dataset: observations/arm_left
            Dataset: observations/arm_right
            Dataset: observations/hand_left
            Dataset: observations/hand_right

            Group: observations/images
            Dataset: observations/images/cam_head_left
            Dataset: observations/images/cam_head_right

    (2) lerobot：
            
            data
            meta
            video

"""

import h5py
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import json
import cv2
import torch
import os
import json
import shutil
import logging
import argparse
from pathlib import Path
from typing import Callable
from functools import partial
from math import ceil
from copy import deepcopy
import einops
from PIL import Image
from tqdm import tqdm
from pprint import pformat
from tqdm.contrib.concurrent import process_map
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import (
    STATS_PATH,
    check_timestamps_sync,
    get_episode_data_index,
    serialize_dict,
    write_json,
)
from scipy.spatial.transform import Rotation as R


# HEAD_COLOR = "head_color.mp4"
CAM_COLOR_LEFT = "cam_color_left.mp4"
CAM_COLOR_RIGHT = "cam_color_right.mp4"

FEATURES = {
    "observation.images.cam_head_left": 
    {
        "dtype": "video",
        "shape": [720, 1280, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 20.0,
        }
    },
    "observation.images.cam_head_right": 
    {
        "dtype": "video",
        "shape": [720, 1280, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 20.0,
        }
    },
    "observation.state": 
    {
        "dtype": "float32",
        "shape": [62],
    },
    "action":
    {
        "dtype": "float32",
        "shape": [26],
    },
    "episode_index": {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
    "frame_index": {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
    "index": {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
    "task_index": {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },

    
}

# def load_local_dataset(episode_id: int, src_path: str, task_id: int) -> list | None:
def load_local_dataset(hdf5_path: str) -> tuple:
    """Load local dataset and return a dict with observations and actions"""

    
    with h5py.File(hdf5_path,"r") as hdf_file:
        # 读取动作数据
        actions = {
            "arm_left": np.array(hdf_file['action/arm_left']),
            "arm_right": np.array(hdf_file['action/arm_right']),
            "hand_left": np.array(hdf_file['action/hand_left']),
            "hand_right": np.array(hdf_file['action/hand_right']),
        }

        # 读取观测数据
        observations = {
            "arm_left": np.array(hdf_file['observations/arm_left']),
            "arm_right": np.array(hdf_file['observations/arm_right']),
            "hand_left": np.array(hdf_file['observations/hand_left']),
            "hand_right": np.array(hdf_file['observations/hand_right']),
        }

        # 读取图像数据
        images = {
            "cam_head_left": [decompress_image(frame) for frame in hdf_file['observations/images/cam_head_left']],
            "cam_head_right": [decompress_image(frame) for frame in hdf_file['observations/images/cam_head_right']],
        }

        # 保存图像帧为 MP4 视频
        v_path = Path(hdf5_path).parent / "videos" /"chunk-000/observation.images"
        v_path.mkdir(parents=True, exist_ok=True)

        cam_head_left_path = v_path / CAM_COLOR_LEFT
        cam_head_right_path = v_path / CAM_COLOR_RIGHT

        save_video(images["cam_head_left"], cam_head_left_path, fps=20, feature_key="observation.images.cam_head_left")
        save_video(images["cam_head_right"], cam_head_right_path, fps=20, feature_key="observation.images.cam_head_right")

        # 计算帧数
        num_frames = len(actions["arm_left"])

        # 生成帧数据
        frames = []
        for i in tqdm(range(num_frames), desc="Processing frames"):

            # state:
            # 提取四元数并转换为轴角
            print("observations['arm_left'][i][9:]:", observations["arm_left"][i][9:])
            print("observations['arm_right'][i][9:].shape:", observations["arm_right"][i][9:].shape)
            state_arm_left_quaternion = observations["arm_left"][i][9:]  # 后四维 (xyzw)
            state_arm_right_quaternion = observations["arm_right"][i][9:]  # 后四维 (xyzw)
            # 转换为轴角表示
            state_arm_left_axis_angle = R.from_quat(state_arm_left_quaternion).as_rotvec()  # 转为轴角
            state_arm_right_axis_angle = R.from_quat(state_arm_right_quaternion).as_rotvec()  # 转为轴角
            # 将轴角替换到原始数组中
            state_arm_left_processed = np.hstack([observations["arm_left"][i][:9], state_arm_left_axis_angle])
            state_arm_right_processed = np.hstack([observations["arm_right"][i][:9], state_arm_right_axis_angle])
            assert state_arm_left_processed.shape == (12,), f"state_arm_left_processed.shape wrong: {state_arm_left_processed.shape}, must be (12,)"
            # action:
            # 提取四元数并转换为轴角
            action_arm_left_quaternion = actions["arm_left"][i][3:]  # 后四维 (xyzw)
            action_arm_right_quaternion = actions["arm_right"][i][3:]  # 后四维 (xyzw)
            # 转换为轴角表示
            action_arm_left_axis_angle = R.from_quat(action_arm_left_quaternion).as_rotvec()  # 转为轴角
            action_arm_right_axis_angle = R.from_quat(action_arm_right_quaternion).as_rotvec()  # 转为轴角
            # 将轴角替换到原始数组中
            action_arm_left_processed = np.hstack([actions["arm_left"][i][:3], action_arm_left_axis_angle])
            action_arm_right_processed = np.hstack([actions["arm_right"][i][:3], action_arm_right_axis_angle])
            assert action_arm_left_processed.shape == (6,), f"action_arm_left_processed.shape wrong: {action_arm_left_processed.shape}, must be (6,)"
            
            print("action_arm_left_processed:", action_arm_left_processed)


            frame = {
                
                "observation.state": np.hstack([
                    state_arm_left_processed,
                    state_arm_right_processed,
                    observations["hand_left"][i],
                    observations["hand_right"][i],
                ]).astype(np.float32),
                "action": np.hstack([
                    action_arm_left_processed,
                    action_arm_right_processed,
                    actions["hand_left"][i],
                    actions["hand_right"][i],
                ]).astype(np.float32),
            }
            frames.append(frame)

        # 视频生成路径
        videos = {
            
            "observation.images.cam_head_left": cam_head_left_path,
            "observation.images.cam_head_right": cam_head_right_path,
        }
    return frames, videos

# 统计计算
def get_stats_einops_patterns(dataset, num_workers=0):
    """These einops patterns will be used to aggregate batches and compute statistics.

    Note: We assume the images are in channel first format
    """

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=2,
        shuffle=False,
    )
    batch = next(iter(dataloader))

    stats_patterns = {}

    for key in dataset.features:
        # sanity check that tensors are not float64
        assert batch[key].dtype != torch.float64

        # if isinstance(feats_type, (VideoFrame, Image)):
        if key in dataset.meta.camera_keys:
            # sanity check that images are channel first
            _, c, h, w = batch[key].shape
            assert (
                c < h and c < w
            ), f"expect channel first images, but instead {batch[key].shape}"
            assert (
                batch[key].dtype == torch.float32
            ), f"expect torch.float32, but instead {batch[key].dtype=}"
            # assert batch[key].max() <= 1, f"expect pixels lower than 1, but instead {batch[key].max()=}"
            # assert batch[key].min() >= 0, f"expect pixels greater than 1, but instead {batch[key].min()=}"
            stats_patterns[key] = "b c h w -> c 1 1"
        elif batch[key].ndim == 2:
            stats_patterns[key] = "b c -> c "
        elif batch[key].ndim == 1:
            stats_patterns[key] = "b -> 1"
        else:
            raise ValueError(f"{key}, {batch[key].shape}")

    return stats_patterns

def compute_stats(dataset, batch_size=8, num_workers=4, max_num_samples=None):
    """Compute mean/std and min/max statistics of all data keys in a LeRobotDataset."""
    if max_num_samples is None:
        max_num_samples = len(dataset)

    # for more info on why we need to set the same number of workers, see `load_from_videos`
    stats_patterns = get_stats_einops_patterns(dataset, num_workers)

    # mean and std will be computed incrementally while max and min will track the running value.
    mean, std, max, min = {}, {}, {}, {}
    for key in stats_patterns:
        mean[key] = torch.tensor(0.0).float()
        std[key] = torch.tensor(0.0).float()
        max[key] = torch.tensor(-float("inf")).float()
        min[key] = torch.tensor(float("inf")).float()

    def create_seeded_dataloader(dataset, batch_size, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=generator,
        )
        return dataloader

    # Note: Due to be refactored soon. The point of storing `first_batch` is to make sure we don't get
    # surprises when rerunning the sampler.
    first_batch = None
    running_item_count = 0  # for online mean computation
    dataloader = create_seeded_dataloader(dataset, batch_size, seed=1337)
    for i, batch in enumerate(
        tqdm(
            dataloader,
            total=ceil(max_num_samples / batch_size),
            desc="Compute mean, min, max",
        )
    ):
        this_batch_size = len(batch["index"])
        running_item_count += this_batch_size
        if first_batch is None:
            first_batch = deepcopy(batch)
        for key, pattern in stats_patterns.items():
            batch[key] = batch[key].float()
            # Numerically stable update step for mean computation.
            batch_mean = einops.reduce(batch[key], pattern, "mean")
            # Hint: to update the mean we need x̄ₙ = (Nₙ₋₁x̄ₙ₋₁ + Bₙxₙ) / Nₙ, where the subscript represents
            # the update step, N is the running item count, B is this batch size, x̄ is the running mean,
            # and x is the current batch mean. Some rearrangement is then required to avoid risking
            # numerical overflow. Another hint: Nₙ₋₁ = Nₙ - Bₙ. Rearrangement yields
            # x̄ₙ = x̄ₙ₋₁ + Bₙ * (xₙ - x̄ₙ₋₁) / Nₙ
            mean[key] = (
                mean[key]
                + this_batch_size * (batch_mean - mean[key]) / running_item_count
            )
            max[key] = torch.maximum(
                max[key], einops.reduce(batch[key], pattern, "max")
            )
            min[key] = torch.minimum(
                min[key], einops.reduce(batch[key], pattern, "min")
            )

        if i == ceil(max_num_samples / batch_size) - 1:
            break

    first_batch_ = None
    running_item_count = 0  # for online std computation
    dataloader = create_seeded_dataloader(dataset, batch_size, seed=1337)
    for i, batch in enumerate(
        tqdm(dataloader, total=ceil(max_num_samples / batch_size), desc="Compute std")
    ):
        this_batch_size = len(batch["index"])
        running_item_count += this_batch_size
        # Sanity check to make sure the batches are still in the same order as before.
        if first_batch_ is None:
            first_batch_ = deepcopy(batch)
            for key in stats_patterns:
                assert torch.equal(first_batch_[key], first_batch[key])
        for key, pattern in stats_patterns.items():
            batch[key] = batch[key].float()
            # Numerically stable update step for mean computation (where the mean is over squared
            # residuals).See notes in the mean computation loop above.
            batch_std = einops.reduce((batch[key] - mean[key]) ** 2, pattern, "mean")
            std[key] = (
                std[key] + this_batch_size * (batch_std - std[key]) / running_item_count
            )

        if i == ceil(max_num_samples / batch_size) - 1:
            break

    for key in stats_patterns:
        std[key] = torch.sqrt(std[key])

    stats = {}
    for key in stats_patterns:
        stats[key] = {
            "mean": mean[key],
            "std": std[key],
            "max": max[key],
            "min": min[key],
        }
    return stats

def get_task_instruction(task_json_path: str) -> dict:
    """Get task language instruction"""
    with open(task_json_path, "r") as f:
        task_info = json.load(f)
    task_name = task_info[0]["task_name"]
    task_init_scene = task_info[0]["init_scene_text"]
    task_instruction = f"{task_name}.{task_init_scene}"
    print(f"Get Task Instruction <{task_instruction}>")
    return task_instruction

# 保存成mp4格式
def save_video(frames, output_path, fps, feature_key):
    """
    保存图像帧为 MP4 视频文件。

    :param frames: 图像帧列表，每帧是一个 NumPy 数组。
    :param output_path: 输出视频文件路径。
    :param fps: 视频帧率。
    :param feature_key: FEATURES 中定义的视频特征键，用于获取视频尺寸。
    """
    # 从 FEATURES 中获取视频尺寸
    height = FEATURES[feature_key]["shape"][0]
    width = FEATURES[feature_key]["shape"][1]

    # 定义视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    try:
        for frame in frames:
            # 确保帧数据为 uint8 类型
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            # 确保像素值在 [0, 255] 范围内
            if frame.max() > 255 or frame.min() < 0:
                frame = np.clip(frame, 0, 255)
            video_writer.write(frame)
    finally:
        video_writer.release()
        print(f"Saved video to {output_path}")



def decompress_image(compressed_data):
    """
    解压缩图像数据，将二进制数据转换为图像数组。

    :param compressed_data: 压缩的二进制图像数据
    :return: 解压缩后的图像数组
    """
    # 将压缩的字节数据转换为 numpy 数组
    np_data = np.frombuffer(compressed_data, np.uint8)
    
    # 使用 OpenCV 解压缩图像
    image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("无法解压缩图像数据")
    
    return image


class AgiBotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        download_videos: bool = True,
        local_files_only: bool = False,
        video_backend: str | None = None,
    ):
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            download_videos=download_videos,
            local_files_only=local_files_only,
            video_backend=video_backend,
        )

    def save_episode(
        self, task: str,  videos: dict | None = None, global_episode_index: int = None
    ) -> None:
        """
        保存 episode 数据，并使用全局 episode 索引。

        :param task: 当前任务名称。
        :param videos: 视频路径字典。
        :param global_episode_index: 全局 episode 索引。
        
        """

        episode_length = self.episode_buffer.pop("size")
        if episode_length == 0:
            raise ValueError("episode_length is 0, please add frames first.")
        episode_index = self.episode_buffer["episode_index"]
        if episode_index != self.meta.total_episodes:
            # TODO(aliberts): Add option to use existing episode_index
            raise NotImplementedError(
                "You might have manually provided the episode_buffer with an episode_index that doesn't "
                "match the total number of episodes in the dataset. This is not supported for now."
            )

        # print(f"task: {task}")
        task_index = self.meta.get_task_index(task)
        if task_index is None:
            raise ValueError(f"Task {task} not found in meta/tasks.jsonl.")


        for key, ft in self.features.items():
            if key == "index":
                self.episode_buffer[key] = np.arange(
                    self.meta.total_frames, self.meta.total_frames + episode_length
                )
            elif key == "episode_index":
                self.episode_buffer[key] = np.full((episode_length,), global_episode_index)
            elif key == "task_index":
                self.episode_buffer[key] = np.full((episode_length,), task_index)
            elif ft["dtype"] in ["image", "video"]:
                continue
            elif len(ft["shape"]) == 1 and ft["shape"][0] == 1:
                self.episode_buffer[key] = np.array(self.episode_buffer[key], dtype=ft["dtype"])
            elif len(ft["shape"]) == 1 and ft["shape"][0] > 1:
                self.episode_buffer[key] = np.stack(self.episode_buffer[key])
            else:
                raise ValueError(key)

        self._wait_image_writer()
        self._save_episode_table(self.episode_buffer, global_episode_index) #保存成 parquet 文件
    
        self.meta.save_episode(global_episode_index, episode_length, task, task_index) #记录episode.jsonl

        for key in self.meta.video_keys:
            print(f"Copying video {key} to {self.root}")
            video_path = self.root / self.meta.get_video_file_path(global_episode_index, key)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(videos[key], video_path)

        self.episode_buffer = self.create_episode_buffer()
        self.consolidated = False

    def consolidate(
        self, run_compute_stats: bool = True, keep_image_files: bool = False
    ) -> None:
        self.hf_dataset = self.load_hf_dataset()
        self.episode_data_index = get_episode_data_index(
            self.meta.episodes, self.episodes
        )
        check_timestamps_sync(
            self.hf_dataset, self.episode_data_index, self.fps, self.tolerance_s
        )
        if len(self.meta.video_keys) > 0:
            self.meta.write_video_info()

        if not keep_image_files:
            img_dir = self.root / "images"
            if img_dir.is_dir():
                shutil.rmtree(self.root / "images")
        video_files = list(self.root.rglob("*.mp4"))
        assert len(video_files) == self.num_episodes * len(self.meta.video_keys)

        parquet_files = list(self.root.rglob("*.parquet"))
        assert len(parquet_files) == self.num_episodes

        if run_compute_stats:
            self.stop_image_writer()
            self.meta.stats = compute_stats(self)
            serialized_stats = serialize_dict(self.meta.stats)
            write_json(serialized_stats, self.root / STATS_PATH)
            self.consolidated = True
        else:
            logging.warning(
                "Skipping computation of the dataset statistics, dataset is not fully consolidated."
            )

    def add_frame(self, frame: dict) -> None:
        """
        This function only adds the frame to the episode_buffer. Apart from images — which are written in a
        temporary directory — nothing is written to disk. To save those frames, the 'save_episode()' method
        then needs to be called.
        """
        # TODO(aliberts, rcadene): Add sanity check for the input, check it's numpy or torch,
        # check the dtype and shape matches, etc.

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        frame_index = self.episode_buffer["size"]
        timestamp = (
            frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        )
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)

        for key in frame:
            if key not in self.features:
                raise ValueError(key)
            item = (
                frame[key].numpy()
                if isinstance(frame[key], torch.Tensor)
                else frame[key]
            )
            self.episode_buffer[key].append(item)

        self.episode_buffer["size"] += 1

def main(tasks_dir: str, output_dir: str,  fps: int = 20):
    """
    将多个任务的 HDF5 文件夹转换为 LeRobot 数据集格式，并统一排序。

    :param tasks_dir: 包含多个任务文件夹的根目录，每个任务文件夹对应一个任务。
    :param output_dir: 输出目录路径。
    :param fps: 视频帧率。
    """
    # 创建输出数据集
    dataset = AgiBotDataset.create(
        repo_id="multi_task_dataset",
        root=output_dir,
        fps=fps,
        robot_type="a2d",
        features=FEATURES,
    )

     # 遍历任务文件夹
    tasks_dir_path = Path(tasks_dir)
    task_folders = [folder for folder in tasks_dir_path.iterdir() if folder.is_dir()]
    if not task_folders:
        raise ValueError(f"No task folders found in directory: {tasks_dir}")

     # 全局 episode 索引
    global_episode_index = 0

    for task_index, task_folder in enumerate(sorted(task_folders)):
        task_name = task_folder.name
        print(f"Processing task: {task_name} (task_index: {task_index})")

        # 遍历任务文件夹下的所有 HDF5 文件
        hdf5_files = sorted(task_folder.glob("*.hdf5"))
        if not hdf5_files:
            print(f"No HDF5 files found in task folder: {task_folder}")
            continue


        for hdf5_file in tqdm(hdf5_files, desc=f"Loading HDF5 files for task {task_name}"):
            print(f"Loading {hdf5_file}")
            # 加载 HDF5 文件中的数据
            raw_dataset = load_local_dataset(str(hdf5_file))

            # 添加帧数据到数据集
            for frame in tqdm(raw_dataset[0], desc=f"Adding frames to {hdf5_file.name}"):
                dataset.add_frame(frame)

            # 保存 episode，使用全局 episode 索引
            dataset.save_episode(
                task=task_name,
                videos=raw_dataset[1],
                global_episode_index=global_episode_index,
            )

            # 更新全局 episode 索引
            global_episode_index += 1

    # 整理数据集
    dataset.consolidate()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="将 Realman 数据集转换为 LeRobot 数据集格式")
    parser.add_argument("--tasks_dir", type=str, required=True, help="Realman 数据集的 HDF5 文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录路径")
    parser.add_argument("--fps", type=int, default=10, help="视频帧率")
    args = parser.parse_args()

    main(
        tasks_dir=args.tasks_dir,
        output_dir=args.output_dir,
        fps=args.fps,
    )
# python mydataset2lerobot_all_task.py --tasks_dir /home/jdtech/Documents/dataset_convert/dataset --output_dir /home/jdtech/Documents/dataset_convert/dataset_lerobot  --fps 20