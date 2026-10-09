"""Local inference adapter for the bundled MPG point-cloud model."""

import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import torch


RUNTIME_DIR = Path(__file__).resolve().parent / "mpg_runtime"
if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))

import environment as env
import transformations as tra
from config import DEVICE
from model import Agent, action_from_logits, load
from augmentation import FPSResampler, Normalize


CHECKPOINT_PATH = Path(__file__).resolve().parent / "weights" / "weights.zip"
NUM_POINTS = 2048
SOURCE_POINTS = 1280
INFERENCE_STEPS = 10


def _raw_transform(normalized_transform, center, scale):
    transform = np.eye(4, dtype=np.float64)
    rotation = normalized_transform[:3, :3]
    transform[:3, :3] = rotation
    transform[:3, 3] = center - rotation @ center + scale * normalized_transform[:3, 3]
    return transform


def _restore_centroid_alignment(aligned_transform, source_alignment, target_alignment):
    """Map an aligned-cloud transform back to the original input frames."""
    return np.linalg.inv(target_alignment) @ aligned_transform @ source_alignment


def register_point_clouds(source_path, target_path, checkpoint_path=CHECKPOINT_PATH,
                          update_callback=None):
    """Return a raw-coordinate source-to-target transform and elapsed time."""
    source_cloud = o3d.io.read_point_cloud(str(source_path))
    target_cloud = o3d.io.read_point_cloud(str(target_path))
    source_points = np.asarray(source_cloud.points, dtype=np.float32)
    target_points = np.asarray(target_cloud.points, dtype=np.float32)
    if len(source_points) < 3 or len(target_points) < 3:
        raise ValueError("Source and target point clouds must each contain at least 3 points.")
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"MPG checkpoint not found: {checkpoint_path}")

    # Preserve each input frame separately; MPG only sees centered clouds.
    source_pre_alignment = np.eye(4, dtype=np.float64)
    source_pre_alignment[:3, 3] = -source_points.mean(axis=0)
    target_pre_alignment = np.eye(4, dtype=np.float64)
    target_pre_alignment[:3, 3] = -target_points.mean(axis=0)
    source_aligned = (
        source_points @ source_pre_alignment[:3, :3].T + source_pre_alignment[:3, 3]
    ).astype(np.float32)
    target_aligned = (
        target_points @ target_pre_alignment[:3, :3].T + target_pre_alignment[:3, 3]
    ).astype(np.float32)

    np.random.seed(666)
    torch.manual_seed(666)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(666)
    sample = {
        "points_src": source_aligned,
        "points_ref": target_aligned,
        "crop_proportion": [SOURCE_POINTS / NUM_POINTS],
    }
    sample = FPSResampler(NUM_POINTS)(sample)
    sample = Normalize()(sample)
    scale = float(sample["normalization"][0, 0])
    center = sample["normalization"][:3, 3].astype(np.float64)

    source = torch.from_numpy(sample["points_src"][:, :3].copy()).unsqueeze(0).to(DEVICE)
    target = torch.from_numpy(sample["points_ref"][:, :3].copy()).unsqueeze(0).to(DEVICE)
    agent = Agent().to(DEVICE)
    load(agent, checkpoint_path)
    agent.eval()

    started = time.perf_counter()
    with torch.inference_mode():
        target_features = agent.prepare_target(target, detach=True)
        pose = torch.eye(4, device=DEVICE).unsqueeze(0)
        current = source
        for _ in range(INFERENCE_STEPS):
            _, logits, _, _ = agent.forward_with_target(current, target_features)
            current, pose = env.step(source, action_from_logits(logits, deterministic=True), pose, True)
            if update_callback is not None:
                global_pose = tra.to_global(pose.clone(), source)
                normalized = np.eye(4, dtype=np.float64)
                normalized[:, :] = global_pose[0].cpu().numpy()
                aligned_transform = _raw_transform(normalized, center, scale)
                update_callback(_restore_centroid_alignment(
                    aligned_transform, source_pre_alignment, target_pre_alignment
                ))
        global_pose = tra.to_global(pose, source)
        normalized = np.eye(4, dtype=np.float64)
        normalized[:, :] = global_pose[0].cpu().numpy()

    aligned_transform = _raw_transform(normalized, center, scale)
    transform = _restore_centroid_alignment(
        aligned_transform, source_pre_alignment, target_pre_alignment
    )
    return transform, time.perf_counter() - started
