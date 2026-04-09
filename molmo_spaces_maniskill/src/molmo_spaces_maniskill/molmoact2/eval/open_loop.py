#!/usr/bin/env python3
"""
Open-Loop Policy Evaluation on LeRobot Datasets

This script evaluates a policy in open-loop mode by:
1. Loading a LeRobot dataset from local path
2. For each episode, feeding images at chunk boundaries to the policy
3. Comparing predicted actions with ground truth actions
4. Plotting GT vs predicted action curves

Usage:
    python -m molmo_spaces_maniskill.molmoact2.eval.open_loop \
        --checkpoint /path/to/checkpoint \
        --dataset-path /home/shuo/research/datasets/molmoact2/sim_bench/yam/maniskill_BimanualYAMPickPlace \
        --n-episodes 5
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from tqdm import tqdm

sys.path.insert(0, "/home/shuo/research/lerobot/src")
from lerobot.datasets.lerobot_dataset import LeRobotDataset

try:
    from .common import (
        EVAL_DIR,
        LeRobotPolicyWrapper,
        save_image,
    )
except ImportError:
    from common import (
        EVAL_DIR,
        LeRobotPolicyWrapper,
        save_image,
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = str(EVAL_DIR / "outputs" / "open_loop")


@dataclass
class OpenLoopConfig:
    """Configuration for open-loop policy evaluation."""
    
    checkpoint: Annotated[str, tyro.conf.arg(aliases=["-c"])]
    """Path to the policy checkpoint (local dir or HuggingFace repo)"""
    
    dataset_path: Annotated[str, tyro.conf.arg(aliases=["-d"])]
    """Path to the LeRobot dataset directory (containing meta/info.json)"""
    
    n_episodes: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 5
    """Number of episodes to evaluate"""
    
    action_chunk_size: Optional[int] = 8
    """Action chunk size. If None, read from policy config."""
    
    device: str = "cuda"
    """Device for policy inference"""
    
    output_dir: str = DEFAULT_OUTPUT_DIR
    """Output directory for plots"""
    
    image_size: tuple[int, int] = (256, 256)
    """Image size (height, width) for policy input"""
    
    seed: int = 42
    """Random seed"""
    
    task_description: Optional[str] = None
    """Task description for VLA models"""
    
    camera_names: list[str] = field(default_factory=lambda: ["base_camera"])
    """Camera names to use for policy input"""
    
    save_images: bool = True
    """Save input images at chunk boundaries"""


def load_episode_data(dataset: LeRobotDataset, episode_idx: int) -> dict:
    """
    Load all frames for a specific episode.
    
    Returns:
        dict with:
            - samples: list of dataset samples
            - actions: (T, action_dim) numpy array of GT actions
            - states: (T, state_dim) numpy array of states
    """
    ep_dataset = LeRobotDataset(
        repo_id=dataset.meta.repo_id,
        root=dataset.root,
        episodes=[episode_idx],
        download_videos=False,
    )
    
    samples = []
    actions = []
    states = []
    
    for i in range(len(ep_dataset)):
        sample = ep_dataset[i]
        samples.append(sample)
        actions.append(sample["action"].numpy())
        if "observation.state" in sample:
            states.append(sample["observation.state"].numpy())
    
    return {
        "samples": samples,
        "actions": np.array(actions),
        "states": np.array(states) if states else None,
    }


def evaluate_episode_open_loop(
    policy: LeRobotPolicyWrapper,
    episode_data: dict,
    chunk_size: int,
) -> dict:
    """
    Evaluate policy in open-loop mode on a single episode.
    
    At each chunk boundary (every chunk_size steps), feed the image to the policy
    and collect the predicted action chunk.
    
    Returns:
        dict with:
            - gt_actions: (T, action_dim) ground truth actions
            - pred_actions: (T, action_dim) predicted actions
            - chunk_indices: list of frame indices where chunks start
            - chunk_images: list of images at chunk boundaries
            - inference_times: list of inference times per chunk
    """
    import time
    
    policy.reset()
    
    samples = episode_data["samples"]
    gt_actions = episode_data["actions"]
    T, action_dim = gt_actions.shape
    
    pred_actions = np.zeros_like(gt_actions)
    chunk_indices = []
    chunk_images = []
    inference_times = []
    
    t = 0
    n_chunks = 0
    while t < T:
        chunk_indices.append(t)
        sample = samples[t]
        
        for key in sample:
            if key.startswith("observation.images."):
                img = sample[key]
                if isinstance(img, torch.Tensor):
                    img = img.permute(1, 2, 0).numpy()
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                chunk_images.append({"frame_idx": t, "camera": key, "image": img})
                break
        
        t0 = time.perf_counter()
        action_chunk = policy.get_action_chunk(sample)
        t1 = time.perf_counter()
        inference_times.append(t1 - t0)
        n_chunks += 1
        
        chunk_len = min(chunk_size, T - t)
        pred_chunk = action_chunk[:chunk_len]
        
        if pred_chunk.shape[0] < chunk_len:
            last_action = pred_chunk[-1:]
            pad_len = chunk_len - pred_chunk.shape[0]
            pred_chunk = np.concatenate([pred_chunk, np.tile(last_action, (pad_len, 1))], axis=0)
        
        pred_actions[t:t+chunk_len] = pred_chunk
        t += chunk_size
    
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    logger.info(f"    Policy called {n_chunks} times, avg inference: {avg_inference_time*1000:.1f}ms")
    
    return {
        "gt_actions": gt_actions,
        "pred_actions": pred_actions,
        "chunk_indices": chunk_indices,
        "chunk_images": chunk_images,
        "inference_times": inference_times,
    }


def plot_actions(
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
    chunk_indices: list[int],
    output_path: Path,
    episode_idx: int,
    action_names: list[str] | None = None,
):
    """
    Plot ground truth vs predicted actions.
    
    Creates a figure with subplots for each action dimension.
    Marks chunk boundaries with vertical lines.
    """
    T, action_dim = gt_actions.shape
    
    n_cols = 2
    n_rows = (action_dim + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows), sharex=True)
    axes = axes.flatten() if action_dim > 1 else [axes]
    
    timesteps = np.arange(T)
    
    for i in range(action_dim):
        ax = axes[i]
        
        ax.plot(timesteps, gt_actions[:, i], 'b-', label='GT', linewidth=1.5, alpha=0.8)
        ax.plot(timesteps, pred_actions[:, i], 'r--', label='Pred', linewidth=1.5, alpha=0.8)
        
        for ci in chunk_indices:
            ax.axvline(x=ci, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
        
        name = action_names[i] if action_names and i < len(action_names) else f"action_{i}"
        ax.set_ylabel(name, fontsize=9)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        mse = np.mean((gt_actions[:, i] - pred_actions[:, i]) ** 2)
        ax.set_title(f"{name} (MSE: {mse:.4f})", fontsize=10)
    
    for i in range(action_dim, len(axes)):
        axes[i].set_visible(False)
    
    axes[-1].set_xlabel("Timestep", fontsize=10)
    if action_dim > 1:
        axes[-2].set_xlabel("Timestep", fontsize=10)
    
    fig.suptitle(f"Episode {episode_idx}: GT vs Predicted Actions (chunk boundaries in gray)", fontsize=12)
    plt.tight_layout()
    
    output_path.mkdir(parents=True, exist_ok=True)
    fig_path = output_path / f"episode_{episode_idx}_actions.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved action plot to {fig_path}")
    return fig_path


def save_chunk_images(
    chunk_images: list[dict],
    output_path: Path,
    episode_idx: int,
):
    """Save images at chunk boundaries."""
    img_dir = output_path / f"episode_{episode_idx}_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    for i, img_data in enumerate(chunk_images):
        frame_idx = img_data["frame_idx"]
        img = img_data["image"]
        save_image(
            frame=img,
            output_dir=img_dir,
            filename=f"chunk_{i:03d}_frame_{frame_idx:04d}.png",
        )
    
    logger.info(f"Saved {len(chunk_images)} chunk images to {img_dir}")


def main(config: OpenLoopConfig):
    """Main evaluation function."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    logger.info("=" * 60)
    logger.info("Open-Loop Policy Evaluation on LeRobot Dataset")
    logger.info("=" * 60)
    
    logger.info(f"Loading dataset from: {config.dataset_path}")
    dataset = LeRobotDataset(
        repo_id=Path(config.dataset_path).name,
        root=config.dataset_path,
        download_videos=False,
    )
    logger.info(f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames, {dataset.fps} fps")
    
    # Print task instructions from dataset
    if hasattr(dataset, 'meta') and hasattr(dataset.meta, 'tasks'):
        logger.info(f"Task instructions in dataset:")
        for task_idx, task_desc in dataset.meta.tasks.items():
            logger.info(f"  [{task_idx}]: {task_desc}")
    elif len(dataset) > 0 and "task" in dataset[0]:
        task_desc = dataset[0]["task"]
        logger.info(f"Task instruction: {task_desc}")
    
    # Print available cameras in dataset
    if len(dataset) > 0:
        sample = dataset[0]
        available_cams = [k.replace("observation.images.", "") 
                         for k in sample.keys() if k.startswith("observation.images.")]
        logger.info(f"Available cameras in dataset: {available_cams}")
    
    policy = LeRobotPolicyWrapper(
        checkpoint_path=config.checkpoint,
        camera_names=config.camera_names,
        image_size=config.image_size,
        device=config.device,
        task_description=config.task_description,
    )
    policy.load()
    logger.info(f"Policy loaded with chunk size: {policy.get_chunk_size()}")
    
    chunk_size = config.action_chunk_size or policy.get_chunk_size()
    logger.info(f"Using action chunk size: {chunk_size}")
    
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    n_episodes = min(config.n_episodes, dataset.num_episodes)
    all_mse = []
    
    for ep_idx in tqdm(range(n_episodes), desc="Evaluating episodes"):
        logger.info(f"\nEvaluating episode {ep_idx}")
        
        episode_data = load_episode_data(dataset, ep_idx)
        logger.info(f"  Episode length: {len(episode_data['samples'])} frames")
        
        result = evaluate_episode_open_loop(
            policy=policy,
            episode_data=episode_data,
            chunk_size=chunk_size,
        )
        
        plot_actions(
            gt_actions=result["gt_actions"],
            pred_actions=result["pred_actions"],
            chunk_indices=result["chunk_indices"],
            output_path=output_path,
            episode_idx=ep_idx,
        )
        
        if config.save_images:
            save_chunk_images(
                chunk_images=result["chunk_images"],
                output_path=output_path,
                episode_idx=ep_idx,
            )
        
        mse = np.mean((result["gt_actions"] - result["pred_actions"]) ** 2)
        all_mse.append(mse)
        logger.info(f"  Episode MSE: {mse:.6f}")
    
    avg_mse = np.mean(all_mse)
    logger.info(f"\n{'='*60}")
    logger.info(f"Open-loop evaluation complete")
    logger.info(f"  Episodes evaluated: {n_episodes}")
    logger.info(f"  Action chunk size: {chunk_size}")
    logger.info(f"  Average MSE: {avg_mse:.6f}")
    logger.info(f"  Results saved to: {output_path}")
    
    summary = {
        "dataset_path": config.dataset_path,
        "checkpoint": config.checkpoint,
        "n_episodes": n_episodes,
        "chunk_size": chunk_size,
        "avg_mse": float(avg_mse),
        "per_episode_mse": [float(m) for m in all_mse],
    }
    
    summary_path = output_path / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Summary saved to: {summary_path}")
    
    return summary


if __name__ == "__main__":
    config = tyro.cli(OpenLoopConfig)
    main(config)
