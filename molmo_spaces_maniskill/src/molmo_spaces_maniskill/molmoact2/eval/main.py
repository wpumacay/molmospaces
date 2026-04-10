#!/usr/bin/env python3
"""
LeRobot Policy Evaluation on ManiSkill Environments (Closed-Loop)

This script loads a pretrained LeRobot policy (ACT, Diffusion, VQ-BeT, etc.)
and evaluates it on custom MolmoAct2 ManiSkill tasks in closed-loop.

Usage:
    python -m molmo_spaces_maniskill.molmoact2.eval.main \
        --checkpoint /path/to/checkpoint \
        --env-id SO100PushCubeSlot-v1 \
        --n-episodes 10

    # Or with HuggingFace model
    python -m molmo_spaces_maniskill.molmoact2.eval.main \
        --checkpoint lerobot/act_so100_push_cube \
        --env-id SO100PushCubeSlot-v1
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Optional

import gymnasium as gym
import numpy as np
import torch
import tyro
from tqdm import tqdm

import mani_skill.envs  # noqa: F401
from molmo_spaces_maniskill.molmoact2.tasks import *  # noqa: F401, F403

try:
    from .common import (
        AVAILABLE_TASKS,
        DEFAULT_TASK_DESCRIPTIONS,
        DEFAULT_OUTPUT_DIR,
        LeRobotPolicyWrapper,
        get_default_robot_uid,
        save_video,
        save_image,
        extract_input_frames,
    )
except ImportError:
    from common import (
        AVAILABLE_TASKS,
        DEFAULT_TASK_DESCRIPTIONS,
        DEFAULT_OUTPUT_DIR,
        LeRobotPolicyWrapper,
        get_default_robot_uid,
        save_video,
        save_image,
        extract_input_frames,
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Configuration for LeRobot policy evaluation on ManiSkill."""
    
    checkpoint: Annotated[str, tyro.conf.arg(aliases=["-c"])]
    """Path to the policy checkpoint (local dir or HuggingFace repo)"""
    
    env_id: Annotated[Optional[str], tyro.conf.arg(aliases=["-e"])] = None
    """Environment ID to evaluate. If None, runs on all tasks."""
    
    task_group: Annotated[Optional[str], tyro.conf.arg(aliases=["-g"])] = None
    """Task group to evaluate: 'so100', 'droid', 'yam'. If None, uses env_id."""
    
    robot_uid: Annotated[str, tyro.conf.arg(aliases=["-r"])] = "yam_bimanual"
    """Robot UID for the environment"""
    
    n_episodes: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 10
    """Number of episodes to evaluate per task"""
    
    max_episode_steps: int = 800
    """Maximum steps per episode"""
    
    seed: int = 42
    """Random seed"""
    
    device: str = "cuda"
    """Device for policy inference"""
    
    use_amp: bool = False
    """Use automatic mixed precision"""
    
    camera_names: list[str] = field(default_factory=lambda: ["base_camera"])
    """Camera names to use for observations"""
    
    image_size: tuple[int, int] = (256, 256)
    """Image size (height, width) for policy input"""
    
    output_dir: str = DEFAULT_OUTPUT_DIR
    """Output directory for results"""
    
    save_video: bool = True
    """Save evaluation videos"""
    
    max_videos: int = 10
    """Maximum number of videos to save per task"""
    
    render: bool = False
    """Render environment during evaluation"""
    
    verbose: bool = True
    """Verbose output"""
    
    task_description: Optional[str] = None
    """Task description for VLA models. If None, uses default based on env_id."""

    chunk_size: Optional[int] = None
    """Override policy chunk_size. If None, uses the value from checkpoint config."""
    
    n_action_steps: Optional[int] = None
    """Override policy n_action_steps. If None, uses the value from checkpoint config."""


def _capture_frame(env: gym.Env) -> np.ndarray | None:
    """Capture a frame for video recording."""
    try:
        frame = env.unwrapped.render_rgb_array()
        if frame is not None:
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            if frame.ndim == 4 and frame.shape[0] == 1:
                frame = frame[0]
            return frame.copy()
    except Exception:
        pass
    
    try:
        frame = env.render()
        if frame is not None and not hasattr(frame, 'window'):
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            if hasattr(frame, 'ndim'):
                if frame.ndim == 4 and frame.shape[0] == 1:
                    frame = frame[0]
                return frame.copy()
    except Exception:
        pass
    
    return None


def evaluate_episode(
    env: gym.Env,
    policy: LeRobotPolicyWrapper,
    obs: dict,
    max_steps: int,
    render: bool = False,
    save_input_frame: bool = True,
) -> dict:
    """Run one evaluation episode."""
    policy.reset()
    
    total_reward = 0.0
    success = False
    frames = []
    input_frames = {}
    
    if save_input_frame:
        input_frames = extract_input_frames(obs)
    
    if render:
        env.unwrapped.render_human()
    
    frame = _capture_frame(env)
    if frame is not None:
        frames.append(frame)
    
    for step in range(max_steps):
        action = policy.get_action(obs, from_maniskill=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if isinstance(reward, torch.Tensor):
            reward = reward.item() if reward.numel() == 1 else reward.sum().item()
        total_reward += reward
        
        success_val = info.get("success", False)
        if isinstance(success_val, torch.Tensor):
            success_val = success_val.item() if success_val.numel() == 1 else success_val.any().item()
        if success_val:
            success = True
        
        if isinstance(terminated, torch.Tensor):
            terminated = terminated.item() if terminated.numel() == 1 else terminated.any().item()
        if isinstance(truncated, torch.Tensor):
            truncated = truncated.item() if truncated.numel() == 1 else truncated.any().item()
        
        if render:
            env.unwrapped.render_human()
        
        frame = _capture_frame(env)
        if frame is not None:
            frames.append(frame)
        
        if terminated or truncated:
            break

    return {
        "total_reward": total_reward,
        "success": success,
        "steps": step + 1,
        "frames": frames,
        "input_frames": input_frames,
    }


def evaluate_task(
    env_id: str,
    policy: LeRobotPolicyWrapper,
    config: EvalConfig,
) -> dict:
    """Evaluate policy on a single task."""
    logger.info(f"Evaluating on {env_id}")
    
    task_desc = config.task_description or DEFAULT_TASK_DESCRIPTIONS.get(env_id, env_id)
    policy.set_task_description(task_desc)
    logger.info(f"Task description: {task_desc}")
    
    robot_uid = config.robot_uid
    default_robot = get_default_robot_uid(env_id)
    if robot_uid != default_robot:
        logger.info(f"Using robot: {robot_uid} (task default: {default_robot})")
    else:
        robot_uid = default_robot
        logger.info(f"Using robot: {robot_uid}")
    
    render_mode = "human" if config.render else "rgb_array"
    env = gym.make(
        env_id,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode=render_mode,
        robot_uids=robot_uid,
        max_episode_steps=config.max_episode_steps,
        reward_mode="none",  # Evaluation doesn't need dense reward
    )
    
    viewer = None
    viewer_initialized = False
    
    results = {
        "env_id": env_id,
        "episodes": [],
        "success_rate": 0.0,
        "avg_reward": 0.0,
        "avg_steps": 0.0,
    }
    
    successes = []
    rewards = []
    steps_list = []
    
    pbar = tqdm(range(config.n_episodes), desc=f"Eval {env_id}", leave=False)
    for ep_idx in pbar:
        seed = config.seed + ep_idx
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        obs, info = env.reset(seed=seed)
        
        # Print available cameras on first episode
        if ep_idx == 0:
            policy.print_available_cameras(obs)
        
        if config.render and not viewer_initialized:
            viewer = env.unwrapped.render_human()
            viewer_initialized = True
        
        ep_result = evaluate_episode(
            env=env,
            policy=policy,
            obs=obs,
            max_steps=config.max_episode_steps,
            render=config.render,
        )
        
        results["episodes"].append({
            "episode_idx": ep_idx,
            "success": ep_result["success"],
            "total_reward": float(ep_result["total_reward"]),
            "steps": ep_result["steps"],
        })
        
        successes.append(ep_result["success"])
        rewards.append(ep_result["total_reward"])
        steps_list.append(ep_result["steps"])
        
        if config.save_video and ep_idx < config.max_videos and len(ep_result["frames"]) > 0:
            save_video(
                frames=ep_result["frames"],
                output_dir=Path(config.output_dir) / "videos" / env_id,
                filename=f"episode_{ep_idx}.mp4",
            )
        
        if ep_result["input_frames"]:
            for cam_name, frame in ep_result["input_frames"].items():
                save_image(
                    frame=frame,
                    output_dir=Path(config.output_dir) / "input_frames" / env_id,
                    filename=f"episode_{ep_idx}_{cam_name}.png",
                )
        
        current_sr = np.mean(successes) * 100
        pbar.set_postfix({"success_rate": f"{current_sr:.1f}%"})
    
    if viewer is not None:
        viewer.close()
    env.close()
    
    results["success_rate"] = float(np.mean(successes))
    results["avg_reward"] = float(np.mean(rewards))
    results["avg_steps"] = float(np.mean(steps_list))
    
    logger.info(
        f"{env_id}: Success Rate = {results['success_rate']*100:.1f}%, "
        f"Avg Reward = {results['avg_reward']:.2f}, "
        f"Avg Steps = {results['avg_steps']:.1f}"
    )
    
    return results


def save_results(results: dict, output_dir: Path):
    """Save evaluation results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "eval_results.json"
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {results_path}")


def main():
    config = tyro.cli(EvalConfig)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir) / timestamp
    config.output_dir = str(output_dir)
    
    logger.info("=" * 60)
    logger.info("LeRobot Policy Evaluation on ManiSkill (Closed-Loop)")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {config.checkpoint}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Output: {output_dir}")
    
    policy = LeRobotPolicyWrapper(
        checkpoint_path=config.checkpoint,
        camera_names=config.camera_names,
        image_size=config.image_size,
        device=config.device,
    )
    policy.load()
    

    if hasattr(policy.config, 'chunk_size'):
        logger.info(f"Policy chunk size: {policy.config.chunk_size}")
    if hasattr(policy.config, 'n_action_steps'):
        logger.info(f"Policy n_action_steps: {policy.config.n_action_steps}")

    # Override chunk_size if specified
    if config.chunk_size is not None:
        if hasattr(policy.config, 'chunk_size'):
            old_chunk_size = policy.config.chunk_size
            policy.config.chunk_size = config.chunk_size
            logger.info(f"Overriding chunk_size: {old_chunk_size} -> {policy.config.chunk_size}")
    if config.n_action_steps is not None:
        if hasattr(policy.config, 'n_action_steps'):
            old_n_action_steps = policy.config.n_action_steps
            policy.config.n_action_steps = config.n_action_steps
            logger.info(f"Overriding n_action_steps: {old_n_action_steps} -> {policy.config.n_action_steps}")
    
    if config.env_id:
        task_list = [config.env_id]
    elif config.task_group:
        if config.task_group not in AVAILABLE_TASKS:
            raise ValueError(f"Unknown task group: {config.task_group}")
        task_list = AVAILABLE_TASKS[config.task_group]
    else:
        task_list = []
        for tasks in AVAILABLE_TASKS.values():
            task_list.extend(tasks)
    
    logger.info(f"Evaluating on {len(task_list)} tasks: {task_list}")
    
    all_results = {
        "config": {
            "checkpoint": config.checkpoint,
            "n_episodes": config.n_episodes,
            "max_episode_steps": config.max_episode_steps,
            "seed": config.seed,
        },
        "tasks": {},
        "overall": {},
    }
    
    all_successes = []
    all_rewards = []
    
    for env_id in task_list:
        try:
            task_results = evaluate_task(env_id, policy, config)
            all_results["tasks"][env_id] = task_results
            all_successes.append(task_results["success_rate"])
            all_rewards.append(task_results["avg_reward"])
        except Exception as e:
            logger.error(f"Failed to evaluate {env_id}: {e}")
            if config.verbose:
                import traceback
                traceback.print_exc()
    
    if all_successes:
        all_results["overall"] = {
            "mean_success_rate": float(np.mean(all_successes)),
            "mean_reward": float(np.mean(all_rewards)),
            "num_tasks": len(all_successes),
        }
        
        logger.info("=" * 60)
        logger.info("Overall Results")
        logger.info("=" * 60)
        logger.info(f"Mean Success Rate: {all_results['overall']['mean_success_rate']*100:.1f}%")
        logger.info(f"Mean Reward: {all_results['overall']['mean_reward']:.2f}")
    
    save_results(all_results, output_dir)
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
