#!/usr/bin/env python3
"""
Common utilities for LeRobot policy evaluation.

This module contains shared components used by both closed-loop (main.py) 
and open-loop (open_loop.py) evaluation scripts.
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, "/home/shuo/research/lerobot/src")

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).parent
DEFAULT_OUTPUT_DIR = str(EVAL_DIR / "outputs")

AVAILABLE_TASKS = {
    "so100": [
        "SO100PushCubeSlot-v1",
        "SO100CloseMicrowave-v1",
    ],
    "droid": [
        "DroidKitchenOpenDrawerPnpFork-v1",
        "DroidKitchenOpenDrawerMem-v1",
        "DroidKitchenPourBottleCup-v1",
    ],
    "yam": [
        "BimanualYAMPickPlace-v1",
        "BimanualYAMLiftPot-v1",
        "BimanualYAMInsert-v1",
        "BimanualYAMLFlipPlate-v1",
        "BimanualYAMMicrowave-v1",
    ],
}

DEFAULT_ROBOT_UIDS = {
    "so100": "so100_wristcam",
    "droid": "fr3_robotiq_wristcam",
    "yam": "yam_bimanual",
}

DEFAULT_TASK_DESCRIPTIONS = {
    "SO100PushCubeSlot-v1": "Push the cube into the slot",
    "SO100CloseMicrowave-v1": "Close the microwave door",
    "DroidKitchenOpenDrawerPnpFork-v1": "open top drawer and put the fork in the pan",
    "DroidKitchenOpenDrawerMem-v1": "Open the drawer",
    "DroidKitchenPourBottleCup-v1": "Pour from the bottle into the cup",
    "BimanualYAMPickPlace-v1": "Pick the red cube and place it to the goal",
    "BimanualYAMLiftPot-v1": "Lift the pot with both arms",
    "BimanualYAMInsert-v1": "Insert the peg into the hole",
    "BimanualYAMLFlipPlate-v1": "Flip the plate on the table.",
    "BimanualYAMMicrowave-v1": "Open the microwave",
}


def get_task_group(env_id: str) -> str | None:
    """Get the task group for an environment ID."""
    for group, tasks in AVAILABLE_TASKS.items():
        if env_id in tasks:
            return group
    return None


def get_default_robot_uid(env_id: str) -> str:
    """Get the default robot UID for an environment ID."""
    group = get_task_group(env_id)
    if group and group in DEFAULT_ROBOT_UIDS:
        return DEFAULT_ROBOT_UIDS[group]
    return "yam_bimanual"


@dataclass
class BasePolicyConfig:
    """Base configuration for policy loading."""
    
    checkpoint: str
    """Path to the policy checkpoint (local dir or HuggingFace repo)"""
    
    device: str = "cuda"
    """Device for policy inference"""
    
    camera_names: list[str] = field(default_factory=lambda: ["base_camera"])
    """Camera names to use for observations"""
    
    image_size: tuple[int, int] = (256, 256)
    """Image size (height, width) for policy input"""
    
    task_description: Optional[str] = None
    """Task description for VLA models"""


class LeRobotPolicyWrapper:
    """
    Wrapper to interface LeRobot policies with various input formats.
    
    Handles:
    - Policy loading from checkpoint
    - Observation preprocessing (ManiSkill or LeRobot dataset format)
    - Action postprocessing
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        camera_names: list[str] | None = None,
        image_size: tuple[int, int] | None = None,
        device: str = "cuda",
        task_description: str | None = None,
        camera_mapping: dict[str, str] | None = None,
    ):
        """
        Args:
            checkpoint_path: Path to checkpoint (HF repo or local path)
            camera_names: Camera names in ManiSkill env. If None, auto-detect from config.
            image_size: Target image size. If None, use from config.
            device: Device to run on
            task_description: Task description for VLM policies
            camera_mapping: Map ManiSkill camera names to policy camera names.
                           e.g., {"hand_camera": "left_hand_camera"}
        """
        self.checkpoint_path = checkpoint_path
        self.camera_names = camera_names or []
        self.image_size = image_size
        self.device = device
        self.task_description = task_description
        self.camera_mapping = camera_mapping or {}
        
        self.policy = None
        self.config = None
        self.preprocessor = None
        self.postprocessor = None
        self.policy_camera_names = []  # Camera names expected by policy
        
    def _parse_checkpoint_path(self) -> tuple[str, str | None]:
        """Parse checkpoint path to extract repo_id and optional subfolder.
        
        Supports:
        - "namespace/repo" -> ("namespace/repo", None)
        - "namespace/repo/subfolder" -> ("namespace/repo", "subfolder")
        - "namespace/repo/path/to/subfolder" -> ("namespace/repo", "path/to/subfolder")
        - Local paths (starts with / or .) -> (path, None)
        """
        path = self.checkpoint_path
        
        # Check if it's a local path
        if path.startswith("/") or path.startswith(".") or Path(path).exists():
            return path, None
        
        # Parse HuggingFace path: namespace/repo[/subfolder...]
        parts = path.split("/")
        if len(parts) < 2:
            return path, None
        elif len(parts) == 2:
            # Standard repo format: namespace/repo
            return path, None
        else:
            # Has subfolder: namespace/repo/subfolder/...
            repo_id = f"{parts[0]}/{parts[1]}"
            subfolder = "/".join(parts[2:])
            return repo_id, subfolder
    
    def load(self):
        """Load the policy from checkpoint."""
        from huggingface_hub import snapshot_download
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        
        repo_id, subfolder = self._parse_checkpoint_path()
        
        if subfolder:
            logger.info(f"Loading policy from repo: {repo_id}, subfolder: {subfolder}")
        else:
            logger.info(f"Loading policy from: {repo_id}")
        
        # Determine local path for loading
        if repo_id.startswith("/") or repo_id.startswith(".") or Path(repo_id).exists():
            # Local path
            local_path = repo_id
        else:
            # Download from HuggingFace Hub
            # If subfolder specified, download only that subfolder
            if subfolder:
                local_path = snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=[f"{subfolder}/*"],
                )
                local_path = Path(local_path) / subfolder
            else:
                local_path = snapshot_download(repo_id=repo_id)
        
        local_path = Path(local_path)
        logger.info(f"Using local path: {local_path}")
        
        # Load config from local path
        self.config = PreTrainedConfig.from_pretrained(str(local_path))
        self.config.device = self.device
        self.config.pretrained_path = local_path
        
        # Extract camera names and image size from config
        self._extract_config_info()
        
        policy_cls = get_policy_class(self.config.type)
        
        # Load policy from local path
        self.policy = policy_cls.from_pretrained(
            str(local_path),
            config=self.config,
        )
        
        self.policy.to(self.device)
        self.policy.eval()
        
        # Create pre/post processors
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.config,
            pretrained_path=str(local_path),
            preprocessor_overrides={
                "device_processor": {"device": self.device},
            },
        )
        
        logger.info(f"Loaded {self.config.type} policy successfully")
        logger.info(f"Policy expects cameras: {self.policy_camera_names}")
        logger.info(f"ManiSkill cameras: {self.camera_names}")
        logger.info(f"Camera mapping: {self.camera_mapping}")
        return self
    
    def _extract_config_info(self):
        """Extract camera names and image size from policy config."""
        # Get camera names from input_features or image_features
        input_features = getattr(self.config, "input_features", {})
        image_features = getattr(self.config, "image_features", [])
        
        # Extract from input_features (pi05 format)
        for key, value in input_features.items():
            if key.startswith("observation.images."):
                cam_name = key.replace("observation.images.", "")
                self.policy_camera_names.append(cam_name)
        
        # Or from image_features list (diffusion format)
        if not self.policy_camera_names and image_features:
            for feat in image_features:
                if feat.startswith("observation.images."):
                    cam_name = feat.replace("observation.images.", "")
                    self.policy_camera_names.append(cam_name)
        
        # Auto-setup camera_names if not provided
        if not self.camera_names:
            self.camera_names = self.policy_camera_names.copy()
            logger.info(f"Auto-detected camera names: {self.camera_names}")
        
        # Auto-setup camera_mapping for matching names
        for policy_cam in self.policy_camera_names:
            if policy_cam in self.camera_names and policy_cam not in self.camera_mapping.values():
                # Direct match - no mapping needed
                pass
        
        # Get image size from config
        if self.image_size is None:
            image_resolution = getattr(self.config, "image_resolution", None)
            if image_resolution:
                self.image_size = tuple(image_resolution)
            else:
                # Try to get from input_features shape
                for key, value in input_features.items():
                    if key.startswith("observation.images.") and "shape" in value:
                        shape = value["shape"]
                        if len(shape) == 3:  # (C, H, W)
                            self.image_size = (shape[1], shape[2])
                            break
            if self.image_size:
                logger.info(f"Auto-detected image size: {self.image_size}")
    
    def reset(self):
        """Reset policy state (call at episode start)."""
        if self.policy is not None:
            self.policy.reset()
    
    def set_task_description(self, task_description: str):
        """Set the task description for VLA models."""
        self.task_description = task_description
    
    def get_chunk_size(self) -> int:
        """Get the action chunk size from policy config."""
        if self.config is None:
            return 1
        if hasattr(self.config, 'chunk_size'):
            return self.config.chunk_size
        if hasattr(self.config, 'n_action_steps'):
            return self.config.n_action_steps
        return 1
    
    def _get_maniskill_camera_name(self, policy_cam_name: str) -> str:
        """Get ManiSkill camera name for a policy camera name."""
        # Check camera_mapping (maniskill_name -> policy_name)
        for maniskill_name, policy_name in self.camera_mapping.items():
            if policy_name == policy_cam_name:
                return maniskill_name
        # Direct match
        return policy_cam_name
    
    def _debug_print_obs_structure(self, obs: dict, prefix: str = ""):
        """Print observation structure for debugging."""
        for key, value in obs.items():
            if isinstance(value, dict):
                logger.info(f"{prefix}{key}/")
                self._debug_print_obs_structure(value, prefix + "  ")
            elif isinstance(value, (np.ndarray, torch.Tensor)):
                shape = value.shape if hasattr(value, 'shape') else 'unknown'
                logger.info(f"{prefix}{key}: {type(value).__name__} {shape}")
            else:
                logger.info(f"{prefix}{key}: {type(value).__name__}")
    
    def print_available_cameras(self, obs: dict):
        """Print all available cameras in the observation."""
        logger.info("=" * 50)
        logger.info("Available cameras in observation:")
        if "sensor_data" in obs:
            for cam_name, cam_data in obs["sensor_data"].items():
                if isinstance(cam_data, dict):
                    keys = list(cam_data.keys())
                    if "rgb" in cam_data:
                        shape = cam_data["rgb"].shape if hasattr(cam_data["rgb"], 'shape') else 'unknown'
                        logger.info(f"  - {cam_name}: rgb {shape}")
                    else:
                        logger.info(f"  - {cam_name}: {keys}")
                else:
                    shape = cam_data.shape if hasattr(cam_data, 'shape') else 'unknown'
                    logger.info(f"  - {cam_name}: {shape}")
        else:
            logger.info("  No sensor_data found in obs")
        logger.info(f"Policy expects: {self.policy_camera_names}")
        logger.info("=" * 50)
    
    def preprocess_maniskill_obs(self, obs: dict) -> dict:
        """
        Convert ManiSkill observation to LeRobot batch format.
        
        ManiSkill obs format (with obs_mode='rgb'):
            - agent/qpos: (B, n_joints) torch.Tensor
            - agent/qvel: (B, n_joints) torch.Tensor
            - sensor_data/{camera}/rgb: (B, H, W, 3) torch.Tensor
            
        Returns LeRobot batch format:
            - observation.state: (B, state_dim) float tensor
            - observation.images.{camera}: (B, C, H, W) float tensor [0, 1]
            - task: list[str] - task descriptions (for VLA models)
        """
        batch = {}
        batch_size = 1
        
        if "agent" in obs and "qpos" in obs["agent"]:
            qpos = obs["agent"]["qpos"]
            if isinstance(qpos, np.ndarray):
                qpos = torch.from_numpy(qpos).float()
            elif isinstance(qpos, torch.Tensor):
                qpos = qpos.float()
            if qpos.dim() == 1:
                qpos = qpos.unsqueeze(0)
            batch_size = qpos.shape[0]
            batch["observation.state"] = qpos.to(self.device)
        
        # Use policy_camera_names to ensure all required cameras are present
        cameras_to_use = self.policy_camera_names if self.policy_camera_names else self.camera_names
        
        missing_cameras = []
        for policy_cam_name in cameras_to_use:
            # Get ManiSkill camera name (may be mapped)
            maniskill_cam_name = self._get_maniskill_camera_name(policy_cam_name)
            
            img = None
            
            # Try to find image in ManiSkill obs
            if "sensor_data" in obs and maniskill_cam_name in obs["sensor_data"]:
                sensor_data = obs["sensor_data"][maniskill_cam_name]
                if isinstance(sensor_data, dict) and "rgb" in sensor_data:
                    img = sensor_data["rgb"]
                elif isinstance(sensor_data, (np.ndarray, torch.Tensor)):
                    img = sensor_data
            elif maniskill_cam_name in obs:
                img = obs[maniskill_cam_name]
            
            if img is not None:
                img = self._process_image(img)
                # Use policy camera name as key (what policy expects)
                batch[f"observation.images.{policy_cam_name}"] = img.to(self.device)
            else:
                missing_cameras.append((policy_cam_name, maniskill_cam_name))
        
        # Strict check: all required cameras must be present
        if missing_cameras:
            available_cams = list(obs.get("sensor_data", {}).keys())
            error_msg = (
                f"Missing required cameras!\n"
                f"  Policy requires: {cameras_to_use}\n"
                f"  Available in obs: {available_cams}\n"
                f"  Missing: {[f'{p} (maniskill: {m})' for p, m in missing_cameras]}\n"
                f"  Camera mapping: {self.camera_mapping}\n"
                f"Hint: Use camera_mapping to map ManiSkill camera names to policy camera names."
            )
            raise ValueError(error_msg)
        
        if self.task_description:
            batch["task"] = [self.task_description] * batch_size
        
        return batch
    
    def preprocess_dataset_sample(self, sample: dict) -> dict:
        """
        Convert a LeRobot dataset sample to policy input batch format.
        
        Dataset sample format:
            - observation.state: (state_dim,) tensor
            - observation.images.{camera}: (C, H, W) tensor [0, 1]
            - action: (action_dim,) tensor
            - task: str
            
        Returns LeRobot batch format (with batch dimension added).
        """
        batch = {}
        
        if "observation.state" in sample:
            state = sample["observation.state"]
            if state.dim() == 1:
                state = state.unsqueeze(0)
            batch["observation.state"] = state.to(self.device)
        
        # Use policy_camera_names to ensure all required cameras are present
        cameras_to_use = self.policy_camera_names if self.policy_camera_names else self.camera_names
        
        missing_cameras = []
        for cam_name in cameras_to_use:
            key = f"observation.images.{cam_name}"
            if key in sample:
                img = sample[key]
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                if self.image_size and img.shape[2:] != self.image_size:
                    img = torch.nn.functional.interpolate(
                        img, size=self.image_size, mode="bilinear", align_corners=False
                    )
                batch[key] = img.to(self.device)
            else:
                missing_cameras.append(cam_name)
        
        # Strict check: all required cameras must be present
        if missing_cameras:
            available_cams = [k for k in sample.keys() if k.startswith("observation.images.")]
            error_msg = (
                f"Missing required cameras in dataset sample!\n"
                f"  Policy requires: {cameras_to_use}\n"
                f"  Available in sample: {available_cams}\n"
                f"  Missing: {missing_cameras}\n"
                f"Hint: Ensure dataset was recorded with all required cameras."
            )
            raise ValueError(error_msg)
        
        if self.task_description:
            batch["task"] = [self.task_description]
        elif "task" in sample:
            task = sample["task"]
            batch["task"] = [task] if isinstance(task, str) else task
        
        return batch
    
    def _process_image(self, img: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Process image to policy input format.
        
        Input: (B, H, W, C) or (H, W, C) - uint8 or float [0, 255]
        Output: (B, C, H, W) - float [0, 1]
        """
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        
        img = img.clone()
        
        if img.dim() == 3:
            img = img.unsqueeze(0)
        
        if img.shape[-1] in [1, 3, 4]:
            img = img.permute(0, 3, 1, 2)
        
        img = img.float()
        
        if img.max() > 1.0:
            img = img / 255.0
        
        if img.shape[2:] != self.image_size:
            img = torch.nn.functional.interpolate(
                img,
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            )
        
        return img
    
    @torch.no_grad()
    def get_action(self, obs: dict, from_maniskill: bool = True) -> np.ndarray:
        """
        Get single action from policy given observation.
        
        Args:
            obs: Observation dict (ManiSkill or dataset format)
            from_maniskill: If True, preprocess as ManiSkill obs; else as dataset sample
        
        Returns:
            action: (action_dim,) numpy array
        """
        if from_maniskill:
            batch = self.preprocess_maniskill_obs(obs)
        else:
            batch = self.preprocess_dataset_sample(obs)
        
        if self.preprocessor is not None:
            batch = self.preprocessor(batch)
        
        action = self.policy.select_action(batch)
        
        if self.postprocessor is not None:
            action = self.postprocessor(action)
        
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        if action.ndim == 2 and action.shape[0] == 1:
            action = action[0]
        
        return action
    
    @torch.no_grad()
    def get_action_chunk(self, sample: dict) -> np.ndarray:
        """
        Get action chunk from policy given a dataset sample.
        
        Returns:
            action_chunk: (chunk_size, action_dim) numpy array
        """
        batch = self.preprocess_dataset_sample(sample)
        
        if self.preprocessor is not None:
            batch = self.preprocessor(batch)
        
        if hasattr(self.policy, 'predict_action_chunk'):
            action_chunk = self.policy.predict_action_chunk(batch)
        else:
            action = self.policy.select_action(batch)
            action_chunk = action.unsqueeze(1) if action.dim() == 2 else action
        
        if self.postprocessor is not None:
            if action_chunk.dim() == 3:
                B, C, D = action_chunk.shape
                processed = []
                for i in range(C):
                    processed.append(self.postprocessor(action_chunk[:, i, :]))
                action_chunk = torch.stack(processed, dim=1)
            else:
                action_chunk = self.postprocessor(action_chunk)
        
        if isinstance(action_chunk, torch.Tensor):
            action_chunk = action_chunk.cpu().numpy()
        
        if action_chunk.ndim == 3 and action_chunk.shape[0] == 1:
            action_chunk = action_chunk[0]
        if action_chunk.ndim == 1:
            action_chunk = action_chunk[np.newaxis, :]
        
        return action_chunk


def save_video(frames: list, output_dir: Path, filename: str, fps: int = 30):
    """Save frames as video."""
    import imageio
    
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / filename
    
    processed_frames = []
    for frame in frames:
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        processed_frames.append(frame)
    
    imageio.mimsave(str(video_path), processed_frames, fps=fps)
    logger.info(f"Saved video to {video_path}")


def save_image(frame: np.ndarray, output_dir: Path, filename: str):
    """Save a single frame as PNG image."""
    import imageio
    
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = output_dir / filename
    
    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
    
    imageio.imwrite(str(frame_path), frame)
    logger.info(f"Saved image to {frame_path}")


def extract_input_frames(obs: dict) -> dict[str, np.ndarray]:
    """Extract ALL camera images from ManiSkill observation for visualization."""
    frames = {}
    
    if "sensor_data" in obs:
        for cam_name, sensor_data in obs["sensor_data"].items():
            img = None
            if isinstance(sensor_data, dict) and "rgb" in sensor_data:
                img = sensor_data["rgb"]
            elif isinstance(sensor_data, (np.ndarray, torch.Tensor)):
                img = sensor_data
            
            if img is not None:
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                if img.ndim == 4 and img.shape[0] == 1:
                    img = img[0]
                if img.dtype != np.uint8:
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                frames[cam_name] = img
    
    return frames
