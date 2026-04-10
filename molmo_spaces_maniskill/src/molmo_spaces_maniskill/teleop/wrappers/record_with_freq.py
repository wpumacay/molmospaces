"""
Recording wrapper with frequency control support.

This wrapper extends RecordEpisode to support:
- Recording at a different frequency than control frequency
- Saving joint position actions
- Tracking true elapsed steps vs recorded steps
"""

import numpy as np
from typing import Optional

from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.utils import common, gym_utils


class RecordEpisodeWithFreq(RecordEpisode):
    """
    Recording wrapper with frequency control support.
    
    Extends RecordEpisode to support recording at a lower frequency than control,
    while preserving joint position actions.
    
    Args:
        env: The environment to wrap
        record_freq: Recording frequency (Hz). If None, records every step.
        control_freq: Control frequency (Hz).
        **kwargs: Additional arguments passed to RecordEpisode
        
    Example:
        >>> env = RecordEpisodeWithFreq(
        ...     env,
        ...     output_dir="demos/",
        ...     record_freq=10.0,   # Record at 10 Hz
        ...     control_freq=50.0,  # Control at 50 Hz
        ... )
    """
    
    def __init__(
        self,
        env,
        record_freq: Optional[float] = None,
        control_freq: float = 50.0,
        **kwargs
    ):
        # Set save_on_reset=False by default for manual control
        kwargs['save_on_reset'] = kwargs.get('save_on_reset', False)
        super().__init__(env, **kwargs)
        
        self.record_freq = record_freq
        self.control_freq = control_freq
        self._step_count = 0
        self._true_elapsed_steps = {}  # Per-environment true step counts
        self._episode_true_steps = []  # True steps for each saved episode
    
    def reset(self, *args, **kwargs):
        """Reset environment and step counters."""
        # Handle trajectory saving if save_on_reset is True
        if self.save_on_reset:
            if self.save_video and self.num_envs == 1:
                self.flush_video()
            if self._trajectory_buffer is not None:
                options = kwargs.get('options')
                if options is None or "env_idx" not in options:
                    self.flush_trajectory(env_idxs_to_flush=np.arange(self.num_envs))
                else:
                    self.flush_trajectory(env_idxs_to_flush=common.to_numpy(options["env_idx"]))
                self._trajectory_buffer = None
        
        # Call parent reset
        obs, info = super().reset(*args, **kwargs)
        
        if info["reconfigure"]:
            self._trajectory_buffer = None
        
        # Reset step counters
        self._step_count = 0
        self._true_elapsed_steps[0] = 0
        
        return obs, info
    
    def step(self, action):
        """
        Execute action and optionally record based on frequency settings.
        
        Args:
            action: Action to execute (e.g., joint positions [7 joints + 1 gripper])
            
        Returns:
            Standard gym step outputs (obs, reward, terminated, truncated, info)
        """
        # Capture initial frame for video
        if self.save_video and self._video_steps == 0:
            self.render_images.append(self.capture_image())
        
        # Execute environment step
        obs, rew, terminated, truncated, info = self.env.step(action)
        
        # Update step counters
        self._step_count += 1
        self._true_elapsed_steps[0] = self._true_elapsed_steps.get(0, 0) + 1
        
        # Determine if this step should be recorded
        should_record = self._should_record_step(info)
        
        # Record trajectory data
        if self.save_trajectory and should_record and self._trajectory_buffer is not None:
            self._record_step(obs, action, rew, terminated, truncated, info)
        
        # Handle video recording
        if self.save_video:
            self._handle_video_step(rew, info)
        
        self._elapsed_record_steps += 1
        
        return obs, rew, terminated, truncated, info
    
    def _should_record_step(self, info: dict) -> bool:
        """
        Determine if current step should be recorded.
        
        Always records on success to ensure the final successful state is captured.
        """
        should_record = True
        
        if self.record_freq is not None and self.record_freq < self.control_freq:
            record_interval = int(self.control_freq / self.record_freq)
            should_record = (self._step_count % record_interval == 0)
        
        # Always record on success
        if info.get("success", False):
            should_record = True
            
        return should_record
    
    def _record_step(self, obs, action, rew, terminated, truncated, info):
        """Record a single step to the trajectory buffer."""
        state_dict = self.base_env.get_state_dict()
        
        if self.record_env_state:
            self._trajectory_buffer.state = common.append_dict_array(
                self._trajectory_buffer.state,
                common.to_numpy(common.batch(state_dict)),
            )
        
        self._trajectory_buffer.observation = common.append_dict_array(
            self._trajectory_buffer.observation,
            common.to_numpy(common.batch(obs)),
        )
        
        # Save original action (e.g., joint positions)
        self._trajectory_buffer.action = common.append_dict_array(
            self._trajectory_buffer.action,
            common.to_numpy(common.batch(action)),
        )
        
        if self.record_reward:
            self._trajectory_buffer.reward = common.append_dict_array(
                self._trajectory_buffer.reward,
                common.to_numpy(common.batch(rew)),
            )
        
        self._trajectory_buffer.terminated = common.append_dict_array(
            self._trajectory_buffer.terminated,
            common.to_numpy(common.batch(terminated)),
        )
        self._trajectory_buffer.truncated = common.append_dict_array(
            self._trajectory_buffer.truncated,
            common.to_numpy(common.batch(truncated)),
        )
        
        done = terminated | truncated
        self._trajectory_buffer.done = common.append_dict_array(
            self._trajectory_buffer.done,
            common.to_numpy(common.batch(done)),
        )
        
        if "success" in info:
            self._trajectory_buffer.success = common.append_dict_array(
                self._trajectory_buffer.success,
                common.to_numpy(common.batch(info["success"])),
            )
        else:
            self._trajectory_buffer.success = None
            
        if "fail" in info:
            self._trajectory_buffer.fail = common.append_dict_array(
                self._trajectory_buffer.fail,
                common.to_numpy(common.batch(info["fail"])),
            )
        else:
            self._trajectory_buffer.fail = None
    
    def _handle_video_step(self, rew, info):
        """Handle video frame capture."""
        self._video_steps += 1
        
        if self.info_on_video:
            scalar_info = gym_utils.extract_scalars_from_info(
                common.to_numpy(info), batch_size=self.num_envs
            )
            scalar_info["reward"] = common.to_numpy(rew)
            if np.size(scalar_info["reward"]) > 1:
                scalar_info["reward"] = [float(r) for r in scalar_info["reward"]]
            else:
                scalar_info["reward"] = float(scalar_info["reward"])
            image = self.capture_image(scalar_info)
        else:
            image = self.capture_image()
            
        self.render_images.append(image)
        
        if (self.max_steps_per_video is not None and 
            self._video_steps >= self.max_steps_per_video):
            self.flush_video()
    
    def flush_trajectory(
        self,
        verbose=False,
        ignore_empty_transition=True,
        env_idxs_to_flush=None,
    ):
        """Flush trajectory and save true elapsed steps."""
        # Save true elapsed steps
        if 0 in self._true_elapsed_steps:
            self._episode_true_steps.append(self._true_elapsed_steps[0])
            self._true_elapsed_steps[0] = 0
        
        # Call parent flush
        super().flush_trajectory(verbose, ignore_empty_transition, env_idxs_to_flush)
    
    def clear_video_buffer(self):
        """Clear video buffer without saving. Call this when restarting episode."""
        self.render_images = []
        self._video_steps = 0
    
    def clear_all_buffers(self):
        """Clear both trajectory and video buffers without saving."""
        self._trajectory_buffer = None
        self.render_images = []
        self._video_steps = 0
        self._step_count = 0
        self._true_elapsed_steps[0] = 0
    
    def close(self) -> None:
        """Close wrapper and update JSON metadata with true step counts."""
        # Call parent close first
        super().close()
        
        # Update JSON metadata if using frequency control
        if (self.save_trajectory and 
            self.record_freq is not None and 
            self.record_freq < self.control_freq):
            self._update_json_metadata()
    
    def _update_json_metadata(self):
        """Update JSON file with true elapsed steps for all episodes."""
        try:
            import json
            from mani_skill.utils.io_utils import dump_json
            
            with open(self._json_path, 'r') as f:
                json_data = json.load(f)
            
            for i, episode in enumerate(json_data['episodes']):
                if i < len(self._episode_true_steps):
                    true_steps = self._episode_true_steps[i]
                    recorded_steps = episode['elapsed_steps']
                    
                    episode['true_elapsed_steps'] = true_steps
                    episode['recorded_steps'] = recorded_steps
                    episode['elapsed_steps'] = true_steps
                    episode['recording_frequency'] = f"{self.record_freq}Hz"
                    episode['sampling_interval'] = int(self.control_freq / self.record_freq)
            
            dump_json(self._json_path, json_data, indent=2)
            print(f"\n✅ Updated metadata for {len(json_data['episodes'])} episodes")
            
        except Exception as e:
            print(f"⚠️  Failed to update JSON metadata: {e}")

    def render_human(self, *args, **kwargs):
        """Proxy render_human to the underlying environment."""
        return self.base_env.render_human(*args, **kwargs)

    @property
    def agent(self):
        """Proxy agent property to the underlying environment."""
        return self.base_env.agent

    @property
    def unwrapped(self):
        """Return the unwrapped base environment."""
        return self.base_env.unwrapped

    @property
    def device(self):
        """Return the device of the underlying environment."""
        return self.base_env.device
