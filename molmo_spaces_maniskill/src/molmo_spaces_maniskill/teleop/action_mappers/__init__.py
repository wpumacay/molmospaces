"""Action mappers for converting device input to robot actions."""

from molmo_spaces_maniskill.teleop.action_mappers.base_mapper import BaseActionMapper
from molmo_spaces_maniskill.teleop.action_mappers.single_arm_mapper import SingleArmActionMapper
from molmo_spaces_maniskill.teleop.action_mappers.dual_arm_mapper import DualArmActionMapper

__all__ = [
    "BaseActionMapper",
    "SingleArmActionMapper",
    "DualArmActionMapper",
    "get_action_mapper",
]


def get_action_mapper(robot_uid: str, **kwargs) -> BaseActionMapper:
    """
    Get appropriate action mapper for a robot.
    
    Args:
        robot_uid: ManiSkill robot UID
        **kwargs: Additional arguments passed to mapper constructor
        
    Returns:
        Appropriate ActionMapper instance
    """
    from molmo_spaces_maniskill.teleop.configs.robot_configs import get_robot_config
    
    config = get_robot_config(robot_uid)
    if config is None:
        raise ValueError(f"Unknown robot: {robot_uid}")
    
    if config.is_bimanual:
        return DualArmActionMapper(robot_uid, **kwargs)
    else:
        return SingleArmActionMapper(robot_uid, **kwargs)

