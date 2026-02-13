from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeVar

import mujoco as mj
import numpy as np
import sapien
from mani_skill.envs.scene import ManiSkillScene
from sapien.render import RenderMaterial, RenderTexture2D
from scipy.spatial.transform import Rotation as R

VISUAL_CLASSES = {"__VISUAL_MJT__", "visual"}
VISUAL_THRESHOLD_DENSITY = 1e-5
VISUAL_THRESHOLD_MASS = 1e-6

CAPSULE_FIX_POSE = sapien.Pose(q=R.from_euler("xyz", [0, np.pi / 2, 0]).as_quat(scalar_first=True))
CYLINDER_FIX_POSE = sapien.Pose(q=R.from_euler("xyz", [0, np.pi / 2, 0]).as_quat(scalar_first=True))

X_AXIS = np.array([1.0, 0.0, 0.0], dtype=np.float64)
Y_AXIS = np.array([0.0, 1.0, 0.0], dtype=np.float64)
Z_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)

WORLD_UP = Z_AXIS.copy()

SceneT = TypeVar("SceneT", sapien.Scene, ManiSkillScene)

THOR_COLLISION_GROUPS: dict[str, list[int]] = {
    "__STRUCTURAL_MJT__": [0b1000, 0b1111, 1, 0],
    "__STRUCTURAL_WALL_MJT__": [0b1000, 0b1111, 1, 0],
    # "__DYNAMIC_MJT__": [0b0001, 0b1111, 1, 0],
    "__ARTICULABLE_DYNAMIC_MJT__": [0b0000, 0b0111, 1, 0],
}


@dataclass
class MjcfTextureInfo:
    name: str
    type: mj.mjtTexture
    rgb1: list
    rgb2: list
    file: Path


@dataclass
class MjcfJointInfo:
    name: str
    type: Literal["free", "fixed", "hinge", "slide", "ball"]
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    axis: np.ndarray = field(default_factory=X_AXIS.copy)
    limited: int = False
    limits: np.ndarray = field(default_factory=lambda: np.array([-np.inf, np.inf]))
    frictionloss: float = 0.0
    damping: float = 0.0


def mjc_joint_type_to_str(
    jnt_type: mj.mjtJoint,
) -> Literal["free", "fixed", "hinge", "slide", "ball"]:
    match jnt_type:
        case mj.mjtJoint.mjJNT_FREE:
            return "free"
        case mj.mjtJoint.mjJNT_HINGE:
            return "hinge"
        case mj.mjtJoint.mjJNT_SLIDE:
            return "slide"
        case mj.mjtJoint.mjJNT_BALL:
            return "ball"
        case _:
            raise RuntimeError(f"Joint of type '{jnt_type}' is not valid")


QUAT_TOLERANCE = 1e-10
AXIS_NORM_TOLERANCE = 1e-3


def vec_to_quat(vec: np.ndarray) -> np.ndarray:
    vec /= np.linalg.norm(vec)

    cross = np.cross(Z_AXIS, vec)
    s = np.linalg.norm(cross)

    if s < QUAT_TOLERANCE:
        return np.array([0.0, 1.0, 0.0, 0.0])
    else:
        cross /= np.linalg.norm(cross)
        ang = np.arctan2(s, vec[2]).item()
        quat = np.array(
            [
                np.cos(ang / 2.0),
                cross[0] * np.sin(ang / 2.0),
                cross[1] * np.sin(ang / 2.0),
                cross[2] * np.sin(ang / 2.0),
            ]
        )
        quat /= np.linalg.norm(quat)
        return quat


def get_frame_axes(axis_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if abs(np.dot(axis_x, [1.0, 0.0, 0.0])) > 0.9:  # noqa: PLR2004
        axis_y = np.cross(axis_x, WORLD_UP)
        axis_y = axis_y / np.linalg.norm(axis_y)
    else:
        axis_y = np.cross(axis_x, X_AXIS)
        axis_y = axis_y / np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y)
    axis_z = axis_z / np.linalg.norm(axis_z)

    return axis_x, axis_y, axis_z


def is_visual(geom: mj.MjsGeom) -> bool:
    return (geom.classname.name in VISUAL_CLASSES) or (geom.contype == 0 and geom.conaffinity == 0)


def get_visual_specs_from_body(mjs_body: mj.MjsBody) -> list[mj.MjsGeom]:
    return [geom for geom in mjs_body.geoms if is_visual(geom)]


def get_collider_specs_from_body(mjs_body: mj.MjsBody) -> list[mj.MjsGeom]:
    return [geom for geom in mjs_body.geoms if not is_visual(geom)]


def get_orientation(body_spec: mj.MjsBody) -> np.ndarray:
    match body_spec.alt.type:
        case mj.mjtOrientation.mjORIENTATION_QUAT:
            return body_spec.quat.copy()
        case mj.mjtOrientation.mjORIENTATION_AXISANGLE:
            axisangle = body_spec.alt.axisangle
            return R.from_rotvec(axisangle[-1] * axisangle[:-1]).as_quat(scalar_first=True)
        case mj.mjtOrientation.mjORIENTATION_XYAXES:
            raise NotImplementedError("Support for xyaxes in orientation is not supported yet")
        case mj.mjtOrientation.mjORIENTATION_ZAXIS:
            raise NotImplementedError("Support for zaxis in orientation is not supported yet")
        case mj.mjtOrientation.mjORIENTATION_EULER:
            euler = body_spec.alt.euler
            return R.from_euler("xyz", euler, degrees=False).as_quat(scalar_first=True)
        case _:
            raise ValueError(f"Orientation type {body_spec.alt.type} is not valid")


def get_rgba_from_geom(mj_spec: mj.MjSpec, mj_geom: mj.MjsGeom) -> np.ndarray:
    rgba = mj_geom.rgba.copy()
    if mj_geom.material:
        mj_mat = mj_spec.material(mj_geom.material)
        if mj_mat is not None:
            rgba = mj_mat.rgba.copy()
    return rgba


def parse_textures(spec: mj.MjSpec, model_dir: Path) -> dict[str, MjcfTextureInfo]:
    textures_info: dict[str, MjcfTextureInfo] = {}
    for tex_spec in spec.textures:
        assert isinstance(tex_spec, mj.MjsTexture)
        if tex_spec.name in textures_info:
            print(f"[WARN]: texture with name {tex_spec.name} already parsed")
            continue
        textures_info[tex_spec.name] = MjcfTextureInfo(
            name=tex_spec.name,
            type=tex_spec.type,
            rgb1=tex_spec.rgb1.tolist(),
            rgb2=tex_spec.rgb2.tolist(),
            file=model_dir / tex_spec.file,
        )

    return textures_info


def parse_materials(spec: mj.MjSpec, model_dir: Path) -> dict[str, RenderMaterial]:
    textures_info = parse_textures(spec, model_dir)

    materials: dict[str, RenderMaterial] = {}
    for mat_spec in spec.materials:
        assert isinstance(mat_spec, mj.MjsMaterial)
        if mat_spec.name in materials:
            print(f"[WARN]: material with name {mat_spec.name} already parsed")
            continue

        rgba = mat_spec.rgba.copy()
        em = mat_spec.emission
        emission_arr = [rgba[0] * em, rgba[1] * em, rgba[2] * em, 1]
        render_material = RenderMaterial(
            emission=emission_arr,
            base_color=mat_spec.rgba.tolist(),
            specular=mat_spec.specular,
            roughness=1.0 - mat_spec.reflectance,
            metallic=mat_spec.shininess,
        )

        texture: RenderTexture2D | None = None
        texture_id = mat_spec.textures[mj.mjtTextureRole.mjTEXROLE_RGB]
        if texture_id != "":
            if texture_id in textures_info:
                texture_filepath = textures_info[texture_id].file
                if texture_filepath.exists() and texture_filepath.is_file():
                    texture = RenderTexture2D(filename=texture_filepath.as_posix())

        if texture is not None:
            render_material.base_color_texture = texture
        materials[mat_spec.name] = render_material

    return materials


def has_any_non_free_joint(root_body: mj.MjsBody) -> bool:
    has_non_free_joint = False
    stack = [root_body]
    while len(stack) > 0:
        body = stack.pop()
        if len(body.joints) > 0:
            if any([jnt.type != mj.mjtJoint.mjJNT_FREE for jnt in body.joints]):
                has_non_free_joint = True
                break
        for child in body.bodies:
            stack.append(child)

    return has_non_free_joint
