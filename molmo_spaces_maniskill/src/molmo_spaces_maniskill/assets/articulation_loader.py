from __future__ import annotations

from pathlib import Path
from typing import Generic, cast

import mujoco as mj
import numpy as np
from mani_skill.envs.scene import ManiSkillScene
from sapien import ArticulationBuilder, Pose, Scene
from sapien.physx import PhysxMaterial
from sapien.render import RenderMaterial
from sapien.wrapper.articulation_builder import LinkBuilder
from scipy.spatial.transform import Rotation as R

from .common import (
    AXIS_NORM_TOLERANCE,
    CAPSULE_FIX_POSE,
    CYLINDER_FIX_POSE,
    THOR_COLLISION_GROUPS,
    MjcfJointInfo,
    SceneT,
    get_collider_specs_from_body,
    get_frame_axes,
    get_orientation,
    get_rgba_from_geom,
    get_visual_specs_from_body,
    has_any_non_free_joint,
    is_visual,
    mjc_joint_type_to_str,
    parse_materials,
    vec_to_quat,
)


def add_colliders_to_sapien_link(
    link_builder: LinkBuilder,
    mj_spec: mj.MjSpec,
    mj_body_name: str,
    mj_model_dir: Path,
) -> None:
    mjs_body = mj_spec.body(mj_body_name)
    if mjs_body is None:
        return
    colliders_specs = get_collider_specs_from_body(mjs_body)
    for idx in range(len(colliders_specs)):
        col_spec: mj.MjsGeom = colliders_specs[idx]
        local_pose = Pose(p=tuple(col_spec.pos), q=tuple(get_orientation(col_spec)))
        physx_material: PhysxMaterial | None = None
        if col_spec.condim == 3:  # noqa: PLR2004
            physx_material = PhysxMaterial(
                static_friction=col_spec.friction[0],
                dynamic_friction=col_spec.friction[0],
                restitution=0,
            )
        elif col_spec.condim == 1:
            physx_material = PhysxMaterial(static_friction=0, dynamic_friction=0, restitution=0)

        if col_spec.type == mj.mjtGeom.mjGEOM_BOX:
            link_builder.add_box_collision(
                pose=local_pose,
                half_size=tuple(col_spec.size),
                density=col_spec.density,
                material=physx_material,
            )
        elif col_spec.type == mj.mjtGeom.mjGEOM_SPHERE:
            link_builder.add_sphere_collision(
                pose=local_pose,
                radius=col_spec.size[0],
                density=col_spec.density,
                material=physx_material,
            )
        elif col_spec.type == mj.mjtGeom.mjGEOM_CAPSULE:
            radius = col_spec.size[0]
            half_length = col_spec.size[1]
            if not np.isnan(col_spec.fromto[0]):
                start, end = col_spec.fromto[:3], col_spec.fromto[3:]
                pos = (end + start) / 2
                quat = vec_to_quat(end - start)
                local_pose = Pose(p=tuple(pos), q=tuple(quat))
                half_length = np.linalg.norm(end - start).item() / 2.0
            link_builder.add_capsule_collision(
                pose=local_pose * CAPSULE_FIX_POSE,
                radius=radius,
                half_length=half_length,
                density=col_spec.density,
                material=physx_material,
            )
        elif col_spec.type == mj.mjtGeom.mjGEOM_CYLINDER:
            radius = col_spec.size[0]
            half_length = col_spec.size[1]
            if not np.isnan(col_spec.fromto[0]):
                start, end = col_spec.fromto[:3], col_spec.fromto[3:]
                pos = (end + start) / 2
                quat = vec_to_quat(end - start)
                local_pose = Pose(p=tuple(pos), q=tuple(quat))
                half_length = np.linalg.norm(end - start).item() / 2.0
            link_builder.add_cylinder_collision(
                pose=local_pose * CYLINDER_FIX_POSE,
                radius=radius,
                half_length=half_length,
                density=col_spec.density,
                material=physx_material,
            )
        elif col_spec.type == mj.mjtGeom.mjGEOM_MESH:
            mesh_spec = mj_spec.mesh(col_spec.meshname)
            if mesh_spec is not None:
                mesh_path = mj_model_dir / mj_spec.meshdir / mesh_spec.file
                link_builder.add_convex_collision_from_file(
                    pose=local_pose,
                    filename=mesh_path.as_posix(),
                    scale=tuple(mesh_spec.scale),
                    density=col_spec.density,
                    material=physx_material,
                )
        else:
            raise ValueError(f"Collider geom type {col_spec.type} not supported")

        if col_groups := THOR_COLLISION_GROUPS.get(col_spec.classname.name):
            link_builder.collision_groups = col_groups


def add_visuals_to_sapien_link(
    link: LinkBuilder,
    mj_spec: mj.MjSpec,
    mj_body_name: str,
    mj_model_dir: Path,
    materials: dict[str, RenderMaterial],
    colliders_are_visuals: bool = False,
) -> None:
    mjs_body = mj_spec.body(mj_body_name)
    if mjs_body is None:
        return
    visual_specs = (
        get_visual_specs_from_body(mjs_body)
        if not colliders_are_visuals
        else get_collider_specs_from_body(mjs_body)
    )
    for idx in range(len(visual_specs)):
        vis_spec: mj.MjsGeom = visual_specs[idx]
        local_pose = Pose(p=tuple(vis_spec.pos), q=tuple(get_orientation(vis_spec)))

        vis_name = vis_spec.name if vis_spec.name != "" else f"{mj_body_name}_visual_{idx}"

        match vis_spec.type:
            case mj.mjtGeom.mjGEOM_BOX:
                link.add_box_visual(
                    pose=local_pose,
                    half_size=tuple(vis_spec.size),
                    material=tuple(get_rgba_from_geom(mj_spec, vis_spec)),
                    name=vis_name,
                )
            case mj.mjtGeom.mjGEOM_SPHERE:
                link.add_sphere_visual(
                    pose=local_pose,
                    radius=vis_spec.size[0],
                    material=tuple(get_rgba_from_geom(mj_spec, vis_spec)),
                    name=vis_name,
                )
            case mj.mjtGeom.mjGEOM_CAPSULE:
                radius = vis_spec.size[0]
                half_length = vis_spec.size[1]
                if not np.isnan(vis_spec.fromto[0]):
                    start, end = vis_spec.fromto[:3], vis_spec.fromto[3:]
                    pos = (end + start) / 2
                    quat = vec_to_quat(end - start)
                    local_pose = Pose(p=tuple(pos), q=tuple(quat))
                    half_length = np.linalg.norm(end - start).item() / 2.0
                link.add_capsule_visual(
                    pose=local_pose * CAPSULE_FIX_POSE,
                    radius=radius,
                    half_length=half_length,
                    material=tuple(get_rgba_from_geom(mj_spec, vis_spec)),
                    name=vis_name,
                )
            case mj.mjtGeom.mjGEOM_CYLINDER:
                radius = vis_spec.size[0]
                half_length = vis_spec.size[1]
                if not np.isnan(vis_spec.fromto[0]):
                    start, end = vis_spec.fromto[:3], vis_spec.fromto[3:]
                    pos = (end + start) / 2
                    quat = vec_to_quat(end - start)
                    local_pose = Pose(p=tuple(pos), q=tuple(quat))
                    half_length = np.linalg.norm(end - start).item() / 2.0
                link.add_cylinder_visual(
                    pose=local_pose * CYLINDER_FIX_POSE,
                    radius=radius,
                    half_length=half_length,
                    material=tuple(vis_spec.rgba[:3]),
                    name=vis_name,
                )
            case mj.mjtGeom.mjGEOM_MESH:
                mesh_spec = mj_spec.mesh(vis_spec.meshname)
                if mesh_spec is not None:
                    mesh_path = mj_model_dir / mj_spec.meshdir / mesh_spec.file
                    material = materials.get(vis_spec.material)
                    link.add_visual_from_file(
                        pose=local_pose,
                        filename=mesh_path.as_posix(),
                        scale=tuple(mesh_spec.scale),
                        material=material,
                        name=vis_name,
                    )
            case _:
                raise ValueError(f"Visual geom type {vis_spec.type} not supported")


class MjcfAssetArticulationLoader(Generic[SceneT]):
    def __init__(self, scene: SceneT | None = None, use_colliders_as_visuals: bool = True):
        self._scene: SceneT | None = scene

        self._spec: mj.MjSpec | None = None
        self._model_dir: Path | None = None
        self._materials: dict[str, RenderMaterial] = {}
        self._num_links: int = 0
        self._num_colliders: int = 0
        self._num_visuals: int = 0
        self._use_colliders_as_visuals = use_colliders_as_visuals

    def set_scene(self, scene: SceneT) -> MjcfAssetArticulationLoader:
        self._scene = scene
        return self

    @property
    def mjspec(self) -> mj.MjSpec | None:
        return self._spec

    def load_from_spec(
        self,
        scene_spec: mj.MjSpec,
        model_dir: Path,
        root_body_name: str | None = None,
        floating_base: bool | None = None,
        materials: dict[str, RenderMaterial] | None = None,
        is_part_of_scene: bool = False,
    ) -> ArticulationBuilder:
        """Loads an articulation from a given MjSpec for a given scene

        The given spec is assumed to be the full MjSpec for a whole scene, and the section that
        corresponds to the articulation we want to create starts at the body that has name given
        by the `root_body_name` parameter. If no root_body_name parameter is given, it's assumed
        that the whole spec corresponds for a single articulated object.

        Args:
            scene_spec: The spec for a whole scene, or for a single articulation
            model_dir: Path to the folder that contains the xml model from which the spec was parsed
            root_body_name (optional): The name of the root of the articulation
            floating_base (optional): Whether or not the base should be free to move
            materials (optional): A cache of parsed materials. If not given, will parse again here

        Returns:
            PhysxArticulation: The generated articulation object

        Raises:
            ValueError: If the root body was not found in the given spec

        """
        assert self._scene is not None, "Must assign a valid Sapien scene before loading"

        self._spec = scene_spec
        self._model_dir = model_dir
        self._num_links = 0
        self._num_colliders = 0
        self._num_visuals = 0

        self._materials = parse_materials(scene_spec, model_dir) if materials is None else materials

        articulation_builder = cast(
            Scene | ManiSkillScene, self._scene
        ).create_articulation_builder()

        mj_root_body: mj.MjsBody | None = None
        if root_body_name is not None:
            mj_root_body = self._spec.body(root_body_name)
            if mj_root_body is None:
                raise ValueError(
                    f"Couldn't find root-body with name {root_body_name} in the given MjSpec"
                )
        else:
            mj_root_body = self._spec.worldbody.first_body()

        assert mj_root_body is not None, "Something went wrong when loading articulated object"

        if not has_any_non_free_joint(mj_root_body):
            print(
                f"[WARN]: the given model {self._spec.modelname} doesn't have any non-free joints. "
                + "You might be better off using MjcfAssetActorLoader instead"
            )

        has_freejoint = any(jnt.type == mj.mjtJoint.mjJNT_FREE for jnt in mj_root_body.joints)

        if has_freejoint:
            for jnt in mj_root_body.joints:
                assert isinstance(jnt, mj.MjsJoint)
                if jnt.type == mj.mjtJoint.mjJNT_FREE:
                    self._spec.delete(jnt)

        dummy_root_link: LinkBuilder = articulation_builder.create_link_builder()
        dummy_root_link.name = root_body_name or "dummy_root_0"

        self._parse_body(mj_root_body, articulation_builder, dummy_root_link, True)

        if not has_freejoint and not floating_base:
            dummy_root_link.set_joint_properties(
                type="fixed",
                limits=None,
                pose_in_parent=Pose(),
                pose_in_child=Pose(),
            )

        # If is not part of a scene, then the whole spec corresponds to this articulation, so we
        # have to parse the other stuff here (not part of the scene loader)
        if not is_part_of_scene:
            # TODO: parse constraints and other stuff here
            pass

        return articulation_builder

    def load_from_xml(
        self,
        xml_model: Path,
        floating_base: bool | None = None,
    ) -> ArticulationBuilder:
        """Loads an articulation from a given mjcf model

        The given model is assumed to correspond to a single articulation. If the mjcf model
        contains more than 1 articulation or actor, you should use MjcfSceneLoader instead. If
        the model corresponds to a single actor that doesn't contain joints, then you should use
        MjcfAssetActorLoader instead.

        Args:
            xml_model: The path to the mjcf model corresponding to an articulation
            floating_base: Whether or not the base should be free

        Returns:
            PhysxArticulation: The generated articulation object

        Raises:
            ValueError: If the mjcf model couldn't be parsed

        """
        spec = mj.MjSpec.from_file(xml_model.as_posix())
        return self.load_from_spec(spec, xml_model.parent, floating_base=floating_base)

    def _parse_body(
        self,
        mjs_body: mj.MjsBody,
        articulation_builder: ArticulationBuilder,
        parent_link_builder: LinkBuilder,
        is_root: bool = False,
    ) -> LinkBuilder:
        assert self._spec is not None, "A valid mjspec object is required for the loader"
        assert self._model_dir is not None, (
            "A valid path to the folder containing the model should be provided"
        )

        link_body_name = mjs_body.name if mjs_body.name != "" else f"link_{self._num_links}"

        joints_info: list[MjcfJointInfo] = []
        for jnt_spec in mjs_body.joints:
            assert isinstance(jnt_spec, mj.MjsJoint)
            limits = (
                np.deg2rad(jnt_spec.range)
                if self._spec.compiler.degree and jnt_spec.type == mj.mjtJoint.mjJNT_HINGE
                else jnt_spec.range
            )
            joints_info.append(
                MjcfJointInfo(
                    name=jnt_spec.name,
                    type=mjc_joint_type_to_str(jnt_spec.type),
                    pos=jnt_spec.pos,
                    axis=jnt_spec.axis,
                    limited=jnt_spec.limited,
                    limits=limits,
                    frictionloss=float(jnt_spec.frictionloss),
                    damping=float(jnt_spec.damping),
                )
            )
        if len(mjs_body.joints) == 0:
            joints_info.append(
                MjcfJointInfo(
                    name=f"{link_body_name}_fixed",
                    type="fixed",
                )
            )

        link_builder = parent_link_builder

        for i, jnt_info in enumerate(joints_info):
            link_builder = articulation_builder.create_link_builder(parent=link_builder)
            link_builder.set_joint_name(jnt_info.name)
            link_body_name = mjs_body.name if mjs_body.name != "" else f"link_{self._num_links}"
            if i == len(joints_info) - 1:
                link_builder.set_name(link_body_name)
                if len(mjs_body.geoms) > 0:
                    has_any_visuals = any(is_visual(geom) for geom in mjs_body.geoms)
                    add_colliders_to_sapien_link(
                        link_builder,
                        self._spec,
                        link_body_name,
                        self._model_dir,
                    )
                    add_visuals_to_sapien_link(
                        link_builder,
                        self._spec,
                        link_body_name,
                        self._model_dir,
                        self._materials,
                        colliders_are_visuals=not has_any_visuals
                        and self._use_colliders_as_visuals,
                    )
            else:
                link_builder.set_name(f"{link_body_name}_dummy_{i}")

            self._num_links += 1

            tf_joint2parent = np.eye(4)
            if i == 0 and not is_root:
                tf_joint2parent[:3, 3] = mjs_body.pos
                tf_joint2parent[:3, :3] = R.from_quat(
                    get_orientation(mjs_body), scalar_first=True
                ).as_matrix()
                if (frame := mjs_body.frame) is not None:
                    tf_local_frame = np.eye(4)
                    tf_local_frame[:3, 3] = frame.pos
                    tf_local_frame[:3, :3] = R.from_quat(
                        get_orientation(frame), scalar_first=True
                    ).as_matrix()
                    tf_joint2parent = tf_local_frame @ tf_joint2parent

            axis = jnt_info.axis
            axis_norm = np.linalg.norm(axis)
            if axis_norm < AXIS_NORM_TOLERANCE:
                axis = np.array([1.0, 0.0, 0.0])
            else:
                axis /= axis_norm
            axis_x, axis_y, axis_z = get_frame_axes(axis)

            tf_axis2joint = np.eye(4)
            tf_axis2joint[:3, 3] = jnt_info.pos
            tf_axis2joint[:3, 0] = axis_x
            tf_axis2joint[:3, 1] = axis_y
            tf_axis2joint[:3, 2] = axis_z

            tf_axis2parent = tf_joint2parent @ tf_axis2joint

            jnt_range_min, jnt_range_max = jnt_info.limits
            jnt_limited = (
                jnt_range_min < jnt_range_max
                if jnt_info.limited == mj.mjtLimited.mjLIMITED_AUTO
                else jnt_info.limited == mj.mjtLimited.mjLIMITED_TRUE
            )
            jnt_range = [jnt_range_min, jnt_range_max] if jnt_limited else [-np.inf, np.inf]

            match jnt_info.type:
                case "hinge":
                    link_builder.set_joint_properties(
                        "revolute_unwrapped" if jnt_limited else "revolute",
                        limits=[jnt_range],
                        pose_in_parent=Pose(tf_axis2parent),
                        pose_in_child=Pose(tf_axis2joint),
                        friction=jnt_info.frictionloss,
                        damping=jnt_info.damping,
                    )
                case "slide":
                    link_builder.set_joint_properties(
                        "prismatic",
                        limits=[jnt_range],
                        pose_in_parent=Pose(tf_axis2parent),
                        pose_in_child=Pose(tf_axis2joint),
                        friction=jnt_info.frictionloss,
                        damping=jnt_info.damping,
                    )
                case "fixed":
                    link_builder.set_joint_properties(
                        "fixed",
                        limits=[],
                        pose_in_parent=Pose(tf_axis2parent),
                        pose_in_child=Pose(tf_axis2joint),
                    )

        for mjs_child in mjs_body.bodies:
            assert isinstance(mjs_child, mj.MjsBody)
            self._parse_body(mjs_child, articulation_builder, link_builder, False)
        return link_builder
