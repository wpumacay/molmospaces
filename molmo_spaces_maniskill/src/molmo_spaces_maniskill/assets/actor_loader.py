from __future__ import annotations

from pathlib import Path
from typing import Generic, cast

import mujoco as mj
import numpy as np
from mani_skill.envs.scene import ManiSkillScene
from sapien import ActorBuilder, Pose, Scene
from sapien.physx import PhysxMaterial
from sapien.render import RenderMaterial

from .common import (
    CAPSULE_FIX_POSE,
    CYLINDER_FIX_POSE,
    THOR_COLLISION_GROUPS,
    SceneT,
    get_collider_specs_from_body,
    get_orientation,
    get_visual_specs_from_body,
    has_any_non_free_joint,
    parse_materials,
    vec_to_quat,
)


def add_visuals_to_actor_builder(
    builder: ActorBuilder,
    mj_spec: mj.MjSpec,
    mj_body: mj.MjsBody,
    rel_pose_to_parent: Pose,
    mj_model_dir: Path,
    materials: dict[str, RenderMaterial],
) -> None:
    visual_specs = get_visual_specs_from_body(mj_body)
    for vis_spec in visual_specs:
        tf_geom_to_body = Pose(p=tuple(vis_spec.pos), q=tuple(get_orientation(vis_spec)))
        match vis_spec.type:
            case mj.mjtGeom.mjGEOM_BOX:
                builder.add_box_visual(
                    pose=rel_pose_to_parent * tf_geom_to_body,
                    half_size=tuple(vis_spec.size),
                    material=tuple(vis_spec.rgba[:3]),
                )
            case mj.mjtGeom.mjGEOM_SPHERE:
                builder.add_sphere_visual(
                    pose=rel_pose_to_parent * tf_geom_to_body,
                    radius=vis_spec.size[0],
                    material=tuple(vis_spec.rgba[:3]),
                )
            case mj.mjtGeom.mjGEOM_CAPSULE:
                radius = vis_spec.size[0]
                half_length = vis_spec.size[1]
                if not np.isnan(vis_spec.fromto[0]):
                    start, end = vis_spec.fromto[:3], vis_spec.fromto[3:]
                    pos = (end + start) / 2
                    quat = vec_to_quat(end - start)
                    tf_geom_to_body = Pose(p=tuple(pos), q=tuple(quat))
                    half_length = np.linalg.norm(end - start).item() / 2.0
                builder.add_capsule_visual(
                    pose=rel_pose_to_parent * tf_geom_to_body * CAPSULE_FIX_POSE,
                    radius=radius,
                    half_length=half_length,
                    material=tuple(vis_spec.rgba[:3]),
                )
            case mj.mjtGeom.mjGEOM_CYLINDER:
                radius = vis_spec.size[0]
                half_length = vis_spec.size[1]
                if not np.isnan(vis_spec.fromto[0]):
                    start, end = vis_spec.fromto[:3], vis_spec.fromto[3:]
                    pos = (end + start) / 2
                    quat = vec_to_quat(end - start)
                    tf_geom_to_body = Pose(p=tuple(pos), q=tuple(quat))
                    half_length = np.linalg.norm(end - start).item() / 2.0
                builder.add_cylinder_visual(
                    pose=rel_pose_to_parent * tf_geom_to_body * CYLINDER_FIX_POSE,
                    radius=radius,
                    half_length=half_length,
                    material=tuple(vis_spec.rgba[:3]),
                )
            case mj.mjtGeom.mjGEOM_MESH:
                mesh_spec = mj_spec.mesh(vis_spec.meshname)
                if mesh_spec is not None:
                    mesh_path = mj_model_dir / mj_spec.meshdir / mesh_spec.file
                    material = materials.get(vis_spec.material)
                    builder.add_visual_from_file(
                        pose=rel_pose_to_parent * tf_geom_to_body,
                        filename=mesh_path.as_posix(),
                        scale=tuple(mesh_spec.scale),
                        material=material,
                    )
            case _:
                raise ValueError(
                    f"Visual geom type {vis_spec.type} not supported, for geom {vis_spec.name}"
                )


def add_colliders_to_actor_builder(
    builder: ActorBuilder,
    mj_spec: mj.MjSpec,
    mj_body: mj.MjsBody,
    rel_pose_to_parent: Pose,
    mj_model_dir: Path,
) -> None:
    colliders_specs = get_collider_specs_from_body(mj_body)
    for col_spec in colliders_specs:
        tf_geom_to_body = Pose(p=tuple(col_spec.pos), q=tuple(get_orientation(col_spec)))
        physx_material: PhysxMaterial | None = None
        if col_spec.condim == 3:  # noqa: PLR2004
            physx_material = PhysxMaterial(
                static_friction=col_spec.friction[0],
                dynamic_friction=col_spec.friction[0],
                restitution=0,
            )
        elif col_spec.condim == 1:
            physx_material = PhysxMaterial(static_friction=0, dynamic_friction=0, restitution=0)

        match col_spec.type:
            case mj.mjtGeom.mjGEOM_BOX:
                builder.add_box_collision(
                    pose=rel_pose_to_parent * tf_geom_to_body,
                    half_size=tuple(col_spec.size),
                    density=col_spec.density,
                    material=physx_material,
                )
            case mj.mjtGeom.mjGEOM_SPHERE:
                builder.add_sphere_collision(
                    pose=rel_pose_to_parent * tf_geom_to_body,
                    radius=col_spec.size[0],
                    density=col_spec.density,
                    material=physx_material,
                )
            case mj.mjtGeom.mjGEOM_CAPSULE:
                radius = col_spec.size[0]
                half_length = col_spec.size[1]
                if not np.isnan(col_spec.fromto[0]):
                    start, end = col_spec.fromto[:3], col_spec.fromto[3:]
                    pos = (end + start) / 2
                    quat = vec_to_quat(end - start)
                    tf_geom_to_body = Pose(p=tuple(pos), q=tuple(quat))
                    half_length = np.linalg.norm(end - start).item() / 2.0
                builder.add_capsule_collision(
                    pose=rel_pose_to_parent * tf_geom_to_body * CAPSULE_FIX_POSE,
                    radius=radius,
                    half_length=half_length,
                    density=col_spec.density,
                    material=physx_material,
                )
            case mj.mjtGeom.mjGEOM_CYLINDER:
                radius = col_spec.size[0]
                half_length = col_spec.size[1]
                if not np.isnan(col_spec.fromto[0]):
                    start, end = col_spec.fromto[:3], col_spec.fromto[3:]
                    pos = (end + start) / 2
                    quat = vec_to_quat(end - start)
                    tf_geom_to_body = Pose(p=tuple(pos), q=tuple(quat))
                    half_length = np.linalg.norm(end - start).item() / 2.0
                builder.add_cylinder_collision(
                    pose=rel_pose_to_parent * tf_geom_to_body * CYLINDER_FIX_POSE,
                    radius=radius,
                    half_length=half_length,
                    density=col_spec.density,
                    material=physx_material,
                )
            case mj.mjtGeom.mjGEOM_MESH:
                mesh_spec = mj_spec.mesh(col_spec.meshname)
                if mesh_spec is not None:
                    mesh_path = mj_model_dir / mj_spec.meshdir / mesh_spec.file
                    builder.add_convex_collision_from_file(
                        pose=rel_pose_to_parent * tf_geom_to_body,
                        filename=mesh_path.as_posix(),
                        scale=tuple(mesh_spec.scale),
                        density=col_spec.density,
                        material=physx_material,
                    )
            case _:
                raise ValueError(
                    f"Collider geom type {col_spec.type} not supported, for geom {col_spec.name}"
                )

        if col_groups := THOR_COLLISION_GROUPS.get(col_spec.classname.name):
            builder.collision_groups = col_groups


class MjcfAssetActorLoader(Generic[SceneT]):
    def __init__(self, scene: SceneT | None = None):
        self._scene: SceneT | None = scene

        self._spec: mj.MjSpec | None = None
        self._model_dir: Path | None = None
        self._materials: dict[str, RenderMaterial] = {}

    def set_scene(self, scene: SceneT) -> MjcfAssetActorLoader:
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
    ) -> ActorBuilder:
        """Loads an actor from a given MjSpec for a given scene

        The given spec is assumed to be the full MjSpec for a whole scene, and the section that
        corresponds to the actor we want to create starts at the body that has name given
        by the :root_body_name: parameter. If no root_body_name parameter is given, it's assumed
        that the whole spec corresponds for a single actor.

        Args:
            scene_spec: The spec for a whole scene, or for a single actor
            model_dir: Path to the folder that contains the xml model from which the spec was parsed
            root_body_name (optional): The name of the root of the actor
            floating_base (optional): Whether or not the base should be free to move
            materials (optional): A cache of parsed materials. If not given, will parse again here

        Returns:
            sapien.Entity: The generated actor

        Raises:
            ValueError: If the root body was not found in the given spec

        """
        assert self._scene is not None, "Must assign a valid Sapien scene before loading"

        self._spec = scene_spec
        self._model_dir = model_dir
        assert self._model_dir is not None, (
            "Must provide valid model_dir to the folder containing the mjcf model"
        )

        self._materials = parse_materials(scene_spec, model_dir) if materials is None else materials

        actor_builder = cast(Scene | ManiSkillScene, self._scene).create_actor_builder()

        mj_root_body: mj.MjsBody | None = None
        if root_body_name is not None:
            mj_root_body = self._spec.body(root_body_name)
            if mj_root_body is None:
                raise ValueError(
                    f"Couldn't find root-body with name {root_body_name} in the given MjSpec"
                )
        else:
            mj_root_body = self._spec.worldbody.first_body()

        assert mj_root_body is not None, "Something went wrong when loading an actor"

        if has_any_non_free_joint(mj_root_body):
            print(
                f"[WARN]: the given body {mj_root_body.name} has some non-free joints. "
                + "You might be better off using MjcfAssetArticulationLoader instead"
            )

        has_freejoint = any([jnt.type == mj.mjtJoint.mjJNT_FREE for jnt in mj_root_body.joints])
        body_type = "dynamic" if has_freejoint or floating_base else "static"

        def make_tree_recursive(
            builder: ActorBuilder,
            spec: mj.MjSpec,
            body: mj.MjsBody,
            rel_pose: Pose,
            model_dir: Path,
            materials: dict[str, RenderMaterial],
        ) -> None:
            add_visuals_to_actor_builder(builder, spec, body, rel_pose, model_dir, materials)
            add_colliders_to_actor_builder(builder, spec, body, rel_pose, model_dir)
            builder.set_physx_body_type(body_type)

            for child in body.bodies:
                body_pose = Pose(p=tuple(child.pos), q=tuple(get_orientation(child)))
                make_tree_recursive(
                    builder, spec, child, rel_pose * body_pose, model_dir, materials
                )

        make_tree_recursive(
            actor_builder, self._spec, mj_root_body, Pose(), self._model_dir, self._materials
        )

        return actor_builder

    def load_from_xml(
        self,
        xml_model: Path,
        floating_base: bool | None = None,
    ) -> ActorBuilder:
        """Loads an actor from a given mjcf model

        The given model is assumed to correspond to a single actor. If the mjcf model constains
        more than one articulation or actor, you should use :MjcfSceneLoader: instead. If the
        model corresponds to an articulation , then you should use :MjcfAssetArticulationLoader:

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
