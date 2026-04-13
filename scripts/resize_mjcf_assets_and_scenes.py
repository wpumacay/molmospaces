from dataclasses import dataclass
from pathlib import Path

import mujoco as mj
import numpy as np
import trimesh
from tqdm import tqdm

MJCF_DIR = Path(__file__).parent.parent / "assets" / "mjcf"
MJCF_ASSETS_THOR_DIR = MJCF_DIR / "objects" / "thor"
MJCF_SCENES_ITHOR_DIR = MJCF_DIR / "scenes" / "ithor"

MJC_VERSION = tuple(map(int, mj.__version__.split(".")))

COLLIDERS_CLASSNAMES = ("__DYNAMIC_MJT__", "__STRUCTURAL_MJT__", "__ARTICULABLE_DYNAMIC_MJT__")

def resize_thor_asset(model_path: Path) -> None:
    modified = False
    try:
        spec = mj.MjSpec.from_file(model_path.as_posix())

        for geom in spec.geoms:
            assert isinstance(geom, mj.MjsGeom)
            if geom.type != mj.mjtGeom.mjGEOM_MESH:
                continue
            if not geom.classname.name in COLLIDERS_CLASSNAMES:
                continue
            mesh_handle = spec.mesh(geom.meshname)
            if mesh_handle is None:
                continue

            mesh_path = model_path.parent / mesh_handle.file
            if not mesh_path.is_file():
                continue

            scale = mesh_handle.scale
            if np.allclose(scale, np.ones_like(scale), atol=1e-3       ):
                continue
            modified = True

            mesh = trimesh.load_mesh(mesh_path.as_posix())
            if isinstance(mesh, list):
                mesh = mesh[0]
            mesh.apply_scale(mesh_handle.scale)

            with open(mesh_path.resolve(), "w") as fhandle:
                trimesh.exchange.export.export_mesh(mesh, fhandle, file_type="obj")

            mesh_handle.scale = np.ones_like(scale)

        if modified:
            _ = spec.compile()
            with open(model_path.resolve(), "w") as fhandle:
                fhandle.write(spec.to_xml())

    except Exception as e:
        print(f"[ERROR]: couldn't resize thor assets '{model_path.stem}', error: {e}")

def update_scenes_scales(scene_path: Path) -> None:
    modified = False
    try:
        spec = mj.MjSpec.from_file(scene_path.as_posix())
        for geom in spec.geoms:
            assert isinstance(geom, mj.MjsGeom)
            if geom.type != mj.mjtGeom.mjGEOM_MESH:
                continue
            if not geom.classname.name in COLLIDERS_CLASSNAMES:
                continue
            mesh_handle = spec.mesh(geom.meshname)
            if mesh_handle is None:
                continue

            if not "objects/thor" in mesh_handle.file:
                continue

            scale = mesh_handle.scale
            if np.allclose(scale, np.ones_like(scale), atol=1e-3       ):
                continue
            modified = True
            mesh_handle.scale = np.ones_like(scale)

        if modified:
            _ = spec.compile()
            with open(scene_path.resolve(), "w") as fhandle:
                fhandle.write(spec.to_xml())


    except Exception as e:
        print(f"[ERROR]: couldn't resize scene '{scene_path.stem}', error: {e}")

def main() -> int:
    models_filepaths: list[Path] = []
    for candidate_xml in MJCF_ASSETS_THOR_DIR.rglob("*.xml"):
        if any(substr in candidate_xml.stem for substr in ("_old", "_fix", "_upt", "_orig", "_sc")):
            continue
        if "_mesh" not in candidate_xml.stem:
            continue
        models_filepaths.append(candidate_xml)

    for model_path in tqdm(models_filepaths):
        resize_thor_asset(model_path)


    scenes_filepaths: list[Path] = []
    for candidate_xml in MJCF_SCENES_ITHOR_DIR.glob("*.xml"):
        if any(substr in candidate_xml.stem for substr in ("_old", "_fix", "_upt", "_orig", "_sc")):
            continue
        scenes_filepaths.append(candidate_xml)

    for scene_path in tqdm(scenes_filepaths):
        update_scenes_scales(scene_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
