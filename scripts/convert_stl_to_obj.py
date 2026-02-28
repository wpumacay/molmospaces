import shutil
from dataclasses import dataclass
from pathlib import Path

import mujoco as mj
import numpy as np
import trimesh
import tyro

@dataclass
class Args:
    model: Path
    output_dir: Path


def main() -> int:
    args = tyro.cli(Args)

    if not args.model.is_file():
        print(f"Given model file @ '{args.model}' is not a valid file")
        return 1
    if not args.output_dir.is_dir():
        args.output_dir.mkdir(exist_ok=True)

    spec = mj.MjSpec.from_file(args.model.as_posix())

    @dataclass
    class MeshInfo:
        name: str
        path: Path
        mesh: mj.MjsMesh

    meshes_stl: list[MeshInfo] = []
    for mesh_spec in spec.meshes:
        assert isinstance(mesh_spec, mj.MjsMesh)
        if mesh_spec.file.lower().endswith(".stl"):
            meshes_stl.append(MeshInfo(
                name=mesh_spec.name,
                path=args.model.parent / spec.meshdir / mesh_spec.file,
                mesh=mesh_spec,
            ))

    for info in meshes_stl:
        stl_path = info.path
        obj_path = stl_path.parent / f"{stl_path.stem}.obj"

        tmesh = trimesh.load(stl_path.resolve())
        tmesh.export(obj_path.resolve(), file_type="obj")

        info.mesh.file = obj_path.relative_to(args.model.parent / spec.meshdir).as_posix()

    _ = spec.compile()
    new_model = args.model.parent / f"{args.model.stem}_new.xml"
    with open(new_model, "w") as fhandle:
        fhandle.write(spec.to_xml())

    new_assets_dir = args.output_dir / "assets"
    new_assets_dir.mkdir(exist_ok=True)

    assets_deps: list[Path] = []
    for asset_dep in mj.mju_getXMLDependencies(new_model.as_posix()):
        asset_path = Path(asset_dep)
        if asset_path.stem == new_model.stem:
            continue
        assets_deps.append(asset_path)

    for src_asset in assets_deps:
        dst_asset = new_assets_dir / src_asset.name
        shutil.copyfile(src_asset, dst_asset)

    dst_model = args.output_dir / args.model.name
    shutil.copyfile(new_model, dst_model)

    print(f"Successfully created {dst_model}")

    return 0




if __name__ == "__main__":
    raise SystemExit(main())
