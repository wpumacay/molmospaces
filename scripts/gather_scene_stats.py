from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import msgspec
import mujoco as mj
import tyro
from p_tqdm import p_uimap
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent.parent
DEFAULT_SCENES_DIR = ROOT_DIR / "assets" / "mjcf" / "scenes"

SCENES_SUFFIXES_TO_SKIP = ["_orig", "_non_settled", "_ceiling"]


@dataclass
class Args:
    dataset: Literal["ithor", "procthor-10k", "procthor-objaverse", "holodeck-objaverse"]
    split: Literal["train", "val", "test"]

    scenes_dir: Path = DEFAULT_SCENES_DIR

    max_workers: int = 1


class ObjMetaInfo(msgspec.Struct):
    hash_name: str
    asset_id: str
    object_id: str
    category: str
    object_enum: str


@dataclass
class SceneStats:
    name: str
    is_ok: bool = True

    assets_ids: set[str] = field(default_factory=set)
    doors_ids: set[str] = field(default_factory=set)


def gather_scene_stats(scene_path: Path) -> SceneStats:
    stats = SceneStats(scene_path.stem)
    scene_metadata_path = scene_path.parent / f"{scene_path.stem}_metadata.json"
    if not scene_metadata_path.is_file():
        stats.is_ok = False
        return stats

    try:
        with open(scene_metadata_path, "rb") as fhandle:
            metadata = msgspec.json.decode(fhandle.read(), type=dict[str, dict[str, ObjMetaInfo]])

        spec = mj.MjSpec.from_file(scene_path.as_posix())

        def is_obj_articulated(body: mj.MjsBody) -> bool:
            if any(
                jnt.type in {mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE}
                for jnt in body.joints
            ):
                return True
            return any(is_obj_articulated(child) for child in body.bodies)

        for obj_name, obj_meta in metadata.get("objects", {}).items():
            stats.assets_ids.add(obj_meta.asset_id)
            if obj_meta.category == "Doorway" and (door_root_body := spec.body(obj_name)):
                if is_obj_articulated(door_root_body):
                    stats.doors_ids.add(obj_meta.asset_id)

    except Exception:
        stats.is_ok = False

    return stats


def main() -> int:
    args = tyro.cli(Args)
    dataset_id = "ithor" if args.dataset == "ithor" else f"{args.dataset}-{args.split}"
    if not args.scenes_dir.is_dir():
        print(f"[ERROR]: given scenes directory {args.scenes_dir} is not a valid directory")
        return 1

    dataset_dir = args.scenes_dir / dataset_id
    if not dataset_dir.is_dir():
        print(f"[ERROR]: dataset dir {dataset_dir} is not a valid directory")
        return 1

    scene_xml_pattern = (
        "FloorPlan*_physics.xml" if args.dataset == "ithor" else f"{args.split}_*.xml"
    )

    def is_valid_scene(scene_path: Path) -> bool:
        return all(suffix not in scene_path.stem for suffix in SCENES_SUFFIXES_TO_SKIP)

    scenes_xmls = [path for path in dataset_dir.glob(scene_xml_pattern) if is_valid_scene(path)]

    doors_ids_to_scenes: defaultdict[str, set[str]] = defaultdict(set)

    if args.max_workers > 1:
        results = p_uimap(gather_scene_stats, scenes_xmls, num_cpus=args.max_workers)
        for scene_stats in results:
            if scene_stats.is_ok:
                for door_id in scene_stats.doors_ids:
                    doors_ids_to_scenes[door_id].add(scene_stats.name)
    else:
        for scene_path in tqdm(scenes_xmls):
            scene_stats = gather_scene_stats(scene_path)
            if scene_stats.is_ok:
                for door_id in scene_stats.doors_ids:
                    doors_ids_to_scenes[door_id].add(scene_stats.name)

    results = dict(doors=dict(stats=dict(), summary=dict()))
    results["doors"]["stats"] = doors_ids_to_scenes
    results["doors"]["summary"]["ids"] = list(doors_ids_to_scenes.keys())

    results_file = ROOT_DIR / f"scene_stats_{dataset_id}.json"
    with open(results_file, "wb") as fhandle:
        fhandle.write(msgspec.json.format(msgspec.json.encode(results), indent=4))

    print("DONE gathering stats!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
