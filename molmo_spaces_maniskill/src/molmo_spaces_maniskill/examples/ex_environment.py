from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import sapien
import tyro


@dataclass
class Args:
    scene_file: Path


def main() -> int:
    args = tyro.cli(Args)

    if not args.scene_file.is_file():
        print(f"[ERROR]: given scene file '{args.scene_file}' is not a valid file")
        return 1

    env = gym.make("MolmoSpacesEnv-v0", render_mode="human", scene_file=args.scene_file)

    _, _ = env.reset()
    viewer = env.render()

    while True:
        try:
            if isinstance(viewer, sapien.utils.Viewer):
                if viewer.window and viewer.window.key_down("q"):
                    viewer.close()
                    break
            env.step(action=None)
            env.render()
        except KeyboardInterrupt:
            print("User stopped the simulation")
            break

    env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
