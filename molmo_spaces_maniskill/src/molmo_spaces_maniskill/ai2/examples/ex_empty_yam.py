from dataclasses import dataclass
from typing import Literal

import gymnasium as gym
import sapien
import tyro

from molmo_spaces_maniskill.ai2.env import MolmoSpacesEmptyEnv


@dataclass
class Args:
    env_id: Literal["MolmoSpacesEmptyEnv-v0"] = "MolmoSpacesEmptyEnv-v0"
    robot_id: Literal["i2rt-yam", "bi-i2rt-yam", "franka-droid"] = "i2rt-yam"


def main() -> int:
    args = tyro.cli(Args)

    env = gym.make(args.env_id, render_mode="human", robot_uids=args.robot_id)
    if not isinstance(env.unwrapped, MolmoSpacesEmptyEnv):
        print("[WARN]: env wrapper should be of type 'MolmoSpacesEmptyEnv'")

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
