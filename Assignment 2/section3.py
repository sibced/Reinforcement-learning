import imageio
import os

from section1 import *
from display_caronthehill import save_caronthehill_image

# TODO: make sure it works

def save_from_policy(policy: "function", suffix: str, length: int = 40):
    
    dir_name = f"frames_{suffix}"

    images = []
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    for i, (state, _, _, _) in enumerate(
        trajectory_yield(get_random_initial_state((1,)), policy, length)
    ):
        file_name = f"{dir_name}/model" + str(i) + ".jpg"
        save_caronthehill_image(float(state[0]), float(state[1]), out_file=file_name)
        images.append(imageio.imread(file_name))

    imageio.mimsave(f"video_model_{suffix}.gif", images)


if __name__ == "__main__":
    save_from_policy(policy_random, suffix="random_policy", length=40)
