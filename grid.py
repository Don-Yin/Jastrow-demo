import os
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm import tqdm

images = os.listdir(Path("generated_samples"))
images = [i for i in images if i.endswith(".png")]

attributes = [i.removesuffix(".png") for i in images]
attributes = [i.split("-") for i in attributes]  # outer radius / thickness / radian


class MakeGrid:
    def __init__(self):
        self.outer_radius = sorted(list(set([float(i[0]) for i in attributes])))
        self.thickness = sorted(list(set([float(i[1]) for i in attributes])))
        self.radian = sorted(list(set([float(i[2]) for i in attributes])))

        self.attribute_names = ["radian", "thickness", "outer_radius"]

        plt.style.use("dark_background")

        # -----------------------------

        self.make_dirs()
        for name in self.attribute_names:
            self.main(name)

    def make_dirs(self):
        for i in self.attribute_names:
            os.makedirs(f"grid_{i}", exist_ok=True)

    def read_image(self, args_dict):  # dict: outer_radius, thickness, radian
        return plt.imread(
            f"generated_samples/{args_dict['outer_radius']}-{args_dict['thickness']}-{args_dict['radian']}.png"
        )

    def main(self, attribute_for_slider: str):
        remaining_attributes = [i for i in self.attribute_names if i != attribute_for_slider]

        for i in tqdm(getattr(self, attribute_for_slider)):
            if not os.path.exists(Path(f"grid_{attribute_for_slider}", f"{i}.png")):
                combs: list[tuple] = list(product(*[getattr(self, j) for j in remaining_attributes]))
                combs: list[dict] = [{attribute_for_slider: i, **dict(zip(remaining_attributes, j))} for j in combs]

                images = [self.read_image(i) for i in tqdm(combs)]

                fig = plt.figure(figsize=(20.0, 20.0), facecolor="black")

                grid = ImageGrid(
                    fig,
                    111,  # similar to subplot(111)
                    nrows_ncols=(20, 20),  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes in inch.
                )

                for ax, im in zip(grid, images):
                    ax.imshow(im, cmap="gray_r")

                plt.savefig(Path(f"grid_{attribute_for_slider}", f"{i}.png"), pad_inches=0.0)

        # images = [
        #     self.read_image(outer_radius=self.outer_radius[0], thickness=self.thickness[0], radian=self.radian[0]),
        #     self.read_image(outer_radius=self.outer_radius[1], thickness=self.thickness[1], radian=self.radian[1]),
        #     self.read_image(outer_radius=self.outer_radius[2], thickness=self.thickness[2], radian=self.radian[2]),
        # ]

        # fig = plt.figure(figsize=(4.0, 4.0), facecolor="black")

        # grid = ImageGrid(
        #     fig,
        #     111,  # similar to subplot(111)
        #     nrows_ncols=(1, 3),  # creates 2x2 grid of axes
        #     axes_pad=0.1,  # pad between axes in inch.
        # )

        # for ax, im in zip(grid, images):
        #     ax.imshow(im, cmap="gray_r")

        # # save plt image
        # plt.savefig("grid.png", pad_inches=0.0)


if __name__ == "__main__":
    MakeGrid()
    pass
