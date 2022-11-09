"""
A script that generates a jastrow shape and saves it to a file.
"""

import itertools
import operator
import os
from math import cos, sin, pi
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.Image import new
from PIL.ImageDraw import Draw
from tqdm import tqdm

configs_constant = {
    "canvas_size_jastrow": 4000,
    "background_size": 8000,
    "overlap_threshold": 0.001,
    "init_angle": 20, # from 0 - 44
    "rotate_step": 1,
}

configs_variable = {
    "radius_outer": 1,  # from 0 - 1 / this basically defines resolution
    "thickness": 0.5,  # from 0 - 1
    "angle": 45, # from 0 - 45; note this is different from the init_angle which is for rotations
}

def update_configs(configs_variable: dict) -> dict:
    """
    Initial update of the configs_variable. 
    Called once at the beginning of the program.
    """
    configs_variable["radius_outer"] *= configs_constant["canvas_size_jastrow"] / 2
    configs_variable["radius_outer"] += configs_constant["canvas_size_jastrow"] / 2


    configs_variable["radius_inner"] = configs_variable["radius_outer"] - (
        (configs_variable["radius_outer"] - (configs_constant["canvas_size_jastrow"] / 2)) * configs_variable["thickness"]
    )

    return configs_variable

def get_vector(angle: float, length: float):
    """Compute the vector of the given angle and length
    params:
        angle (float): angle in radians
        length (float): length of the vector
    """
    angle = -angle
    angle -= 90
    angle = angle * pi / 180
    x, y = length * cos(angle), length * sin(angle)
    return (x, y)

def generate_shape(configs_variable: dict = configs_variable) -> Image:
    """
    Generate a uncropped/un-rotated jastrow shape using the 
    configs_variable.
    """
    # make empty canvas
    canvas = new("RGBA", tuple([configs_constant["canvas_size_jastrow"]] * 2), color=(0, 0, 0, 0))

    # draw the large circle
    Draw(canvas).pieslice(
        (
            *[(configs_constant["canvas_size_jastrow"] - configs_variable["radius_outer"])] * 2,
            *[configs_variable["radius_outer"]] * 2,
        ),
        start=-90 - configs_variable["angle"],
        end=-90 + configs_variable["angle"],
        fill=(255, 255, 255, 255),
    )

    # small circle
    Draw(canvas).pieslice(
        (
            *[(configs_constant["canvas_size_jastrow"] - configs_variable["radius_inner"])] * 2,
            *[configs_variable["radius_inner"]] * 2,
        ),
        start=0,
        end=360,
        fill=(0, 0, 0, 0),
    )

    return canvas


def compute_coordinates(canvas: Image) -> np.ndarray:
    """
    Compute the coordinates of the upper left and lower left 
    corners of the canvas' white pixel.
    """
    # translate into arrays
    canvas_array = np.array(canvas)
    # coordinates of the white pixels at the lowest four y values
    white_pixels = np.where(canvas_array[:, :, 3] == 255)
    white_pixels = np.array([white_pixels[0], white_pixels[1]])
    white_pixels = white_pixels[:, white_pixels[1].argsort()]
    coordinates_upper_left = white_pixels[:, :1]

    # coordinates of the white pixels at the highest four x values
    white_pixels = np.where(canvas_array[:, :, 3] == 255)
    white_pixels = np.array([white_pixels[0], white_pixels[1]])
    white_pixels = white_pixels[:, white_pixels[0].argsort()]
    white_pixels = white_pixels[:, ::-1]
    white_pixels = white_pixels[:, :4]
    # select the one with the lowest y value
    white_pixels = white_pixels[:, white_pixels[1].argsort()]
    coordinates_lower_left = white_pixels[:, :1]

    return coordinates_upper_left, coordinates_lower_left


def rotate_and_crop(canvas: Image, degree: int = 44) -> Image:
    """Rotate and crop the canvas to the smallest possible size.
        Params:
            canvas: the canvas to rotate and crop
            degree: the degree to rotate the canvas
        Returns:
            canvas_cropped: the cropped canvas
            canvas_rotated: the rotated and cropped canvas
    """
    coordinates_upper_left, coordinates_lower_left = compute_coordinates(canvas)

    canvas_cropped = canvas.crop(canvas.getbbox())
    canvas_rotated = canvas.rotate(degree, center=(coordinates_lower_left[1], coordinates_lower_left[0]), expand=True)
    canvas_rotated = canvas_rotated.crop(canvas_rotated.getbbox())

    return canvas_cropped, canvas_rotated


def count_size(canvas) -> int:
    canvas = np.array(canvas)
    return np.sum(canvas[:, :, 3] == 255)


def make_jastrow(shape_original: Image, shape_rotated: Image, configs_variable: dict, distance: float = None) -> Image:
    # place the first shape at the center of the background
    half_shape_size = tuple(i // 2 for i in shape_original.size)
    center_coordinates = tuple(i // 2 for i in (configs_constant["background_size"],) * 2)
    paste_coordinates = tuple(operator.sub(*i) for i in zip(center_coordinates, half_shape_size))

    background = new("RGBA", tuple([configs_constant["background_size"]] * 2), color=(0, 0, 0, 0))
    background.paste(shape_original, paste_coordinates, shape_original)

    # compute the motion vector for moving the second shape over the first shape
    O_coordinates_upper_left, O_coordinates_lower_left = compute_coordinates(shape_original)
    R_coordinates_upper_left, R_coordinates_lower_left = compute_coordinates(shape_rotated)
    motion_vector = O_coordinates_upper_left - R_coordinates_lower_left

    # find a new paste coordinate so that the previous lower left coordinate is now the upper left coordinate
    paste_coordinates = (int(paste_coordinates[0] + motion_vector[1]), int(paste_coordinates[1] + motion_vector[0]))

    if distance:
        """
        Testify the overlap of the two shapes here by pasting the rotated shape on the background.
        Then apply the second motion vector.
        """
        # compute the motion vector for moving the second shape away from the first shape
        motion_vector = get_vector(
            angle=configs_variable['angle'] / 2,
            length=distance,
        )
        paste_coordinates = (int(paste_coordinates[0] + motion_vector[0]), int(paste_coordinates[1] + motion_vector[1]))

    # # paste the second image
    background.paste(shape_rotated, paste_coordinates, shape_rotated)

    return background

def get_area_size(canvas: Image, color: str) -> int:
    canvas_array = np.array(canvas)
    match color:
        case "white":
            return np.sum(np.logical_and(canvas_array[:, :, 0] == 255, canvas_array[:, :, 1] == 255, canvas_array[:, :, 2] == 255))
        case "black":
            return np.sum(np.logical_and(canvas_array[:, :, 0] == 0, canvas_array[:, :, 1] == 0, canvas_array[:, :, 2] == 0))
        case "red":
            return np.sum(canvas_array[:, :, 0] == 255)
        case "green":
            return np.sum(canvas_array[:, :, 1] == 255)
        case "blue":
            return np.sum(canvas_array[:, :, 2] == 255)
        case "yellow":
            return np.sum(canvas_array[:, :, 0] == 255) + np.sum(canvas_array[:, :, 1] == 255)
        case _:
            raise ValueError(f"Unknown color {color}")

class RotateRecorder():
    def __init__(self):
        self.init_angle = configs_constant['init_angle']
        self.overlapped = 0
    
    def update_overlap(self, overlap: int):
        self.overlapped = overlap
    
    def update_angle(self):
        self.init_angle -= configs_constant['rotate_step']

def chunk_list(target_list: list, n: int):
    return [target_list[i:i + n] for i in range(0, len(target_list), n)] 

if __name__ == "__main__":

    # make grid
    grid_radius_outer = [0.8] # from 0 - 1; basically resolution
    grid_thickness = np.linspace(0.1, 0.9, 20) # from 0 - 1
    grid_angle = np.linspace(5, 45, 20) # from 1 - 45

    # distances
    num_distances = 40
    distances = np.linspace(0, 0.1, num_distances) * configs_constant['canvas_size_jastrow'] # from 0 - 1

    # make combinations of the settings
    combinations = list(itertools.product(grid_radius_outer, grid_thickness, grid_angle))
    configurations = [{"radius_outer": i[0], "thickness": i[1], "angle": i[2]} for i in combinations]
    configurations = [update_configs(config) for config in configurations]
    
    # divide the list into 3 pieces for parallel processing
    # configuration_chunked = chunk_list(target_list=configurations, n=int(len(configurations) / 3)) 

    for c in tqdm(configurations):
        if len([i for i in os.listdir(Path("samples")) if i.startswith(f"{c['radius_outer']}-{c['thickness']}-{c['angle']}")]) < num_distances:
            # init the original canvas + rotate recorder
            canvas_uncropped = generate_shape(configs_variable=c)
            rotate_recorder = RotateRecorder()

            # iteratively reduce the angle of the second shape until it touches (overlap) with the first shape
            # or stop when the angle is 0 (totally horizontal)
            while rotate_recorder.overlapped < configs_constant['overlap_threshold'] and rotate_recorder.init_angle > 0:
                canvas_cropped, canvas_rotated = rotate_and_crop(canvas_uncropped, degree=rotate_recorder.init_angle)

                # compute sum of the pixel size of both shapes:
                    # original
                    # rotated
                size_cropped = get_area_size(canvas=canvas_cropped, color="white")
                size_rotated = get_area_size(canvas=canvas_rotated, color="white")
                size_sum = size_cropped + size_rotated

                # make a jastrow by concatenating the shapes
                # compute pixel size of the jastrow
                jastrow = make_jastrow(canvas_cropped, canvas_rotated, configs_variable=c)
                size_jastrow = get_area_size(canvas=jastrow, color="white")

                # save the settings
                previous_config = c
                previous_angle = rotate_recorder.init_angle

                # update overlap area / angle
                rotate_recorder.update_overlap(overlap= 1 - (size_jastrow / size_sum))
                rotate_recorder.update_angle()

                # print(f"Angle: {rotate_recorder.init_angle}, Overlap: {rotate_recorder.overlapped}")

            # make the actual shape here
            canvas_cropped, canvas_rotated = rotate_and_crop(canvas_uncropped, degree=previous_angle)

            # build with the same settings + various distances
            for distance in distances:
                actual_jastrow = make_jastrow(canvas_cropped, canvas_rotated, configs_variable=previous_config, distance=distance)
                actual_jastrow = actual_jastrow.crop(actual_jastrow.getbbox())
                actual_jastrow.save(Path("samples", f"{c['radius_outer']}-{c['thickness']}-{c['angle']}-{distance}.png"))

# streamlit.io
# heroku
