from os import path

import pandas as pd
import rasterio as rio
from laspy import file as lp_file
from rasterio import enums, features
from scipy import ndimage as ndi
from shapely import geometry

HIGH_VEG_VAL = 15
LIDAR_TREE_VALUES = [4, 5]
NUM_OPENING_ITERATIONS = 2
NUM_DILATION_ITERATIONS = 2
OUTPUT_DTYPE = "uint8"
OUTPUT_TREE_VAL = 255
OUTPUT_NODATA = 0


def get_lidar_filename(tile_filepath):
    return f"lidar-{path.splitext(path.basename(tile_filepath))[0]}.las"


def postprocess_canopy_mask(
    canopy_mask_arr,
    tree_threshold,
    num_opening_iterations,
    num_dilation_iterations,
    output_dtype,
    output_tree_val,
):
    return (
        ndi.binary_dilation(
            ndi.binary_opening(
                canopy_mask_arr >= tree_threshold, iterations=num_opening_iterations
            ),
            iterations=num_dilation_iterations,
        ).astype(output_dtype)
        * output_tree_val
    )
