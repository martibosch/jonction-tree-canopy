from os import path

import pandas as pd
import rasterio as rio
from laspy import file as lp_file
from rasterio import enums, features
from scipy import ndimage as ndi
from shapely import geometry

HIGH_VEG_VAL = 15
NUM_OPENING_ITERATIONS = 2
NUM_DILATION_ITERATIONS = 2
OUTPUT_DTYPE = 'uint8'
OUTPUT_TREE_VAL = 255
OUTPUT_NODATA = 0


def get_lidar_filename(tile_filepath):
    return f'lidar-{path.splitext(path.basename(tile_filepath))[0]}.las'


class LidarTileGenerator:
    def __init__(self,
                 high_veg_val=None,
                 num_opening_iterations=None,
                 num_dilation_iterations=None,
                 output_dtype=None,
                 output_tree_val=None,
                 output_nodata=None,
                 log_method=None):
        self.high_veg_val = high_veg_val if high_veg_val else HIGH_VEG_VAL
        self.num_opening_iterations = num_opening_iterations if \
            num_opening_iterations else NUM_OPENING_ITERATIONS
        self.num_dilation_iterations = num_dilation_iterations if \
            num_dilation_iterations else NUM_DILATION_ITERATIONS
        self.output_dtype = output_dtype if output_dtype else OUTPUT_DTYPE
        self.output_tree_val = output_tree_val if output_tree_val else \
            OUTPUT_TREE_VAL
        self.output_nodata = output_nodata if output_nodata else OUTPUT_NODATA

        self.log_method = log_method if log_method else print

    def make_response_tile(self, tile_filepath, lidar_filepath, dst_filepath):
        with lp_file.File(lidar_filepath, mode='r') as src:
            c = src.get_classification()
            x = src.x
            y = src.y

        cond = ((c == 4) ^ (c == 5))
        lidar_df = pd.DataFrame({
            'class_val': c[cond],
            'x': x[cond],
            'y': y[cond]
        })

        with rio.open(tile_filepath) as src:
            arr = features.rasterize(shapes=[
                (geom, class_val) for geom, class_val in zip([
                    geometry.Point(x, y)
                    for x, y in zip(lidar_df['x'], lidar_df['y'])
                ], lidar_df['class_val'])
            ],
                                     out_shape=src.shape,
                                     transform=src.transform,
                                     merge_alg=enums.MergeAlg('ADD'))
            meta = src.meta.copy()

        output_arr = ndi.binary_dilation(
            ndi.binary_opening(arr >= self.high_veg_val,
                               iterations=self.num_opening_iterations),
            iterations=self.num_dilation_iterations).astype(
                self.output_dtype) * self.output_tree_val

        meta.update(dtype=self.output_dtype,
                    count=1,
                    nodata=self.output_nodata)
        # response_tile_filepath = path.join(self.response_dir,
        #                                    path.basename(tile_filepath))
        with rio.open(dst_filepath, 'w', **meta) as dst:
            dst.write(output_arr, 1)
        log_string = 'Dumped response tile for ' + \
            f'{tile_filepath} to {dst_filepath}'
        self.log_method(log_string)

        return dst_filepath
