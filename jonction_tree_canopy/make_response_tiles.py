import logging
from os import path

import click
import pandas as pd

from jonction_tree_canopy import lidar_utils, settings


@click.command()
@click.argument('split_csv_filepath', type=click.Path(exists=True))
@click.argument('lidar_dir', type=click.Path(exists=True))
@click.argument('response_dir', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--high-veg-val', type=int, required=False)
@click.option('--num-opening-iterations', type=int, required=False)
@click.option('--num-dilation-iterations', type=int, required=False)
@click.option('--output-dtype', required=False)
@click.option('--output-tree-val', type=int, required=False)
@click.option('--output-nodata', type=int, required=False)
def main(split_csv_filepath, lidar_dir, response_dir, output_filepath,
         high_veg_val, num_opening_iterations, num_dilation_iterations,
         output_dtype, output_tree_val, output_nodata):
    logger = logging.getLogger(__name__)

    split_df = pd.read_csv(split_csv_filepath)
    tile_filepaths = split_df[split_df['train']]['img_filepath']

    ltg = lidar_utils.LidarTileGenerator(
        high_veg_val=high_veg_val,
        num_opening_iterations=num_opening_iterations,
        num_dilation_iterations=num_dilation_iterations,
        output_dtype=output_dtype,
        output_tree_val=output_tree_val,
        output_nodata=output_nodata)
    response_tile_filepaths = []
    for tile_filepath in tile_filepaths:
        lidar_filepath = path.join(
            lidar_dir, lidar_utils.get_lidar_filename(tile_filepath))
        response_tile_filepath = ltg.make_response_tile(
            tile_filepath, lidar_filepath,
            path.join(response_dir, path.basename(tile_filepath)))
        response_tile_filepaths.append(response_tile_filepath)

    pd.Series(response_tile_filepaths).to_csv(output_filepath, header=False)
    logger.info("Dumped list of response tiles to %s", output_filepath)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=settings.DEFAULT_LOG_FMT)

    main()
