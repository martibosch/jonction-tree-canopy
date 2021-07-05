import glob
import logging
from os import path

import click
import detectree as dtr
import joblib as jl
import numpy as np
import pandas as pd
import rasterio as rio

from jonction_tree_canopy import lidar_utils, settings


@click.command()
@click.argument("tile_filepath", type=click.Path(exists=True))
@click.argument("split_filepath", type=click.Path(exists=True))
@click.argument("models_dir", type=click.Path(exists=True))
@click.argument("lidar_dir", type=click.Path(exists=True))
@click.argument("validation_tiles_dir", type=click.Path(exists=True))
@click.argument("dst_filepath", type=click.Path())
@click.option("--high-veg-val", type=int, required=False)
@click.option("--num-opening-iterations", type=int, required=False)
@click.option("--num-dilation-iterations", type=int, required=False)
@click.option("--output-dtype", required=False)
@click.option("--output-tree-val", type=int, required=False)
@click.option("--output-nodata", type=int, required=False)
def main(
    tile_filepath,
    split_filepath,
    models_dir,
    lidar_dir,
    validation_tiles_dir,
    dst_filepath,
    high_veg_val,
    num_opening_iterations,
    num_dilation_iterations,
    output_dtype,
    output_tree_val,
    output_nodata,
):
    logger = logging.getLogger(__name__)

    # predict the tile using the trained classifier
    split_df = pd.read_csv(split_filepath, index_col=0)
    tile_cluster = split_df[split_df["img_filepath"] == tile_filepath][
        "img_cluster"
    ].iloc[0]
    pred_arr = dtr.Classifier().classify_img(
        tile_filepath, jl.load(path.join(models_dir, f"{tile_cluster}.joblib"))
    )

    # load lidar default settings
    high_veg_val = high_veg_val if high_veg_val else lidar_utils.HIGH_VEG_VAL
    num_opening_iterations = (
        num_opening_iterations
        if num_opening_iterations
        else lidar_utils.NUM_OPENING_ITERATIONS
    )
    num_dilation_iterations = (
        num_dilation_iterations
        if num_dilation_iterations
        else lidar_utils.NUM_DILATION_ITERATIONS
    )
    output_dtype = output_dtype if output_dtype else lidar_utils.OUTPUT_DTYPE
    output_tree_val = (
        output_tree_val if output_tree_val else lidar_utils.OUTPUT_TREE_VAL
    )
    output_nodata = output_nodata if output_nodata else lidar_utils.OUTPUT_NODATA

    # estimate the "ground-truth" mask with LIDAR data
    lidar_filepath = path.join(lidar_dir, lidar_utils.get_lidar_filename(tile_filepath))
    validation_tile_filepath = path.join(
        validation_tiles_dir, path.basename(tile_filepath)
    )
    _ = dtr.LidarToCanopy(
        three_threshold=high_veg_val,
        output_dtype=output_dtype,
        output_tree_val=output_tree_val,
        output_nodata=output_nodata,
    ).to_canopy_mask(
        lidar_filepath,
        lidar_utils.LIDAR_TREE_VALUES,
        tile_filepath,
        output_filepath=validation_tile_filepath,
        postprocess_func=lidar_utils.postprocess_canopy_mask,
        postprocess_func_args=[
            high_veg_val,
            num_opening_iterations,
            num_dilation_iterations,
            output_dtype,
            output_tree_val,
        ],
    )
    with rio.open(validation_tile_filepath) as src:
        obs_arr = src.read(1)

    # compute the confusion matrix and dump it to a file
    obs_ser = pd.Series(obs_arr.flatten(), name="obs")
    pred_ser = pd.Series(pred_arr.flatten(), name="pred")
    df = pd.crosstab(obs_ser, pred_ser) / len(obs_ser)
    logger.info("estimated accuracy score is %f", np.trace(df))
    df.to_csv(dst_filepath)
    logger.info("dumped confusion data frame to %s", dst_filepath)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=settings.DEFAULT_LOG_FMT)

    main()
