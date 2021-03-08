import logging
import os

import click
import detectree as dtr
import geopandas as gpd
import rasterio as rio
from rasterio import features
from shapely import geometry

from jonction_tree_canopy import settings

CRS = 'epsg:2056'


@click.command()
@click.argument('orthophoto_filepath', type=click.Path(exists=True))
@click.argument('ref_raster_filepath', type=click.Path(exists=True))
@click.argument('dst_tiles_dir', type=click.Path(exists=True))
@click.argument('dst_filepath', type=click.Path())
@click.option('--tile-width', type=int, default=512, required=False)
@click.option('--tile-height', type=int, default=512, required=False)
@click.option('--dst-filename', required=False)
@click.option('--only-full-tiles', default=True, required=False)
@click.option('--keep-empty-tiles', default=False, required=False)
def main(orthophoto_filepath, ref_raster_filepath, dst_tiles_dir, dst_filepath,
         tile_width, tile_height, dst_filename, only_full_tiles,
         keep_empty_tiles):
    logger = logging.getLogger(__name__)

    img_filepaths = dtr.split_into_tiles(orthophoto_filepath,
                                         dst_tiles_dir,
                                         tile_width=tile_width,
                                         tile_height=tile_height,
                                         output_filename=dst_filename,
                                         only_full_tiles=only_full_tiles,
                                         keep_empty_tiles=keep_empty_tiles,
                                         custom_meta={
                                             'crs': CRS,
                                             'driver': 'GTiff',
                                             'nodata': 255
                                         })
    logger.info("dumped %d tiles to %s", len(img_filepaths), dst_tiles_dir)

    # get only the tiles that intersect the agglomeration extent (not to be
    # confused with the extent's bounding box)
    tile_gdf = gpd.GeoDataFrame(img_filepaths,
                                columns=['img_filepath'],
                                geometry=list(
                                    map(
                                        lambda img_filepath: geometry.box(
                                            *rio.open(img_filepath).bounds),
                                        img_filepaths)))

    with rio.open(ref_raster_filepath) as src:
        # Stay tuned to https://github.com/geopandas/geopandas/issues/921
        tile_ser = tile_gdf[tile_gdf.intersects([
            geometry.shape(geom)
            for geom, val in features.shapes(src.dataset_mask(),
                                             transform=src.transform)
            if val != 0
        ][0])]['img_filepath']

    tiles_to_rm_ser = tile_gdf['img_filepath'].loc[~tile_gdf.index.
                                                   isin(tile_ser.index)]
    for img_filepath in tiles_to_rm_ser:
        os.remove(img_filepath)
    logger.info("removed %d tiles that do not intersect with the extent of %s",
                len(tiles_to_rm_ser), dst_filepath)

    tile_ser.to_csv(dst_filepath, index=False, header=False)
    logger.info("dumped list of tile filepaths to %s", dst_filepath)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=settings.DEFAULT_LOG_FMT)

    main()
