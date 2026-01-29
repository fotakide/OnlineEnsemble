"""
This module contains functions required to retrieve the tiling of the EMT Cube study area defined in a shapefile.
Dec 2024.
Eastern Macedonia and Thrace Data Cube edits and updates: Vangelis Fotakidis, Aristotle University of Thessaloniki

"""

import os
import logging

from typing import Tuple, Optional, Dict
import geopandas as gpd
from datacube.utils.geometry import CRS, Geometry
from datacube.model import GridSpec
from odc.io.text import split_and_check, parse_range_int

# ====================
# sources: 
# 1. https://github.com/opendatacube/odc-tools/blob/dff7b984464a4cc9d6bd9f6f444ef4a292c730d0/libs/dscache/odc/dscache/tools/tiling.py#L13-L41
# 2. https://github.com/digitalearthafrica/deafrica-waterbodies/blob/3517fc6985d89fa006c1644d2080ce73ada054f6/deafrica_waterbodies/tiling.py#L10
# ====================


epsg3035 = CRS("epsg:3035")

GRIDS = {
    **{
        f"lambert_gr_{n}": GridSpec(
            crs=epsg3035,
            tile_size=(48_000.0, 48_000.0),
            resolution=(-n, n),
            # origin=(2016000.0, 5376000.0),
            origin=(2015985.0, 5375985.0),
        )
        for n in (10, 20, 25, 30, 60)
    },
}


def _parse_gridspec_string(s: str) -> GridSpec:
    """
    "epsg:3035;30;1600"
    "epsg:3035;-30x30;1600x1600"
    """

    crs, res, shape = split_and_check(s, ";", 3)
    try:
        if "x" in res:
            res = tuple(float(v) for v in split_and_check(res, "x", 2))
        else:
            res = float(res)
            res = (-res, res)

        if "x" in shape:
            shape = parse_range_int(shape, separator="x")
        else:
            shape = int(shape)
            shape = (shape, shape)
    except ValueError:
        raise ValueError(f"Failed to parse gridspec: {s}") from None

    tsz = tuple(abs(n * res) for n, res in zip(res, shape))

    return GridSpec(crs=CRS(crs), tile_size=tsz, resolution=res, origin=(0, 0))

def _norm_gridspec_name(s: str) -> str:
    return s.replace("-", "_")


def parse_gridspec_with_name(
    s: str, grids: Optional[Dict[str, GridSpec]] = None
) -> Tuple[str, GridSpec]:
    if grids is None:
        grids = GRIDS

    named_gs = grids.get(_norm_gridspec_name(s))
    if named_gs is not None:
        return (s, named_gs)

    gs = _parse_gridspec_string(s)
    s = s.replace(";", "_")
    return (s, gs)


def get_tiles(resolution, aoi_gdf_path):
    grid_name = f"lambert_gr_{resolution}"
    grid, gridspec = parse_gridspec_with_name(grid_name)

    print(f"Using the grid {grid} with {gridspec}")
   
    # From the GridSpec get the crs and resolution.
    crs = gridspec.crs
    resolution = abs(gridspec.resolution[0])

    area_footprint = gpd.read_file(aoi_gdf_path).to_crs(crs)
    
    # Get the product footprint geopolygon.
    area_footprint = Geometry(geom=area_footprint.geometry[0], crs=crs)

    tiles = gridspec.tiles_from_geopolygon(geopolygon=area_footprint)
    tiles = list(tiles)

    # Get the individual tile geometries.
    tile_geometries = []
    tile_ids = []
    for tile in tiles:
        tile_idx, tile_idy = tile[0]
        tile_geometry = tile[1].extent.geom

        tile_geometries.append(tile_geometry)
        tile_ids.append(f"x{tile_idx:02d}_y{tile_idy:02d}")

    tiles_gdf = gpd.GeoDataFrame(data={"tile_ids": tile_ids, "geometry": tile_geometries}, crs=crs)
    
    tiles_gdf = tiles_gdf.to_crs(crs=4326)
    
    if not os.path.isfile("../geojson/emt_grid_v1.geojson"):
        tiles_gdf.to_file("../geojson/emt_grid_v1.geojson", driver="GeoJSON")

    return tiles_gdf

if __name__ == "__main__":
    
    resolution = 30
    aoi_gdf_path = "../../geojson/EMT_EL51_NUTS_RG_01M_2021_3035.geojson"
    
    get_tiles(resolution, aoi_gdf_path)
    
    grid_name = f"lambert_gr_{resolution}"
    grid, gridspec = parse_gridspec_with_name(grid_name)
    print(f"Using the grid {grid} with {gridspec}")
    
    crs = gridspec.crs
    resolution = abs(gridspec.resolution[0])

    area_footprint = gpd.read_file(aoi_gdf_path).to_crs(crs)
    # Get the product footprint geopolygon.
    area_footprint = Geometry(geom=area_footprint.geometry[0], crs=crs)
    
    tiles = gridspec.tiles_from_geopolygon(geopolygon=area_footprint)
    tiles = list(tiles)
    print(len(tiles))
    
    
    tile_geometries = []
    tile_geometries_3035 = []
    region_code = []
    ix = []
    iy = []
    for tile in tiles:
        tile_idx, tile_idy = tile[0]
        tile_geometry = tile[1].extent.geom

        tile_geometries_3035.append(tile_geometry)
        tile_geometries.append(tile_geometry)
        region_code.append(f"x{tile_idx:02d}_y{tile_idy:02d}")
        ix.append(tile_idx)
        iy.append(tile_idy)

    tiles_gdf = gpd.GeoDataFrame(data={"region_code": region_code, "ix":ix, "iy":iy,"geometry":tile_geometries}, crs=4326)

    if not os.path.isfile("../../geojson/emt_grid_v1.geojson"):
        tiles_gdf.to_file("../../geojson/emt_grid_v1.geojson", driver="GeoJSON")