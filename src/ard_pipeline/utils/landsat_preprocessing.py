import logging

import pystac_client
from pystac import Item, ItemCollection
from odc.stac import load, configure_s3_access

import odc.geo.xr
from odc.geo.geom import BoundingBox
from odc.geo.geobox import GeoBox

import xarray as xr
import numpy as np
import geopandas as gpd

from odc.algo import to_f32, mask_cleanup
from dea_tools.spatial import xr_rasterize

import odc.geo.xr

from utils import nbar
from utils.stac_catalog import http_to_s3_url

import enum

from xrspatial import slope, aspect

import yaml
import gc
import tempfile
from pathlib import Path

class QABits(enum.Enum):
    # https://github.com/jgrss/geowombat/blob/a3088c6ef61260b5665a0187dfd8ffbf275524e1/src/geowombat/radiometry/qa.py
    # https://github.com/flowers-huang/cs325b-wildfire/blob/ee7614fb94ba247ce5cd030ce2a766a23922bd91/src/composite.py#L28
    """QA bits.

    Reference:
        https://www.usgs.gov/landsat-missions/landsat-project-documents
    """

    landsat_c2_l2 = {
        'fill': 0,
        'dilated_cloud': 1,
        'cirrus': 2,
        'cloud': 3,
        'cloud_shadow': 4,
        'snow': 5,
        'clear': 6,
        'water': 7,
    }

def load_scene(items_sr, items_ang, geobox, bands='all', resampling='nearest'):
    if bands=='all':
        bands=["blue", "green", "red", "nir08", "swir16", "swir22", "qa_pixel"]
    
    logging.debug("Pre-processing: Loading SR BANDS with odc-stac")
    sr_bands = load(
        items=items_sr,
        bands=bands,
        like=geobox,
        resampling={
            "*": resampling, #bilinear
        },
        groupby='time', fail_on_error=True,
        chunks=dict(time=1, y=2048, x=2048),
        patch_url=http_to_s3_url,
    ).compute()

    logging.debug("Pre-processing: Loading ANGLES with odc-stac")
    angles = load(
        items=items_ang,
        bands=["SZA", "SAA", "VAA", "VZA"],
        like=geobox,
        resampling={
            "*": resampling, #bilinear
        },
        groupby='time', fail_on_error=True,
        chunks=dict(time=1, y=2048, x=2048),
        patch_url=http_to_s3_url,
    ).compute()

    ds_scene = xr.merge([sr_bands, angles])

    for band in ["blue", "green", "red", "nir08", "swir16", "swir22"]:
        ds_scene[band] = ds_scene[band].astype('uint16')
        ds_scene[band].attrs['nodata'] = 0
    for band in ["qa_pixel"]:
        ds_scene[band] = ds_scene[band].where(~ds_scene[band].isnull(), other=1)
        ds_scene[band] = ds_scene[band].astype('uint16')
        ds_scene[band].attrs['nodata'] = 1
    for band in ["SZA", "SAA", "VAA", "VZA"]:
        ds_scene[band] = ds_scene[band].astype('float32')
        ds_scene[band].attrs['nodata'] = np.nan

    return ds_scene.isel(time=0)


def intercalibration_Roy_to_OLI(ds_nbart, platform):
    # source: https://github.com/google/earthengine-community/blob/master/tutorials/landsat-etm-to-oli-harmonization/script.js
    # Define slopes and intercepts for each band
    if platform=='LANDSAT_5':
        slopes = [0.9785, 0.9542, 0.9825, 1.0073, 1.0171, 0.9949]
        intercepts = [-0.0095, -0.0016, -0.0022, -0.0021, -0.0030, 0.0029]
    elif platform=='LANDSAT_7':
        slopes = [0.8474, 0.8483, 0.9047, 0.8462, 0.8937, 0.9071]
        intercepts = [0.0003, 0.0088, 0.0061, 0.0412, 0.0254, 0.0172]

    bands = ['nbart_blue', 'nbart_green', 'nbart_red', 'nbart_nir08', 'nbart_swir16', 'nbart_swir22']
    # Apply harmonization
    for band, slope, intercept in zip(bands, slopes, intercepts):
        attrs = ds_nbart[band].attrs.copy()
        ds_nbart[band] = ds_nbart[band] * slope + (intercept*10000)
        ds_nbart[band] = ds_nbart[band].where(ds_nbart[band]>0, other=-999).astype('int16')
        ds_nbart[band].attrs = attrs
    
    return ds_nbart


def mask_scene(ds_scene):
    logging.debug("Pre-processing: Building the mask")
    # Get cloud and cloud shadow mask
    mask_items = ['fill','dilated_cloud','cirrus','cloud','cloud_shadow','snow']
    mask_bitfields = [
        getattr(QABits, "landsat_c2_l2").value[mask_item] for mask_item in mask_items
        ]
    # Source: https://stackstac.readthedocs.io/en/v0.3.0/examples/gif.html
    bitmask = 0
    for field in mask_bitfields:
        bitmask |= 1 << field

    # Get cloud mask
    bad_pixels_mask = ds_scene["qa_pixel"].astype(int) & bitmask != 0
    dilated_bad_pixels_mask = mask_cleanup(bad_pixels_mask, [("opening", 5), ("dilation", 6)])

    # Get the nodata mask, just for the main band
    nodata_mask = (ds_scene.nir08.isnull())
    combined_mask = (dilated_bad_pixels_mask | nodata_mask)

    for band in ["blue", "green", "red", "nir08", "swir16", "swir22"]:
        ds_scene[band] = ds_scene[band].astype('int16').where(~combined_mask, other=-999)
        ds_scene[band].attrs['nodata'] = -999
    
    return_bands = [b for b in ds_scene.data_vars if b!='qa_pixel']
    return ds_scene[return_bands]


def remove_null(ds_scene):
    # Get the nodata mask, just for the main band
    nodata_mask = (ds_scene.nir08.isnull())

    for band in ["blue", "green", "red", "nir08", "swir16", "swir22"]:
        ds_scene[band] = ds_scene[band].astype('int16').where(~nodata_mask, other=-999)
        ds_scene[band].attrs['nodata'] = -999
    
    return_bands = [b for b in ds_scene.data_vars if b!='qa_pixel']
    return ds_scene[return_bands]


def generate_nbar(ds_scene, tmp_netcdf_dir=None):
    """
    Only for Landsat
    """
    # Handle memory -- maybe delete
    crs = ds_scene.odc.crs
    coreg_attrs = ds_scene.attrs.copy()

    with tempfile.NamedTemporaryFile(delete=False, suffix='.nc', dir=tmp_netcdf_dir) as tmp_file:
        tmp_netcdf = tmp_file.name
        logging.info(f"Temporary netCDF file created at: {tmp_netcdf}")
        ds_scene.to_netcdf(tmp_netcdf)

    logging.info("Pre-processing: opening NETCDF file")
    with xr.open_dataset(tmp_netcdf).chunk(dict(y=512,x=512)) as ds_scene:
        ds_scene = ds_scene.drop_vars(['spatial_ref']).odc.assign_crs(crs)
        ds_scene.attrs = coreg_attrs

        logging.info("Pre-processing: NBAR calculation")
        sr_bands = ["blue", "green", "red", "nir08", "swir16", "swir22"]
        ang_bands = ["SZA", "SAA", "VAA", "VZA"]

        xr_cube_nbar = nbar.correct_brdf_c_factor(
            ds_sr_bands=ds_scene[sr_bands].chunk(dict(y=512,x=512)), 
            ds_angles=ds_scene[ang_bands].chunk(dict(y=512,x=512)), 
            ).compute()

        logging.info("Pre-processing: Replace nan with -999 to be compatible with the odc-product")
        for band in sr_bands:
            xr_cube_nbar[band] = xr_cube_nbar[band].where(~xr_cube_nbar[band].isnull(), other=-999).astype('int16')
            xr_cube_nbar[band].attrs['nodata'] = -999

        for band in ["view_zenith", "solar_zenith", "relative_azimuth", "solar_azimuth"]:
            xr_cube_nbar[band].attrs['nodata'] = np.nan

        rename_dict = {band: f"nbar_{band}" for band in sr_bands}
        xr_cube_nbar = xr_cube_nbar.rename(rename_dict) # Rename variables

    gc.collect()

    return xr_cube_nbar


def correct_topo(ds_scene, dem, method='empirical-rotation', tmp_netcdf_dir=None, robust=False, njobs=1):
    """
    Only for Landsat

    method (Optional[str]): The method to apply. Choices are ['scs+c', 'c-correction', 'empirical-rotation'].
    """
    # Handle memory -- maybe delete
    crs = ds_scene.odc.crs
    coreg_attrs = ds_scene.attrs.copy()

    with tempfile.NamedTemporaryFile(delete=False, suffix='.nc', dir=tmp_netcdf_dir) as tmp_file:
        tmp_netcdf = tmp_file.name
        logging.info(f"Temporary netCDF file created at: {tmp_netcdf}")
        ds_scene.to_netcdf(tmp_netcdf)

    logging.info("Pre-processing: opening NETCDF file")
    with xr.open_dataset(tmp_netcdf).chunk(dict(y=512,x=512)) as ds_scene:
        ds_scene = ds_scene.drop_vars(['spatial_ref']).odc.assign_crs(crs)
        ds_scene.attrs = coreg_attrs

        logging.info(f"Pre-processing: NBAR-T calculation ({method})")
        sr_bands = ["nbar_blue", "nbar_green", "nbar_red", "nbar_nir08", "nbar_swir16", "nbar_swir22"]
        ang_bands = ["view_zenith", "solar_zenith", "relative_azimuth", "solar_azimuth"]

        if method=='empirical-rotation':
            xr_cube_nbart = nbar.correct_topo_empirical_rotation(
                ds_nbart_bands=ds_scene[sr_bands].chunk(dict(y=512,x=512)), 
                ds_angles=ds_scene[ang_bands].chunk(dict(y=512,x=512)), 
                ds_dem=dem.chunk(dict(y=512,x=512)), 
                robust=robust, njobs=njobs
                ).compute()
        elif method=='c-correction':
            xr_cube_nbart = nbar.correct_topo_c_correction(
                ds_nbart_bands=ds_scene[sr_bands].chunk(dict(y=512,x=512)), 
                ds_angles=ds_scene[ang_bands].chunk(dict(y=512,x=512)), 
                ds_dem=dem.chunk(dict(y=512,x=512)),
                robust=robust, njobs=njobs
                ).compute()
        elif method=='scs+c':
            xr_cube_nbart = nbar.correct_topo_sun_canopy_sencor_c(
                ds_nbart_bands=ds_scene[sr_bands].chunk(dict(y=512,x=512)), 
                ds_angles=ds_scene[ang_bands].chunk(dict(y=512,x=512)), 
                ds_dem=dem.chunk(dict(y=512,x=512)),
                robust=robust, njobs=njobs
                ).compute()
        else:
            raise RuntimeError('Check the mothod you asked for. Available options: empirical-rotation, c-correction, scs+c')

        logging.info("Pre-processing: Replace nan with -999 to be compatible with the odc-product")
        for band in sr_bands:
            xr_cube_nbart[band] = xr_cube_nbart[band].where(~xr_cube_nbart[band].isnull(), other=-999).astype('int16')
            xr_cube_nbart[band].attrs['nodata'] = -999

        for band in ["view_zenith", "solar_zenith", "relative_azimuth", "solar_azimuth"]:
            xr_cube_nbart[band].attrs['nodata'] = np.nan

        rename_dict = {band: band.replace("nbar_", "nbart_") for band in sr_bands}
        xr_cube_nbart = xr_cube_nbart.rename(rename_dict) # Rename variables

        xr_cube_nbart.attrs['calibration'] = 'topographic-adjusted'
        xr_cube_nbart.attrs['method'] = method

    gc.collect()

    return xr_cube_nbart
    

def scale_landsat(ds_landsat, reflect_scale_info):
    """
    Scales reflectance values to the range [0, 1] from original unsigned 16-bit integers.
    Use the scaling info retrieved from STAC for a surface reflectance band.
    """
    # ---------------- Scale ---------------------------
    quality_band = ['qa_pixel']
    crs_info = ds_landsat.odc.crs

    ds_landsat = ds_landsat.drop_vars(['spatial_ref'])
    bands = [b for b in list(ds_landsat.keys()) if b not in (quality_band)]
    scale_multiplier = 1

    # https://www.usgs.gov/faqs/how-do-i-use-a-scale-factor-landsat-level-2-science-products
    # https://github.com/brazil-data-cube/metadata/blob/48539ea11a4b2f61689bea2b98e4b872284a2b5f/landsat.json#L266

    ds_landsat[bands] = ds_landsat[bands].where((ds_landsat[bands]>=7273) & (ds_landsat[bands]<=43636))
    ds_landsat[bands] = to_f32(ds_landsat[bands], scale=reflect_scale_info["scale"], offset=reflect_scale_info["offset"]) * scale_multiplier # https://www.mdpi.com/2072-4292/15/14/3636

    return ds_landsat.odc.assign_crs(crs_info)


# def lazy_odc_stac_cloudfree_scaled_landsat(items_filt, roi_bbox, emt, epsg, bands='all', resampling='cubic'):

#     if bands=='all':
#         bands=["red", "green", "blue", "nir08", "swir16", "swir22", "qa_pixel"]
    
#     logging.debug("Pre-processing: Loading with odc-stac")
#     xr_cube = load(
#         items=items_filt,
#         # geobox=
#         bbox=roi_bbox,
#         bands=bands,
#         chunks=dict(y=1024, x=1024),
#         crs=f'epsg:{epsg}',
#         # resolution=30,
#         groupby='time',
#         fail_on_error=False,
#         resampling={
#             "*": resampling, #bilinear
#         },
#     ).compute()

#     logging.debug("Pre-processing: Masking")
#     mask_qa = (qa_mask(xr_cube.qa_pixel.data,'cloud')
#             | qa_mask(xr_cube.qa_pixel.data,'mid cloud')
#             | qa_mask(xr_cube.qa_pixel.data,'snow') 
#             | qa_mask(xr_cube.qa_pixel.data,'shadow')
#             | qa_mask(xr_cube.qa_pixel.data,'high cirrus') 
#             | qa_mask(xr_cube.qa_pixel.data,'mid cirrus')
#     )
#     mask0qa = (xr_cube.qa_pixel == 0).data
#     mask0bn = (xr_cube.red == 0).data

#     mask = (mask_qa | mask0qa | mask0bn)

#     bands = [b for b in xr_cube.data_vars if b!='qa_pixel']
#     xr_cube_cf = (xr_cube.astype('int16'))[bands].where(~mask, other=-999)

#     xr_cube_cf = xr_cube_cf.where(xr_cube_cf>0, np.nan)

#     reflect_scale_info = items_filt[0].assets["red"].extra_fields["raster:bands"][0]
#     xr_cube_cf = scale_landsat(xr_cube_cf, reflect_scale_info)

#     if emt is not None:
#         xr_cube_cf = apply_mask(xr_cube_cf, emt)

#     return xr_cube_cf


def apply_mask(ds, mask_gdf):
    xr_mask = xr_rasterize(mask_gdf, da=ds)   
    for band in ds.data_vars:
        ds[band] = ds[band].where(xr_mask, other=ds[band].nodata)
    return ds


def median_int16(ds, nodata=-999):
    crs = ds.odc.crs
    ds = ds.median(dim='time').compute() # in case it hassn't been computed
    ds = xr.where(ds>0, ds*10000, nodata).astype('int16')
    for var in ds.data_vars:
        ds[var].attrs['nodata'] = nodata
    return ds.odc.assign_crs(crs)


def landsat_scene_lazy_odc_stac_cloudfree_scaled(one_item, geobbox_intersection, emt, resampling='cubic'):
    
    landsat_scene = load(
        items=ItemCollection([one_item]),
        geobox=geobbox_intersection,
        bands=["red", "green", "blue", "nir08", "swir16", "swir22", "qa_pixel"],
        chunks=dict(y=1024, x=1024),
        groupby='time',
        fail_on_error=True,
        resampling={
            "*": resampling,
        },
    )

    mask_qa = (qa_mask(landsat_scene.qa_pixel.data,'cloud')
            | qa_mask(landsat_scene.qa_pixel.data,'mid cloud')
            | qa_mask(landsat_scene.qa_pixel.data,'snow') 
            | qa_mask(landsat_scene.qa_pixel.data,'shadow')
            | qa_mask(landsat_scene.qa_pixel.data,'high cirrus') 
            | qa_mask(landsat_scene.qa_pixel.data,'mid cirrus')
    )
    mask0qa = (landsat_scene.qa_pixel == 0).data
    mask0bn = (landsat_scene.red == 0).data

    mask = (mask_qa | mask0qa | mask0bn)

    bands = [b for b in landsat_scene.data_vars if b!='qa_pixel']
    landsat_scene_cf = (landsat_scene.astype('int16'))[bands].where(~mask, other=-999)

    landsat_scene_cf = landsat_scene_cf.where(landsat_scene_cf>0, np.nan)

    reflect_scale_info = one_item.assets["red"].extra_fields["raster:bands"][0]
    landsat_scene_cf = scale_landsat(landsat_scene_cf, reflect_scale_info)

    emt_mask = xr_rasterize(emt, da=landsat_scene_cf)   

    landsat_scene_cf = landsat_scene_cf.where(emt_mask)

    for var in landsat_scene_cf.data_vars:
        landsat_scene_cf[var].attrs['nodata'] = -999

    return landsat_scene_cf


def geomedian_with_count_int16(geomedian, count, tile_processing, emt_mask):
    roi_bbox = BoundingBox(
        left=tile_processing.geometry.bounds[0],
        right=tile_processing.geometry.bounds[2],
        bottom=tile_processing.geometry.bounds[1],
        top=tile_processing.geometry.bounds[3]
    )
    emt_mask = xr_rasterize(emt_mask, da=count)

    count = count.where(emt_mask, other=-999)

    geomedian = geomedian.sel(x=slice(roi_bbox.left, roi_bbox.right), 
                            y=slice(roi_bbox.top, roi_bbox.bottom))
    count = count.sel(x=slice(roi_bbox.left, roi_bbox.right), 
                    y=slice(roi_bbox.top, roi_bbox.bottom))
    
    # Convert to int16
    geomedian = (geomedian*10000)
    geomedian = geomedian.where((~geomedian.isnull())&(geomedian>0), other=-999)
    geomedian = (geomedian).astype('int16')

    geomedian['count'] = count.astype('int16')

    return geomedian


def assign_xr_attrs(dataset, stac_items, tile, is_composite=False, **kwagrs):
    platforms = []
    instruments = []
    for item in stac_items:
        platforms.append(item.properties['platform'])
        instruments.append(item.properties['instruments'][0])
    platforms = list(np.unique(platforms))
    instruments = list(np.unique(instruments))
    instruments = [sensor.replace('+', '') for sensor in instruments]

    if is_composite:
        dataset.attrs['dtr:start_datetime'] = kwagrs["start_datetime"]
        dataset.attrs['dtr:end_datetime'] = kwagrs["end_datetime"]

    dataset.attrs['odc:region_code'] = tile

    dataset.attrs['eo:instrument'] = instruments
    dataset.attrs['eo:platform'] = platforms
    dataset.attrs['eo:gsd'] =  dataset.odc.geobox.resolution.x # Alternative: stac_items[0].properties['gsd'] # Get ground sampling distance from 
    
    return dataset


def load_base_for_zone(epsg, geom, roi_bbox, mask_path, lcc=0.001, scale=False):
    intersecting_geom = geom.geometry.simplify(tolerance=0.01, preserve_topology=True)

    logging.info(f'Loading NIR Reference Image for {epsg} ...')  
    catalog = pystac_client.Client.open("https://landsatlook.usgs.gov/stac-server/")

    mask_items = ['fill','dilated_cloud','cirrus','cloud','cloud_shadow','snow','water']
    mask_bitfields = [
        getattr(QABits, "landsat_c2_l2").value[mask_item] for mask_item in mask_items
        ]
    # Source: https://stackstac.readthedocs.io/en/v0.3.0/examples/gif.html
    bitmask = 0
    for field in mask_bitfields:
        bitmask |= 1 << field

    items_base = catalog.search(collections="landsat-c2l2-sr",
                                intersects=intersecting_geom,
                                # bbox=roi_bbox,
                                datetime="/".join((str(1999), str(2001))), 
                                query={"landsat:cloud_cover_land": {"lt": lcc},
                                        "proj:epsg": {"in": [epsg]}}
                                        ).item_collection()
    logging.info(f'Loading NIR Reference Image for {epsg} ... found {len(items_base)} items')  
    roi_geobox = GeoBox.from_bbox(roi_bbox.to_crs(crs=f'epsg:{epsg}'), crs=f'epsg:{epsg}', resolution=30)
    base = load(
        items=items_base,
        bands=['nir08', 'qa_pixel'],
        like=roi_geobox,
        groupby='time', fail_on_error=False,
        chunks=dict(time=1, y=2048, x=2048),
        patch_url=http_to_s3_url,
    )

    bad_data_mask = base["qa_pixel"].astype(int) & bitmask != 0
    nodata_mask = (base.nir08.isnull())
    combined_mask = (bad_data_mask | nodata_mask)
    base = base[['nir08']].where(~combined_mask)

    if scale:
        base = ((base*0.275) - 2000)

    base = base.median('time').astype('int16').compute()
    logging.info(f'Loading NIR Reference Image for {epsg} ... done')

    base = apply_mask(base, gpd.read_file(mask_path))
    base = base.astype('int16')
    base.nir08.attrs['nodata'] = 0
    
    return base
