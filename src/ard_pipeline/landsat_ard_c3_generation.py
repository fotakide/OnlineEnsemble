# Area definition
import geopandas as gpd
from odc.geo.geom import BoundingBox
from odc.geo.geobox import GeoBox

# Data retrieval from Planetary Computer
# from datacube.utils.aws import configure_s3_access
from odc.stac import configure_rio
from odc.stac import configure_s3_access
from odc.stac import load
from planetary_computer import sign_inplace
import pystac_client
from pystac import ItemCollection
from rasterio.errors import RasterioIOError

# Parallelization
from dask.distributed import Client, LocalCluster

# COG storage
import boto3
from datacube.utils.aws import s3_dump

# eo3 and STAC metadata
import json

# Open Data Cube for output indexing
import datacube
from datacube.index.hl import Doc2Dataset
from eodatasets3 import serialise
from pathlib import Path

# Libraries for matrix operations
import numpy as np
import pandas as pd
import xarray as xr
import datetime
from scipy.ndimage import gaussian_filter
from xrspatial import slope, aspect

# Import utils
from utils import landsat_preprocessing
from utils import utils
from utils import metadata
from utils import stac_catalog
import pytz
import os
import gc
import shutil
import time

from utils import arosics_coregistration
from utils import utils
from utils import metadata
from utils import bandindices


def landsat_collection_3_gen_with_arosics_coreg_and_export_date_composite(parameters_dict):
    """
    """
    try:
        # Set up logger.
        _log = utils.setup_logger(logger_name='coreg_',
                                  logger_path=f'../logs/coreg_{datetime.datetime.now(pytz.timezone("Europe/Athens")).strftime("%Y%m%dT%H%M%S")}.log', 
                                  logger_format='%(asctime)s - %(levelname)s - %(message)s',
                                  )

        # 1) =========================== SETUP ==============================================================
        _log.info(os.getcwd())

        _log.info('Initializing...')
        _log.info(parameters_dict)
        
        _log.info('Setup...')
        # 1.1) Get parameters from JSON input file
        # 1.1.2) Get Time range
        START_YEAR = parameters_dict['START_YEAR']
        geomedian_year = stac_catalog.get_year_group(START_YEAR)

        # 1.1.3) Working in cloud or locally
        S3_BUCKET = "s3://emtc_production" #parameters_dict['S3_BUCKET']
        S3_BUCKET = "D:/Phd/BUCKET/derivative"
        BIN = f"D:/Phd/BUCKET/derivative/bin/arosics/{START_YEAR}"
        utils.mkdir(BIN)
        PATH_TMP_NC = os.path.join(BIN, 'tmp.nc')
        PATH_TMP_JPEG = os.path.join(BIN, 'CRL_GCPs.jpeg')
        BROKEN_DATES_JSON_PATH = '../jsons/broken_landsat_ard.json'
        FAILED_IDS = []
        FAILED_IDS_PATH = f'../jsons/failed_ids_{START_YEAR}.txt'

        # Determine if working on the cloud
        WORKING_ON_CLOUD = S3_BUCKET.startswith("s3://")
        if WORKING_ON_CLOUD:
            s3_session = boto3.Session(profile_name='eo4natura')
            s3_session_client = s3_session.client('s3')

        operation = parameters_dict['OPERATION'] #'filling-storage' or 'real-time'
        
        # 1.1.4) Get the Area Of Interest
        tiles_path = parameters_dict['TILES_VECTOR_PATH']
        emt_path = parameters_dict['EMT_VECTOR_PATH']
        tiles = gpd.read_file(tiles_path).to_crs(epsg=3035)
        emt = gpd.read_file(emt_path).to_crs(epsg=4326).iloc[0]

        roi_bbox = BoundingBox(
            left=emt.geometry.bounds[0],
            right=emt.geometry.bounds[2],
            bottom=emt.geometry.bounds[1],
            top=emt.geometry.bounds[3],
            crs='epsg:4326'
            )
        # # Buffer by 1km to catch data in coregistration
        # roi_bbox = roi_bbox.to_crs(crs="EPSG:3035").buffered(xbuff=1000, ybuff=1000).to_crs(crs="EPSG:4326")
        roi_geobox = GeoBox.from_bbox(roi_bbox.to_crs(crs='epsg:3035'), crs="epsg:3035", resolution=30)
        _log.info('Loaded parameters')

        # 1.2) Connection to Eastern Macedonia and Thrace Data Cube
        dc = datacube.Datacube(app='Landsat ARD generation v3.0.0', env='emtc', config='../.datacube.conf')
        PRODUCT_NAME = "emt_lnd_ard_3_nbart"

        # 1.3) Set up Dask Cluster for Parallelization
        _log.info('Initiated Dask cluster')
        cluster = LocalCluster(n_workers=1, threads_per_worker=16, processes=False)
        dask_client = Client(cluster)

        # 1.4) Congigure S3 access from .aws/credentials
        _log.info("Congigure S3 access and assign Dask client")
        _s3 = configure_s3_access(cloud_defaults=True, requester_pays=True, profile='default', aws_unsigned=False) # for landsatlook
        # Rest of access configurations
        # _s3 = configure_s3_access(cloud_defaults=True, requester_pays=True, profile='default', client=client) # For MPC
        # configure_s3_access(aws_unsigned=False, requester_pays=True, client=client)
        # configure_rio(cloud_defaults=True, client=client) # For Planetary Computer
        # configure_rio(cloud_defaults=True, aws={"aws_unsigned": True}, client=client) # for Earth Search
        _log.info(dask_client.dashboard_link)

        _log.info('Prepare base images ...')
        _log.info('Loading Global Reference Images ...')       
        # base_34N = landsat_preprocessing.load_base_for_zone(epsg=32634, geom=emt, roi_bbox=roi_bbox, mask_path=emt_path, lcc=1)
        # base_35N = landsat_preprocessing.load_base_for_zone(epsg=32635, geom=emt, roi_bbox=roi_bbox, mask_path=emt_path)
        
        roi_geobox_34N = GeoBox.from_bbox(roi_bbox.to_crs(crs='epsg:32634'), crs="epsg:32634", resolution=30, anchor='center') # to match Landsat origin: anchor=center
        roi_geobox_35N = GeoBox.from_bbox(roi_bbox.to_crs(crs='epsg:32635'), crs="epsg:32635", resolution=30, anchor='center') # to match Landsat origin: anchor=center

        roi_geobox_34N_10m = GeoBox.from_bbox(roi_bbox.to_crs(crs='epsg:32634'), crs="epsg:32634", resolution=10, anchor='center')
        roi_geobox_35N_10m = GeoBox.from_bbox(roi_bbox.to_crs(crs='epsg:32635'), crs="epsg:32635", resolution=10, anchor='center')

        base_34N = dc.load(
            product='gri',
            region_code=[['34TGK','34TGL','34TGM']],
            # relative_orbit='050',
            like=roi_geobox_34N,
            resampling='bilinear')
        base_34N = base_34N.where(base_34N>0).min('time').astype('int16')
        base_34N.red.attrs['nodata'] = 0

        base_35N = dc.load(
            product='gri',
            region_code=[['35TKE','35TKF','35TKG','35TLE','35TLF',
                          '35TLG','35TMF','35TMG']],
            # relative_orbit='050',
            like=roi_geobox_35N,
            resampling='bilinear')
        base_35N = base_35N.where(base_35N>0).min('time').astype('int16')
        base_35N.red.attrs['nodata'] = 0
        _log.info("Loading Global Reference Images ... done")

        _log.info('Loading DEM...')
        # https://planetarycomputer.microsoft.com/dataset/nasadem
        # https://planetarycomputer.microsoft.com/dataset/cop-dem-glo-30
        # Planetary Computer
        mpc_catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=sign_inplace,
        )
        dem_from = 'cop'
        if dem_from == 'srtm':
            items_dem = mpc_catalog.search(collections=["nasadem"], bbox=roi_bbox).item_collection() 
            dem_band="elevation"
        elif dem_from == 'cop':
            items_dem = mpc_catalog.search(collections=["cop-dem-glo-30"], bbox=roi_bbox).item_collection()
            dem_band="data"
        dem = load(
            items_dem,
            bands=[dem_band],
            like=roi_geobox_34N,
            chunks=dict(time=1, y=2048, x=2048),
            resampling='bilinear'
        ).compute().isel(time=0)
        _log.info('Loading DEM... Apply Gausian low-pass filter (3 × 3 kernel)')
        sigma = 1  # Standard deviation for Gaussian kernel, corresponds to a 3×3 window
        dem_lowpass = xr.apply_ufunc(
            gaussian_filter,
            dem[dem_band],
            input_core_dims=[["y", "x"]],
            output_core_dims=[["y", "x"]],
            kwargs={"sigma": sigma},
            dask="allowed",
        )
        dem_34N = xr.merge([slope(dem_lowpass), aspect(dem_lowpass)])
        dem = load(
            items_dem,
            bands=[dem_band],
            like=roi_geobox_35N,
            chunks=dict(time=1, y=2048, x=2048),
            resampling='bilinear'
        ).compute().isel(time=0)
        _log.info('Loading DEM... Apply Gausian low-pass filter (3 × 3 kernel)')
        dem_lowpass = xr.apply_ufunc(
            gaussian_filter,
            dem[dem_band],
            input_core_dims=[["y", "x"]],
            output_core_dims=[["y", "x"]],
            kwargs={"sigma": sigma},
            dask="allowed",
        )
        dem_35N = xr.merge([slope(dem_lowpass), aspect(dem_lowpass)])
        _log.info('Loading DEM...done')
        
        # 2) =========================== SCENE PROCESSING ==============================================================
        # Working per date for a given year within the study area
        # Based on
        # Saah, D., Tenneson, K., Poortinga, A., Nguyen, Q., Chishtie, F., San Aung, K., ... & Ganz, D. (2020). 
        # Primitives as building blocks for constructing land cover maps. 
        # International Journal of Applied Earth Observation and Geoinformation, 85, 101979.
        # https://doi.org/10.1016/j.rse.2011.01.019
        # https://www.sciencedirect.com/science/article/pii/S0303243419306270
        _log.info('Starting processing...')
        
        for date in pd.date_range(start=f"{START_YEAR}-01-01", end=f"{START_YEAR}-12-31", freq="D"):
            # 2.1) Find scenes in EMT region
            _log.info(f'Searching datetime {str(date.date())}.')
            items_c2l2sr = stac_catalog.open_and_search_landsat_in_stac_catalog_single_date(
                roi_bbox=roi_bbox, date=str(date.date()), catalog_endpoint="landsatlook", collection="landsat-c2l2-sr")
            if items_c2l2sr: # Load in native projection, coregister and crop into tiles
                _log.info(f'Found {len(items_c2l2sr)} Items in {str(date.date())}.')
                for scene_item in items_c2l2sr:
                    _log.info(f'Processing item datetime {scene_item.datetime}.')

                    gc.collect()
                    utils.try_remove_tmpfiles(BIN)

                    # Retreive basic metadata
                    platform = scene_item.properties['platform']
                    landsat_id = scene_item.properties['landsat:scene_id']
                    dt = scene_item.datetime
                    native_proj = scene_item.properties["proj:epsg"]

                    # Do not process Landsat-7 past 6 April 2022 (https://www.usgs.gov/landsat-missions/landsat-7)
                    # Nominal Science Mission Ends: April 6, 2022
                    if platform == 'LANDSAT_7' and date > pd.Timestamp("2022-04-06"):
                        _log.info(f'Skipping Landsat-7 past 2022-04-06.')
                        continue

                    if native_proj == 32634:
                        base = base_34N
                        geobox_zone = roi_geobox_34N
                        dem_zone = dem_34N
                    elif native_proj == 32635:
                        base = base_35N
                        geobox_zone = roi_geobox_35N
                        dem_zone = dem_35N
                    else:
                        raise RuntimeError(f"No BASE image for EPSG:{native_proj}. Currecntly supporting UTM Zones 34N and 35N")

                    # =============================================================================
                    #                               SEARCH ANGLES
                    # =============================================================================
                    _log.info(f"Searching angles for {landsat_id}...")
                    angle_item = stac_catalog.open_and_search_landsat_in_stac_catalog_single_date(
                        roi_bbox=roi_bbox, 
                        date=str(date.date()), 
                        catalog_endpoint="landsatlook", 
                        collection="landsat-c2l1",
                        known_id=landsat_id
                        )
                    if not angle_item:
                        _log.error(f"Couldn't find angles for {landsat_id}. Terminating, need for debug.")
                        break
                    # =============================================================================
                    #                               SCENE LOADING (native EPSG)
                    # =============================================================================
                    _log.info(f"Loading Landsat scene {landsat_id} ...")
                    try:
                        landsat_scene = landsat_preprocessing.load_scene(
                            ItemCollection([scene_item]),
                            ItemCollection([angle_item]),
                            geobox_zone, 
                            bands='all', 
                            resampling='bilinear')
                    except RasterioIOError as e:
                        if "CURL error: Connection died" in str(e):
                            FAILED_IDS.append(landsat_id)
                        else:
                            _log.error(f"Unexpected error while loading {landsat_id}: {e}.")
                            break

                    # =============================================================================
                    #                               COREGISTRATION
                    # =============================================================================
                    _log.info(f"Coregistration of {landsat_id} ...")
                    # 2.4) Compute the global coregistration (X/Y/Θ shifts in local areas of the image)
                    # PATH_TMP_JPEG to bin and later copy to new path for each tile
                    # https://github.com/GFZ/arosics/issues/50
                    # The recommended window size (256x256) should work fine for Landsat-8/Sentinel-2. 
                    # For the grid resolution, I would set a value which lets you end up with less than 1000 tie points. 
                    # This should be absolutely enough, a denser grid is not needed.
                    landsat_scene_coregistered, err = arosics_coregistration.arosics_local_coreg(
                        scene=landsat_scene, base=base, 
                        bname2match='red', bname2refer='red',
                        tar_wsize=256, grid_res=150,
                        resamp_alg='bilinear', jpeg_path=PATH_TMP_JPEG,
                        )
                    if err:
                        _log.warning(f"Coregistration failed for {landsat_id}.")
                        _log.info(err)
                    else:
                        _log.info(f"Coregistration for {landsat_id}....successful")
                    
                    # If Coregistration failed pass to next date:
                    if not landsat_scene_coregistered:
                        _log.warning(f"Coregistration failed for {landsat_id}.")
                        continue

                    # =============================================================================
                    #                            MASKING (drops QA band)
                    # =============================================================================
                    _log.info(f"Masking Landsat scene {landsat_id} ...")
                    landsat_scene_coregistered = landsat_preprocessing.mask_scene(landsat_scene_coregistered)

                    # =============================================================================
                    #                   Normalized BRDF Adjusted Reflectance (NBAR)
                    # =============================================================================
                    # https://github.com/brazil-data-cube/sensor-harm/blob/master/sensor_harm/harmonization_model.py

                    _log.info(f"Normalized BRDF Adjusted Reflectance of Landsat scene {landsat_id} ...")
                    try:
                        landsat_scene_nbar = landsat_preprocessing.generate_nbar(landsat_scene_coregistered,
                                                                                 tmp_netcdf_dir=BIN)
                    except Exception as e:
                        _log.warning(f"Normalized BRDF Adjusted Reflectance of Landsat scene {landsat_id} ...failed: {e}")
                        _log.info("Skip to the next iteration.")
                        _log.exception("Exception details:")
                        continue
                    # =============================================================================
                    #                 NBAR + Terrain Illumination Correction (NBAR-T)
                    # =============================================================================
                    # https://github.com/jgrss/geowombat/blob/main/src/geowombat/radiometry/topo.py

                    _log.info(f"Terrain Illumination Correction of Landsat scene {landsat_id} ...")
                    try:
                        landsat_scene_nbart = landsat_preprocessing.correct_topo(
                            landsat_scene_nbar, 
                            dem_zone, 
                            method='scs+c',
                            tmp_netcdf_dir=BIN,
                            robust=False, njobs=-1
                            )
                    except Exception as e:
                        _log.warning(f"Terrain Illumination Correction of Landsat scene {landsat_id} ...failed: {e}")
                        _log.info("Skip to the next iteration.")
                        _log.exception("Exception details:")
                        continue
                    
                    # =============================================================================
                    #                               Intercalibration
                    # =============================================================================
                    # https://www.sciencedirect.com/science/article/pii/S0034425715302455
                    if platform in ['LANDSAT_5','LANDSAT_7']:
                        try:
                            _log.info(f"Intercalibration of Landsat scene {landsat_id} ...")
                            landsat_scene_nbart = landsat_preprocessing.intercalibration_Roy_to_OLI(
                                ds_nbart=landsat_scene_nbart,
                                platform=platform
                            )
                        except Exception as e:
                            _log.warning(f"Intercalibration of Landsat scene {landsat_id} ...failed: {e}")
                            _log.info("Skip to the next iteration.")
                            _log.exception("Exception details:")
                            continue
                    # =============================================================================
                    #                              SPECTRAL INDICES
                    # =============================================================================               
                    # 2.5) Compute spectral indices NDVI, kNDVI, EVI, SAVI, NBR, NDMI, TCB, TCW, TSG
                    _log.info("Compute basic spectral indices...")
                    basic_spectral_indices = ['NDVI', 'kNDVI', 'EVI', 'SAVI', 'NBR', 'NDMI', 'TCW', 'TCG', 'TCB']
                    landsat_scene_nbart = bandindices.calculate_landsat_indices(
                        ds=landsat_scene_nbart,
                        index=basic_spectral_indices,
                        nbart=True
                        )

                    # =============================================================================
                    #                             MASKING TO EMT REGION
                    # =============================================================================
                    # Mask to EMT region bounds in EPSG:3035
                    _log.info("Masking to EMT region")
                    landsat_scene_nbart = landsat_preprocessing.apply_mask(landsat_scene_nbart, gpd.read_file(emt_path))
                    
                    # =============================================================================
                    #                                   WARP
                    # =============================================================================
                    _log.info("Warp (reproject) to EPSG:3035 (native projection of cube)")
                    coreg_attrs = landsat_scene_nbart.attrs.copy()
                    # reproject loses attributes and dtypes
                    landsat_scene_nbart = landsat_scene_nbart.odc.reproject(
                        how='EPSG:3035',
                        resampling='bilinear'
                    )
                    landsat_scene_nbart.attrs = coreg_attrs
                    for band in ['view_zenith', 'solar_zenith', 'relative_azimuth', 'solar_azimuth']:
                        landsat_scene_nbart[band].attrs['nodata'] = np.nan

                    # =============================================================================
                    #                            ARRANGE INTO TILING
                    # =============================================================================
                    _log.info("Crop into tiles...")
                    for idx, tile_ovr in tiles.iterrows():
                        tile = tile_ovr.tile_ids
                        tile_bbox = BoundingBox(
                            left=tile_ovr.geometry.bounds[0],
                            right=tile_ovr.geometry.bounds[2],
                            bottom=tile_ovr.geometry.bounds[1],
                            top=tile_ovr.geometry.bounds[3],
                            crs='epsg:3035'
                        )

                        ls_scene_tile_crop = landsat_scene_nbart.sel(x=slice(tile_bbox.left, tile_bbox.right), 
                                                                     y=slice(tile_bbox.top, tile_bbox.bottom))
                        
                        # Set attributes for eo3 metadata search fileds
                        ls_scene_tile_crop.attrs['odc:region_code'] = tile
                        ls_scene_tile_crop.attrs['eo:platform'] = platform
                        ls_scene_tile_crop.attrs['landsat:scene_id'] = landsat_id
                        
                        if not all((ls_scene_tile_crop.nbart_red == ls_scene_tile_crop.nbart_red.nodata).all().values.flatten()):
                            _log.info(f'Crop into tiles... tile {tile}')
                            # Naming the dataset
                            DATASET_NAME = f"{PRODUCT_NAME}_{landsat_id}_{tile}_{str(date.date())}"
                            FOLDER_NAME = f"{PRODUCT_NAME}/{tile}/{date.year}/{date.month}/{date.day}/{landsat_id}"
                            DESTINATION_PATH = f"{S3_BUCKET}/{FOLDER_NAME}"
                            utils.mkdir(DESTINATION_PATH)
                            _log.info(f"Path for dataset: {DESTINATION_PATH}")

                            # Copy GCPs picture to dataset path if CRL successful
                            if os.path.isfile(PATH_TMP_JPEG):
                                cor_cgps_jpeg = f"{DESTINATION_PATH}/{DATASET_NAME}_CRL_GCPs.jpeg"
                                shutil.copy(PATH_TMP_JPEG, cor_cgps_jpeg)
                            else:
                                cor_cgps_jpeg = None

                            _log.info("Writing COGs...")
                            name_measurements = []
                            uris_measurements = []
                            for var in ls_scene_tile_crop.data_vars:
                                FILE_NAME = f"{DATASET_NAME}_{var}.tif"
                                FILE_URI = f"{DESTINATION_PATH}/{FILE_NAME}"

                                if WORKING_ON_CLOUD:
                                    name_measurements.append(FILE_NAME)
                                    var_cog_bytes = ls_scene_tile_crop[var].odc.to_cog(
                                        blocksize=1024, 
                                        overview_resampling='nearest', 
                                        overview_levels=[2,4,8,16,32], 
                                        use_windowed_writes=True)
                                    _log.info(f'Uploading {var} to {FILE_URI}')
                                    s3_dump(data=var_cog_bytes, url=FILE_URI, s3=s3_session_client)
                                else:
                                    name_measurements.append(FILE_URI)
                                    ls_scene_tile_crop[var].odc.write_cog(FILE_URI, overwrite=True)

                                uris_measurements.append(FILE_URI)
                            _log.info("Writing COGs... done")

                            _log.info("Writing eo3 and stac metadata...")
                            collection_path = f"{S3_BUCKET}/{FOLDER_NAME}"
                            eo3_path = f'{collection_path}/{DATASET_NAME}.odc-metadata.yaml'
                            stac_path = f'{collection_path}/{DATASET_NAME}.stac-metadata.json'
                            
                            if WORKING_ON_CLOUD:
                                eo3_doc, stac_doc = metadata.prepare_eo3_metadata_S3BUCKET(
                                    s3_session=s3_session,
                                    xr_cube=ls_scene_tile_crop,
                                    s3_collection_path=collection_path,
                                    dataset_name=DATASET_NAME,
                                    product_name=PRODUCT_NAME,
                                    product_family="ard",
                                    s3_bands=list(ls_scene_tile_crop.data_vars),
                                    s3_name_measurements=name_measurements,
                                    datetime_list=[dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond],
                                    version=3,
                                    has_local_coreg_info=True,
                                    cor_cgps_jpeg=cor_cgps_jpeg
                                )

                                serialise.to_path(Path(eo3_path), eo3_doc)
                                
                                s3_dump(
                                    data=json.dumps(stac_doc, default=False),
                                    url=stac_path,
                                    s3=s3_session_client,
                                    ACL="bucket-owner-full-control",
                                    ContentType="text/vnd.json",
                                    )
                            else:
                                eo3_doc, stac_doc = metadata.prepare_eo3_metadata_LOCAL( # Returns eodatasets3.model.DatasetDoc
                                    xr_cube=ls_scene_tile_crop,
                                    collection_path=collection_path,
                                    dataset_name=DATASET_NAME,
                                    product_name=PRODUCT_NAME,
                                    product_family="ard",
                                    bands=list(ls_scene_tile_crop.data_vars),
                                    name_measurements=uris_measurements,
                                    datetime_list=[dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond],
                                    version=3,
                                    has_local_coreg_info=True,
                                    cor_cgps_jpeg=cor_cgps_jpeg
                                ) 

                                serialise.to_path(Path(eo3_path), eo3_doc)
                                

                                with open(stac_path, 'w') as json_file:
                                    json.dump(stac_doc, json_file, indent=4, default=False)
                            _log.info("Writing eo3 and stac metadata... done")
                            
                            # 3.4.5) Index tile dataset into EMT Data Cube using eo3 metadata
                            # Create a dataset model. Product must be already in the data cube.
                            _log.info("Indexing using eo3 metadata...")
                            _log.info("...creating dataset model...")
                            
                            # see also: https://github.com/brazil-data-cube/stac2odc/blob/cd55a511f63c8ef9d3c89ad654cfcc1285dc27ba/stac2odc/cli.py#L92
                            uri = eo3_path if WORKING_ON_CLOUD else f"file:///{eo3_path}"

                            resolver = Doc2Dataset(dc.index)
                            dataset_tobe_indexed, err = resolver(doc_in=serialise.to_doc(eo3_doc), uri=uri)
                            
                            # Indexing
                            dc.index.datasets.add(dataset=dataset_tobe_indexed, with_lineage=False)

                            _log.info("Indexing using eo3 metadata... done")
                    _log.info("Crop into tiles...done")
        
        _log.info('Closing Dask client.')
        dask_client.close()

        if os.path.isfile(PATH_TMP_JPEG):
            os.remove(PATH_TMP_JPEG)

        utils.try_remove_tmpfiles(BIN)
        
        with open(FAILED_IDS_PATH, 'w') as f:
            for landsat_id in FAILED_IDS:
                f.write(f"{landsat_id}\n")

    except Exception as e:
        _log.error(f"An error occurred: {e}")
        _log.exception("Exception details:")
    else:
        _log.info('Exiting Landsat NBART generation process.')


def main():
    args = utils.get_sys_argv()
    json_path = args['json_file']  # Accept either a single JSON file or a folder

    # Check if the path is a folder or a file
    if os.path.isdir(json_path):
        # List all JSON files in the directory
        json_files = [os.path.join(json_path, f) for f in os.listdir(json_path) if f.endswith('.json')]
    else:
        # Single file case
        json_files = [json_path]

    # Loop through and process each JSON file
    for json_file in json_files:
        try:
            with open(json_file) as f:
                parameters_dict = json.load(f)
                landsat_collection_3_gen_with_arosics_coreg_and_export_date_composite(parameters_dict)
                print(f"Processed {json_file}")
                gc.collect()
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

        gc.collect()  # Force garbage collection
        time.sleep(15)


if __name__ == "__main__":
    main()