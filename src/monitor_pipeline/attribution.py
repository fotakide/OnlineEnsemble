# System
import os
import gc
import json
import datetime
import pytz
from pprint import pprint
from pathlib import Path

# Array operations
import xarray as xr
import pandas as pd
import numpy as np
import rasterio.features

# Data Cube
import datacube

# CRS and area definition
import odc.geo.xr
from odc.geo.geom import BoundingBox
from odc.geo.geobox import GeoBox

# Parallel processing
from dask.distributed import Client, LocalCluster

# eo3 and STAC metadata
from datacube.index.hl import Doc2Dataset
from eodatasets3 import serialise

# Geospatial data
import geopandas as gpd
from dea_tools.spatial import xr_rasterize
from esda.moran import Moran_Local
from libpysal.weights import lat2W

# CatBoost model
import catboost as cb

# STAC catalogs
import pystac_client
from planetary_computer import sign_inplace
from odc.stac import configure_rio
from odc.stac import configure_s3_access
from odc.stac import load

# Graphics
import matplotlib.pyplot as plt
import seaborn as sns

# Utilities
from utils import utils, metadata, stac_catalog

import warnings
import joblib


# Suppress all warnings
warnings.filterwarnings("ignore")



def classify_disturbances(year: str, region_code: str, storage_path: str, version: str, on_cloud=False):
    
    _log = utils.setup_logger(logger_name='class_',
                            logger_path=f'../logs/class_{datetime.datetime.now(pytz.timezone("Europe/Athens")).strftime("%Y%m%dT%H%M%S")}.log', 
                            logger_format='%(asctime)s - %(levelname)s - %(message)s',
                            )
    
    cluster = LocalCluster(n_workers=1, threads_per_worker=16, processes=False)
    dask_client = Client(cluster)
    _log.info('Configure Raster I/O from S3 connection for Dask')
    _s3 = configure_s3_access(cloud_defaults=True, requester_pays=True, profile='default', aws_unsigned=False) # for landsatlook
    
    try:
        # =============================================================================
        #                          SETUP PARAMETERS
        # =============================================================================
        TILE_ID = region_code
        YEAR = year
        WORKING_ON_CLOUD = on_cloud
        S3_BUCKET = storage_path
        
        _log.info('Initializing...')
        _log.info(f'YEAR: {YEAR}')
        _log.info(f'TILE_ID: {TILE_ID}')
        _log.info(f'WORKING_ON_CLOUD: {WORKING_ON_CLOUD}')
        _log.info(f'S3_BUCKET: {S3_BUCKET}')
        
        
        # =============================================================================
        #                          LOAD AUXILIARY DATA
        # =============================================================================
        _log.info('Loading auxiliary data')
        roi_geojson = '../../geojson/EMT_EL51_NUTS_RG_01M_2021_3035.geojson'
        roi = gpd.read_file(roi_geojson)
        ilots_mask = gpd.read_file('../../auxiliary/ILOTS/ILOTS_2018_GR_onlyCrops_EEA_dissolved.shp')

        sea = gpd.read_file('../../geojson/GR_coastline_poly.shp')
        sea = sea.dissolve()
                
        cbc_name = f'cbm_ens'
        cbc_model_path = f"./{cbc_name}/{cbc_name}.cbm"
        cat_bo_classifier = cb.CatBoostClassifier()
        cat_bo_classifier.load_model(cbc_model_path)
        

        # =============================================================================
        #                          CONNECT TO THE DATA CUBE
        # =============================================================================
        _log.info('Connect to datacube')
        dc = datacube.Datacube(app='Continuous monitoring', env='emtcold', config='../.datacube_new.conf')        
        
        # =============================================================================
        #                     LOAD THE TIME SERIES ANALYSIS RESULTS
        # =============================================================================
        _log.info('Loading time series analysis results')
        ds = dc.load(
            product='emt_lnd_contmon',
            time=year, 
            region_code=region_code,
            resolution=30
        ).squeeze()
        
        for var in list(ds.data_vars):
            if var.endswith('breakpoints_days'):
                base = var.replace('breakpoints_days', '')
                mag_var = base + 'magnitudes'
                
                ds[mag_var] = ds[mag_var].where(ds[var] > 0, np.nan)

        _log.info('Masking bounaries and sea')
        sea_mask_xr = xr_rasterize(sea, ds.kndvi_mosum_breakpoints)
        boundary_xr = xr_rasterize(roi, ds.kndvi_mosum_breakpoints)
        ilots_mask_xr = xr_rasterize(ilots_mask, ds.kndvi_ccdc_breakpoints)
        ds = ds.where(sea_mask_xr == 1)
        ds = ds.where(ilots_mask_xr == 0)

        _log.info('Convert SITS analysis to dataframe')
        df_nrt_cb = ds.to_dataframe()
        
        _log.info('keeping columns')
        
        PRODUCT_NAME = 'emt_lnd_contmon'
        print('BP + Magn + Tr')
        df_nrt_cb = df_nrt_cb.drop(columns=[col for col in df_nrt_cb.columns if col.endswith(('b1', 'b2', 'b3', 'a1', 'a2', 'a3'))])
        df_nrt_cb = df_nrt_cb.drop(columns=[col for col in df_nrt_cb.columns if col in ['elevation', 'slope', 'aspect']])

        gc.collect()
        
        
        # =============================================================================
        #                          PREDICTION OF DRIVER OF CHANGE
        # =============================================================================
        _log.info('Predicting driver of change via trained model')
        target = 'driver'
        prediction_nrt_cat = cat_bo_classifier.predict(df_nrt_cb)
        predict_proba_nrt_cat = cat_bo_classifier.predict_proba(df_nrt_cb)

        df_cat_probabilities = pd.DataFrame(predict_proba_nrt_cat, columns=['p_stable', 'p_wildfire', 'p_drought', 'p_clearing'], index=df_nrt_cb.index)
        df_nrt_cb['prediction_nrt_cat'] = prediction_nrt_cat
        df_nrt_cb['prediction_nrt_cat_proba_0_stable'] = df_cat_probabilities['p_stable']
        df_nrt_cb['prediction_nrt_cat_proba_1_wildfire'] = df_cat_probabilities['p_wildfire']
        df_nrt_cb['prediction_nrt_cat_proba_2_drought'] = df_cat_probabilities['p_drought']
        df_nrt_cb['prediction_nrt_cat_proba_3_clearing'] = df_cat_probabilities['p_clearing']
        
        fused_breakpoints_bo_cat = df_nrt_cb['prediction_nrt_cat'].to_xarray()
        fused_breakpoints_bo_cat = fused_breakpoints_bo_cat.where(fused_breakpoints_bo_cat>0, -1)
        fused_breakpoints_bo_cat = fused_breakpoints_bo_cat.where(sea_mask_xr)

        breakpoints_bo_cat = df_cat_probabilities.to_xarray()
        breakpoints_bo_cat = breakpoints_bo_cat.where(sea_mask_xr)

        # Idx of max probability
        bp_class = breakpoints_bo_cat.to_dataarray(dim='class')
        bp_class['class'] = [0, 1, 2, 3]

        # Assign to class based on max probability
        _log.info('Assigning class based on max change probability')
        class_proba = bp_class.idxmax(dim='class')
        class_proba = class_proba.where(boundary_xr)
        
        filtered_da = class_proba.where(ilots_mask_xr == 0, -1)
        
        # =============================================================================
        #                PREPARE BANDS AND ATTRIBUTES FOR INDEXING
        # =============================================================================
        _log.info('Aligning bands and attributes to product definition')
        classified_tile = breakpoints_bo_cat.copy()

        classified_tile['classification'] = filtered_da.fillna(-1) 
        classified_tile['classification'] = classified_tile['classification'].fillna(-1).astype('int8')
        classified_tile['classification'].attrs['nodata'] = -1

        classified_tile['p_drought'] = classified_tile['p_drought'].astype('float32')
        classified_tile['p_drought'].attrs['nodata'] = np.nan

        classified_tile['p_clearing'] = classified_tile['p_clearing'].astype('float32')
        classified_tile['p_clearing'].attrs['nodata'] = np.nan

        classified_tile['p_stable'] = classified_tile['p_stable'].astype('float32')
        classified_tile['p_stable'].attrs['nodata'] = np.nan

        classified_tile['p_wildfire'] = classified_tile['p_wildfire'].astype('float32')
        classified_tile['p_wildfire'].attrs['nodata'] = np.nan
        gc.collect()
        
        
        # =============================================================================
        #                EXPORT COG RESULTS 
        # =============================================================================
        # -----------  Combine all indices into a single dataset  -----------           
        # Add geospatial information
        classified_tile.attrs['odc:region_code'] = TILE_ID
        classified_tile.attrs['dtr:start_datetime'] = f"{YEAR}-01-01"
        classified_tile.attrs['dtr:end_datetime'] = f"{YEAR}-12-31"
        classified_tile.attrs['cbm:version'] = f"{version}"
        _log.info(f'Convert to a single dataset...done')

        # -----------  Prepare for indexing  -----------
        _log.info(f'Prepare for indexing...')
        # ---- Naming the dataset
        DATASET_NAME = f"{PRODUCT_NAME}_{TILE_ID}_{YEAR}_cbm{version}"
        FOLDER_NAME = f"{PRODUCT_NAME}/{TILE_ID}/{YEAR}"
        DESTINATION_PATH = f"{S3_BUCKET}/{FOLDER_NAME}"
        utils.mkdir(DESTINATION_PATH)
        _log.info(f"Path for dataset: {DESTINATION_PATH}")

        _log.info("Writing COGs...")
        # ---- Export each variable to a COG file
        name_measurements = []
        uris_measurements = []
        for var in classified_tile.data_vars:
            FILE_NAME = f"{DATASET_NAME}_{var}.tif"
            FILE_URI = f"{DESTINATION_PATH}/{FILE_NAME}"

            name_measurements.append(FILE_URI)
            classified_tile[var].odc.write_cog(FILE_URI, overwrite=True)

            uris_measurements.append(FILE_URI)
        _log.info("Writing COGs... done")

        # =============================================================================
        #                WRITING METADATA FILES
        # =============================================================================

        _log.info("Writing eo3 and stac metadata...")
        # ---- Create the names for the dataset (folder and file names)
        collection_path = f"{S3_BUCKET}/{FOLDER_NAME}"
        eo3_path = f'{collection_path}/{DATASET_NAME}.odc-metadata.yaml'
        stac_path = f'{collection_path}/{DATASET_NAME}.stac-metadata.json'
        
        # ---- Write the metadata
        eo3_doc, stac_doc = metadata.prepare_eo3_metadata_LOCAL( # Returns eodatasets3.model.DatasetDoc
            xr_cube=classified_tile,
            collection_path=collection_path,
            dataset_name=DATASET_NAME,
            product_name=PRODUCT_NAME,
            product_family="landsat",
            bands=list(classified_tile.data_vars),
            name_measurements=uris_measurements,
            datetime_list=[int(YEAR), 1, 1],
            set_range=True, 
            lineage_path=None,
            version=1,
            has_class_info=True
        ) 

        serialise.to_path(Path(eo3_path), eo3_doc)

        with open(stac_path, 'w') as json_file:
            json.dump(stac_doc, json_file, indent=4, default=False)
        _log.info("Writing eo3 and stac metadata... DONE")
        
        
        # =============================================================================
        #                INDEXING INTO DATA CUBE
        # =============================================================================
        # ---- Index tile dataset into EMT Data Cube using eo3 metadata
        # Create a dataset model. Product must be already in the data cube.
        _log.info("Creating dataset model...")
        # see also: https://github.com/brazil-data-cube/stac2odc/blob/cd55a511f63c8ef9d3c89ad654cfcc1285dc27ba/stac2odc/cli.py#L92
        uri = eo3_path if WORKING_ON_CLOUD else f"file:///{eo3_path}"

        resolver = Doc2Dataset(dc.index)
        dataset_tobe_indexed, err  = resolver(doc_in=serialise.to_doc(eo3_doc), uri=uri)       
        if err != None:
            _log.error(f"Indexing failed: {err}")
        else:
            _log.info("Creating dataset model... DONE")
        
        # 2.3.2.6) Indexing
        _log.info("Indexing using eo3 metadata...")
        dc.index.datasets.add(dataset=dataset_tobe_indexed, with_lineage=False)

        _log.info("Indexing using eo3 metadata... DONE")
        
        # =============================================================================
        #                CLOSING EVERYTHING
        # =============================================================================
        _log.info('Closing Dask client.')
        dask_client.close()
        cluster.close()
    
    except Exception as e:
        _log.info('Closing Dask client.')
        dask_client.close()
        cluster.close()
        _log.error(f"An error occurred: {e}")
        _log.exception("Exception details:")
    else:
        _log.info(f'Classification process for calendar year {YEAR} | tile {region_code} completed successfully.')


if __name__ == "__main__":
    args = utils.get_sys_argv()
    json_path = args['json_file']  


    if os.path.isdir(json_path):
        json_files = [os.path.join(json_path, f) for f in os.listdir(json_path) if f.endswith('.json')]
    else:
        json_files = [json_path]
        
    # process each JSON file
    for json_file in json_files:
        try:
            with open(json_file) as f:
                parameters_dict = json.load(f)

                YEAR = parameters_dict['YEAR']
                STORAGE_PATH = parameters_dict['STORAGE_PATH']
                REGIONS = parameters_dict['REGIONS_GEOJSON']
                VERSION = ' '
                
                tiles = gpd.read_file(REGIONS)

                for idx, tile_processing in tiles.iterrows():
                    tile_id = tile_processing.tile_ids
                    print(f'{YEAR}: Processing tile {tile_id}')
                    classify_disturbances(year=str(YEAR), region_code=str(tile_id), 
                                          storage_path=str(STORAGE_PATH), version=str(VERSION), on_cloud=False)
                
                print(f"Processed {json_file}")
        except Exception as e:
            print(f"Error processing {json_file}: {e}")