# Area definition
import geopandas as gpd
from odc.geo.geom import BoundingBox

# Data retrieval from Planetary Computer
from datacube.utils.aws import configure_s3_access
from odc.stac import configure_rio

# Geometric median algorithm
from odc.algo import xr_geomedian

# Parallelization
from dask.distributed import Client, LocalCluster

# COG storage
import boto3
from datacube.utils.aws import s3_dump

# eo3 and STAC metadata
import yaml
import json

# Open Data Cube for output indexing
import datacube
from datacube.index.hl import Doc2Dataset
from urllib.parse import urlparse
from eodatasets3 import serialise
from pathlib import Path

# Libraries for matrix operations
import xarray as xr
import numpy as np
import pandas as pd
import datetime

# Import utils
from utils import landsat_preprocessing
from utils import utils
from utils import metadata
from utils import stac_catalog
import pytz
import os

import logging


def run_geomedian_composite_generation_service(parameters_dict):
    """
    Template based on https://explorer.dea.ga.gov.au/products/ga_ls8cls9c_gm_cyear_3
    """
    try:
        # Set up logger.
        _log = utils.setup_logger(logger_name='comp_gen',
                                  logger_path=f'../logs/compgen_{datetime.datetime.now(pytz.timezone("Europe/Athens")).strftime("%Y%m%dT%H%M%S")}.log', 
                                  logger_format='%(asctime)s - %(levelname)s - %(message)s',
                                  )

        # 1) =========================== SETUP ==============================================================
        _log.info(os.getcwd())

        _log.info('Initializing...')
        _log.info(parameters_dict)
        
        _log.info('Setup...')
        # 1.1) Get parameters from JSON input file
        # 1.1.2) Get Time range
        start_year = parameters_dict['START_YEAR']

        # 1.1.3) Working in cloud or locally
        S3_BUCKET = "s3://emtc_production" #parameters_dict['S3_BUCKET']
        S3_BUCKET = "D:/Phd/BUCKET/derivative"
        NC_BIN = f"D:/Phd/BUCKET/derivative/bin/{start_year}"
        utils.mkdir(NC_BIN)
        PATH_TMP_NC = os.path.join(NC_BIN, 'tmp.nc')

        # Determine if working on the cloud
        WORKING_ON_CLOUD = S3_BUCKET.startswith("s3://")
        if WORKING_ON_CLOUD:
            s3_session = boto3.Session(profile_name='eo4natura')
            s3_session_client = s3_session.client('s3')

        operation = parameters_dict['OPERATION'] #'filling-storage' or 'real-time'
        
        # 1.1.4) Get the Area Of Interest
        aoi = parameters_dict['TILES_VECTOR_PATH']
        emt = parameters_dict['EMT_VECTOR_PATH']
        aoi = gpd.read_file(aoi)
        emt = gpd.read_file(emt)

        _log.info('Loaded parameters')

        # 1.2) Connection to Eastern Macedonia and Thrace Data Cube
        dc = datacube.Datacube(app='Compute and index geomedian composites', env='emtc', config='../.datacube.conf')
        PRODUCT_NAME = "ls_gm_cyear_1"

        # 1.3) Set up Dask Cluster for Parallelization
        _log.info('Initiated Dask cluster')
        cluster = LocalCluster(n_workers=1, threads_per_worker=16, processes=False)
        client = Client(cluster)
        # configure_s3_access(aws_unsigned=False, requester_pays=True, client=client)
        # configure_rio(cloud_defaults=True, client=client) # For Planetary Computer
        # configure_rio(cloud_defaults=True, aws={"aws_unsigned": True}, client=client) # for Earth Search

        # 2) =========================== TILE PROCESSING ==============================================================
        _log.info('Starting tile processing...')
        for t in range(len(aoi)):
            # 2.1) Get the tile ID (region code) and its gemetry bounds. Convert geometry to STAC friendly format (lat/lon)
            tile_processing = aoi.iloc[t]
            tile = tile_processing.tile_ids
            _log.info(f'Processing tile {tile}')
 
            # 2.2) Load from EMTC
            _log.info('Loading from EMTC...')
            cube = dc.load(
                product='emt_lnd_ard_3_nbart',
                time=str(start_year),
                region_code=tile,
                measurements=['nbart_blue', 'nbart_green', 'nbart_red', 'nbart_nri08', 'nbart_swir16', 'nbart_swir22'],
                dask_chunks=dict(time=1, y=512, x=512)
            )
                            
            # 2.3) Exclude nodata and replace with np.nan
            _log.info("Computing observation count...")
            count=(cube.nbart_red!=cube.nbart_red.nodata).sum(dim='time').compute()

            _log.info("Computing GeoMedian...")
            geomedian = xr_geomedian(cube).compute()
            _log.info("Computing GeoMedian... done")
                       
            # 2.4) Convert to int16
            _log.info("Converting to int16...")
            geomedian = geomedian.where((~geomedian.isnull())&(geomedian>0), other=-999)
            geomedian = (geomedian).astype('int16')
            geomedian['count'] = count.astype('int16')
            geomedian.attrs['dtr:start_datetime'] = f"{start_year}-01-01"
            geomedian.attrs['dtr:end_datetime'] = f"{start_year}-12-31"
            geomedian.attrs['odc:region_code'] = tile
            geomedian.attrs['eo:gsd'] =  geomedian.odc.geobox.resolution.x
            
            # Naming the dataset
            DATASET_NAME = f"{PRODUCT_NAME}_{tile}_{start_year}"
            FOLDER_NAME = f"{PRODUCT_NAME}/{tile}/{start_year}"
            DESTINATION_PATH = f"{S3_BUCKET}/{FOLDER_NAME}"
            utils.mkdir(DESTINATION_PATH)
            _log.info(f"Path for dataset: {DESTINATION_PATH}")
            
            _log.info("Writing COGs...")
            name_measurements = []
            uris_measurements = []
            for var in geomedian.data_vars:
                FILE_NAME = f"{DATASET_NAME}_{var}.tif"
                FILE_URI = f"{DESTINATION_PATH}/{FILE_NAME}"

                if WORKING_ON_CLOUD:
                    name_measurements.append(FILE_NAME)
                    var_cog_bytes = geomedian[var].odc.to_cog(blocksize=1024, 
                                                            overview_resampling='nearest', 
                                                            overview_levels=[2,4,8,16,32], 
                                                            use_windowed_writes=True)
                    _log.info(f'Uploading {var} to {FILE_URI}')
                    s3_dump(data=var_cog_bytes, url=FILE_URI, s3=s3_session_client)
                else:
                    name_measurements.append(FILE_URI)
                    geomedian[var].odc.write_cog(FILE_URI, overwrite=True)

                uris_measurements.append(FILE_URI)
            _log.info("Writing COGs... done")

            _log.info("Writing eo3 and stac metadata...")
            collection_path = f"{S3_BUCKET}/{FOLDER_NAME}"
            eo3_path = f'{collection_path}/{DATASET_NAME}.odc-metadata.yaml'
            stac_path = f'{collection_path}/{DATASET_NAME}.stac-metadata.json'
            
            if WORKING_ON_CLOUD:
                eo3_doc, stac_doc = metadata.prepare_eo3_metadata_S3BUCKET(
                    s3_session=s3_session,
                    xr_cube=geomedian,
                    s3_collection_path=collection_path,
                    dataset_name=DATASET_NAME,
                    product_name=PRODUCT_NAME,
                    product_family="geomedian",
                    s3_bands=list(geomedian.data_vars),
                    s3_name_measurements=name_measurements,
                    datetime_list=[start_year, 1, 1],
                    set_range=True,
                    s3_lineage_path=None
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
                    xr_cube=geomedian,
                    collection_path=collection_path,
                    dataset_name=DATASET_NAME,
                    product_name=PRODUCT_NAME,
                    product_family="geomedian",
                    bands=list(geomedian.data_vars),
                    name_measurements=uris_measurements,
                    datetime_list=[start_year, 1, 1],
                    set_range=True,
                    lineage_path=None
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
            dataset_tobe_indexed, err  = resolver(doc_in=serialise.to_doc(eo3_doc), uri=uri)
            
            # Indexing
            dc.index.datasets.add(dataset=dataset_tobe_indexed, with_lineage=False)

            _log.info("Indexing using eo3 metadata... done")
        
        _log.info('Closing Dask client.')
        client.close()
    except Exception as e:
        _log.error(f"An error occurred: {e}")
        _log.exception("Exception details:")
    else:
        _log.info('Geomedian composite generation process completed successfully.')


if __name__ == "__main__":
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
                run_geomedian_composite_generation_service(parameters_dict)
                print(f"Processed {json_file}")
        except Exception as e:
            print(f"Error processing {json_file}: {e}")