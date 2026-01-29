# Data Cube connection
import datacube
import cdsapi

# Geospatial operations
import odc.geo.xr
from odc.geo.geom import BoundingBox
import xarray as xr
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio

# EO3 metadata
from utils import metadata
from eodatasets3 import serialise
from datacube.index.hl import Doc2Dataset
from pathlib import Path


# Continuous monitoring
from nrt.monitor.mosum import MoSum
from nrt.monitor.ewma import EWMA
from nrt.monitor.ccdc import CCDC
from utils import nrt_utils

# RBF 
# outlier detection
from utils.rbfkernel_outlier_detection import rbf_outlier_detection
# interpolation
from utils.rbftimeseriesfitting import rbfEnsemble_xr

# Dask Cluster for parallel processing
from dask.distributed import Client, LocalCluster

# System operations
from utils import utils
import os
import json
import datetime
from dateutil.relativedelta import relativedelta
import pytz
import gc

import warnings
warnings.filterwarnings("ignore")


def add_geobox(ds, crs=None):
    if ds.odc.crs is None and crs is not None:
        ds = ds.odc.assign_crs(crs)
    elif ds.odc.crs is None and crs is None:
        raise ValueError(
            "Unable to determine `ds`'s coordinate "
            "reference system (CRS). Please provide a "
            "CRS using the `crs` parameter "
            "(e.g. `crs='EPSG:3577'`)."
        )

    return ds


def xr_rasterize(
    gdf,
    da,
    attribute_col=None,
    crs=None,
    name=None,
    verbose=True,
    **rasterio_kwargs,
):
    # Add GeoBox and odc.* accessor to array using `odc-geo`
    da = add_geobox(da, crs)

    # Reproject vector data to raster's CRS
    gdf_reproj = gdf.to_crs(crs=da.odc.crs)

    # If an attribute column is specified, rasterise using vector
    # attribute values. Otherwise, rasterise into a boolean array
    if attribute_col is not None:
        # Use the geometry and attributes from `gdf` to create an iterable
        shapes = zip(gdf_reproj.geometry, gdf_reproj[attribute_col])
    else:
        # Use geometry directly (will produce a boolean numpy array)
        shapes = gdf_reproj.geometry

    # Rasterise shapes into a numpy array
    im = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=da.odc.geobox.shape,
        transform=da.odc.geobox.transform,
        **rasterio_kwargs,
    )

    # Convert numpy array to a full xarray.DataArray
    # and set array name if supplied
    da_rasterized = odc.geo.xr.wrap_xr(im=im, gbox=da.odc.geobox)
    da_rasterized = da_rasterized.rename(name)

    return da_rasterized


def continuous_monitoring_all_tiles(MONITORING_START_YEAR,
                                    TILES_VECTOR_PATH,
                                    EMT_VECTOR_PATH,
                                    BUCKET,
                                    OPERATION): #'filling-storage' or 'real-time'
    """
    """
    try:
        # Set up logger.
        _log = utils.setup_logger(logger_name='contmon_',
                                  logger_path=f'../logs/contmon_{datetime.datetime.now(pytz.timezone("Europe/Athens")).strftime("%Y%m%dT%H%M%S")}.log', 
                                  logger_format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                                  )
        
        # 1) =========================== SETUP ==============================================================
        # 1.1) Loading parameters
        _log.info(os.getcwd())

        _log.info('Initializing...')
        _log.info(f'MONITORING_START_YEAR: {MONITORING_START_YEAR}')
        _log.info(f'TILES_VECTOR_PATH: {TILES_VECTOR_PATH}')
        _log.info(f'EMT_VECTOR_PATH: {EMT_VECTOR_PATH}')
        _log.info(f'BUCKET: {BUCKET}')
        _log.info(f'OPERATION: {OPERATION}')
        
        BIN = f"../../BUCKET/derivative/bin/contmon/binnc"
        utils.mkdir(BIN)
        

        # 1.2) Read the Area Of Interest
        tiles = gpd.read_file(TILES_VECTOR_PATH).to_crs(epsg=3035)
        emt = gpd.read_file(EMT_VECTOR_PATH).to_crs(epsg=4326).iloc[0]

        sea_mask = gpd.read_file('../../geojson/GR_coastline_poly.shp')
        sea_mask = sea_mask.dissolve()
        
        _log.info('Loaded parameters')
        
        # 1.3) Connection to Eastern Macedonia and Thrace Data Cube
        dc = datacube.Datacube(app='Continuous monitoring', env='emtc', config='../.datacube.conf')
        PRODUCT_NAME = "emt_lnd_contmon"

        # 1.4) Set up Dask Cluster for Parallelization
        _log.info('Initiated Dask cluster')
        cluster = LocalCluster(
            n_workers=4, 
            threads_per_worker=2, 
            # memory_limit='4GB',
            )
        client = Client(cluster)

        _log.info(client.dashboard_link)

        # 1.5) Define dates of historic and moniroting periods (widen by 3-month shoulders to improve RBF kernels)
        start_history_extented = datetime.datetime.fromisoformat(f"{MONITORING_START_YEAR-4}-01-01") - relativedelta(months=5)
        start_monitor = datetime.datetime.fromisoformat(f"{MONITORING_START_YEAR}-01-01")
        end_monitor_extented = datetime.datetime.fromisoformat(f"{MONITORING_START_YEAR}-12-31") + relativedelta(months=5)

        # 1.6) Indices to be monitored
        indices2monitor = ['kndvi', 'nbr', 'ndmi']

        # 2) =========================== CONTINUOUS MONITORING ===============================
        fit_methods = ['OLS', 'RIRLS', 'CCDC-stable', 'ROC'] # LASSO not yet implemented
        screen_outliers_methods = ['Shewhart', 'CCDC_RIRLS', None]

        # Working per tile for a single monitoring period
        _log.info('Starting processing...')
        for idx, tile_processing in tiles.iterrows():
            tile_id = tile_processing.tile_ids
            _log.info(f'Processing tile {tile_id} ({idx+1}/{len(tiles)})')
            skip_tile = False # initialize flag in case of insufficient data points

            # 2.1) initialize an empty list to append each index monitoring output
            df_indices2monitor_list = []
            for index in indices2monitor:
                # =============================================================================
                #                               LOADING DATA CUBE
                # =============================================================================
                # 2.1.1) loading the data cube of the spectral index to be monitored
                gc.collect()
                utils.try_remove_tmpfiles(BIN)
                
                _log.info(f'Loading cube for {index}')
                
                find_datasets = dc.find_datasets(
                    product='emt_lnd_ard_3_nbart',
                    time=(str(start_history_extented), str(end_monitor_extented)),
                    region_code=tile_id,
                    )
                
                datasets = []
                for dataset in find_datasets:
                    # Keep Landsat-7 scenes only if it is in 2012 or 2013
                    if dataset.metadata.platform == 'landsat-7':
                        if dataset.metadata.time.begin.year in [2012, 2013]:
                            datasets.append(dataset)
                    else:
                        datasets.append(dataset)

                ds = dc.load(
                    datasets=datasets,
                    measurements=[index],
                    dask_chunks=dict(time=25, x=512,y=512),
                )
                
                sea_mask_xr = xr_rasterize(sea_mask, ds[index])
                
                ds = ds.where(sea_mask_xr == 1, np.nan)

                # =============================================================================
                #                             SIGNAL PRE-PROCESSING
                # =============================================================================
                # 2.1.2) create np.nan for nodata
                _log.info(f'Minor preprocessing for Continuous Monitoring...')
                ds = ds.where(ds!=ds[index].nodata, np.nan)
                ds = ds.where(ds!=0, np.nan)
                _log.info(f'Minor preprocessing for Continuous Monitoring...done')

                # 2.1.2) detect the outliers based RBF filters
                _log.info(f'Detect outliers with Gaussian convolution filter...')
                sigma=3.0
                    
                ds = rbf_outlier_detection(ds=ds, sigma=sigma, dim="time").compute()
                _log.info(f'Detect outliers with Gaussian convolution filter...done')

                # 2.1.2) gap-filling by interpolation and smoothing using RBF convolutional filters (EnMAP-Box)
                # Following: https://doi.org/10.1016/j.jag.2016.06.019 
                # For 16-day use widths: [8, 16, 32] like FORCE

                INT_DAY = 16
                _log.info(f'RBF interpolation to {INT_DAY}-days interval...')
                # dataset's decimal time
                X = (ds.time.dt.year + (ds.time.dt.dayofyear - 1) / 365.25).values
                # target decimal time
                target_time = ds.isel(x=0,y=0).resample(time=f'{INT_DAY}D').asfreq().time
                X2 = (target_time.dt.year + (target_time.dt.dayofyear - 1) / 365.25).values
                rbfCutOffValue = 0.035  # minimal value concidered as weights in convolution (value between 0 and 1)
                rbfFwhms = [day / 365.25 for day in [1.0*16, 2.5*16, 4.5*16]] # [8, 16, 32] RBF kernel sizes [decimal years] based on: https://doi.org/10.1016/j.jag.2016.06.019
                rbfUserWeights = [10., 3., 1.] #[10, 3, 1]

                ds = rbfEnsemble_xr(ds, X, X2, rbfFwhms, rbfUserWeights, rbfCutOffValue, index, target_time, BIN)
                
                utils.try_remove_tmpfiles(BIN)
                _log.info(f'RBF interpolation to {INT_DAY}-days interval...done')

                _log.info(f'Slicing the cube to historic and monitoring periods')
                if MONITORING_START_YEAR in [2004, 2016]:
                    start_history = start_history_extented
                else:
                    start_history = start_history_extented + relativedelta(months=5)
                    
                if MONITORING_START_YEAR==str(2024):
                    end_monitor = end_monitor_extented
                else:
                    end_monitor = end_monitor_extented- relativedelta(months=5) 
                    
                da_history = ds[index].sel(time=slice(start_history, start_monitor))
                da_monitor = ds[index].sel(time=slice(start_monitor, end_monitor))

                # =============================================================================
                #                MONITORING FIXED INTERVAL (MoSum, EWMA, CCDC)
                # =============================================================================
                gc.collect()
                _log.info(f'Find the percentage of NaNs in the time series ...')
                nan_fraction = da_history.isnull().sum(dim="time") / len(da_history.time)
                # masknan = da_history.isnull().all(dim='time').astype('int8')
                masknan = (nan_fraction>0.60).astype('int8')
                masknan = xr.where(masknan==1, 4, 1) # 4 for no data, 1 for rest
                if (masknan == 4).all().item():
                    _log.warning(f'DataArray {tile_id} - {index} {MONITORING_START_YEAR} has no time-series with sufficient (12) data points.')
                    _log.warning(f'Tile {tile_id} will be skipped entirely. (break)')
                    skip_tile = True  # Set flag
                    break # Exit inner loop
                nans_count = da_history.isnull().sum(dim="time")
                clean_count = len(da_history.time) - nans_count
                clean_fraction = 1 - (nans_count / len(da_history.time))
                
                # Values are as follow:
                    # 0: 'Not monitored'
                    # 1: 'monitored'
                    # 2: 'Unstable history'
                    # 3: 'Confirmed break - no longer monitored'
                    # 4: 'Not enough observations - not monitored'
                
                # 2.2.1) ----------- MOSUM test -----------
                # Side note: mask=~masknan.data may create mismatch issues when .fit()
                _log.info(f'Monitoring with MOSUM model...')
                fit_method = fit_methods[-1] # ROC is suggested by Verbesselt
                alpha = 0.05
                MoSumMonitor = MoSum(trend=True, harmonic_order=3, mask=masknan.data, sensitivity=0.05, h=0.5) #1 not 1.0, if mask then ~masknan.data
                                                    # h controls minimum segment size for break detection (as a fraction of total time series length).
                                                    # which means that successive breakpoints will be at least h × total_length away.
                                                    # For example:
                                                    #     Total time series length = 5 years (4 history + 1 monitor)
                                                    #     h = 0.25 → 0.25 × 5 = 1.25 years → at most 1 break every ~1.25 years
                                                    #     h = 0.5  → 0.5 × 5 = 2.5 years → at most 1 break every 2.5 years
                                                    #     h = 1.0  → 1.0 × 5 = 5 years → only 1 break allowed in entire 5-year period
                                                    # Choose h based on desired sensitivity: smaller h = more frequent breaks, larger h = fewer breaks, conservative
                try:
                    MoSumMonitor.fit(dataarray=da_history, method=fit_method, alpha=alpha, n_threads=12)
                except Exception as ex:
                    msg = f'{tile_id} - {index}: MoSum fails with error: {ex}'
                    client.close()
                    cluster.close()
                    _log.error(msg)
                    raise RuntimeError(msg)
                # We use da_history.where(~(nan_fraction >= 0.01), np.nan) only for MOSUM as the moving manner of the test would fails if it has too many consecutive nan values
                # 0.05 equals to 7 consecutive nan observations, which seems to work
                fit_kwargs = {
                    "fit_method":fit_method,
                    "alpha":alpha
                }
                ds_model_mosum, _, _, _, _, _ = nrt_utils.get_monitoring_datasets(
                    nrtmodel=MoSumMonitor, 
                    monitoring=da_monitor, reference_xr=ds, aoi_mask=None, magnitude_method='exact',
                    start_history=start_history, start_monitor=start_monitor, end_monitor=end_monitor, **fit_kwargs)
                _log.info(f'Monitoring with MOSUM model...done')

                # 2.2.2) ----------- CCDC test  -----------
                _log.info(f'Monitoring with CCDC model...')
                CCDCMonitor = CCDC(trend=True, mask=masknan.data, harmonic_order=3, sensitivity=1, boundary=6)
                CCDCMonitor.fit(dataarray=da_history,
                                method=fit_methods[2], #'CCDC-stable'
                                # green=history.green,
                                # swir=history.swir16,
                                screen_outliers=None,
                                scaling_factor=1, 
                                n_threads=12)
                
                fit_kwargs = {
                    "fit_method":'CCDC-stable',
                    "screen_outliers":'CCDC_RIRLS'
                }
                
                ds_model_ccdc, _, _, _, _, _ = nrt_utils.get_monitoring_datasets(
                    nrtmodel=CCDCMonitor, 
                    monitoring=da_monitor, reference_xr=ds, aoi_mask=None, magnitude_method='exact',
                    start_history=start_history, start_monitor=start_monitor, end_monitor=end_monitor, **fit_kwargs)
                _log.info(f'Monitoring with CCDC model...done')

                # 2.2.3) ----------- EWMA test  -----------
                _log.info(f'Monitoring with EWMA model...')
                lambda_ = 0.15 #λ, A higher 𝜆 gives more weight to recent observations.   
                L = 3 #np.log(da_history.time.size) #np.log(history.time.size) #sensitivity, high sensetivity more certain results
                # L is typically set to 3 or slightly smaller depending on the value (https://doi.org/10.1016/j.isprsjprs.2018.07.002)

                fit_method = fit_methods[0] # ['OLS', 'RIRLS', 'CCDC-stable', 'ROC']
                EwmaMonitor = EWMA(trend=True, harmonic_order=3, lambda_=lambda_, sensitivity=L, threshold_outlier=10, mask=masknan.data, save_fit_start=True,update_mask=True)
                EwmaMonitor.fit(dataarray=da_history, method=fit_method, screen_outliers="Shewhart", n_threads=12)
                
                ds_model_ewma, _, _, _, _, _ = nrt_utils.get_monitoring_datasets(
                    nrtmodel=EwmaMonitor, 
                    monitoring=da_monitor, reference_xr=ds, magnitude_method='exact',
                    aoi_mask=None, start_history=start_history, start_monitor=start_monitor, end_monitor=end_monitor, **fit_kwargs)
                _log.info(f'Monitoring with EWMA model...done')

                # 2.2.4) ----------- Convert to a single dataset -----------
                _log.info(f'Convert to a single dataset...')
                df_mosum = ds_model_mosum.to_dataframe()
                del ds_model_mosum
                df_mosum.columns = [f'{index}_mosum_{col}' for col in df_mosum.columns if col != 'spatial_ref']
                df_ccdc = ds_model_ccdc.to_dataframe()
                del ds_model_ccdc
                df_ccdc.columns = [f'{index}_ccdc_{col}' for col in df_ccdc.columns if col != 'spatial_ref']
                df_ewma = ds_model_ewma.to_dataframe()
                del ds_model_ewma
                df_ewma.columns = [f'{index}_ewma_{col}' for col in df_ewma.columns if col != 'spatial_ref']

                gc.collect()
                df_nrt_idx = pd.concat([df_ccdc, df_ewma, df_mosum], axis=1)               
                df_indices2monitor_list.append(df_nrt_idx)
            
            if skip_tile: # When the dataarray is completely empty skip the entire tile_processing
                continue  # Skip the rest of the outer loop iteration
            # =============================================================================
            #                EXPORT COG RESULTS & INDEX TO DATA CUBE
            # =============================================================================
            # 2.3.1) -----------  Combine all indices into a single dataset  -----------           
            df_nrt = pd.concat(df_indices2monitor_list, axis=1)
            ds_nrt = df_nrt.to_xarray()
            # Add geospatial information
            ds_nrt = ds_nrt.odc.assign_crs(ds.odc.crs)
            ds_nrt.attrs['odc:region_code'] = tile_id
            ds_nrt.attrs['dtr:start_datetime'] = f"{MONITORING_START_YEAR}-01-01"
            ds_nrt.attrs['dtr:end_datetime'] = f"{MONITORING_START_YEAR}-12-31"
            _log.info(f'Convert to a single dataset...done')

            # 2.3.2) -----------  Prepare for indexing  -----------
            _log.info(f'Prepare for indexing...')
            # 2.3.2.1) Naming the dataset
            # Example: /ls_cont_3cyear_mon_1/x03_y01/2020/ls_cont_mon_3cyear_1_x03_y01_2020_monitor_var.tif
            DATASET_NAME = f"{PRODUCT_NAME}_{tile_id}_{start_monitor.year}_contmon"
            FOLDER_NAME = f"{PRODUCT_NAME}/{tile_id}/{start_monitor.year}"
            DESTINATION_PATH = f"{BUCKET}/{FOLDER_NAME}"
            utils.mkdir(DESTINATION_PATH)
            _log.info(f"Path for dataset: {DESTINATION_PATH}")

            _log.info("Writing COGs...")
            # 2.3.2.2) Export each variable to a COG file
            name_measurements = []
            uris_measurements = []
            for var in ds_nrt.data_vars:
                FILE_NAME = f"{DATASET_NAME}_{var}.tif"
                FILE_URI = f"{DESTINATION_PATH}/{FILE_NAME}"

                name_measurements.append(FILE_URI)
                ds_nrt[var].odc.write_cog(FILE_URI, overwrite=True)

                uris_measurements.append(FILE_URI)
            _log.info("Writing COGs... done")

            _log.info("Writing eo3 and stac metadata...")
            # 2.3.2.3) Create the names for the dataset (folder and file names)
            collection_path = f"{BUCKET}/{FOLDER_NAME}"
            eo3_path = f'{collection_path}/{DATASET_NAME}.odc-metadata.yaml'
            stac_path = f'{collection_path}/{DATASET_NAME}.stac-metadata.json'
            
            # 2.3.2.4) Write the metadata
            eo3_doc, stac_doc = metadata.prepare_eo3_metadata_LOCAL( # Returns eodatasets3.model.DatasetDoc
                xr_cube=ds_nrt,
                collection_path=collection_path,
                dataset_name=DATASET_NAME,
                product_name=PRODUCT_NAME,
                product_family="landsat",
                bands=list(ds_nrt.data_vars),
                name_measurements=uris_measurements,
                datetime_list=[start_monitor.year, 1, 1],
                set_range=False, 
                lineage_path=None,
                version=1
            ) 

            serialise.to_path(Path(eo3_path), eo3_doc)

            with open(stac_path, 'w') as json_file:
                json.dump(stac_doc, json_file, indent=4, default=False)
            _log.info("Writing eo3 and stac metadata... done")
            
            # 2.3.2.5) Index tile dataset into EMT Data Cube using eo3 metadata
            # Create a dataset model. Product must be already in the data cube.
            _log.info("Indexing using eo3 metadata...")
            _log.info("...creating dataset model...")
            
            # see also: https://github.com/brazil-data-cube/stac2odc/blob/cd55a511f63c8ef9d3c89ad654cfcc1285dc27ba/stac2odc/cli.py#L92
            uri = f"file:///{eo3_path}"

            resolver = Doc2Dataset(dc.index)
            dataset_tobe_indexed, err  = resolver(doc_in=serialise.to_doc(eo3_doc), uri=uri)
            
            # 2.3.2.6) Indexing
            dc.index.datasets.add(dataset=dataset_tobe_indexed, with_lineage=False)

            _log.info("Indexing using eo3 metadata... done")
            
        _log.info('Closing Dask client.')
        client.close()
        cluster.close()
        
        utils.try_remove_tmpfiles(BIN)
        
    except Exception as e:
        utils.try_remove_tmpfiles(BIN)
        gc.collect()
        client.close()
        cluster.close()
        _log.error(f"An error occurred: {e}")
        _log.exception("Exception details:")
    else:
        _log.info(f'Continuous monitoring process for calendar year {start_monitor.year} completed successfully.')


if __name__ == "__main__":

    args = utils.get_sys_argv()
    json_path = args['json_file']  

    if os.path.isdir(json_path):
        json_files = [os.path.join(json_path, f) for f in os.listdir(json_path) if f.endswith('.json')]
    else:
        json_files = [json_path]
        
    for json_file in json_files:
        try:
            with open(json_file) as f:
                parameters_dict = json.load(f)

                MONITORING_START_YEAR = parameters_dict['MONITORING_START_YEAR']
                BUCKET = parameters_dict['BUCKET']
                TILES_VECTOR_PATH = parameters_dict['TILES_VECTOR_PATH']
                EMT_VECTOR_PATH = parameters_dict['EMT_VECTOR_PATH']
                OPERATION = parameters_dict['OPERATION']

                continuous_monitoring_all_tiles(MONITORING_START_YEAR=MONITORING_START_YEAR,
                                                TILES_VECTOR_PATH=TILES_VECTOR_PATH,
                                                EMT_VECTOR_PATH=EMT_VECTOR_PATH,
                                                BUCKET=BUCKET,
                                                OPERATION=OPERATION)
                print(f"Processed {json_file}")
        except Exception as e:
            print(f"Error processing {json_file}: {e}")