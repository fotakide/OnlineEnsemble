import rasterio as rio
from eodatasets3 import DatasetPrepare, DatasetDoc, ValidDataMethod
from eodatasets3.model import ProductDoc, AccessoryDoc
from eodatasets3 import serialise
from eodatasets3.stac import to_stac_item

from shapely import Polygon

import datetime
import pandas as pd
from pathlib import Path


def prepare_eo3_metadata_S3BUCKET(s3_session, xr_cube, s3_collection_path, dataset_name, product_name, product_family, 
                                  s3_bands, s3_name_measurements, datetime_list, set_range=False, s3_lineage_path=None,
                                  version=1, has_coreg_info=False, has_nrt_info=False, cor_cgps_jpeg=None):
    """
    Prepare eo3 metadata with relative paths for cloud deployments on a S3 bucket
    """
    if has_nrt_info and has_coreg_info:
        raise RuntimeError("The metadata cannot have both NRT and COR attributes")
    
    y,m,d = datetime_list
    with rio.Env(rio.session.AWSSession(s3_session)) as env:
        with DatasetPrepare(
            dataset_location=f'{s3_collection_path}/',
            metadata_path=f'{s3_collection_path}/{dataset_name}.odc-metadata.yaml',
            allow_absolute_paths=False,
            naming_conventions="default"
        ) as preparer:

            preparer.valid_data_method = ValidDataMethod.bounds

            preparer.product_name = product_name
            preparer.product_family = product_family
            preparer.product_maturity = "stable"
            preparer.dataset_version = str(version) # if used without product_name then 'product': {'name': 'productname_1'}

            preparer.datetime = datetime.datetime(y,m,d)
            if set_range:
                preparer.datetime_range = [datetime.datetime(int(xr_cube.attrs['dtr:start_datetime'][0:4]),1,1,0,0,0),
                                        datetime.datetime(int(xr_cube.attrs['dtr:end_datetime'][0:4]),12,31,23,59,59)]
            preparer.processed_now()

            if hasattr(xr_cube, 'odc:region_code'):
                preparer.properties["odc:region_code"] = xr_cube.attrs['odc:region_code']
            
            preparer.properties["odc:file_format"] = "GeoTIFF"
            preparer.properties["odc:processing_datetime"] = datetime.datetime.now().isoformat()

            if hasattr(xr_cube, 'eo:instrument'):
                preparer.properties["eo:instrument"] = xr_cube.attrs['eo:instrument']
            if hasattr(xr_cube, 'eo:platform"'):
                preparer.properties["eo:platform"] = xr_cube.attrs['eo:platform']
            preparer.properties["eo:gsd"] = xr_cube.attrs['eo:gsd']
                
            # if has coregistration attributes
            if has_coreg_info:
                preparer.properties["cor:x_mean_shifts_px"] = xr_cube.attrs["cor:x_mean_shifts_px"]
                preparer.properties["cor:x_stddev_shifts_px"] = xr_cube.attrs["cor:x_stddev_shifts_px"]
                preparer.properties["cor:y_stddev_shifts_px"] = xr_cube.attrs["cor:y_stddev_shifts_px"]
                preparer.properties["cor:x_stddev_shifts_map"] = xr_cube.attrs["cor:x_stddev_shifts_map"]
                preparer.properties["cor:y_stddev_shifts_map"] = xr_cube.attrs["cor:y_stddev_shifts_map"]
                preparer.properties["cor:stddev_abs_shift_map"] = xr_cube.attrs["cor:stddev_abs_shift_map"]
                preparer.properties["cor:mean_abs_shift_map"] = xr_cube.attrs["cor:mean_abs_shift_map"]
                preparer.properties["cor:angle_shift"] = xr_cube.attrs["cor:angle_shift"]
                preparer.properties["cor:mean_reliability"] = xr_cube.attrs["cor:mean_reliability"]
                preparer.properties["cor:stddev_reliability"] = xr_cube.attrs["cor:stddev_reliability"]
                preparer.properties["cor:GCPs_num"] = xr_cube.attrs["cor:GCPs_num"]
                preparer.properties["cor:success"] = xr_cube.attrs["cor:success"]
                if cor_cgps_jpeg:
                    preparer.add_accessory_file("cor_cgps", cor_cgps_jpeg)

            if has_nrt_info:
                preparer.properties["nrt:version"] = xr_cube.attrs["nrt:version"]
                preparer.properties["nrt:mosum_fit_method"] = xr_cube.attrs["nrt:mosum_fit_method"]
                preparer.properties["nrt:mosum_screen_outliers_method"] = xr_cube.attrs["nrt:mosum_screen_outliers_method"]
                preparer.properties["nrt:mosum_harmonic_order"] = xr_cube.attrs["nrt:mosum_harmonic_order"]
                preparer.properties["nrt:mosum_sensitivity"] = xr_cube.attrs["nrt:mosum_sensitivity"]
                preparer.properties["nrt:mosum_hfrac"] = xr_cube.attrs["nrt:mosum_hfrac"]
                preparer.properties["nrt:mosum_alpha"] = xr_cube.attrs["nrt:mosum_alpha"]
                preparer.properties["nrt:ccdc_fit_method"] = xr_cube.attrs["nrt:ccdc_fit_method"]
                preparer.properties["nrt:ccdc_screen_outliers_method"] = xr_cube.attrs["nrt:ccdc_screen_outliers_method"]
                preparer.properties["nrt:ccdc_trend"] = xr_cube.attrs["nrt:ccdc_trend"]
                preparer.properties["nrt:ccdc_harmonic_order"] = xr_cube.attrs["nrt:ccdc_harmonic_order"]
                preparer.properties["nrt:ccdc_sensitivity"] = xr_cube.attrs["nrt:ccdc_sensitivity"]
                preparer.properties["nrt:ccdc_boundary"] = xr_cube.attrs["nrt:ccdc_boundary"]
                preparer.properties["nrt:ewma_fit_method"] = xr_cube.attrs["nrt:ewma_fit_method"]
                preparer.properties["nrt:ewma_screen_outliers_method"] = xr_cube.attrs["nrt:ewma_screen_outliers_method"]
                preparer.properties["nrt:ewma_trend"] = xr_cube.attrs["nrt:ewma_trend"]
                preparer.properties["nrt:ewma_harmonic_order"] = xr_cube.attrs["nrt:ewma_harmonic_order"]
                preparer.properties["nrt:ewma_lambda"] = xr_cube.attrs["nrt:ewma_lambda"]
                preparer.properties["nrt:ewma_sensitivity"] = xr_cube.attrs["nrt:ewma_sensitivity"]
                preparer.properties["nrt:ewma_outlier_threshold"] = xr_cube.attrs["nrt:ewma_outlier_threshold"]


            if s3_lineage_path is not None:
                preparer.add_accessory_file("lineage", s3_lineage_path) # For composites, path to json with S2 IDs

            # if uuid_lineage is not None:
            #     preparer.note_source_datasets(product_lineage, uuid_lineage) # As in ("ard", metadata["id"]), UUIDs from datacube schema

            polygon_geometry = Polygon(xr_cube.odc.geobox.boundingbox.polygon.boundary.coords)
            preparer.geometry = polygon_geometry

            for name, path in zip(s3_bands, s3_name_measurements):
                preparer.note_measurement(name, path, relative_to_dataset_location=True) # else: (name, f'{granule_dir}/{path}', relative_to_dataset_location=False)

            eo3_doc = preparer.to_dataset_doc()
            preparer.done() 

            crs, grid_docs, measurement_docs = preparer._measurements.as_geo_docs()

            eo3 = DatasetDoc(
                id=preparer.dataset_id,
                label=preparer.label,
                product=ProductDoc(
                    name=preparer.names.product_name, href=preparer.names.product_uri
                ),
                crs=preparer._crs_str(crs) if crs is not None else None,
                geometry=polygon_geometry,
                grids=grid_docs,
                properties=preparer.properties,
                accessories={
                    name: AccessoryDoc(path, name=name)
                    for name, path in preparer._accessories.items()
                },
                measurements=measurement_docs,
                # lineage=preparer._lineage, # Preparer does not have _lineage
            )

            for measurement in eo3.measurements.values():
                if measurement.grid is None:
                    measurement.grid = 'default'

    s3_stac_path = f'{s3_collection_path}/{dataset_name}.stac-metadata.json'
    stac_doc = to_stac_item(dataset=eo3, stac_item_destination_url=s3_stac_path, collection_url=s3_collection_path)

    return eo3_doc, stac_doc


def prepare_eo3_metadata_LOCAL(xr_cube, collection_path, dataset_name, product_name, product_family, bands, 
                               name_measurements, datetime_list, set_range=False, lineage_path=None, version=1,
                               has_coreg_info=False, has_nrt_info=False, has_class_info=False, cor_cgps_jpeg=None) -> tuple[DatasetDoc, dict]:
    """
    Prepare eo3 metadata with absolute paths for local deployments
    """
    if has_nrt_info and has_coreg_info:
        raise RuntimeError("The metadata cannot have both NRT and COR attributes")

    y,m,d = datetime_list
    
    with DatasetPrepare(
        dataset_location=Path(collection_path),                 #  A string location is expected to be a URL or VSI path.
        metadata_path=Path(f'{collection_path}/{dataset_name}.odc-metadata.yaml'), #  A string location is expected to be a URL or VSI path.
        allow_absolute_paths=True,
        naming_conventions="default"
    ) as preparer:

        preparer.valid_data_method = ValidDataMethod.bounds

        preparer.product_name = product_name
        preparer.product_family = product_family
        preparer.product_maturity = "stable"
        preparer.dataset_version = str(version) # if used without product_name then 'product': {'name': 'productname_1'}

        preparer.datetime = datetime.datetime(y,m,d)
        if set_range:
            preparer.datetime_range = [datetime.datetime(int(xr_cube.attrs['dtr:start_datetime'][0:4]),1,1,0,0,0),
                                    datetime.datetime(int(xr_cube.attrs['dtr:end_datetime'][0:4]),12,31,23,59,59)]
        preparer.processed_now()

        preparer.properties["odc:region_code"] = xr_cube.attrs['odc:region_code']
        preparer.properties["odc:file_format"] = "GeoTIFF"
        preparer.properties["odc:processing_datetime"] = datetime.datetime.now().isoformat()

        if hasattr(xr_cube, 'eo:instrument"'):
            preparer.properties["eo:instrument"] = xr_cube.attrs['eo:instrument']
        if hasattr(xr_cube, 'eo:platform"'):
            preparer.properties["eo:platform"] = xr_cube.attrs['eo:platform']
        preparer.properties["eo:gsd"] = int(abs(xr_cube.odc.geobox.resolution.x))

        # if has coregistration attributes
        if has_coreg_info:
            preparer.properties["cor:x_mean_shifts_px"] = xr_cube.attrs["cor:x_mean_shifts_px"]
            preparer.properties["cor:x_stddev_shifts_px"] = xr_cube.attrs["cor:x_stddev_shifts_px"]
            preparer.properties["cor:y_stddev_shifts_px"] = xr_cube.attrs["cor:y_stddev_shifts_px"]
            preparer.properties["cor:x_stddev_shifts_map"] = xr_cube.attrs["cor:x_stddev_shifts_map"]
            preparer.properties["cor:y_stddev_shifts_map"] = xr_cube.attrs["cor:y_stddev_shifts_map"]
            preparer.properties["cor:stddev_abs_shift_map"] = xr_cube.attrs["cor:stddev_abs_shift_map"]
            preparer.properties["cor:mean_abs_shift_map"] = xr_cube.attrs["cor:mean_abs_shift_map"]
            preparer.properties["cor:angle_shift"] = xr_cube.attrs["cor:angle_shift"]
            preparer.properties["cor:mean_reliability"] = xr_cube.attrs["cor:mean_reliability"]
            preparer.properties["cor:stddev_reliability"] = xr_cube.attrs["cor:stddev_reliability"]
            preparer.properties["cor:gcp_num"] = xr_cube.attrs["cor:gcp_num"]
            preparer.properties["cor:success"] = xr_cube.attrs["cor:success"]
            preparer.properties["cor:validity_check_level"] = xr_cube.attrs["cor:validity_check_level"]
            preparer.properties["cor:min_reliability"] = xr_cube.attrs["cor:min_reliability"]
            if cor_cgps_jpeg:
                preparer.add_accessory_file("cor_cgps", cor_cgps_jpeg)
        
        if has_nrt_info:
            preparer.properties["nrt:version"] = xr_cube.attrs["nrt:version"]
            preparer.properties["nrt:mosum_fit_method"] = xr_cube.attrs["nrt:mosum_fit_method"]
            preparer.properties["nrt:mosum_screen_outliers_method"] = xr_cube.attrs["nrt:mosum_screen_outliers_method"]
            preparer.properties["nrt:mosum_harmonic_order"] = xr_cube.attrs["nrt:mosum_harmonic_order"]
            preparer.properties["nrt:mosum_sensitivity"] = xr_cube.attrs["nrt:mosum_sensitivity"]
            preparer.properties["nrt:mosum_hfrac"] = xr_cube.attrs["nrt:mosum_hfrac"]
            preparer.properties["nrt:mosum_alpha"] = xr_cube.attrs["nrt:mosum_alpha"]
            preparer.properties["nrt:ccdc_fit_method"] = xr_cube.attrs["nrt:ccdc_fit_method"]
            preparer.properties["nrt:ccdc_screen_outliers_method"] = xr_cube.attrs["nrt:ccdc_screen_outliers_method"]
            preparer.properties["nrt:ccdc_trend"] = xr_cube.attrs["nrt:ccdc_trend"]
            preparer.properties["nrt:ccdc_harmonic_order"] = xr_cube.attrs["nrt:ccdc_harmonic_order"]
            preparer.properties["nrt:ccdc_sensitivity"] = xr_cube.attrs["nrt:ccdc_sensitivity"]
            preparer.properties["nrt:ccdc_boundary"] = xr_cube.attrs["nrt:ccdc_boundary"]
            preparer.properties["nrt:ewma_fit_method"] = xr_cube.attrs["nrt:ewma_fit_method"]
            preparer.properties["nrt:ewma_screen_outliers_method"] = xr_cube.attrs["nrt:ewma_screen_outliers_method"]
            preparer.properties["nrt:ewma_trend"] = xr_cube.attrs["nrt:ewma_trend"]
            preparer.properties["nrt:ewma_harmonic_order"] = xr_cube.attrs["nrt:ewma_harmonic_order"]
            preparer.properties["nrt:ewma_lambda"] = xr_cube.attrs["nrt:ewma_lambda"]
            preparer.properties["nrt:ewma_sensitivity"] = xr_cube.attrs["nrt:ewma_sensitivity"]
            preparer.properties["nrt:ewma_outlier_threshold"] = xr_cube.attrs["nrt:ewma_outlier_threshold"]
            
        if has_class_info:
            preparer.properties["cbm:version"] = xr_cube.attrs["cbm:version"]

        if lineage_path:
            preparer.add_accessory_file("lineage", lineage_path) # For composites, path to json with S2 IDs

        # if uuid_lineage:
        #     preparer.note_source_datasets(product_lineage, uuid_lineage) # As in ("ard", metadata["id"]), UUIDs from datacube schema

        polygon_geometry = Polygon(xr_cube.odc.geobox.boundingbox.polygon.boundary.coords)
        preparer.geometry = polygon_geometry

        for name, path in zip(bands, name_measurements):
            preparer.note_measurement(name, str(Path(path).resolve()), relative_to_dataset_location=False) # else: (name, f'{granule_dir}/{path}', relative_to_dataset_location=False)

        eo3_doc = preparer.to_dataset_doc()

        crs, grid_docs, measurement_docs = preparer._measurements.as_geo_docs()

        eo3 = DatasetDoc(
            id=preparer.dataset_id,
            label=preparer.label,
            product=ProductDoc(
                name=preparer.names.product_name, href=preparer.names.product_uri
            ),
            crs=preparer._crs_str(crs) if crs is not None else None,
            geometry=polygon_geometry,
            grids=grid_docs,
            properties=preparer.properties,
            accessories={
                name: AccessoryDoc(path, name=name)
                for name, path in preparer._accessories.items()
            },
            measurements=measurement_docs,
            # lineage=preparer._lineage, # Preparer does not have _lineage
        )

        for measurement in eo3.measurements.values():
            if measurement.grid is None:
                measurement.grid = 'default'

    stac_path = f'{collection_path}/{dataset_name}.stac-metadata.json'
    stac_doc = to_stac_item(dataset=eo3, stac_item_destination_url=stac_path, collection_url=f'file://{collection_path}')

    return eo3_doc, stac_doc