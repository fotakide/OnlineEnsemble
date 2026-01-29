import xarray as xr
import numpy as np

import arosics
from arosics import COREG, COREG_LOCAL, DESHIFTER
from geoarray import GeoArray

import odc.geo.xr

import logging

from skimage.exposure import match_histograms

# def _match_histograms(ref_ndarr, tgt_ndarr, nodata):
#     r = np.where(ref_ndarr!=nodata, ref_ndarr, np.nan)
#     t = np.where(tgt_ndarr!=nodata, tgt_ndarr, np.nan)

#     m = match_histograms(image=r, reference=t)
#     m = np.where((~np.isnan(r)&(r>0)), m, nodata).astype('int16')

#     return m

def arosics_local_coreg(scene, base, tar_wsize = 3200, grid_res = 100, resamp_alg='nearest',
                        bname2match='nir08', bname2refer='nir08', jpeg_path=None):
    """
    Scene must be int16 and each variable must have the nodata value attribute
    """
    coreg_attributes = {
        'cor:x_mean_shifts_px': -9999.0,
        'cor:y_mean_shifts_px': -9999.0,
        'cor:x_mean_shifts_map': -9999.0,
        'cor:y_mean_shifts_map': -9999.0,
        'cor:x_stddev_shifts_px': -9999.0,
        'cor:y_stddev_shifts_px': -9999.0,
        'cor:x_stddev_shifts_map': -9999.0,
        'cor:y_stddev_shifts_map': -9999.0,
        'cor:stddev_abs_shift_map': -9999.0,
        'cor:mean_abs_shift_map': -9999.0,
        'cor:mean_angle_shift': -9999.0,
        'cor:success': 'False',
        'cor:mean_reliability': -9999.0,
        'cor:stddev_reliability': -9999.0,
        'cor:gcp_num': -9999,
        'cor:bin_window_size': -9999,
        'cor:validity_check_level': -9999,
        'cor:min_reliability': -9999
        }

    attrs = {var: scene[var].attrs for var in scene.data_vars}
    
    # Make a copy of the scene to store the new values after successful coregistration
    coregistered = scene.copy(deep=True)

    if (base.odc.crs != scene.odc.crs):
        base = base.odc.assign_crs(scene.odc.crs)

    bands = list(scene.data_vars)
    bands.remove(bname2match)

    if ('time' in base.dims) | ('time' in scene.dims):
        raise AttributeError('`time` dimension must not be in scene or base dimensions.')

    if (base.odc.crs != scene.odc.crs):
        base = base.odc.assign_crs(scene.odc.crs)

    # Prepare GeoArrays
    tgt_ndarray_base = scene[bname2match].values
    tgt_gt_base = scene.odc.transform.to_gdal()
    tgt_prj_base = scene.odc.geobox.boundingbox.crs.to_wkt()
    geoArr_target = GeoArray(tgt_ndarray_base, tgt_gt_base, tgt_prj_base, nodata=scene[bname2match].nodata)

    # geomedian reference
    ref_ndarray = base[bname2refer].values
    # ref_ndarray = _match_histograms(ref_ndarray, tgt_ndarray_base, nodata=scene[bname2refer].nodata)
    # ref_ndarray = reference.nir08.values
    ref_gt = base.odc.transform.to_gdal()  # (5664990.0, 30.0, 0.0, 2160960.0, 0.0, -30.0)
    ref_prj = base.odc.geobox.boundingbox.crs.to_wkt()
    geoArr_reference = GeoArray(ref_ndarray, ref_gt, ref_prj, nodata=base[bname2refer].nodata)

    if (geoArr_target.epsg != geoArr_reference.epsg):
        raise RuntimeError("Input projections are not equal.")

    # Coregistration
    try:
        # The Near-Infrared (NIR) band is often preferred due
        #  to its pronounced land-water contrast and vegetation sensitivity.
        CRL = COREG_LOCAL(
            im_ref=geoArr_reference, im_tgt=geoArr_target,
            grid_res=grid_res, r_b4match=1, s_b4match=1,
            q=False, max_iter=10, max_shift=5, #maximum shift distance in reference image pixel units (default: 5 px, or 50m for 10m resolution)
            nodata=(base[bname2refer].nodata, scene[bname2match].nodata),
            align_grids=True,
            # footprint_poly_ref=footprint_poly_ref, 
            # footprint_poly_tgt=, 
            tieP_filter_level = 3,
            min_reliability = 50,
            match_gsd=False, window_size=(tar_wsize, tar_wsize),
            CPUs=8, # default: None, which means ‘all CPUs available’
            resamp_alg_calc=resamp_alg, resamp_alg_deshift=resamp_alg,
            outFillVal=scene[bname2match].nodata,
            v=False, ignore_errors=True,
            max_points=1000
        )

        try:
            CRL.calculate_spatial_shifts()
        except Exception as err: # RuntimeError for caculated shifts being abnormally large
            logging.info('AROSICS: calculated shift is too large to be valid (>150m)')
            return None, err
        
        if CRL is not None:
            coreg_info = CRL.correct_shifts()
            if (len(CRL.coreg_info["GCPList"])>40) & (CRL.window_size[0]>=48):
                logging.info(
                    'AROSICS: X shift = %0.3fm | Y shift = %0.3fm | Reliability = %0.1f%% | GCPs: %d' % (
                        CRL.coreg_info['mean_shifts_map']['x'].item(),
                        CRL.coreg_info['mean_shifts_map']['y'].item(),
                        CRL.CoRegPoints_table['RELIABILITY'][CRL.CoRegPoints_table['RELIABILITY'] != CRL.outFillVal].mean().item(),
                        len(CRL.coreg_info["GCPList"])
                    )
                )
            else:
                err = (f'AROSICS: insufficient number of GCPs, match not found. Scene is marked erroneous '
                       f'and sorted out. Found {len(CRL.coreg_info["GCPList"])} GCPs and winsize={CRL.window_size[0]}.')
                logging.warning(err)
                return None, err

    except Exception as err: #if for some reason COREG_LOCAL breaks
        logging.error(f"An error occurred: {err}")
        logging.exception("Exception details:")
        return None, err

    if not CRL.success:
        err = 'CR.success = False'
        return None, err
    
    # save GCPs to an auxiliary file (path will be added in metadata)
    # CRL.view_CoRegPoints(shapes2plot='points', savefigPath=jpeg_path, showFig=False, savefigDPI=200) #shapes2plot='vectors'
    CRL.view_CoRegPoints(shapes2plot='vectors', vector_scale=2, hide_filtered=False,  backgroundIm='tgt', savefigDPI=200, savefigPath=jpeg_path)
    
    # replace coregistered-reprojected (cubic resampling) values
    coregistered[bname2match].data = coreg_info["arr_shifted"]

    for band in bands:
        tgt_ndarray_b = scene[band].values
        tgt_gt_b = scene.odc.transform.to_gdal()
        tgt_prj_b = scene.odc.geobox.boundingbox.crs.to_wkt()
        geoArr_target_b = GeoArray(tgt_ndarray_b, tgt_gt_b, tgt_prj_b, nodata=scene[bname2match].nodata)

        band_corrected_arr = DESHIFTER(
            im2shift=geoArr_target_b,
            coreg_results=CRL.coreg_info,
            align_grids=True,
            cliptoextent=False,
            q=True,
            v=False,
            nodata=scene[band].nodata,
            resamp_alg=resamp_alg,
            progess=False,
            min_points_local_corr=5
        ).correct_shifts()

        coregistered[band].data = band_corrected_arr['arr_shifted']

    # Coregisted scene is now int16 with nodata=-999
    # Reassign the nodata attributes
    for var, attr in attrs.items():
        coregistered[var].attrs = attr

    # Assign CRS
    coregistered = coregistered.odc.assign_crs(scene.odc.crs)

    # Store the coregistration shifts in xarray attributes
    coreg_attributes = dict()
    # https://github.com/GFZ/arosics/blob/main/arosics/Tie_Point_Grid.py#L170
    coreg_attributes['cor:x_mean_shifts_px'] = CRL.coreg_info['mean_shifts_px']['x'].item()
    coreg_attributes['cor:y_mean_shifts_px'] = CRL.coreg_info['mean_shifts_px']['y'].item()
    coreg_attributes['cor:x_mean_shifts_map'] = CRL.coreg_info['mean_shifts_map']['x'].item()
    coreg_attributes['cor:y_mean_shifts_map'] = CRL.coreg_info['mean_shifts_map']['y'].item()
    coreg_attributes['cor:x_stddev_shifts_px'] = CRL.CoRegPoints_table['X_SHIFT_PX'][CRL.CoRegPoints_table['X_SHIFT_PX']!=CRL.outFillVal].std()
    coreg_attributes['cor:y_stddev_shifts_px'] = CRL.CoRegPoints_table['Y_SHIFT_PX'][CRL.CoRegPoints_table['Y_SHIFT_PX']!=CRL.outFillVal].std()
    coreg_attributes['cor:x_stddev_shifts_map'] = CRL.CoRegPoints_table['X_SHIFT_M'][CRL.CoRegPoints_table['X_SHIFT_M']!=CRL.outFillVal].std()
    coreg_attributes['cor:y_stddev_shifts_map'] = CRL.CoRegPoints_table['Y_SHIFT_M'][CRL.CoRegPoints_table['Y_SHIFT_M']!=CRL.outFillVal].std()
    coreg_attributes['cor:stddev_abs_shift_map'] = CRL.CoRegPoints_table['ABS_SHIFT'][CRL.CoRegPoints_table['ABS_SHIFT']!=CRL.outFillVal].std()
    coreg_attributes['cor:mean_abs_shift_map'] = CRL.CoRegPoints_table['ABS_SHIFT'][CRL.CoRegPoints_table['ABS_SHIFT']!=CRL.outFillVal].mean().item()
    coreg_attributes['cor:mean_angle_shift'] = CRL.CoRegPoints_table['ANGLE'][CRL.CoRegPoints_table['ANGLE']!=CRL.outFillVal].mean().item()
    coreg_attributes['cor:success'] = str(CRL.coreg_info['success'])
    coreg_attributes['cor:mean_reliability'] = CRL.CoRegPoints_table['RELIABILITY'][CRL.CoRegPoints_table['RELIABILITY']!=CRL.outFillVal].mean().item()
    coreg_attributes['cor:stddev_reliability'] = CRL.CoRegPoints_table['RELIABILITY'][CRL.CoRegPoints_table['RELIABILITY']!=CRL.outFillVal].std()
    coreg_attributes['cor:gcp_num'] = len(CRL.coreg_info["GCPList"])
    coreg_attributes['cor:validity_check_level'] = CRL.tieP_filter_level
    coreg_attributes['cor:min_reliability'] = CRL.min_reliability
    coreg_attributes['cor:bin_window_size'] = CRL.window_size[0]
    coregistered.attrs = coreg_attributes
        
    return coregistered, None


def arosics_global_coreg(scene, base, tar_wsize = 1500, resamp_alg='nearest', 
                         jpeg_path=None, bname2match='nir08', bname2refer='nir08'):
    """
    Scene must be int16 and each variable must have the nodata value attribute
    """
    coreg_attributes = {
        'cor:vec_length_map': -9999.0,
        'cor:vec_angle_deg': -9999.0,
        'cor:x_shift_map': -9999.0,
        'cor:y_shift_map': -9999.0,
        'cor:x_shift_px': -9999.0,
        'cor:y_shift_px': -9999.0,
        'cor:ssim_improved': 'False',
        'cor:ssim_orig': -9999.0,
        'cor:ssim_deshifted': -9999.0,
        'cor:shift_reliability': -9999.0,
        'cor:success': 'False',
        'cor:overlap_percentage': -9999.0,
        'cor:win_size_XY': 0,
        'cor:arosics_version': '1.21.1'
        }

    attrs = {var: scene[var].attrs for var in scene.data_vars}
    
    # Make a copy of the scene to store the new values after successful coregistration
    coregistered = scene.copy(deep=True)

    if (base.odc.crs != scene.odc.crs):
        base = base.odc.assign_crs(scene.odc.crs)

    bands = list(scene.data_vars)
    bands.remove(bname2match)

    if ('time' in base.dims) | ('time' in scene.dims):
        raise AttributeError('`time` dimension must not be in scene or base dimensions.')

    if (base.odc.crs != scene.odc.crs):
        base = base.odc.assign_crs(scene.odc.crs)

    # Prepare GeoArrays
    tgt_ndarray_base = scene[bname2match].values
    tgt_gt_base = scene.odc.transform.to_gdal()
    tgt_prj_base = scene.odc.geobox.boundingbox.crs.to_wkt()
    geoArr_target = GeoArray(tgt_ndarray_base, tgt_gt_base, tgt_prj_base, nodata=scene[bname2match].nodata)

    # reference
    ref_ndarray = base[bname2refer].values
    # ref_ndarray = _match_histograms(ref_ndarray, tgt_ndarray_base, nodata=scene[bname2refer].nodata)
    # ref_ndarray = reference[bname2refer].values
    ref_gt = base.odc.transform.to_gdal()  # (5664990.0, 30.0, 0.0, 2160960.0, 0.0, -30.0)
    ref_prj = base.odc.geobox.boundingbox.crs.to_wkt()
    geoArr_reference = GeoArray(ref_ndarray, ref_gt, ref_prj, nodata=base[bname2refer].nodata)

    if (geoArr_target.epsg != geoArr_reference.epsg):
        raise RuntimeError("Input projections are not equal.")

    # Coregistration
    try:
        # The Near-Infrared (NIR) band is often preferred due
        #  to its pronounced land-water contrast and vegetation sensitivity.
        # Max shift is 1 pixel to get sub-pixel coregistration only.
        CR = COREG(
            im_ref=geoArr_reference, im_tgt=geoArr_target,
            r_b4match=1, s_b4match=1,
            q=False, max_iter=10, max_shift=3, #maximum shift distance in reference image pixel units (default: 5 px, or 50m for 10m resolution)
            nodata=(base[bname2refer].nodata, scene[bname2match].nodata),
            align_grids=True, # Default is false, but since we work with gridded time series data, we warp, as a (i,j) pixel drill should have the same center coordinates
            # footprint_poly_tgt=, footprint_poly_tgt=, 
            match_gsd=False, ws=(tar_wsize, tar_wsize),
            CPUs=8, # default: None, which means ‘all CPUs available’
            resamp_alg_deshift=resamp_alg, #bilinear
            v=False, ignore_errors=True,
            # target_xyGrid=[[20,50], [20,50]] #[[x],[y]]
        )
        try:
            CR.calculate_spatial_shifts()
        except: # RuntimeError for caculated shifts being abnormally large
            logging.info('AROSICS: calculated shift is too large to be valid (>150m)')

        coreg_info = CR.correct_shifts()
        logging.info('AROSICS: X shift = %0.3fm | Y shift = %0.3fm | Reliability = %0.1f%%' % (CR.x_shift_map, CR.y_shift_map, CR.shift_reliability))
    except Exception as err: #if for some reason COREG_LOCAL breaks
        return None, err

    if not CR.success:
        err = 'CR.success = False'
        return None, err
    elif CR.shift_reliability<40:
        err = f'CR.shift_reliability = {CR.shift_reliability} < 40'
        return None, err
    elif CR.ssim_orig<0.20:
        err = f'CR.ssim_orig = {CR.ssim_orig} < 20%'
        return None, err
       
    # replace coregistered-reprojected (cubic resampling) values
    coregistered[bname2match].data = coreg_info["arr_shifted"]

    for band in bands:
        tgt_ndarray_b = scene[band].values
        tgt_gt_b = scene.odc.transform.to_gdal()
        tgt_prj_b = scene.odc.geobox.boundingbox.crs.to_wkt()
        geoArr_target_b = GeoArray(tgt_ndarray_b, tgt_gt_b, tgt_prj_b, nodata=scene[bname2match].nodata)

        band_corrected_arr = DESHIFTER(
            im2shift=geoArr_target_b,
            coreg_results=CR.coreg_info,
            align_grids=True,
            cliptoextent=False,
            q=True,
            v=False,
            nodata=scene[band].nodata,
            resamp_alg=resamp_alg,
            progess=False,
            min_points_local_corr=5
        ).correct_shifts()

        coregistered[band].data = band_corrected_arr['arr_shifted']

    # Coregisted scene is now int16 with nodata=-999
    # Reassign the nodata attributes
    for var, attr in attrs.items():
        coregistered[var].attrs = attr

    # Assign CRS
    coregistered = coregistered.odc.assign_crs(scene.odc.crs)

    # Store the coregistration shifts in xarray attributes
    coreg_attributes = dict()
    # https://github.com/GFZ/arosics/blob/main/arosics/Tie_Point_Grid.py#L170
    
    coreg_attributes['cor:vec_length_map'] = CR.vec_length_map
    coreg_attributes['cor:vec_angle_deg'] = CR.vec_angle_deg
    coreg_attributes['cor:x_shift_map'] = CR.x_shift_map.item()
    coreg_attributes['cor:y_shift_map'] = CR.y_shift_map.item()
    coreg_attributes['cor:x_shift_px'] = CR.x_shift_px.item()
    coreg_attributes['cor:y_shift_px'] = CR.y_shift_px.item()
    coreg_attributes['cor:ssim_improved'] = str(bool(CR.ssim_improved))
    coreg_attributes['cor:ssim_orig'] = CR.ssim_orig.item()
    coreg_attributes['cor:ssim_deshifted'] = CR.ssim_deshifted.item()
    coreg_attributes['cor:shift_reliability'] = CR.shift_reliability.item() if hasattr(CR.shift_reliability, 'item') else CR.shift_reliability
    coreg_attributes['cor:success'] = str(CR.success)
    coreg_attributes['cor:overlap_percentage'] = CR.overlap_percentage
    coreg_attributes['cor:win_size_XY'] = max(CR.matchWin.shape) #int(CR.win_size_XY[0])
    coreg_attributes['cor:arosics_version'] = arosics.__version__
    coregistered.attrs = coreg_attributes
        
    return coregistered, None


# if to make a new xr datarray
# align_grids=False,
# target_xyGrid=[[0,30], [0,30]] #[[x],[y]]
# coreg_info = CR.correct_shifts()
# data = coreg_info["arr_shifted"]
# x_min, x_res, _, y_max, _, y_res = coreg_info['updated geotransform']
# coreg_info['updated projection']
# # Compute the coordinates for the x and y axes
# height, width = data.shape
# x_coords = np.arange(x_min, x_min + (width * x_res), x_res)
# y_coords = np.arange(y_max, y_max + (height * y_res), y_res)
# data_array = xr.DataArray(
#     data,
#     dims=["y", "x"],
#     coords={"x": x_coords, "y": y_coords},
#     name='red',
#     attrs={
#         "crs": coreg_info['updated projection'],
#         "geotransform": coreg_info['updated geotransform'],
#         "nodata":0
#     },
# )
# data_array = data_array.where(data_array!=0, other=landsat_scene[band].nodata).astype(str(landsat_scene.red.dtype))