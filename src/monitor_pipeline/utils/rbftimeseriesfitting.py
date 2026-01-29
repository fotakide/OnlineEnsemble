import numpy as np
import xarray as xr
import gc
from tqdm import tqdm
from dask import delayed, compute
import tempfile

def rbfEnsemble_xr(ds, X, X2, rbfFwhms, rbfUserWeights, rbfCutOffValue, index, target_time, bin='./binnc'):
    """
    Calculate and return the weighted average of RBF-interpolated data.
    No need to chunk beforehand as it utilized delayed functions for parallel processing.
    Parameters:
        ds (xarray.Dataset): The dataset containing the time series data.
        X (numpy.array): Source decimal times.
        X2 (numpy.array): Target decimal times.
        rbfFwhms (list): List of RBF kernel sizes (decimal years).
        rbfUserWeights (list): List of user-defined weights for the kernels.
        rbfCutOffValue (float): Minimum weight threshold for convolution.
        index (str): The variable name in the dataset to process.

    Returns:
        xarray.DataArray: Weighted average of the interpolated RBF data.
    """

    @delayed
    def process_target_time(tt, wsum_tt, valid_mask_da, weights_da, ds, data_nan_mask, rbfUserWeight):
        # Extract valid indexes and relevant slices
        valid_indexes = np.where(valid_mask_da.isel(target_time=tt).data)[0]
        ii, ij = valid_indexes[0], valid_indexes[-1]
        wi = weights_da.isel(target_time=tt).isel(time=slice(ii, ij + 1))
        sl = ds[index].isel(time=slice(ii, ij + 1))
        
        # Perform computations
        slSum = (sl * wi).sum('time')
        miSum = wi.expand_dims({"y": ds.y, "x": ds.x}).where(data_nan_mask).sum('time')
        
        # Compute target data arrays
        slmean = slSum / miSum
        D = miSum / wsum_tt * rbfUserWeight
        
        return slmean.astype('int32'), (D*100000).astype('int32')

    @delayed
    def return_nan_templates():
        return nan_template, nan_template
    
    def rbfWeights(decimalYears, rbfCenter, rbfFwhm):
        """Compute RBF weights for given decimalYears."""
        sigma = rbfFwhm / 2.355
        return np.exp(-(decimalYears - rbfCenter) ** 2 / (2 * sigma ** 2))
    
    def process_results(results, index):
        rbf = xr.concat([results[index][i][0] for i in range(len(results[0]))], dim="time").drop_vars(['target_time'])
        weights = xr.concat([results[index][i][1] for i in range(len(results[0]))], dim="time").drop_vars(['target_time'])

        rbf.name = 'rbfi'
        weights.name = 'weighti'
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nc', dir=bin) as rbf_tmp_netcdf_obj:
            rbf_tmp_netcdf = rbf_tmp_netcdf_obj.name
            rbf.to_netcdf(rbf_tmp_netcdf)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nc', dir=bin) as wght_tmp_netcdf_obj:
            wght_tmp_netcdf = wght_tmp_netcdf_obj.name
            weights.to_netcdf(wght_tmp_netcdf)
        
        return rbf_tmp_netcdf, wght_tmp_netcdf
    
    nan_template = xr.DataArray(
        data=np.full(ds[index].isel(time=0).shape, 0),  # All NaN
        dims=("y", "x"),  # Add the time dimension
        coords={
            "y": ds["y"],  # Copy x coordinate
            "x": ds["x"],  # Copy y coordinate
            "spatial_ref": 3035,
            "target_time": 0.0
        },
    )

    delayed_tasks = []
    
    for rbfFwhm, rbfUserWeight in zip(rbfFwhms, rbfUserWeights):
        X_da = xr.DataArray(X, dims="time", coords={"time": X})
        X2_da = xr.DataArray(X2, dims="target_time", coords={"target_time": X2})

        # Compute weights for all combinations of X and X2
        weights_da = rbfWeights(X_da, X2_da, rbfFwhm)

        # Compute the valid mask (2D for each target time in X2)
        valid_mask_da = weights_da >= rbfCutOffValue
        weights_da = weights_da.where(valid_mask_da)
        weights_da['time'] = ds.time
        weightsSum_da = [weights_da.isel(target_time=tt).sum().data.item() for tt in range(len(target_time))]
        data_nan_mask = ~ds[index].isnull()

        # Collect delayed tasks for this FWHM/User Weight
        slmean_tasks = []

        # for tt, wsum_tt in enumerate(tqdm(weightsSum_da, desc="Processing weights")):
        for tt, wsum_tt in enumerate(weightsSum_da):
            if wsum_tt == 0:
                task = return_nan_templates()
            else:
                task = process_target_time(tt, wsum_tt, valid_mask_da, weights_da, ds, data_nan_mask, rbfUserWeight)
            
            # Append the delayed task
            slmean_tasks.append(task)

        delayed_tasks.append(slmean_tasks)
    
    results = compute(*delayed_tasks, scheduler="threads")
    
    rbfs_tmpfiles, weights_tmpfiles = [], []
    for i in range(len(rbfFwhms)): # How many filters we have, useual len(rbfFwhms)=3
        rbf, w = process_results(results, i)
        rbfs_tmpfiles.append(rbf)
        weights_tmpfiles.append(w)

    del results
    gc.collect()
    
    # Stack the rbfs and weights into 3D DataArrays along a new 'rbf' dimension
    # Use `with` to delete after computation    
    with xr.open_mfdataset(rbfs_tmpfiles, chunks={'x': 400, 'y': 400}, engine="netcdf4", 
                               concat_dim='rbf', combine='nested') as ds_y2, \
         xr.open_mfdataset(weights_tmpfiles, chunks={'x': 400, 'y': 400}, engine="netcdf4", 
                           concat_dim='rbf', combine='nested') as ds_d:

        # Select variables
        Y2 = ds_y2.rbfi
        D = ds_d.weighti

        # Compute weighted sum
        Y2_weighted_avg = (Y2 * D).sum(dim='rbf') / D.sum(dim='rbf')

        # Assign time coordinate and finalize dataset
        Y2_weighted_avg = Y2_weighted_avg.assign_coords(time=target_time.data)
        Y2_weighted_avg.name = index
        Y2_weighted_avg = Y2_weighted_avg.to_dataset()
        Y2_weighted_avg = Y2_weighted_avg[['time', 'y', 'x', index]]
        Y2_weighted_avg = Y2_weighted_avg.astype('float32')

        return Y2_weighted_avg.compute()  # Compute before exiting the 'with' block