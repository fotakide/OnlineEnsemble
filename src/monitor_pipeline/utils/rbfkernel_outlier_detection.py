from math import nan
from typing import List, Tuple

import numpy as np
import xarray as xr

from astropy.convolution import Gaussian1DKernel, convolve

# def rbfoutlierdetection(xr_da, sigma=5):
#     """
#     Based on https://doi.org/10.1016/j.jag.2016.06.019
    
#     Args:
#         sigma (int, optional): _description_. Defaults to 5.
#     """
#     kernel = Gaussian1DKernel(stddev=sigma)
#     smoothed = xr.apply_ufunc(
#         convolve, 
#         xr_da, 
#         kernel, 
#         dask="parallelized",  # Allow dask to handle the computation if the dataset is large
#         vectorize=True,  # Vectorize the operation to make it compatible with xarray
#     )
    
#     # Detect outliers (Obs deviations > 1σ from the smoothed signal)
#     outliers = np.abs(xr_da - smoothed) > np.std(smoothed, axis=0, keepdims=True)
    
#     return xr_da.where(~outliers)


def rbf_outlier_detection(ds, sigma=5, dim="time"):
    """
    Apply Gaussian convolution to all variables in an xarray Dataset along the specified dimension
    and detect outliers based on deviations greater than 1σ from the smoothed signal.

    Based on https://doi.org/10.1016/j.jag.2016.06.019
    
    Args:
        ds (xr.Dataset): Input Dataset with multiple variables.
        sigma (int, optional): Standard deviation of the Gaussian kernel. Defaults to 5.
        dim (str, optional): Dimension along which to apply the convolution. Defaults to "time".

    Returns:
        xr.Dataset: Dataset with smoothed variables and NaNs for detected outliers.
    """
    kernel = Gaussian1DKernel(stddev=sigma).array  # Create Gaussian kernel

    def smooth_and_detect(var):
        """
        Apply convolution and outlier detection to a DataArray.
        """
        convolved = xr.apply_ufunc(
            convolve,
            var,
            kernel,
            input_core_dims=[[dim], ["kernel_dim"]],
            output_core_dims=[[dim]],
            vectorize=True,  # Vectorize for xarray compatibility
            dask="parallelized",  # Enable Dask for large datasets
            dask_gufunc_kwargs=dict(allow_rechunk=True), # dimension time on 0th function argument to apply_ufunc with dask='parallelized'
            kwargs={"boundary": "extend"}  # Handle boundary conditions
        )
        # Detect outliers as deviations > 1σ from the smoothed signal
        outliers = np.abs(var - convolved) > convolved.std(dim)
        return var.where(~outliers)  # Mask outliers as NaN

    # Apply the smoothing and outlier detection to each variable in the dataset
    smoothed_ds = ds.map(smooth_and_detect)
    smoothed_ds = smoothed_ds.astype('float32')

    return smoothed_ds


def rbf_outlier_detection_pixelwise(ds: xr.Dataset,
                                    sigma_map: xr.DataArray,
                                    dim: str = "time",
                                    boundary: str = "extend",
                                    sigma_quantize: float | None = None) -> xr.Dataset:
    """
    Like rbf_outlier_detection, but sigma can vary per pixel.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with a time dimension `dim` and spatial dims (e.g., y, x).
    sigma_map : xr.DataArray
        Per-pixel sigma values. Typically dims ('y','x'), broadcastable to all
        variables in `ds` except the `dim` dimension.
    dim : str
        Name of the time dimension to smooth along. Default "time".
    boundary : str
        Boundary handling for astropy.convolution.convolve. Default "extend".
    sigma_quantize : float or None
        If set (e.g., 0.1), sigmas are rounded to this step before building kernels.
        This enables kernel caching and can speed up large runs.

    Returns
    -------
    xr.Dataset
        Same variables as `ds`, with outliers (|obs-smoothed| > 1σ of smoothed)
        set to NaN.
    """

    # small kernel cache so identical/quantized sigmas reuse the same kernel
    kernel_cache: dict[float, np.ndarray] = {}

    def _smooth_and_detect_1d(a_1d: np.ndarray, sigma_val: float) -> np.ndarray:
        # (time,) -> (time,)  for a single pixel’s series
        if sigma_quantize is not None:
            sigma_val = float(np.round(float(sigma_val) / sigma_quantize) * sigma_quantize)
        else:
            sigma_val = float(sigma_val)

        if not np.isfinite(sigma_val) or sigma_val <= 0:
            return a_1d  # no-op if sigma invalid

        # build/reuse kernel for this sigma
        k = kernel_cache.get(sigma_val)
        if k is None:
            k = Gaussian1DKernel(stddev=sigma_val).array
            kernel_cache[sigma_val] = k

        sm = convolve(a_1d, k, boundary=boundary)
        sstd = np.nanstd(sm)
        if not np.isfinite(sstd) or sstd == 0:
            return a_1d

        outliers = np.abs(a_1d - sm) > sstd
        return np.where(outliers, np.nan, a_1d)

    if not isinstance(sigma_map, xr.DataArray):
        raise TypeError("sigma_map must be an xarray.DataArray (e.g., dims ('y','x')).")

    # per-variable apply: (time, y, x) × (y, x) -> (time, y, x)
    def _apply(var: xr.DataArray) -> xr.DataArray:
        return xr.apply_ufunc(
            _smooth_and_detect_1d,
            var,
            sigma_map,
            input_core_dims=[[dim], []],
            output_core_dims=[[dim]],
            vectorize=True,
            dask="parallelized",
            dask_gufunc_kwargs=dict(allow_rechunk=True),
        )

    out = ds.map(_apply)
    return out.astype("float32")