import nrt.monitor
import xarray as xr
import numpy as np
import pandas as pd
import nrt
from typing import Union
import datetime
import matplotlib.dates as mdates

from nrt.monitor.cusum import CuSum
from nrt.monitor.mosum import MoSum
from nrt.monitor.ewma import EWMA
from nrt.monitor.ccdc import CCDC

# Function to get values (https://github.com/ec-jrc/nrt/issues/4)
def _get_monitoring_state(nrtInstance, date, array):
    """Retrieve monitoring parameters for all pixels in the array."""
    process = nrtInstance.process
    boundary = nrtInstance.boundary
    model = nrtInstance.predict(date)
    value = array
    mask = nrtInstance.mask #{0: 'Not monitored', 1: 'monitored', 2: 'Unstable history', 3: 'Confirmed break - no longer monitored'}
    return value, model, process, boundary, mask


def get_monitoring_datasets(nrtmodel: nrt.monitor.BaseNrt, monitoring: Union[xr.Dataset, xr.DataArray], 
                            reference_xr: xr.Dataset, start_history, start_monitor, end_monitor,
                            magnitude_method='bfm', **kwargs):
    """
    To 

    Parameters:
    - nrtmodel: 

    """
    if not isinstance(monitoring, (xr.Dataset, xr.DataArray)):
        raise TypeError("The `monitoring` parameter must be of type xr.Dataset or xr.DataArray")
    
    if not isinstance(reference_xr, xr.Dataset):
        raise TypeError("The `reference_xr` parameter must be of type xr.Dataset")
    
    if not isinstance(nrtmodel, nrt.monitor.BaseNrt):
        raise TypeError("The `nrtmodel` parameter must be of type nrt.monitor.BaseNrt")

    # Update attributes based on NRT model selection and fitting method
    global_attrs = reference_xr.attrs.copy()
    global_attrs.update({
        'spectral_index': monitoring.name.upper(),
        'nrt_version': nrt.__version__,
        'nrt_monitoring_strategy': nrtmodel.monitoring_strategy,
        'nrt_trend': nrtmodel.trend,
        'nrt_harmonic_order': nrtmodel.harmonic_order,
        'nrt_start_history': start_history.strftime('%Y-%m-%d'),
        'nrt_start_monitor': start_monitor.strftime('%Y-%m-%d'),
        'nrt_end_monitor': end_monitor.strftime('%Y-%m-%d'),
    })
    
    # List of Model specific attributes to check and update
    model_specific_attrs = [
        'sensitivity', # MoSum, CuSum, CCDC
        'critval', # MoSum, CuSum
        'update_mask', # MoSum, CuSum, EWMA
        'h', # MoSum
        'boundary', #CCDC
        'lambda_', 'threshold' # EWMA
        ]
    
    for attr in model_specific_attrs:
        if hasattr(nrtmodel, attr):
            value = getattr(nrtmodel, attr)
            if not isinstance(value, np.ndarray):
                global_attrs[f'nrt_{attr}'] = value

    kwargs_attrs = {
        'nrt_fit_method': kwargs.get('fit_method', 'OLS'),
        'nrt_fit_screen_outliers': kwargs.get('screen_outliers'),
        'nrt_fit_alpha': kwargs.get('alpha')
    }

    global_attrs.update({k: v for k, v in kwargs_attrs.items() if v is not None})

    if nrtmodel.monitoring_strategy == 'EWMA':
        global_attrs['nrt_monitoring_strategy_paper_doi'] = 'https://doi.org/10.1109/TGRS.2013.2272545'
    elif nrtmodel.monitoring_strategy in ['MOSUM', 'CUSUM']:
        global_attrs['nrt_monitoring_strategy_paper_doi'] = 'https://doi.org/10.1016/j.rse.2012.02.022'
    elif nrtmodel.monitoring_strategy == 'CCDC':
        global_attrs['nrt_monitoring_strategy_paper_doi'] = 'https://doi.org/10.1016/j.rse.2014.01.011'
    else:
        pass

    # Assuming the shape of the array is known
    array_shape = monitoring.values[0].shape
    num_observations = len(monitoring.values)

    # Pre-allocate arrays to store results for all pixels and all observations
    process_array = np.zeros((num_observations, *array_shape))
    boundary_array = np.zeros((num_observations, *array_shape))
    model_array = np.zeros((num_observations, *array_shape))
    value_array = np.zeros((num_observations, *array_shape))
    date_array = np.zeros(num_observations, dtype='datetime64[s]')
    mask_array = np.zeros((num_observations, *array_shape))

    # Monitor new observations
    for i, (array, date) in enumerate(zip(monitoring.values, monitoring.time.values.astype("M8[s]").astype(datetime.datetime))):
        
        nrtmodel.monitor(array=array, date=date)
        
        value, model, process, boundary, mask = _get_monitoring_state(nrtmodel, date, array)
        process_array[i] = process
        boundary_array[i] = boundary
        model_array[i] = model
        value_array[i] = value
        date_array[i] = date
        mask_array[i] = mask


    dates_model = monitoring.time.values.astype("M8[s]").astype(datetime.datetime)
    detection_dates = nrtmodel.detection_date
    # Initialize an array to store breakpoint indices
    breakpoint_indices = np.empty_like(detection_dates, dtype=np.int64)
    date_19700101 = datetime.datetime(1970, 1, 1)

    # Iterate over each pixel (pxx, pxy) and find the breakpoint index
    for pxx in range(detection_dates.shape[0]):
        for pxy in range(detection_dates.shape[1]):
            detection_date = detection_dates[pxx, pxy]
            detection_datetime = datetime.timedelta(days=float(detection_date)) + date_19700101
            
            # Find the breakpoint index for the current pixel
            bp_idx = next((index for index, dt in enumerate(dates_model) if dt.date() == detection_datetime.date()), None)
            
            # Store the breakpoint index in the array
            breakpoint_indices[pxx, pxy] = bp_idx if bp_idx is not None else -1  # Use -1 if breakpoint index is not found

    # Initialize arrays to store magnitudes
    ds_model_array_monperiod = (
        xr.Dataset(
            {
                "y_pred": (("time", "y", "x"), model_array)
             }
        )
        .assign_coords(
            time=reference_xr.sel(time=slice(start_monitor, end_monitor)).time,
            y=reference_xr.y,
            x=reference_xr.x,
        )
    )

    # Calculate magnitude for each pixel
    # The magnitude and direction of the disturbance are estimated by deriving the difference between the median of the
    # fitted season-trend model and the new observations during the monitoring period. 
    # Paragraph 2.6: Near real-time disturbance detection using satellite image time series) http://dx.doi.org/10.1016/j.rse.2012.02.022)
    if magnitude_method=='bfm':
        BaseNRT_magnitudes = (monitoring - ds_model_array_monperiod).median('time').compute()
        proxy_name = list(BaseNRT_magnitudes.keys())[0]
    elif magnitude_method=='exact':
        # stacked_time_idx = breakpoint_indices.stack(z=("y", "x"))  # shape (z,)
        stacked_time_idx = xr.DataArray(
                                breakpoint_indices,
                                dims=("y", "x"),
                                coords={"y": monitoring.y, "x": monitoring.x}
                            ).stack(z=("y", "x"))
        
        stacked_monitoring = monitoring.stack(z=("y", "x"))  # shape (time, z)
        stacked_fitted = ds_model_array_monperiod.stack(z=("y", "x"))  # shape (time, z)
        

        selected_monitoring = stacked_monitoring.isel(time=stacked_time_idx)  # shape (z,)
        selected_monitoring = selected_monitoring.unstack("z")  # back to (y, x)
        
        selected_fitted = stacked_fitted.isel(time=stacked_time_idx)  # shape (z,)
        selected_fitted = selected_fitted.unstack("z")  # back to (y, x)

        BaseNRT_magnitudes = (selected_monitoring - selected_fitted).compute()
        proxy_name = list(BaseNRT_magnitudes.keys())[0]

    # Find the model_array for all history+monitor
    # ===============================
    
    # Create an empty y_pred variable with the (time, y, x) order
    # by taking the dimensions' sizes of the reference_xr dataset
    # use `.sel` to take care of adjustable time series length
    y_pred_data = np.empty([reference_xr.sel(time=slice(start_history, end_monitor)).sizes[dim] for dim in reference_xr.dims])

    # Iterate over each time value and update y_pred with nrtmodel.predict
    # replace with `start_monitor` if you want only for monitoring peiod and `model_array`
    time_from_start_his_to_end_mon = reference_xr.sel(time=slice(start_history, end_monitor)).time
    for i, time_value in enumerate(time_from_start_his_to_end_mon.data): 
        datetime_dt = np.datetime64(time_value).astype(datetime.datetime)  # Convert numpy.datetime64 to datetime.datetime
        y_pred_data[i] = nrtmodel.predict(datetime_dt)
    # ===============================

    ds_model_data =  {
        "breakpoints": (("y", "x"), breakpoint_indices), #np.where(nrtmodel.detection_date > 0, nrtmodel.detection_date, -1)),
        "breakpoints_days": (("y", "x"), detection_dates),
        "magnitudes": BaseNRT_magnitudes[proxy_name],
        }
    
    if hasattr(nrtmodel, 'sigma'):
        ds_model_data["sigma"] = (("y", "x"), nrtmodel.sigma)
    
    if hasattr(nrtmodel, 'rmse'):
        ds_model_data["rmse"] = (("y", "x"), nrtmodel.rmse)

    for coeff in range(nrtmodel.beta.shape[0]):
        if nrtmodel.trend:
            if coeff==0:
                ds_model_data['intercept'] = (("y", "x"), nrtmodel.beta[coeff,:,:])
            elif coeff==1:
                ds_model_data['trend'] = (("y", "x"), nrtmodel.beta[coeff,:,:])
            elif coeff%2==0:
                ds_model_data[f'a{int(coeff/2)}'] = (("y", "x"), nrtmodel.beta[coeff,:,:])
            elif coeff%2!=0:
                ds_model_data[f'b{int(coeff//2)}'] = (("y", "x"), nrtmodel.beta[coeff,:,:])
        else:
            if coeff==0:
                ds_model_data['intercept'] = (("y", "x"), nrtmodel.beta[coeff,:,:])
            elif coeff%2!=0:
                ds_model_data[f'a{int((coeff + 1) // 2)}'] = (("y", "x"), nrtmodel.beta[coeff,:,:])
            elif coeff%2==0:
                ds_model_data[f'b{int(coeff/2)}'] = (("y", "x"), nrtmodel.beta[coeff,:,:])

    ds_model = (
        xr.Dataset(
            ds_model_data
        )
        .assign_coords(y=reference_xr.y, x=reference_xr.x)
    )
    
    ds_model['breakpoints_days'] = ds_model['breakpoints_days'].where(ds_model['breakpoints_days']>0, -999)
    if magnitude_method=='exact':
        ds_model = ds_model.drop(['time', 'spatial_ref'])
    elif magnitude_method=='exact':
        ds_model = ds_model.drop(['spatial_ref'])
    
    ds_model.attrs = global_attrs

    # This dataset contains all model prediction both from history and monitor.
    # If I want only the monitoring period, replace `y_pred_data` with `model_array` and change `start_history` with `start_monitor``
    ds_model_y_pred = (
        xr.Dataset(
            {
                "y_pred": (("time", "y", "x"), y_pred_data)
             }
        )
        .assign_coords(
            time=reference_xr.sel(time=slice(start_history, end_monitor)).time,
            y=reference_xr.y,
            x=reference_xr.x,
        )
    )

    ds_model_process = (
        xr.Dataset(
            {
                "process": (("time", "y", "x"), process_array)
            }
        )
        .assign_coords(
            time=reference_xr.sel(time=slice(start_monitor, end_monitor)).time,
            y=reference_xr.y,
            x=reference_xr.x,
        )
    )

    ds_model_bounds = (
        xr.Dataset(
            {
                "bounds": (("time", "y", "x"), boundary_array) #np.flip(boundary_array, axis=1)
            }
        )
        .assign_coords(
            time=reference_xr.sel(time=slice(start_monitor, end_monitor)).time,
            y=reference_xr.y,
            x=reference_xr.x,
        )
    )

    df_model_zeros = pd.DataFrame(
        {
            "time": reference_xr.sel(time=slice(start_history, end_monitor)).time.values,
            "zeros": 0,
        }
    )
    df_model_zeros.set_index("time", inplace=True)

    bp_dates_model = dates_model[(dates_model >= start_monitor) & (dates_model < end_monitor)]

    ds_model.attrs = global_attrs
    ds_model_y_pred.attrs = global_attrs
    ds_model_bounds.attrs = global_attrs
    ds_model_process.attrs = global_attrs

    return ds_model, ds_model_y_pred, ds_model_process, ds_model_bounds, df_model_zeros, bp_dates_model


import matplotlib.pyplot as plt

def plot_monitoring_results(ds_indices, ds_model, ds_y_pred, ds_process, ds_bounds, bp_dates, df_zeros, monitoring_method,
                            start_history, start_monitor, end_monitor, xx, yy, index='ndvi', plot_scatter=False, plot_line=True, 
                            plot_grid=False, return_fig=False, show=True, savefigpath=None):
    """
    Plot monitoring results including history, monitor, fitted data, breakpoints, and critical boundaries. 
    Monitoring processes: MoSum, CuSum, EWMA, CCDC, BFAST Monitor.

    Parameters:
    - ds_indices: Smoothed dataset of indices.
    - ds_model: Dataset containing the breakpoints and magnitudes.
    - ds_y_pred: Predicted values dataset.
    - ds_process: Process dataset.
    - ds_bounds: Bounds dataset.
    - df_zeros: DataFrame of zeros for plotting reference.
    - monitoring_method: str of method.
    - bp_dates: Breakpoint dates from the monitoring process.
    - start_history: Start date for the historical period.
    - start_monitor: Start date for the monitoring period.
    - end_monitor: End date for the monitoring period.
    - xx, yy: Coordinates for data extraction.
    - index: Index name to be used in labels.
    - plot_scatter: Bool for plotting time series points (Default=False)
    - return_fig: Bool for returning the figure object instead of just displaying the plot (Default=False)
    """

    # Create subplots
    fig, ax = plt.subplots(2, 1, figsize=(22, 8))

    # Plot historical and monitoring data
    if plot_line:
        ds_indices[index].sel(x=xx, y=yy, method="nearest").sel(time=slice(start_history, start_monitor)).plot(ax=ax[0], label=f"{index.upper()} history", c='#1f77b4')
        ds_indices[index].sel(x=xx, y=yy, method="nearest").sel(time=slice(start_monitor, end_monitor)).plot(ax=ax[0], label=f"{index.upper()} monitor", c='#ff7f0e')
    if plot_scatter:
        ds_indices[index].sel(x=xx, y=yy, method="nearest").sel(time=slice(start_history, start_monitor)).plot.scatter(ax=ax[0], x='time', label="_nolegend_", marker='o', s=19, c='#1f77b4')
        ds_indices[index].sel(x=xx, y=yy, method="nearest").sel(time=slice(start_monitor, end_monitor)).plot.scatter(ax=ax[0], x='time', label="_nolegend_", marker='o', s=19, c='#ff7f0e')

    
    # Plot fitted history data
    ds_y_pred.y_pred.sel(x=xx, y=yy, method="nearest").plot(ax=ax[0], label="Fitted history", linestyle="--", c='#2ca02c')

    # Determine the breakpoint and plot it if it exists
    bp = int(ds_model.breakpoints.sel(x=xx, y=yy, method="nearest").values.flatten()[0])
    if bp >= 0:
        ax[0].axvline(bp_dates[bp], color="red", linestyle="--", linewidth=2, label="Breakpoint")
        ax[0].set_title(f'Breakpoint detection at {bp_dates[bp].strftime("%Y-%m-%d")} (Magnitude = {round(ds_model.sel(x=xx, y=yy, method="nearest").magnitudes.data.flatten()[0],4)})')
    else:
        ax[0].set_title("No Breakpoint detected")

    # Mark the start of the monitoring period
    ax[0].axvline(start_monitor, color="k", linestyle="-", linewidth=1, label="Start Monitor")

    # Customize first subplot
    ax[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # Plot the monitoring process data
    ds_process.process.sel(x=xx, y=yy, method="nearest").plot(ax=ax[1], label=monitoring_method.upper())

    # Plot the breakpoint on the second subplot if it exists
    if bp >= 0:
        ax[1].axvline(bp_dates[bp], color="red", linestyle="--", linewidth=2, label="Breakpoint")
    
    # Mark the start of the monitoring period on the second subplot
    ax[1].axvline(start_monitor, color="k", linestyle="-", linewidth=1, label="Start Monitor")

    # Plot critical boundaries
    ds_bounds.bounds.sel(x=xx, y=yy, method="nearest").plot(ax=ax[1], color="orange", label="Critical boundary")
    (-1 * ds_bounds).bounds.sel(x=xx, y=yy, method="nearest").plot(ax=ax[1], color="orange", label="_nolegend_")

    # Plot zeros for reference
    # df_zeros.plot(ax=ax[1], color="gray", label="_nolegend_")
    ax[1].plot(df_zeros.index, df_zeros.values, color="gray", label=None)

    # Customize second subplot
    ax[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # Adjust tick labels for the second subplot
    ax[1].tick_params(axis="x", labelrotation=0)
    for tick in ax[1].get_xticklabels():
        tick.set_horizontalalignment("center")

    # Highlight the monitoring period in both subplots
    ax[0].axvspan(start_monitor, end_monitor, color="gray", alpha=0.11)
    ax[1].axvspan(start_monitor, end_monitor, color="gray", alpha=0.11)
    
    ax[0].set_ylabel(f"{index.upper()}")
    ax[1].set_ylabel(f"Score")
    ax[1].set_title(f"Test statistic") # {monitoring_method.upper()}

    # Set the main title for the figure and y lable for the second figure
    if monitoring_method=='MOSUM':
        method_title = 'BFM'
        # method_test = 'MOSUM'
    elif monitoring_method=='EWMA':
        method_title = 'EWMACD'
        # method_test = 'EWMA'
    elif monitoring_method=='CCDC':
        method_title = 'CCDC'
    else:
        method_title = monitoring_method.upper()
        # method_test = monitoring_method.upper()
    
    
    fig.suptitle(f"{index.upper()} — {method_title} Results", fontsize=14)

    if plot_grid:
        for axis in ax:
            # axis.grid(True, linestyle='--', alpha=0.6)
            # Major ticks every year
            axis.xaxis.set_major_locator(mdates.YearLocator())
            axis.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

            # Minor ticks every 3 months
            axis.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))

            # Apply grid to both major and minor ticks
            axis.grid(True, which='major', linestyle='--', alpha=0.9)
            axis.grid(True, which='minor', linestyle=':', alpha=0.4)

    
    # Adjust layout and show the plot
    plt.tight_layout()
    
    if savefigpath:
        plt.savefig(savefigpath, dpi=300, format='tiff')
        
    if show:
        plt.show()

    plt.close()
    
    if return_fig:
        return fig
    

def process_nrt_datasets(
    ds_base, start_history, start_monitor, end_monitor, aoi_mask, ds_dem,
    for_xgb=True, pelt_mask_for_nrt_annual=None,
    run_mosum=True, run_cusum=True, run_ccdc=True, run_ewma=True, **kwargs):
    
    fit_methods = ['OLS', 'RIRLS', 'CCDC-stable', 'ROC']  # LASSO not yet implemented
    alpha = 0.05
    df_idx_list = []

    for fit_idx in list(ds_base.keys()):
        history = ds_base[fit_idx].sel(time=slice(start_history, start_monitor))
        monitoring = ds_base[fit_idx].sel(time=slice(start_monitor, end_monitor))

        # MOSUM
        if run_mosum:
            mosum_params = kwargs.get('mosum_params', {})
            MoSumMonitor = MoSum(**mosum_params)
            MoSumMonitor.fit(dataarray=history, method=fit_methods[0], alpha=alpha, n_threads=6)

            ds_model_mosum, _, _, _, _, _ = get_monitoring_datasets(
                nrtmodel=MoSumMonitor,
                monitoring=monitoring, reference_xr=ds_base, aoi_mask=aoi_mask,
                start_history=start_history, start_monitor=start_monitor, end_monitor=end_monitor, fit_method=fit_methods[0], alpha=alpha)
            if 'spatial_ref' in ds_model_mosum:
                ds_model_mosum = ds_model_mosum.drop_vars('spatial_ref')
            df_mosum = ds_model_mosum.to_dataframe()
            df_mosum.columns = [f'{fit_idx}_mosum_{col}' for col in df_mosum.columns]
        else:
            df_mosum = None

        # CUSUM
        if run_cusum:
            cusum_params = kwargs.get('cusum_params', {})
            CuSumMonitor = CuSum(**cusum_params)
            CuSumMonitor.fit(dataarray=history, method=fit_methods[0], alpha=alpha, n_threads=6)

            ds_model_cusum, _, _, _, _, _ = get_monitoring_datasets(
                nrtmodel=CuSumMonitor,
                monitoring=monitoring, reference_xr=ds_base, aoi_mask=aoi_mask,
                start_history=start_history, start_monitor=start_monitor, end_monitor=end_monitor, fit_method=fit_methods[0], alpha=alpha)
            if 'spatial_ref' in ds_model_cusum:
                ds_model_cusum = ds_model_cusum.drop_vars('spatial_ref')
            df_cusum = ds_model_cusum.to_dataframe()
            df_cusum.columns = [f'{fit_idx}_cusum_{col}' for col in df_cusum.columns]
        else:
            df_cusum = None

        # CCDC
        if run_ccdc:
            ccdc_params = kwargs.get('ccdc_params', {})
            CCDCMonitor = CCDC(**ccdc_params)
            CCDCMonitor.fit(dataarray=history, method='CCDC-stable', screen_outliers=None, scaling_factor=10000, n_threads=6)

            ds_model_ccdc, _, _, _, _, _ = get_monitoring_datasets(
                nrtmodel=CCDCMonitor,
                monitoring=monitoring, reference_xr=ds_base, aoi_mask=aoi_mask,
                start_history=start_history, start_monitor=start_monitor, end_monitor=end_monitor, fit_method='CCDC-stable')
                # Drop 'spatial_ref' if it exists
            if 'spatial_ref' in ds_model_ccdc:
                ds_model_ccdc = ds_model_ccdc.drop_vars('spatial_ref')

            df_ccdc = ds_model_ccdc.to_dataframe()
            df_ccdc.columns = [f'{fit_idx}_ccdc_{col}' for col in df_ccdc.columns]
        else:
            df_ccdc = None

        # EWMA
        if run_ewma:
            ewma_params = kwargs.get('ewma_params', {})
            EwmaMonitor = EWMA(**ewma_params)
            EwmaMonitor.fit(dataarray=history, method=fit_methods[0], screen_outliers="Shewhart", n_threads=6)

            ds_model_ewma, _, _, _, _, _ = get_monitoring_datasets(
                nrtmodel=EwmaMonitor,
                monitoring=monitoring, reference_xr=ds_base, aoi_mask=aoi_mask,
                start_history=start_history, start_monitor=start_monitor, end_monitor=end_monitor, fit_method=fit_methods[0])
            if 'spatial_ref' in ds_model_ewma:
                ds_model_ewma = ds_model_ewma.drop_vars('spatial_ref')
            df_ewma = ds_model_ewma.to_dataframe()
            df_ewma.columns = [f'{fit_idx}_ewma_{col}' for col in df_ewma.columns]
        else:
            df_ewma = None

        # PELT
        if pelt_mask_for_nrt_annual:
            df_pelt = pelt_mask_for_nrt_annual[fit_idx].sel(time=f"{monitoring.time.dt.year.max().item()}", method='nearest').to_dataframe(name='pelt_breakpoints').drop(columns=['time'])
            df_pelt.columns = [f'{fit_idx}_{col}' for col in df_pelt.columns]
        else:
            df_pelt = None

        # Process for XGBoost
        if for_xgb:
            if df_mosum is not None:
                df_mosum[f'{fit_idx}_mosum_breakpoints'] = df_mosum[f'{fit_idx}_mosum_breakpoints'].apply(lambda x: 1 if x > 0 else x)
            if df_cusum is not None:
                df_cusum[f'{fit_idx}_cusum_breakpoints'] = df_cusum[f'{fit_idx}_cusum_breakpoints'].apply(lambda x: 1 if x > 0 else x)
            if df_ccdc is not None:
                df_ccdc[f'{fit_idx}_ccdc_breakpoints'] = df_ccdc[f'{fit_idx}_ccdc_breakpoints'].apply(lambda x: 1 if x > 0 else x)
            if df_ewma is not None:
                df_ewma[f'{fit_idx}_ewma_breakpoints'] = df_ewma[f'{fit_idx}_ewma_breakpoints'].apply(lambda x: 1 if x > 0 else x)
            if df_pelt is not None:
                df_pelt[f'{fit_idx}_pelt_breakpoints'] = df_pelt[f'{fit_idx}_pelt_breakpoints'].apply(lambda x: 1 if x > 0 else x)

        # Append DataFrames
        list_concat = [df for df in [df_ccdc, df_ewma, df_cusum, df_mosum, df_pelt] if df is not None]
        df_nrt_idx = pd.concat(list_concat, axis=1)
        df_idx_list.append(df_nrt_idx)

    # Add DEM data
    if ds_dem:
        df_dem = ds_dem.to_dataframe().drop(columns=['spatial_ref'])
        df_idx_list.append(df_dem)

    # Final concatenation
    df_nrt = pd.concat(df_idx_list, axis=1)
    return df_nrt


def sample_mask_with_clc_and_merge(mask, clc, classes, df_nrt, sample_fraction=0.3, random_state=None):
    rng = np.random.default_rng(random_state)  # Create a random number generator with a seed

    # Flatten the mask and CLC DataArray, but retain the x and y coordinates
    flat_mask = mask.values.flatten()
    flat_clc = clc.values.flatten()
    
    # Get the corresponding x and y coordinates
    y_coords, x_coords = np.meshgrid(mask['y'].values, mask['x'].values, indexing='ij')
    flat_x_coords = x_coords.flatten()
    flat_y_coords = y_coords.flatten()

    # Create a mapping from class values to class names
    value_to_class = {val: name for name, values in classes.items() for val in values}

    # Determine 'changed' (non-zero) and 'unchanged' (zero) values
    changed_indices = np.where(flat_mask > 0)[0]
    unchanged_indices = np.where(flat_mask == 0)[0]

    # Determine the minimum sample size to ensure 50-50 distribution
    sample_size = min(int(len(changed_indices) * sample_fraction), int(len(unchanged_indices) * sample_fraction))

    # Sample from changed and unchanged indices with equal sizes
    sampled_changed_indices = rng.choice(changed_indices, sample_size, replace=False)
    sampled_unchanged_indices = rng.choice(unchanged_indices, sample_size, replace=False)

    # Combine sampled indices
    sampled_indices = np.concatenate([sampled_changed_indices, sampled_unchanged_indices])

    # Create labels and class names
    # labels = np.array([change_idx] * sample_size + [-1] * sample_size)
    labels = np.array([1] * sample_size + [0] * sample_size)
    class_names = []

    for idx in sampled_indices:
        value = flat_clc[idx]
        class_name = value_to_class.get(value, np.nan)
        class_names.append(value)

    # Shuffle labels and class names
    combined = list(zip(sampled_indices, labels, class_names))
    rng.shuffle(combined)

    # Unzip the combined list
    sampled_indices, labels, class_names = zip(*combined)

    # Get the x and y coordinates corresponding to the sampled indices
    sampled_x_coords = flat_x_coords[list(sampled_indices)]
    sampled_y_coords = flat_y_coords[list(sampled_indices)]

    # Create DataFrame with sampled data and set MultiIndex with 'x' and 'y' coordinates
    sampled_data = pd.DataFrame({
        'label': labels,
        # 'class_name': class_names
    }, index=[sampled_y_coords, sampled_x_coords])

    # # # Set the index names to 'x' and 'y'
    sampled_data.index.names = ['y', 'x']

    # Select only the rows from df_nrt that match the sampled x, y coordinates
    merged_data = df_nrt.loc[sampled_data.index]
    
    # Combine the data from df_nrt with the sampled data
    sampled_data = pd.concat([sampled_data, merged_data], axis=1)

    return sampled_data


import planetary_computer
import pystac_client
from pystac_client.stac_api_io import StacApiIO
from urllib3 import Retry
import time

# TODO: Make this changeable...
def http_to_s3_url(http_url):
    """Convert a USGS HTTP URL to an S3 URL"""
    s3_url = http_url.replace(
        "https://landsatlook.usgs.gov/data", "s3://usgs-landsat"
    ).rstrip(":1")
    return s3_url


def connect_to_catalog(catalog_endpoint="landsatlook"):
    # Open a client 
    retry = Retry(
        total=10,
        backoff_factor=1,
        status_forcelist=[404, 502, 503, 504],
        allowed_methods=None, # {*} for CORS
    )
        
    stac_api_io = StacApiIO(max_retries=retry)

    logging.info("Initialize the STAC client")
    if catalog_endpoint=='planetary_computer':
        # Planetary Computer
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
            stac_io=stac_api_io
        )
    elif catalog_endpoint=='earth_search':
        # AWS
        catalog = pystac_client.Client.open(
            "https://earth-search.aws.element84.com/v1",
            stac_io=stac_api_io
            )
    elif catalog_endpoint=='landsatlook':
        catalog = pystac_client. Client.open(
            "https://landsatlook.usgs.gov/stac-server/",
            stac_io=stac_api_io
        )
    else:
        logging.error("You must provide a catalog endpoint alias, available [planetary_computer, earth_search, landsatlook]")

    return catalog