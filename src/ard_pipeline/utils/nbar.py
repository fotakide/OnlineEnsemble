# https://github.com/brazil-data-cube/compendium-harmonization/blob/a54585e8ba8d0d8edf49cb727b62dda1580de76d/tools/auxiliary-library/research_processing/nbar.py

import logging
import numpy
import xarray
import math
from sklearn.linear_model import LinearRegression, TheilSenRegressor

br_ratio = 1.0  # shape parameter
hb_ratio = 2.0  # crown relative height
DE2RA = math.pi/180  # Degree to Radian proportion (0.0174532925199432956)

# Coeffients in  Roy, D. P., Zhang, H. K., Ju, J., Gomez-Dans, J. L., Lewis, P. E., Schaaf, C. B., Sun Q., Li J., Huang H., & Kovalskyy, V. (2016).
# A general method to normalize Landsat reflectance data to nadir BRDF adjusted reflectance.
# Remote Sensing of Environment, 176, 255-271.
brdf_coefficients = {
    'blue': {
        'fiso': 774,
        'fgeo': 79,
        'fvol': 372
    },
    'green': {
        'fiso': 1306,
        'fgeo': 178,
        'fvol': 580
    },
    'red': {
        'fiso': 1690,
        'fgeo': 227,
        'fvol': 574
    },
    'nir08': {
        'fiso': 3093,
        'fgeo': 330,
        'fvol': 1535
    },
    'swir16': {
        'fiso': 3430,
        'fgeo': 453,
        'fvol': 1154
    },
    'swir22': {
        'fiso': 2658,
        'fgeo': 387,
        'fvol': 639
    }
}


def sec(angle):
    """Calculate secant.

    Args:
        angle (numpy array): raster band of angle.

    Returns:
        secant : numpy.array.
    """
    return 1./numpy.cos(angle) #numpy.divide(1./numpy.cos(angle))


def calc_cos_t(hb_ratio, d, theta_s_i, theta_v_i, relative_azimuth):
    """Calculate t cossine.

    Args:
        hb_ratio (int): h/b.
        d (numpy array): d.
        theta_s_i (numpy array): theta_s_i.
        theta_v_i (numpy array): theta_v_i.
        relative_azimuth (numpy array): relative_azimuth.

    Returns:
        cos_t : numpy.array.
    """
    return hb_ratio * numpy.sqrt(d*d + numpy.power(numpy.tan(theta_s_i)*numpy.tan(theta_v_i)*numpy.sin(relative_azimuth), 2)) / (sec(theta_s_i) + sec(theta_v_i))

def calc_d(theta_s_i, theta_v_i, relative_azimuth):
    """Calculate d.

    Args:
        theta_s_i (numpy array): theta_s_i.
        theta_v_i (numpy array): theta_v_i.
        relative_azimuth (numpy array): relative_azimuth.

    Returns:
        d : numpy.array.
    """
    return numpy.sqrt(
    numpy.tan(theta_s_i)*numpy.tan(theta_s_i) + numpy.tan(theta_v_i)*numpy.tan(theta_v_i) - 2*numpy.tan(theta_s_i)*numpy.tan(theta_v_i)*numpy.cos(relative_azimuth))


def calc_theta_i(angle, br_ratio):
    """Calculate calc_theta_i.

    Args:
        angle (numpy array): theta_s_i.
        br_ratio (int): b/r.

    Returns:
        theta_i : numpy.array.
    """
    return numpy.arctan(br_ratio * numpy.tan(angle))

def li_kernel(view_zenith, solar_zenith, relative_azimuth):
    """Calculate Li Kernel - Li X. and Strahler A. H., (1986) - Geometric-Optical Bidirectional Reflectance Modeling of a Conifer Forest Canopy.

    Args:
        view_zenith (numpy array): view zenith.
        solar_zenith (numpy array): solar zenith.
        relative_azimuth (numpy array): relative_azimuth.

    Returns:
        li_kernel : numpy.array.
    """
    # ref 1986
    theta_s_i = calc_theta_i(solar_zenith, br_ratio)
    theta_v_i = calc_theta_i(view_zenith, br_ratio)
    d = calc_d(theta_s_i, theta_v_i, relative_azimuth)
    cos_t = calc_cos_t(hb_ratio, d, theta_s_i, theta_v_i, relative_azimuth)
    t = numpy.arccos(numpy.maximum(-1., numpy.minimum(1., cos_t)))
    big_o = (1./numpy.pi)*(t-numpy.sin(t)*cos_t)*(sec(theta_v_i)*sec(theta_s_i))
    cos_e_i = numpy.cos(theta_s_i)*numpy.cos(theta_v_i) + numpy.sin(theta_s_i)*numpy.sin(theta_v_i)*numpy.cos(relative_azimuth)

    return big_o - sec(theta_s_i) - sec(theta_v_i) + 0.5*(1. + cos_e_i)*sec(theta_v_i)*sec(theta_s_i)


def ross_kernel(view_zenith, solar_zenith, relative_azimuth):
    """Calculate Ross-Thick Kernel.

    Args:
        view_zenith (numpy array): view zenith.
        solar_zenith (numpy array): solar zenith.
        relative_azimuth (numpy array): relative_azimuth.

    Returns:
        ross_thick_kernel : numpy.array.
    """
    cos_e = numpy.cos(solar_zenith)*numpy.cos(view_zenith) + numpy.sin(solar_zenith)*numpy.sin(view_zenith)*numpy.cos(relative_azimuth)
    e = numpy.arccos(cos_e)
    return ((((numpy.pi / 2.) - e)*cos_e + numpy.sin(e)) / (numpy.cos(solar_zenith) + numpy.cos(view_zenith))) - (numpy.pi / 4)


def calc_brf(view_zenith, solar_zenith, relative_azimuth, band_coef):
    """Calculate brf.

    Args:
        view_zenith (numpy array): view zenith.
        solar_zenith (numpy array): solar zenith.
        relative_azimuth (numpy array): relative_azimuth.
        band_coef (float): MODIS band coefficient.

    Returns:
        brf : numpy.array.
    """
    logging.debug('Calculating Li Sparce Reciprocal Kernel')
    li = li_kernel(view_zenith, solar_zenith, relative_azimuth)
    logging.debug('Calculating Ross Thick Kernel')
    ross = ross_kernel(view_zenith, solar_zenith, relative_azimuth)

    return band_coef['fiso'] + band_coef['fvol']*ross +band_coef['fgeo']*li


def prepare_angles(sza, saa, vza, vaa):
    relative_azimuth = ((vaa - saa) / 100) * DE2RA
    solar_zenith = (sza / 100) * DE2RA
    view_zenith = (vza / 100) * DE2RA
    solar_azimuth = (saa / 100) * DE2RA
    return view_zenith, solar_zenith, relative_azimuth, solar_azimuth


def correct_brdf_c_factor(ds_sr_bands, ds_angles):
    """
    Generate a Nadir corrected Bi-directional reflectance distribution function Adjusted Reflectance (NBAR)
    product through the c-factor approach for a single scene. The c-factor efficiency is well-proven for 
    medium spatial resolution products.
    https://doi.org/10.3389/frsen.2023.1254242

    References:

        See :cite:`roy_etal_2016` for the c-factor method.

        For further background on BRDF:

            :cite:`li_strahler_1992`

            :cite:`roujean_etal_1992`

            :cite:`schaaf_etal_2002`

    References:
        Ross-thick Li-sparse model in:
        Lucht, W., Schaaf, C. B., & Strahler, A. H. (2000).
        An algorithm for the retrieval of albedo from space using semiempirical BRDF models.
        IEEE Transactions on Geoscience and Remote Sensing, 38(2), 977-998.

    Sources:
        - https://github.com/brazil-data-cube/sensor-harm/blob/master/sensor_harm/harmonization_model.py
    """
    view_zenith, solar_zenith, relative_azimuth, solar_azimuth = prepare_angles(saa=ds_angles.SAA.data,
                                                                                sza=ds_angles.SZA.data,
                                                                                vaa=ds_angles.VAA.data,
                                                                                vza=ds_angles.VZA.data)
    for band_name in ds_sr_bands.data_vars:
        logging.info(f"Harmonizing band {band_name} ...")
        # Load angle bands
        band_coef = brdf_coefficients[band_name]

        # Compute c-factor
        brf_sensor = calc_brf(view_zenith, solar_zenith, relative_azimuth, band_coef)
        brf_ref = calc_brf(numpy.zeros(view_zenith.shape), solar_zenith, numpy.zeros(view_zenith.shape), band_coef)
        c_factor = brf_ref/brf_sensor

        # Apply scale for Landsat Collection-2
        # Rescale data to 0-10000 -> ((raster_arr * 0.0000275)-0.2)
        ds_sr_bands[band_name] =  ((ds_sr_bands[band_name] * 0.275) - 2000) 

        # Produce NBAR band
        ds_sr_bands[band_name] = ds_sr_bands[band_name] * c_factor

        # Remove values outside the valid range (0-10,000), setting the nodata value to `nan`
        invalid_ard_values = ((ds_sr_bands[band_name] < 0) | (ds_sr_bands[band_name] > 10000))
        ds_sr_bands[band_name] = ds_sr_bands[band_name].where(~invalid_ard_values)

    # Add angles to the dataset
    ds_sr_bands['view_zenith'] = xarray.DataArray(view_zenith, dims=("y", "x"), coords=ds_sr_bands.coords)
    ds_sr_bands['solar_zenith'] = xarray.DataArray(solar_zenith, dims=("y", "x"), coords=ds_sr_bands.coords)
    ds_sr_bands['relative_azimuth'] = xarray.DataArray(relative_azimuth, dims=("y", "x"), coords=ds_sr_bands.coords)
    ds_sr_bands['solar_azimuth'] = xarray.DataArray(solar_azimuth, dims=("y", "x"), coords=ds_sr_bands.coords)
    
    return ds_sr_bands


def _regress_a(X, y, robust, n_jobs):

    """Calculates the slope and intercept."""

    if robust:
        model = TheilSenRegressor(n_jobs=n_jobs)
    else:
        model = LinearRegression(n_jobs=n_jobs)

    model.fit(X, y)

    slope_m = model.coef_[0]
    intercept_b = model.intercept_

    return slope_m, intercept_b


def correct_topo_c_correction(ds_nbart_bands, ds_angles, ds_dem, robust, njobs):
    """
    Generate a Nadir corrected Bi-directional reflectance distribution function Adjusted Reflectance (NBAR)
    product through the c-factor approach for a single scene. The c-factor efficiency is well-proven for 
    medium spatial resolution products. However, in addition, this product applies terrain illumination correction 
    to correct for varying terrain.
    https://doi.org/10.3389/frsen.2023.1254242

    Additionally, normalizes terrain using the C-correction method.

    sources: 
    1. https://github.com/jgrss/geowombat/blob/main/src/geowombat/radiometry/topo.py
    2. https://github.com/brazil-data-cube/sensor-harm/blob/master/sensor_harm/harmonization_model.py

    References:
        C-correction method in:
        P.M. Teillet, B. Guindon & D.G. Goodenough (1982).
        On the Slope-Aspect Correction of Multispectral Scanner Data.
        Canadian Journal of Remote Sensing, 8(2), 84–106.
        https://doi.org/10.1080/07038992.1982.10855028
    """

    logging.info(f"Preparing IC and Cos(z) ...")
    dem_rad = ds_dem * DE2RA

    cos_z = numpy.cos(ds_angles.solar_zenith)
    sin_z = numpy.sin(ds_angles.solar_zenith)
    illumination_angle = numpy.cos(dem_rad.slope.data) * cos_z + numpy.sin(dem_rad.slope.data) * sin_z * numpy.cos(ds_angles.solar_azimuth - dem_rad.aspect.data)
    
    da_illumination_angle = ds_dem['slope'].copy()
    da_illumination_angle.data = illumination_angle
    da_illumination_angle.name = 'illumination_angle'

    for band_name in ds_nbart_bands.data_vars:
        logging.info(f"Correcting Terrain illumination of band {band_name} ...")
        nodata_mask = (ds_nbart_bands[band_name]==-999) | (da_illumination_angle.isnull())
        X = da_illumination_angle.where(~nodata_mask).data.flatten()
        y = ds_nbart_bands[band_name].where(~nodata_mask).data.flatten()
        X = X[~numpy.isnan(X)][:, numpy.newaxis]
        y = y[~numpy.isnan(y)]
        logging.info(f"Correcting Terrain illumination of band {band_name} ... linear regression")
        slope_m, intercept_b = _regress_a(X, y, robust, njobs)

        # compute C-factor
        c = intercept_b / slope_m

        # compute A-factor
        a_factor = (cos_z + c) / (illumination_angle + c)

        logging.info(f"Producing NBAR-T band {band_name} using the C-correction method...")
        # Produce NBAR-T band
        ds_nbart_bands[band_name] = ds_nbart_bands[band_name] * a_factor
        
        # Remove values outside the valid range (0-10,000), setting the nodata value to `nan`
        invalid_ard_values = ((ds_nbart_bands[band_name] < 0) | (ds_nbart_bands[band_name] > 10000))
        ds_nbart_bands[band_name] = ds_nbart_bands[band_name].where(~invalid_ard_values)

    # Add angles to the dataset
    nbart_scene = xarray.merge([ds_nbart_bands, ds_angles])

    return nbart_scene


def correct_topo_empirical_rotation(ds_nbart_bands, ds_angles, ds_dem, robust, njobs):
    """
    Generate a Nadir corrected Bi-directional reflectance distribution function Adjusted Reflectance (NBAR)
    product through the c-factor approach for a single scene. The c-factor efficiency is well-proven for 
    medium spatial resolution products. However, in addition, this product applies terrain illumination correction 
    to correct for varying terrain.
    https://doi.org/10.3389/frsen.2023.1254242

    Additionally, normalizes terrain using the Empirical Rotation method.

    sources: 
    1. https://github.com/jgrss/geowombat/blob/main/src/geowombat/radiometry/topo.py
    2. https://github.com/brazil-data-cube/sensor-harm/blob/master/sensor_harm/harmonization_model.py

    References:
        Empirical Rotation method in:
        B. Tan, R. Wolfe, J. Masek, F. Gao and E. F. Vermote (2010).
        An illumination correction algorithm on Landsat-TM data.
        IEEE International Geoscience and Remote Sensing Symposium, Honolulu, HI, USA, 2010, pp. 1964-1967.
        https://doi.org/10.1109/IGARSS.2010.5653492
    """
    logging.info(f"Preparing IC and Cos(z) ...")
    dem_rad = ds_dem * DE2RA

    cos_z = numpy.cos(ds_angles.solar_zenith)
    sin_z = numpy.sin(ds_angles.solar_zenith)
    illumination_angle = numpy.cos(dem_rad.slope.data) * cos_z + numpy.sin(dem_rad.slope.data) * sin_z * numpy.cos(ds_angles.solar_azimuth - dem_rad.aspect.data)
    
    da_illumination_angle = ds_dem['slope'].copy()
    da_illumination_angle.data = illumination_angle
    da_illumination_angle.name = 'illumination_angle'

    # Clustering instead of all pixels
    # 90th percentile of NDVI and NBR
    # Stratified random sampling across the range of IC values
    # if <5 cloud free pixels for a cluster abort

    for band_name in ds_nbart_bands.data_vars:
        logging.info(f"Correcting Terrain illumination of band {band_name} ...")
        nodata_mask = (ds_nbart_bands[band_name]==-999) | (da_illumination_angle.isnull())
        X = da_illumination_angle.where(~nodata_mask).data.flatten()
        y = ds_nbart_bands[band_name].where(~nodata_mask).data.flatten()
        X = X[~numpy.isnan(X)][:, numpy.newaxis]
        y = y[~numpy.isnan(y)]
        logging.info(f"Correcting Terrain illumination of band {band_name} ... linear regression")
        slope_m, intercept_b = _regress_a(X, y, robust, njobs)

        logging.info(f"Producing NBAR-T band {band_name} using the Empirical method...")
        # Produce NBAR-T band
        ds_nbart_bands[band_name] = ds_nbart_bands[band_name] - slope_m * (illumination_angle - cos_z)
        
        # Remove values outside the valid range (0-10,000), setting the nodata value to `nan`
        invalid_ard_values = ((ds_nbart_bands[band_name] < 0) | (ds_nbart_bands[band_name] > 10000))
        ds_nbart_bands[band_name] = ds_nbart_bands[band_name].where(~invalid_ard_values)

    # Add angles to the dataset
    nbart_scene = xarray.merge([ds_nbart_bands, ds_angles])

    return nbart_scene


def correct_topo_sun_canopy_sencor_c(ds_nbart_bands, ds_angles, ds_dem, robust, njobs):
    """
    Sun-Canopy-Sensor Correction + C

    Generate a Nadir corrected Bi-directional reflectance distribution function Adjusted Reflectance (NBAR)
    product through the c-factor approach for a single scene. The c-factor efficiency is well-proven for 
    medium spatial resolution products. However, in addition, this product applies terrain illumination correction 
    to correct for varying terrain.
    https://doi.org/10.3389/frsen.2023.1254242

    Additionally, normalizes terrain using the Sun-Canopy-Sensor Correction + C method.

    sources: 
    1. https://github.com/jgrss/geowombat/blob/main/src/geowombat/radiometry/topo.py
    2. https://github.com/brazil-data-cube/sensor-harm/blob/master/sensor_harm/harmonization_model.py

    References:
        SCS+C method in:
        Soenen, S. A., Peddle, D. R., & Coburn, C. A. (2005). 
        SCS+ C: A modified sun-canopy-sensor topographic correction in forested terrain. 
        IEEE Transactions on geoscience and remote sensing, 43(9), 2148-2159.
        https://ieeexplore.ieee.org/document/1499030
    """

    logging.info(f"Preparing IC and Cos(z) ...")
    dem_rad = ds_dem * DE2RA

    cos_z = numpy.cos(ds_angles.solar_zenith)
    sin_z = numpy.sin(ds_angles.solar_zenith)
    cos_a = numpy.cos(dem_rad.slope.data)
    illumination_angle = cos_a * cos_z + numpy.sin(dem_rad.slope.data) * sin_z * numpy.cos(ds_angles.solar_azimuth - dem_rad.aspect.data)
    
    da_illumination_angle = ds_dem['slope'].copy()
    da_illumination_angle.data = illumination_angle
    da_illumination_angle.name = 'illumination_angle'

    
    for band_name in ds_nbart_bands.data_vars:
        logging.info(f"Correcting Terrain illumination of band {band_name} ...")
        nodata_mask = (ds_nbart_bands[band_name]==-999) | (da_illumination_angle.isnull())
        X = da_illumination_angle.where(~nodata_mask).data.flatten()
        y = ds_nbart_bands[band_name].where(~nodata_mask).data.flatten()
        X = X[~numpy.isnan(X)][:, numpy.newaxis]
        y = y[~numpy.isnan(y)]
        logging.info(f"Correcting Terrain illumination of band {band_name} ... linear regression")
        slope_m, intercept_b = _regress_a(X, y, robust, njobs)
        # ds_sr_bands[band_name] = ds_sr_bands[band_name] - slope_m * (illumination_angle - cos_z)

        # compute C-factor
        c = intercept_b / slope_m

        # compute scs+c factor
        scs_c_factor = ((cos_a * cos_z) + c) / (illumination_angle + c)

        logging.info(f"Producing NBAR-T band {band_name} using the SCS+C method...")
        # Produce NBAR-T band
        ds_nbart_bands[band_name] = ds_nbart_bands[band_name] * scs_c_factor
        
        # Remove values outside the valid range (0-10,000), setting the nodata value to `nan`
        invalid_ard_values = ((ds_nbart_bands[band_name] < 0) | (ds_nbart_bands[band_name] > 10000))
        ds_nbart_bands[band_name] = ds_nbart_bands[band_name].where(~invalid_ard_values)

    # Add angles to the dataset
    nbart_scene = xarray.merge([ds_nbart_bands, ds_angles])

    return nbart_scene