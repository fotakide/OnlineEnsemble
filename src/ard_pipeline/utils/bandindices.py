# Based on https://github.com/GeoscienceAustralia/dea-notebooks/blob/develop/Tools/dea_tools/bandindices.py
import xarray as xr
import numpy as np

def calculate_landsat_indices(ds,
                              index,
                              nbart=False,
                              nbar=False
                              ):
    
    if nbart:
        ds = ds.rename({
            'nbart_blue': 'blue',
            'nbart_green': 'green',
            'nbart_red': 'red',
            'nbart_nir08': 'nir08',
            'nbart_swir16': 'swir16',
            'nbart_swir22': 'swir22',
        })
    elif nbar:
        ds = ds.rename({
            'nbar_blue': 'blue',
            'nbar_green': 'green',
            'nbar_red': 'red',
            'nbar_nir08': 'nir08',
            'nbar_swir16': 'swir16',
            'nbar_swir22': 'swir22',
        })
    
    # Dictionary containing remote sensing index band recipes
    index_dict = {
                  # Normalised Difference Vegation Index, Rouse 1973
                  'NDVI': lambda ds: (ds.nir08 - ds.red) /
                                     (ds.nir08 + ds.red),
        
                  # Non-linear Normalised Difference Vegation Index,
                  # Camps-Valls et al. 2021
                  'kNDVI': lambda ds: np.tanh(((ds.nir08 - ds.red) /
                                               (ds.nir08 + ds.red)) ** 2),

                  # Enhanced Vegetation Index, Huete 2002
                  'EVI': lambda ds: ((2.5 * (ds.nir08 - ds.red)) /
                                     (ds.nir08 + 6 * ds.red -
                                      7.5 * ds.blue + 1)),

                  # Leaf Area Index, Boegh 2002
                  'LAI': lambda ds: (3.618 * ((2.5 * (ds.nir08 - ds.red)) /
                                     (ds.nir08 + 6 * ds.red -
                                      7.5 * ds.blue + 1)) - 0.118),

                  # Soil Adjusted Vegetation Index, Huete 1988
                  'SAVI': lambda ds: ((1.5 * (ds.nir08 - ds.red)) /
                                      (ds.nir08 + ds.red + 0.5)),
      
                  # Mod. Soil Adjusted Vegetation Index, Qi et al. 1994
                  'MSAVI': lambda ds: ((2 * ds.nir08 + 1 - 
                                      ((2 * ds.nir08 + 1)**2 - 
                                       8 * (ds.nir08 - ds.red))**0.5) / 2),    

                  # Normalised Difference Moisture Index, Gao 1996
                  'NDMI': lambda ds: (ds.nir08 - ds.swir16) /
                                     (ds.nir08 + ds.swir16),

                  # Normalised Burn Ratio, Lopez Garcia 1991
                  'NBR': lambda ds: (ds.nir08 - ds.swir22) /
                                    (ds.nir08 + ds.swir22),

                  # Burn Area Index, Martin 1998
                  'BAI': lambda ds: (1.0 / ((0.10 - ds.red) ** 2 +
                                            (0.06 - ds.nir08) ** 2)),
        
                 # Normalised Difference Chlorophyll Index, 
                 # (Mishra & Mishra, 2012)
                  'NDCI': lambda ds: (ds.red_edge_1 - ds.red) /
                                     (ds.red_edge_1 + ds.red),

                  # Normalised Difference Snow Index, Hall 1995
                  'NDSI': lambda ds: (ds.green - ds.swir16) /
                                     (ds.green + ds.swir16),

                  # Normalised Difference Tillage Index,
                  # Van Deventer et al. 1997
                  'NDTI': lambda ds: (ds.swir16 - ds.swir22) /
                                     (ds.swir16 + ds.swir22),
        
                  # Normalised Difference Turbidity Index,
                  # Lacaux et al., 2007
                  'NDTI2': lambda ds: (ds.red - ds.green) /
                                     (ds.red + ds.green),

                  # Normalised Difference Water Index, McFeeters 1996
                  'NDWI': lambda ds: (ds.green - ds.nir08) /
                                     (ds.green + ds.nir08),

                  # Modified Normalised Difference Water Index, Xu 2006
                  'MNDWI': lambda ds: (ds.green - ds.swir16) /
                                      (ds.green + ds.swir16),
      
                  # Normalised Difference Built-Up Index, Zha 2003
                  'NDBI': lambda ds: (ds.swir16 - ds.nir08) /
                                     (ds.swir16 + ds.nir08),
      
                  # Built-Up Index, He et al. 2010
                  'BUI': lambda ds:  ((ds.swir16 - ds.nir08) /
                                      (ds.swir16 + ds.nir08)) -
                                     ((ds.nir08 - ds.red) /
                                      (ds.nir08 + ds.red)),
      
                  # Built-up Area Extraction Index, Bouzekri et al. 2015
                  'BAEI': lambda ds: (ds.red + 0.3) /
                                     (ds.green + ds.swir16),
      
                  # New Built-up Index, Jieli et al. 2010
                  'NBI': lambda ds: (ds.swir16 + ds.red) / ds.nir08,
      
                  # Bare Soil Index, Rikimaru et al. 2002
                  'BSI': lambda ds: ((ds.swir16 + ds.red) - 
                                     (ds.nir08 + ds.blue)) / 
                                    ((ds.swir16 + ds.red) + 
                                     (ds.nir08 + ds.blue)),

                  # Automated Water Extraction Index (no shadows), Feyisa 2014
                  'AWEI_ns': lambda ds: (4 * (ds.green - ds.swir16) -
                                        (0.25 * ds.nir08 * + 2.75 * ds.swir22)),

                  # Automated Water Extraction Index (shadows), Feyisa 2014
                  'AWEI_sh': lambda ds: (ds.blue + 2.5 * ds.green -
                                         1.5 * (ds.nir08 + ds.swir16) -
                                         0.25 * ds.swir22),

                  # Water Index, Fisher 2016
                  'WI': lambda ds: (1.7204 + 171 * ds.green + 3 * ds.red -
                                    70 * ds.nir08 - 45 * ds.swir16 -
                                    71 * ds.swir22),

                  # Tasseled Cap Wetness, Crist 1985
                  'TCW_TM': lambda ds: (0.0315 * ds.blue + 0.2021 * ds.green +
                                     0.3102 * ds.red + 0.1594 * ds.nir08 +
                                    -0.6806 * ds.swir16 + -0.6109 * ds.swir22),

                  # Tasseled Cap Greeness, Crist 1985
                  'TCG_TM': lambda ds: (-0.1603 * ds.blue + -0.2819 * ds.green +
                                     -0.4934 * ds.red + 0.7940 * ds.nir08 +
                                     -0.0002 * ds.swir16 + -0.1446 * ds.swir22),

                  # Tasseled Cap Brightness, Crist 1985
                  'TCB_TM': lambda ds: (0.2043 * ds.blue + 0.4158 * ds.green +
                                     0.5524 * ds.red + 0.5741 * ds.nir08 +
                                     0.3124 * ds.swir16 + -0.2303 * ds.swir22),

                  # Tasseled Cap Wetness, Zhai 2022
                  'TCW': lambda ds: (0.1301 * ds.blue + 0.2280 * ds.green +
                                     0.3492 * ds.red + 0.1795 * ds.nir08 +
                                    -0.6270 * ds.swir16 + -0.6195 * ds.swir22),

                  # Tasseled Cap Greeness, Zhai 2022
                  'TCG': lambda ds: (-0.2365 * ds.blue + -0.2836 * ds.green +
                                     -0.4257 * ds.red + 0.8097 * ds.nir08 +
                                     0.0043 * ds.swir16 + -0.1638 * ds.swir22),

                  # Tasseled Cap Brightness, Zhai 2022
                  'TCB': lambda ds: (0.3443 * ds.blue + 0.4057 * ds.green +
                                     0.4667 * ds.red + 0.5347 * ds.nir08 +
                                     0.3936 * ds.swir16 + 0.2412 * ds.swir22),
                  
                  # Tasseled Cap Transformations with Sentinel-2 coefficients 
                  # after Nedkov 2017 using Gram-Schmidt orthogonalization (GSO)
                  # Tasseled Cap Wetness, Nedkov 2017
                  'TCW_GSO': lambda ds: (0.0649 * ds.blue + 0.2802 * ds.green +
                                         0.3072 * ds.red + -0.0807 * ds.nir08 +
                                        -0.4064 * ds.swir16 + -0.5602 * ds.swir22),

                  # Tasseled Cap Greeness, Nedkov 2017
                  'TCG_GSO': lambda ds: (-0.0635 * ds.blue + -0.168 * ds.green +
                                         -0.348 * ds.red + 0.3895 * ds.nir08 +
                                         -0.4587 * ds.swir16 + -0.4064 * ds.swir22),

                  # Tasseled Cap Brightness, Nedkov 2017
                  'TCB_GSO': lambda ds: (0.0822 * ds.blue + 0.136 * ds.green +
                                         0.2611 * ds.red + 0.5741 * ds.nir08 +
                                         0.3882 * ds.swir16 + 0.1366 * ds.swir22),

                  # Clay Minerals Ratio, Drury 1987
                  'CMR': lambda ds: (ds.swir16 / ds.swir22),

                  # Ferrous Minerals Ratio, Segal 1982
                  'FMR': lambda ds: (ds.swir16 / ds.nir08),

                  # Iron Oxide Ratio, Segal 1982
                  'IOR': lambda ds: (ds.red / ds.blue),
    }

    # If index supplied is not a list, convert to list. This allows us to
    # iterate through either multiple or single indices in the loop below
    indices = index if isinstance(index, list) else [index]

    # Calculate for each index in the list of indices supplied (indexes)
    ds_indices_to_be_placed_in_ds = {}
    for index in indices:
        # Select an index function from the dictionary
        index_func = index_dict.get(str(index))

        if index_func is None:
            raise ValueError(f"The selected index '{index}' is not one of the "
                              "valid remote sensing index options. \nPlease "
                              "refer to the function documentation for a full "
                              "list of valid options for `index`")
        
        # Apply index function 
        try:
            _ds = ds.copy(deep=True)

            for var_name, da in _ds.data_vars.items():
                nodata_value = da.attrs.get('nodata', None)
                _ds[var_name] = da.where(da != nodata_value, np.nan)

            if index in ['WI', 'BAEI', 'AWEI_ns', 'AWEI_sh', 'EVI', 'LAI', 'SAVI', 'MSAVI']:
                normalise = True
            else:
                normalise = False
            
            # If normalised=True, divide data by 10,000 before applying func
            mult = 10000.0 if normalise else 1.0
            # Multiply float by 1,000 to keep 3 decimal places before converting to integer 16-bit
            if index not in ['TCW', 'TCB', 'TCG', 'TCW_GSO', 'TCB_GSO', 'TCG_GSO']:
                index_array = index_func(_ds / mult) * 1000 
            else:
                index_array = index_func(_ds / mult)
        except AttributeError:
            raise ValueError(f'Please verify that all bands required to '
                             f'compute {index} are present in `ds`.')

        # Replace np.nan with -9999 and convert to integer 16-bit
        nodata = -9999
        index_array = index_array.where(~index_array.isnull(), nodata).astype('int16')
        index_array.attrs['nodata'] = nodata

        ds_indices_to_be_placed_in_ds[index.lower()] = index_array
        
    # Add remote sensing spectral indices in dataset
    for var, arr in ds_indices_to_be_placed_in_ds.items():
        ds[var] = arr

    if nbart:
        ds = ds.rename({
            'blue': 'nbart_blue',
            'green': 'nbart_green',
            'red': 'nbart_red',
            'nir08': 'nbart_nir08',
            'swir16': 'nbart_swir16',
            'swir22': 'nbart_swir22',
        })
    elif nbar:
        ds = ds.rename({
            'blue': 'nbar_blue',
            'green': 'nbar_green',
            'red': 'nbar_red',
            'nir08': 'nbar_nir08',
            'swir16': 'nbar_swir16',
            'swir22': 'nbar_swir22',
        })

    return ds