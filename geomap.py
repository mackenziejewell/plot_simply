# Functions for importing and processing NSIDC Polar Pathfinder sea ice drift data (NSIDC-0116, doi: 10.5067/INAWUWO7QH7B)

# DEPENDENCIES:
import xarray as xr
import numpy as np
import numpy.ma as ma
import cartopy
import cartopy.crs as ccrs
from datetime import datetime, timedelta
from metpy.units import units


# FUNCTIONS:
#---------------------------------------------------------------------
def fix_cartopy_vectors(u, v, uvlats):
    
    """Function to output vector components for plotting in cartopy. 
    
    Reads in vectors and associated latitudes, return fixed vectors. 
    
    Cartopy doesn't know meters per degree increase toward the pole 
    in zonal direction as cosine(lat) when reprojecting for vectors 
    given as m/s (where  a meter covers different amount of a degree 
    depending on latitude), we need to rescale the u (east-west) speeds. 
    otherwise at high latitudes, u will be drawn much larger than 
    they should which will give incorrect angles

INPUT: 
- u: (N x M) array of eastward velocity component (m/s)
- v: (N x M) array of northward velocity component (m/s)
- uvlats: (N x M) array latitudes associated with u,v vectors

OUTPUT:
- u_fixed: (N x M) array of u with correct angle and magnitude for plotting
- v_fixed: (N x M) array of v with correct angle and magnitude for plotting

Latest recorded update:
12-17-2024
    """
    
    # FIX ANGLE
    # fix u scale to be correct relative to v scale
    #----------------------------------------------
    # for u (m/s), m/degree decreases as cos(lat)
    u_fixed = u/np.cos(uvlats/180*np.pi) 
    v_fixed = v  # v does not change with latitude

    # FIX MAGNITUDE
    # scale fixed u,v to have correct magnitude 
    #-----------------------
    # original magnitude 
    orig_mag = ma.sqrt(u**2+v**2)
    # new magnitude
    fixed_mag = ma.sqrt(u_fixed**2+v_fixed**2)
    u_fixed = u_fixed*(orig_mag/fixed_mag)
    v_fixed = v_fixed*(orig_mag/fixed_mag)

    return u_fixed, v_fixed
