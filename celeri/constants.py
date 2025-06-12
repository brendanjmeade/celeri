import numpy as np
import pyproj

GEOID = pyproj.Geod(ellps="WGS84")
KM2M = 1.0e3
M2MM = 1.0e3
RADIUS_EARTH = np.float64((GEOID.a + GEOID.b) / 2)
DEG_PER_MYR_TO_RAD_PER_YR = 1 / 1e3
# The conversion should be 1 / 1e3. Linear units for Cartesian conversions are
# in meters, but we need to convert them to mm to be consistent with mm/yr
# geodetic constraints units. Rotation constraints are expressed in deg/Myr,
# and when applied in celeri we are effectively using m*rad/Myr. To convert
# to the right rate units, we need 1e-3*m*rad/Myr. This conversion is applied in
# get_data_vector (JPL 12/31/23)
N_MESH_DIM = 3
EPS = np.finfo(float).eps
