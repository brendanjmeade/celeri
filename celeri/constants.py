import numpy as np
import pyproj

GEOID = pyproj.Geod(ellps="WGS84")
KM2M = 1.0e3
M2MM = 1.0e3
RADIUS_EARTH = np.float64((GEOID.a + GEOID.b) / 2)
"""Radius of the earth in meters."""

DEG_PER_MYR_TO_RAD_PER_YR = 1 / 1e3
# The conversion should be 1 / 1e3. Linear units for Cartesian conversions are
# in meters, but we need to convert them to mm to be consistent with mm/yr
# geodetic constraints units. Rotation constraints are expressed in deg/Myr,
# and when applied in celeri we are effectively using m*rad/Myr. To convert
# to the right rate units, we need 1e-3*m*rad/Myr. This conversion is applied in
# get_data_vector (JPL 12/31/23)
N_MESH_DIM = 3
EPS = np.finfo(float).eps

VERTICAL_DIP_TOLERANCE_DEG = 0.05
"""Dips within this many degrees of 90 are treated as vertical.

The triangular dislocation solution (cutde, Nikkhoo & Walter 2015) that both
the segment and the mesh elastic operators use loses accuracy for a triangle
side that is close to, but not exactly, vertical: from about 1e-7 degrees
(below which cutde treats the side as vertical) to a few hundredths of a
degree, with errors up to ~1e9 times the peak displacement near 1e-6 degrees
(issue #536). Segments within this tolerance are snapped to 90 degrees, mesh
element sides within it are made vertical in the elastic calculation, and
mesh elements within it take no kinematic dip slip.

UNITS: [degrees]
"""

NEAR_VERTICAL_AMPLIFICATION_WARNING = 10.0
"""Warn about dipping segments and elements whose kinematic dip slip is the
fault-normal relative block motion amplified by more than this factor,
1 / |cos(dip)| (about 5.7 degrees from vertical; see issue #514).
"""
