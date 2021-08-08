import copy
import numpy as np
from pyproj import Geod
from matplotlib import path

import celeri


RADIUS_EARTH = np.float64(6371e3)  # m
GEOID = Geod(ellps="WGS84")


def sph2cart(lon, lat, radius):
    x = radius * np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))
    y = radius * np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon))
    z = radius * np.sin(np.deg2rad(lat))
    return x, y, z


def process_station(station, command):
    if command["unit_sigmas"] == "yes":  # Assign unit uncertainties, if requested
        station.east_sig = np.ones_like(station.east_sig)
        station.north_sig = np.ones_like(station.north_sig)
        station.up_sig = np.ones_like(station.up_sig)

    station["dep"] = np.zeros_like(
        station.lon
    )  # Add a "dep" field of all zeros, to be used with project_tri_coords
    station["x"], station["y"], station["z"] = celeri.sph2cart(
        station.lon, station.lat, celeri.RADIUS_EARTH
    )
    station = station[station.tog == True]  # Keep only the stations that are toggled on
    return station


def locking_depth_manager(segment, command):
    """
    This function assigns the locking depths given in the command file to any
    segment that has the same locking depth flag.  Segments with flag =
    0, 1 are untouched.
    """
    segment = segment.copy(deep=True)
    segment.locking_depth.values[segment.locking_depth_flag == 2] = command[
        "locking_depth_flag2"
    ]
    segment.locking_depth.values[segment.locking_depth_flag == 3] = command[
        "locking_depth_flag3"
    ]
    segment.locking_depth.values[segment.locking_depth_flag == 4] = command[
        "locking_depth_flag4"
    ]
    segment.locking_depth.values[segment.locking_depth_flag == 5] = command[
        "locking_depth_flag5"
    ]

    if command["locking_depth_override_flag"] == "yes":
        segment.locking_depth.values = command["locking_depth_override_value"]
    return segment


def order_endpoints_sphere(segment):
    """
    Endpoint ordering function, placing west point first.
    This converts the endpoint coordinates from spherical to Cartesian,
    then takes the cross product to test for ordering (i.e., a positive z
    component of cross(point1, point2) means that point1 is the western
    point). This method works for both (-180, 180) and (0, 360) longitude
    conventions.
    """
    segment_copy = copy.deepcopy(segment)
    x1, y1, z1 = celeri.sph2cart(segment.lon1, segment.lat1, 1)
    x2, y2, z2 = celeri.sph2cart(segment.lon1, segment.lat2, 1)
    for i in range(x1.size):
        cross_product = np.cross(
            [x1[i], y1[i], z1[i]], [x2[i], y2[i], z2[i]]
        )  # TODO: Need to work on this!!!
        if cross_product[2] <= 0:
            segment_copy.lon1.values[i] = segment.lon2.values[i]
            segment_copy.lat1.values[i] = segment.lat2.values[i]
            segment_copy.lon2.values[i] = segment.lon1.values[i]
            segment_copy.lat2.values[i] = segment.lat1.values[i]
    return segment_copy


def segment_centroids(segment):
    """Calculate segment centroids."""
    segment["centroid_x"] = np.zeros_like(segment.lon1)
    segment["centroid_y"] = np.zeros_like(segment.lon1)
    segment["centroid_z"] = np.zeros_like(segment.lon1)
    segment["centroid_lon"] = np.zeros_like(segment.lon1)
    segment["centroid_lat"] = np.zeros_like(segment.lon1)

    for i in range(len(segment)):
        segment_forward_azimuth, _, _ = celeri.GEOID.inv(
            segment.lon1[i], segment.lat1[i], segment.lon2[i], segment.lat2[i]
        )
        segment_down_dip_azimuth = segment_forward_azimuth + 90.0 * np.sign(
            np.cos(np.deg2rad(segment.dip[i]))
        )
        azx = (segment.y2[i] - segment.y1[i]) / (segment.x2[i] - segment.x1[i])
        azx = np.arctan(-1.0 / azx)  # TODO: FIX THIS VARIABLE NAME
        segment.centroid_z.values[i] = (
            segment.locking_depth[i] - segment.burial_depth[i]
        ) / 2.0
        segment_down_dip_distance = segment.centroid_z[i] / np.abs(
            np.tan(np.deg2rad(segment.dip[i]))
        )
        (
            segment.centroid_lon.values[i],
            segment.centroid_lat.values[i],
            _,
        ) = celeri.GEOID.fwd(
            segment.mid_lon[i],
            segment.mid_lat[i],
            segment_down_dip_azimuth,
            segment_down_dip_distance,
        )
        segment.centroid_x.values[i] = segment.mid_x[i] + np.sign(
            np.cos(np.deg2rad(segment.dip[i]))
        ) * segment_down_dip_distance * np.cos(azx)
        segment.centroid_y.values[i] = segment.mid_y[i] + np.sign(
            np.cos(np.deg2rad(segment.dip[i]))
        ) * segment_down_dip_distance * np.sin(azx)
    segment.centroid_lon.values[segment.centroid_lon < 0.0] += 360.0
    return segment


def process_segment(segment, command):
    segment = celeri.order_endpoints_sphere(segment)
    segment["x1"], segment["y1"], segment["z1"] = celeri.sph2cart(
        segment.lon1, segment.lat1, celeri.RADIUS_EARTH
    )
    segment["x2"], segment["y2"], segment["z2"] = celeri.sph2cart(
        segment.lon2, segment.lat2, celeri.RADIUS_EARTH
    )
    segment["mid_lon_plate_carree"] = (segment.lon1.values + segment.lon2.values) / 2.0
    segment["mid_lat_plate_carree"] = (segment.lat1.values + segment.lat2.values) / 2.0
    segment["mid_lon"] = np.zeros_like(segment.lon1)
    segment["mid_lat"] = np.zeros_like(segment.lon1)

    for i in range(len(segment)):
        segment.mid_lon.values[i], segment.mid_lat.values[i] = celeri.GEOID.npts(
            segment.lon1[i], segment.lat1[i], segment.lon2[i], segment.lat2[i], 1
        )[0]
    segment.mid_lon.values[segment.mid_lon < 0.0] += 360.0

    segment["mid_x"], segment["mid_y"], segment["mid_z"] = celeri.sph2cart(
        segment.mid_lon, segment.mid_lat, celeri.RADIUS_EARTH
    )
    segment = celeri.locking_depth_manager(segment, command)
    # segment.locking_depth.values = PatchLDtoggle(segment.locking_depth, segment.patch_file_name, segment.patch_flag, Command.patchFileNames) % Set locking depth to zero on segments that are associated with patches # TODO: Write this after patches are read in.
    segment = celeri.segment_centroids(segment)
    return segment


def inpolygon(xq, yq, xv, yv):
    """From:
    https://stackoverflow.com/questions/31542843/inpolygon-for-python-examples-of-matplotlib-path-path-contains-points-method
    """
    shape = xq.shape
    xq = xq.reshape(-1)
    yq = yq.reshape(-1)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)
    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
    p = path.Path([(xv[i], yv[i]) for i in range(xv.shape[0])])
    return p.contains_points(q).reshape(shape)


# Trying a direct port of BlockLabel from Blocks
# This is the old Cartesian version
# function [s, b, st] = BlockLabel(s, b, st)
# %

# nseg = numel(s.lon1);

# % make sure western vertex is the start point
# [segx, i] = sort([s.lon1(:) s.lon2(:)], 2);
# segy = [s.lat1(:) s.lat2(:)];
# i = (i-1)*nseg + repmat((1:nseg)', 1, 2);
# segy = segy(i);

# % make sure there are no hanging segments
# allc 								= [segx(:) segy(:)];
# allc                       = [[s.lon1(:) s.lat1(:)]; [s.lon2(:) s.lat2(:)]];
# [cou, i1]					 	= unique(allc, 'rows', 'first');
# [cou, i2]						= unique(allc, 'rows', 'last');
# if isempty(~find(i2-i1, 1))
# 	fprintf(1, '*** All blocks are not closed! ***\n');
# else
# 	fprintf(1, 'No hanging segments found');
# end

# % Carry out a few operations on all segments

# % Find unique points and indices to them
# [unp, unidx, ui] = unique(allc, 'rows', 'first');
# us = ui(1:nseg); ue = ui(nseg+1:end);

# % Calculate the azimuth of each fault segment
# % Using atan instead of azimuth because azimuth breaks down for very long segments
# az1 = rad2deg(atan2(segx(:, 1) - segx(:, 2), segy(:, 1) - segy(:, 2)));
# az2 = rad2deg(atan2(segx(:, 2) - segx(:, 1), segy(:, 2) - segy(:, 1)));
# az = [az2 az1];
# az(az < 0) = az(az < 0) + 360;

# % Declare array to store polygon segment indices
# poly_ver                   = zeros(1, nseg);
# trav_ord                   = poly_ver;
# seg_poly_ver               = zeros(nseg);
# seg_trav_ord					= seg_poly_ver;


# for i = 1:nseg
# 	% establish starting coordinates
# 	cs = i; % current segment start
# 	cp = us(i); % current point: start point of the current segment
# 	se = 1; % flag indicating that it's looking towards ending point
#    starti = cs; % index of the starting point
#    seg_cnt = 1;

# 	clear poly_vec trav_ord

#    while 1
# 		matchss = (us == cp); % starts matching current
# 		matchss(cs) = 0;
# 		matchss = find(matchss);
# 		matches = (ue == cp); % ends matching current
# 		matches(cs) = 0;
# 		matches = find(matches);

#       match = [matchss; matches];

#       % If it's a multiple intersection, find which path to take
#       if numel(match) > 1
#       	daz = az(cs, se) - [az(matchss, 2); az(matches, 1)];
#          daz(find(abs(daz) > 180)) = daz(find(abs(daz) > 180)) - sign(daz(find(abs(daz) > 180)))*360;
#          [maz, mi] = max(daz);
#       else
#          mi = 1;
#       end
#       match = match(mi);

#       % Determine the new starting point
#       cs = match; % current index
#       if mi <= numel(matchss) % if the index is a start-current match
#       	cp = ue(cs); % the new point is the match's ending point
#       	se = 2; % looking towards the start point
#       else
#       	cp = us(cs); % otherwise it's the match's starting point
#       	se = 1; % looking towards the end point
#       end

#       % Prevent endless loops
#       if seg_cnt > nseg
#       	disp(sprintf('Cannot close block starting with segment: %s', s.name(starti, :)))
#    break;
#       end

#       % Break upon returning to the starting segment
#       if match == starti && seg_cnt > 1
#          seg_cnt              = 1;
#          poly_vec             = [poly_vec, starti];
#          trav_ord             = [trav_ord, se];
#    break;
#       else
#          poly_vec(seg_cnt)    = cs;
#          trav_ord(seg_cnt)    = se;
#          seg_cnt              = seg_cnt + 1;
#       end
#    end

#    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#    %%  Put poly_vec into seg_poly_ver                     %%
#    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#    seg_poly_ver(i, 1:length(poly_vec)) = poly_vec;
#    seg_trav_ord(i, 1:length(trav_ord)) = trav_ord;
# end

# % Determine the unique block polygons
# [so, blockrows] = unique(sort(seg_poly_ver, 2), 'rows');
# seg_poly_ver = seg_poly_ver(blockrows, :);
# seg_trav_ord = seg_trav_ord(blockrows, :);
# z = find(so == 0);
# so(z) = NaN;
# so = sort(so, 2);
# [so, blockrows] = sortrows(so);
# seg_poly_ver = seg_poly_ver(blockrows, :);
# seg_trav_ord = seg_trav_ord(blockrows, :);

# % Determine number of blocks
# nblock = size(seg_poly_ver, 1);

# % Calculate block area and label each block
# %bcx = zeros(nblock, nseg); % make an array for holding the circulation coordinates
# %bcy = bcx;
# barea = zeros(nblock, 1);
# alabel = zeros(nblock, 1);
# ext = 0;

# el = zeros(nseg, 1);
# wl = el;
# stl = zeros(numel(st.lon), 1);
# dLon = 1e-6;


# for i = 1:nblock
# 	% Take block coordinates from the traversal order matrix
# 	sib = seg_poly_ver(i, (seg_poly_ver(i, :) ~= 0)); % segments in block
# 	ooc = seg_trav_ord(i, (seg_trav_ord(i, :) ~= 0)); % order in which the segments are traversed
# 	cind = (ooc-1)*nseg + sib; % convert index pairs to linear index
# %  bcx(i, 1:numel(cind)) = segx(cind);
# %	bcy(i, 1:numel(cind)) = segy(cind);
#    bcx = segx(cind)';
#    bcy = segy(cind)';
#    barea(i) = polyarea(bcx, bcy);
# 	% Test which block interior points lie within the current circulation
# 	bin = inpolygon(b.interiorLon, b.interiorLat, bcx, bcy);
#    % Now test the segments for labeling east and west sides
#    testlon = s.midLon(sib) + dLon; % perturbed midpoint longitude
#    cin = inpolygon(testlon, s.midLat(sib), bcx, bcy); % test to see which perturbed coordinates lie within the current block
#    % Now test the station coordinates for block identification
#    stin = inpolygon(st.lon, st.lat, bcx, bcy);
# 	if numel(find(bin)) > 1 % exterior block or error
#       if barea(i) == max(barea) && ext == 0 % if the area is the largest and exterior hasn't yet been assigned
# 	      alabel(find(~bin)) = i; % ...assign this block as the exterior
#    	   ext = i; % and specifically declare the exterior label
#    	elseif ext > 0
#    		disp('Interior points do not uniquely define blocks!')
#          break;
#       end
#    else % if there is only one interior point within the segment polygon (i.e., all other blocks)...
#       alabel(find(bin)) = i; % assign that block associate label to the current block
#       el(sib(cin > 0)) = i; % segments within the polygon are assigned this block as their east label
#       wl(sib(cin == 0)) = i; % those that don't are assigned this block as their west label
#       stl(stin > 0) = i; % associate stations with the block
#    end

#    % Add ordered polygons to the blocks structure
#    b.orderLon{i} = bcx;
#    b.orderLat{i} = bcy;

# end

# if ext == 0 % Special case for a single block
#    ext = 2;
#    alabel = [1 2];
# end

# % treat exterior block segment labels - set exterior block for yet undefined segment labels
# el(el == 0) = ext;
# wl(wl == 0) = ext;
# % treat exterior block stations
# stl(stl == 0) = ext;

# % Final outputs
# s.eastLabel = el;
# s.westLabel = wl;

# [st.blockLabel, st.blockLabelUnused] = deal(stl);
# % Reorder block properties
# b = BlockReorder(alabel, b);
# b.associateLabel = alabel;
# b.exteriorBlockLabel = ext;
