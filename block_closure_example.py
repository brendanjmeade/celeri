import copy
import json
import scipy.spatial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from importlib import reload

import celeri

celeri = reload(celeri)

with open("./data/western_north_america/command.json", "r") as f:
    command = json.load(f)
st = pd.read_csv(command["station_file_name"])
s = pd.read_csv(command["segment_file_name"])
b = pd.read_csv(command["block_file_name"])
st = celeri.process_station(st, command)
s = celeri.process_segment(s, command)

### Start of BlockLabel.m translation ###

# Split any prime meridian-crossing segments
split = celeri.split_segments_crossing_meridian(s)
nseg = len(split)

# Make sure western vertex is the start point
S = celeri.order_endpoints_sphere(split)
sego = np.array([S.lon1.to_numpy(), S.lon2.to_numpy()]).T
sega = np.array([S.lat1.to_numpy(), S.lat2.to_numpy()]).T
segx = np.array([S.x1.to_numpy(), S.x2.to_numpy()]).T
segy = np.array([S.y1.to_numpy(), S.y2.to_numpy()]).T
segy[np.where(np.abs(segy) < 1e-6)] = 0
segz = np.array([S.z1.to_numpy(), S.z2.to_numpy()]).T
i = np.array(
    [np.arange(0, len(S)), np.arange(len(S), 2 * len(S))]
).T  # TODO: Not sure if this should start at 0 or 1

# Check for hanging segments #TODO: Why is this done with x,y,z instead of lon, lat?
allc = np.array([segx.T.flatten(), segy.T.flatten(), segz.T.flatten()]).T
if any(np.unique(allc, axis=0, return_counts=True)[1] == 1):
    print("Found hanging segments!")

# Find unique points and indices to them
ui = np.unique(allc, axis=0, return_index=True, return_inverse=True)[2]
us = ui[0:nseg]
ue = ui[nseg + 1 : -1]


# % Calculate the azimuth of each fault segment
# % Using atan instead of azimuth because azimuth breaks down for very long segments
# % az1 = rad2deg(atan2(segx(:, 1) - segx(:, 2), segy(:, 1) - segy(:, 2)));
# % az2 = rad2deg(atan2(segx(:, 2) - segx(:, 1), segy(:, 2) - segy(:, 1)));

# % Make local azimuths for every endpoint. The problem is indeed for really long segments;
# % the junction is essentially approached along an azimuth that doesn't sensibly fall between
# % the "exit" azimuths. So, for each end point, create a new test point 1 degree away along
# % the great circle, but because it's closer, the great circle azimuth will be "local" and
# % therefore makes the junction decision more sensibly.
# %
# % An example problem segment is the northern boundary of North America in the Blocks release
# % segment file. Its azimuth eastward, away from the northern Pacific Ocean, is about 38, but
# % westward away from Greenland, it's 315. This ends up affecting the Pacific and exterior blocks.

# % Local azimuths originating from endpoint 1
# az2 = sphereazimuth(sego(:, 1), sega(:, 1), sego(:, 2), sega(:, 2)); % Whole segment azimuth
# [tlo2, tla2] = gcpoint(sego(:, 1), sega(:, 1), az2, 1);   % Local test point coordinates
# az21 = sphereazimuth(sego(:, 1), sega(:, 1), tlo2, tla2); % Toward test point
# az22 = sphereazimuth(tlo2, tla2, sego(:, 1), sega(:, 1)); % From test point

# % Local azimuths originating from endpoint 2
# az1 = sphereazimuth(sego(:, 2), sega(:, 2), sego(:, 1), sega(:, 1)); % Local test point coordinates
# [tlo1, tla1] = gcpoint(sego(:, 2), sega(:, 2), az1, 1);   % Whole segment azimuth
# az11 = sphereazimuth(sego(:, 2), sega(:, 2), tlo1, tla1); % Toward test point
# az12 = sphereazimuth(tlo1, tla1, sego(:, 2), sega(:, 2)); % From test point
# saz = [az21 az11];
# eaz = [az22 az12];
# saz(saz < 0) = saz(saz < 0) + 360;
# eaz(eaz < 0) = eaz(eaz < 0) + 360;

# % Declare array to store polygon segment indices
# poly_ver = zeros(1, nseg);
# trav_ord = poly_ver;
# seg_poly_ver = zeros(nseg);
# seg_trav_ord = seg_poly_ver;

# for i = 1:nseg
#     % establish starting coordinates
#     cs = i;      % current segment start
#     cp = us(i);  % current point: start point of the current segment
#     se = 1;      % flag indicating that it's looking towards ending point
#     starti = cs; % index of the starting point
#     seg_cnt = 1;

#     clear poly_vec trav_ord

#     while 1
#         matchss = (us == cp); % starts matching current
#         matchss(cs) = 0;
#         matchss = find(matchss);
#         matches = (ue == cp); % ends matching current
#         matches(cs) = 0;
#         matches = find(matches);

#         match = [matchss; matches];

#         % If it's a multiple intersection, find which path to take
#         if numel(match) > 1
#             daz = saz(cs, se) - [eaz(matchss, 1); saz(matches, 1)];
#             daz(abs(daz) > 180) = daz(abs(daz) > 180) - sign(daz(abs(daz) > 180))*360;
#             [~, mi] = max(daz);
#         else
#             mi = 1;
#         end
#         match = match(mi);

#         % Determine the new starting point
#         cs = match; % current index
#         if mi <= numel(matchss) % if the index is a start-current match
#             cp = ue(cs); % the new point is the match's ending point
#             se = 2; % looking towards the start point
#         else
#             cp = us(cs); % otherwise it's the match's starting point
#             se = 1; % looking towards the end point
#         end

#         % Prevent endless loops
#         if seg_cnt > nseg
#             disp(sprintf('Cannot close block starting with segment: %s', s.name(starti, :)))
#             break;
#         end

#         % Break upon returning to the starting segment
#         if match == starti && seg_cnt > 1
#             seg_cnt = 1;
#             poly_vec = [poly_vec, starti];
#             trav_ord = [trav_ord, se];
#             break;
#         else
#             poly_vec(seg_cnt) = cs;
#             trav_ord(seg_cnt) = se;
#             seg_cnt = seg_cnt + 1;
#         end
#     end
#     % Put poly_vec into seg_poly_ver                     %%
#     seg_poly_ver(i, 1:length(poly_vec)) = poly_vec;
#     seg_trav_ord(i, 1:length(trav_ord)) = trav_ord;
# end

# % Determine the unique block polygons
# [so, blockrows] = unique(sort(seg_poly_ver, 2), 'rows');
# seg_poly_ver = seg_poly_ver(blockrows, :);
# seg_trav_ord = seg_trav_ord(blockrows, :);
# z = ~so;
# so(z) = NaN;
# so = sort(so, 2);
# [~, blockrows] = sortrows(so);
# seg_poly_ver = seg_poly_ver(blockrows, :);
# seg_trav_ord = seg_trav_ord(blockrows, :);

# % Determine number of blocks
# nblock = size(seg_poly_ver, 1);

# % Calculate block area and label each block
# barea = zeros(nblock, 1);
# alabel = zeros(nblock, 1);
# ext = 0;

# el = zeros(nseg, 1);
# wl = el;
# stl = zeros(numel(st.lon), 1);
# dLon = 1e-6;

# % Convert interior points to Cartesian
# [bix, biy, biz] = sph2cart(DegToRad(b.interiorLon), DegToRad(b.interiorLat), 6371);

# for i = 1:nblock
#     % Take block coordinates from the traversal order matrix
#     sib = seg_poly_ver(i, (seg_poly_ver(i, :) ~= 0)); % segments in block
#     ooc = seg_trav_ord(i, (seg_trav_ord(i, :) ~= 0)); % order in which the segments are traversed
#     cind = (ooc-1)*nseg + sib; % convert index pairs to linear index
#     bco = sego(cind)';
#     bca = sega(cind)';
#     % Test which interior points lie within the current block
#     [bin, saz] = inpolygonsphere(b.interiorLon, b.interiorLat, bco, bca);
#     if sum(bin) > 1
#        [bin, saz] = inpolygonsphere(b.interiorLon, b.interiorLat, bco, bca, saz-180);
#     end
#     barea(i) = polyareasphere(bco, bca);
#     % Now test the segments for labeling east and west sides
#     testlon = S.midLon(sib) + dLon; % perturbed midpoint longitude
#     cin = inpolygonsphere(testlon, S.midLat(sib), bco, bca, saz); % test to see which perturbed coordinates lie within the current block
#     % Now test the station coordinates for block identification
#     stin = inpolygonsphere(st.lon, st.lat, bco, bca, saz);
#     if sum(bin) > 1 % exterior block or error
#         if abs(barea(i)) == max(abs(barea)) && ext == 0 % if the area is the largest and exterior hasn't yet been assigned
#             alabel(find(~bin)) = i; % ...assign this block as the exterior
#             ext = i; % and specifically declare the exterior label
#         elseif ext > 0
#             disp('Interior points do not uniquely define blocks!')
#             break;
#         end
#     else % if there is only one interior point within the segment polygon (i.e., all other blocks)...
#         alabel(find(bin)) = i; % assign that block associate label to the current block
#         el(sib(cin > 0)) = i; % segments within the polygon are assigned this block as their east label
#         wl(sib(cin == 0)) = i; % those that don't are assigned this block as their west label
#         stl(stin > 0) = i; % associate stations with the block
#     end

#     % Add ordered polygons to the blocks structure
#     b.orderLon{i} = [bco(:); bco(1)];
#     b.orderLat{i} = [bca(:); bca(1)];
# end

# if ext == 0 % If exterior block is unassigned,
#    if length(b.interiorLon) <= 2 % Special case for a single block
#       ext = 2;
#       alabel = [1 2];
#       b.orderLon{2} = b.orderLon{1};
#       b.orderLat{2} = b.orderLat{1};
#    else
#       ext = setdiff(1:nblock, alabel); % Special case for a north pole block
#    end
# end

# if isempty(ext) % Final test if exterior block has not been labeled; label it by area
#    [~, ext] = max(abs(barea));
# end

# % treat exterior block segment labels - set exterior block for yet undefined segment labels
# el(el == 0) = ext;
# wl(wl == 0) = ext;
# % treat exterior block stations
# stl(stl == 0) = ext;

# % Final outputs
# S.eastLabel = el;
# S.westLabel = wl;

# [st.blockLabel, st.blockLabelUnused] = deal(stl);

# % Reorder block properties
# alabel(alabel == 0) = ext;
# b = BlockReorder(alabel, b);
# b.associateLabel = alabel;
# b.exteriorBlockLabel = ext;
