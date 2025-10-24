# %% Imports
import matplotlib.pyplot as plt  # noqa
import matplotlib_inline.backend_inline
import numpy as np

import celeri

matplotlib_inline.backend_inline.set_matplotlib_formats("retina")


# %% Read the segment file and search for hanging segments


# %% Read model config
config_file_name = "./data/config/wna_config_constraints.json"
model = celeri.build_model(config_file_name)

# %% Plot some diagnostics
plt.figure(figsize=(10, 10))
for i in range(len(model.segment)):
    plt.plot(
        [model.segment.lon1[i], model.segment.lon2[i]],
        [model.segment.lat1[i], model.segment.lat2[i]],
        "-b",
        linewidth=0.5,
    )

# Plot block interior points with labels
for i in range(len(model.block)):
    plt.text(
        model.block.interior_lon[i],
        model.block.interior_lat[i],
        f"{model.block.block_label[i]}",
    )
plt.show(block=False)

# Plot stations on a specific block
current_block_idx = np.where(model.station.block_label == 92)[0]
plt.plot(
    model.station.lon[current_block_idx], model.station.lat[current_block_idx], "r+"
)
