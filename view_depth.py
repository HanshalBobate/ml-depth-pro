import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

NPY_FILE = "./data/RL_depth/frame_000000.npy"

depth = np.load(NPY_FILE)

# Same visualization approach as Depth Pro CLI
inverse_depth = 1.0 / depth

max_invdepth = min(inverse_depth.max(), 1 / 0.1)
min_invdepth = max(1 / 250, inverse_depth.min())

normalized = (
    inverse_depth - min_invdepth
) / (
    max_invdepth - min_invdepth
)

plt.imshow(normalized, cmap="turbo")
plt.axis("off")
plt.show()