import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

IMAGE_DIR = "./data/RL_feed"
DEPTH_DIR = "./data/RL_depth"

FPS = 1
FRAME_TIME = 1.0 / FPS

# Find frames that have both RGB and depth
frames = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
    and os.path.exists(
        os.path.join(
            DEPTH_DIR,
            os.path.splitext(f)[0] + ".npy"
        )
    )
])

if not frames:
    raise RuntimeError("No matching RGB + depth frames found.")

print(f"Found {len(frames)} frames.")
print("Playing at 1 FPS.")
print("Press Ctrl+C to stop.")

plt.ion()

fig, (ax_rgb, ax_depth) = plt.subplots(1, 2, figsize=(14, 6))

try:
    while True:

        for filename in frames:

            start = time.perf_counter()

            # -------------------------
            # Load RGB frame
            # -------------------------
            rgb_path = os.path.join(IMAGE_DIR, filename)
            rgb = Image.open(rgb_path).convert("RGB")

            # -------------------------
            # Load depth
            # -------------------------
            depth_path = os.path.join(
                DEPTH_DIR,
                os.path.splitext(filename)[0] + ".npy"
            )

            depth = np.load(depth_path)

            # -------------------------
            # Depth Pro visualization
            # -------------------------
            inverse_depth = 1.0 / np.maximum(depth, 1e-6)

            max_invdepth = min(
                inverse_depth.max(),
                1 / 0.1
            )

            min_invdepth = max(
                1 / 250,
                inverse_depth.min()
            )

            normalized = (
                inverse_depth - min_invdepth
            ) / (
                max_invdepth - min_invdepth + 1e-8
            )

            # -------------------------
            # Display
            # -------------------------
            ax_rgb.clear()
            ax_depth.clear()

            ax_rgb.imshow(rgb)
            ax_rgb.set_title(f"RGB — {filename}")
            ax_rgb.axis("off")

            ax_depth.imshow(
                normalized,
                cmap="turbo"
            )
            ax_depth.set_title("Depth Pro")
            ax_depth.axis("off")

            fig.tight_layout()
            plt.pause(0.001)

            # -------------------------
            # Maintain 1 FPS
            # -------------------------
            elapsed = time.perf_counter() - start
            remaining = FRAME_TIME - elapsed

            if remaining > 0:
                time.sleep(remaining)

except KeyboardInterrupt:
    print("\nStopped.")

finally:
    plt.close("all")