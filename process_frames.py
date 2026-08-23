import os
import time
import numpy as np
import torch
import depth_pro

INPUT_DIR = "./data/RL_feed"
OUTPUT_DIR = "./data/RL_depth"
WARMUP = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading Depth Pro...")
model, transform = depth_pro.create_model_and_transforms(
    device="cuda:0",
    precision=torch.half,
)
model.eval()

files = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

print(f"Frames found: {len(files)}")

# -------------------------
# Warmup
# -------------------------

print("Warming up...")

first_path = os.path.join(INPUT_DIR, files[0])
image, _, f_px = depth_pro.load_rgb(first_path)
image = transform(image).cuda()

with torch.inference_mode():
    for _ in range(WARMUP):
        model.infer(image, f_px=f_px)

torch.cuda.synchronize()

# -------------------------
# Process frames
# -------------------------

times = []

for i, filename in enumerate(files):

    path = os.path.join(INPUT_DIR, filename)

    image, _, f_px = depth_pro.load_rgb(path)
    image = transform(image).cuda()

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        prediction = model.infer(image, f_px=f_px)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    times.append(elapsed)

    depth = prediction["depth"].cpu().numpy()

    output_name = os.path.splitext(filename)[0] + ".npy"
    np.save(os.path.join(OUTPUT_DIR, output_name), depth)

    print(
        f"[{i+1:03d}/{len(files)}] "
        f"{filename} → "
        f"{elapsed * 1000:.1f} ms"
    )

# -------------------------
# Results
# -------------------------

avg = np.mean(times)
fps = 1 / avg

print()
print("=" * 40)
print("RESULT")
print("=" * 40)
print(f"Frames processed : {len(files)}")
print(f"Average inference: {avg * 1000:.2f} ms")
print(f"Min inference    : {min(times) * 1000:.2f} ms")
print(f"Max inference    : {max(times) * 1000:.2f} ms")
print(f"Effective FPS    : {fps:.2f}")
print(f"Output directory : {OUTPUT_DIR}")