import time
import torch
from PIL import Image
import depth_pro

IMAGE = "./data/example.jpg"
WARMUP = 5
ITERATIONS = 20


print("Loading model...")
model, transform = depth_pro.create_model_and_transforms()
model.eval()
model = model.cuda()

image, _, f_px = depth_pro.load_rgb(IMAGE)
image = transform(image).cuda()
print("Model device:", next(model.parameters()).device)
print("Image device:", image.device)
# Warm up CUDA + model
print(f"Warming up ({WARMUP} runs)...")
with torch.inference_mode():
    for _ in range(WARMUP):
        model.infer(image, f_px=f_px)

torch.cuda.synchronize()

# Actual benchmark
times = []

print(f"Benchmarking ({ITERATIONS} runs)...")

with torch.inference_mode():
    for _ in range(ITERATIONS):
        torch.cuda.synchronize()
        start = time.perf_counter()

        prediction = model.infer(image, f_px=f_px)

        torch.cuda.synchronize()
        end = time.perf_counter()

        times.append((end - start) * 1000)

avg = sum(times) / len(times)
fps = 1000 / avg

print()
print(f"Average inference: {avg:.2f} ms")
print(f"Min inference:     {min(times):.2f} ms")
print(f"Max inference:     {max(times):.2f} ms")
print(f"Approx FPS:        {fps:.2f}")