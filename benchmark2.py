import time
import torch
import depth_pro

IMAGE = "./data/real_img.jpg"
WARMUP = 3
ITERATIONS = 10

model, transform = depth_pro.create_model_and_transforms()
model.eval().cuda()

image, _, f_px = depth_pro.load_rgb(IMAGE)
image = transform(image).cuda()

print("Input tensor:", image.shape)

with torch.inference_mode():
    for _ in range(WARMUP):
        model.infer(image, f_px=f_px)

torch.cuda.synchronize()

times = []

with torch.inference_mode():
    for _ in range(ITERATIONS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        model.infer(image, f_px=f_px)

        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

avg = sum(times) / len(times)

print(f"Average: {avg:.2f} ms")
print(f"FPS: {1000 / avg:.2f}")