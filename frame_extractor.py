import cv2
import os

VIDEO_PATH = r"D:\OBS-RECORDINGS\2026-08-14 20-07-22.mp4"
OUTPUT_DIR = r"C:\Users\bobat\OneDrive\Desktop\Hackathon\SIH2026\ml-depth-pro\data\RL_feed"

START_TIME = 6 * 60 + 40   # 06:40
END_TIME = 6 * 60 + 55     # 06:55

FPS = 5
WIDTH = 854
HEIGHT = 480

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

video_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / video_fps

print(f"Video FPS: {video_fps}")
print(f"Video duration: {duration:.2f} seconds")
print(f"Extracting: {START_TIME}s -> {END_TIME}s")
print(f"Output: {WIDTH}x{HEIGHT} @ {FPS} FPS")

# Start at requested timestamp using frame indices
start_frame = int(START_TIME * video_fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

frame_interval = 1.0 / FPS
next_frame_time = START_TIME

frame_number = 0

while True:
    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    current_time = current_frame / video_fps

    if current_time >= END_TIME:
        break

    ret, frame = cap.read()

    if not ret:
        break

    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    current_time = current_frame / video_fps

    # Only save frames at 5 FPS
    if current_time >= next_frame_time:
        # STRICT 480p: 854 x 480
        frame = cv2.resize(
            frame,
            (WIDTH, HEIGHT),
            interpolation=cv2.INTER_AREA
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            f"frame_{frame_number:06d}.jpg"
        )

        cv2.imwrite(
            output_path,
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

        frame_number += 1
        next_frame_time += frame_interval

        if frame_number % 100 == 0:
            print(f"Saved {frame_number} frames...")

cap.release()

print("\nDone.")
print(f"Total frames saved: {frame_number}")
print(f"Location: {OUTPUT_DIR}")