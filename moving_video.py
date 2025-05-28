import cv2
import numpy as np

# Input and output video paths
input_path = 'drone-tracking-datasets/dataset1/cam0/cam0.mp4'
output_path = 'output.mp4'

# Open the input video
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

# Get frame dimensions and FPS
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Or use 'XVID'
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Parameters for ShiTomasi corner detection
maxCorners = 30
qualityLevel = 0.3
minDistance = 7
blockSize = 7

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Random colors for tracking points
color = np.random.randint(0, 255, (maxCorners, 3))

# Read the first frame and detect corners
ret, old_frame = cap.read()
if not ret:
    print("Error: Can't read the first frame.")
    cap.release()
    out.release()
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, maxCorners=maxCorners,
                             qualityLevel=qualityLevel, minDistance=minDistance,
                             blockSize=blockSize)

# Create a mask for drawing tracks
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None and st is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

        img = cv2.add(frame, mask)
        out.write(img)

        # Update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    else:
        break

# Release everything
cap.release()
out.release()
print(f"Done. Output saved to {output_path}")
