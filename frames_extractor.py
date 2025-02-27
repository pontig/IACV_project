import cv2 as cv
import os
import sys
import argparse

def extract_frames(video_path):
    output_folder = os.path.dirname(video_path)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv.imwrite(frame_filename, frame)
        frame_count += 1
        
        if frame_count % 200 == 0:
            print(f"Extracted {frame_count} frames")
    
    cap.release()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    parser.add_argument("dataset_no", type=int, help="Dataset number")
    parser.add_argument("camera_no", type=int, help="Camera number")
    args = parser.parse_args()
    
    video_path = f"drone-tracking-datasets/dataset{args.dataset_no}/cam{args.camera_no}/cam{args.camera_no}.mp4"
    print(f"Extracting frames from {video_path}")
    extract_frames(video_path)
    print("Frames extracted.")