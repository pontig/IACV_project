import imageio
import cv2
import os
from PIL import Image

# Configuration
input_path = "plots/drone_tracking_output_optical_flow.mp4"  # Can be .webm or .mp4
output_dir = "output_dir"
max_frames = 10000  # Maximum number of frames to process
min_frames = 0

# Supported video formats
SUPPORTED_FORMATS = ('.webm', '.mp4')

# Check input file extension
if not input_path.lower().endswith(SUPPORTED_FORMATS):
    raise ValueError(f"Input file must be one of: {SUPPORTED_FORMATS}")

# Create output directory
os.makedirs(output_dir, exist_ok=True)

def get_video_info_opencv(video_path):
    """Get video info using OpenCV"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"🎥 Video opened succesfully at {fps} fps")
    
    cap.release()
    return fps, total_frames, (width, height)

def extract_frames(video_path, output_dir, max_frames):
    """Extract full frames sequentially using imageio"""
    try:
        reader = imageio.get_reader(video_path)
        total_frames = len(reader)
        frames_to_process = min(max_frames, total_frames)

        print(f"📽️ Extracting {frames_to_process} frames...")

        for idx in range(frames_to_process):
            if idx < min_frames:
                continue
            frame = reader.get_data(idx)
            output_path = os.path.join(output_dir, f"frame_{idx:03}.png")
            imageio.imwrite(output_path, frame)
            print(f"✅ Saved frame {idx} to '{output_path}'")

        reader.close()
        print("🎉 Finished saving frames.")

    except Exception as e:
        print(f"❌ Error extracting frames: {e}")

# Main execution
fps, total_frames, resolution = get_video_info_opencv(input_path)
if fps is not None:
    print(f"📐 Video Info: {resolution[0]}x{resolution[1]} @ {fps:.2f} FPS, {total_frames} total frames")

extract_frames(input_path, output_dir, max_frames)
