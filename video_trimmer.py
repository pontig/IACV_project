import imageio_ffmpeg as ffmpeg
import subprocess
import sys
import os

def trim_webm(input_path, output_path, trim_seconds):
    # Ensure input file exists
    if not os.path.isfile(input_path):
        print(f"Input file '{input_path}' does not exist.")
        return

    # Create the ffmpeg command
    cmd = [
        ffmpeg.get_ffmpeg_exe(),
        '-ss', str(trim_seconds),
        '-i', input_path,
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # force even dimensions
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-strict', 'experimental',
        '-y',
        output_path
    ]

    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print(f"Trimmed video saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error trimming video: {e}")

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python video_trimmer.py <input.webm> <seconds_to_trim> <output.mp4>")
        sys.exit(1)

    input_video = sys.argv[1]
    seconds_to_trim = float(sys.argv[2])
    output_video = sys.argv[3]

    trim_webm(input_video, output_video, seconds_to_trim)
