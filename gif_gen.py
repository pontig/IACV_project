import moviepy.editor as mp

def mp4_to_gif(input_file, output_file, seconds_start, seconds_end):
    """
    Convert a portion of an MP4 file to GIF.
    
    Args:
        input_file (str): Path to the input MP4 file
        output_file (str): Path to save the output GIF file
        seconds_start (float): Start time in seconds
        seconds_end (float): End time in seconds
    """
    # Load the video file
    video = mp.VideoFileClip(input_file)
    
    # Extract the subclip
    clip = video.subclip(seconds_start, seconds_end)
    
    # Write to GIF file
    clip.write_gif(output_file, fps=10)
    
    # Clean up
    video.close()
    clip.close()

# Example usage:
# mp4_to_gif("input_video.mp4", "output.gif", 10, 20)
mp4_to_gif("plots/dataset1_cam2_tracking_output.mp4", "documentation/imgs/tracking.gif", 25, 35)
mp4_to_gif("plots/dataset1_cam2_tracking_output_optical_flow.mp4", "documentation/imgs/tracking_optical_flow.gif", 25, 35)
mp4_to_gif("plots/dataset1_cam2_yolo_tracking_output.mp4", "documentation/imgs/tracking_yolo.gif", 5, 15)