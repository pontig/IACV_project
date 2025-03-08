def compute_global_time(frame_indices, alpha, beta = 0):
    """
    Convert frame indices to global timestamps.
    
    frame_indices: List or array of frame indices
    alpha: Time scale factor (frame rate correction)
    beta: Time offset (initial shift)
    
    Returns: Global timestamps
    """
    return frame_indices / alpha + beta