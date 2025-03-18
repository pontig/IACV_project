import numpy as np

alphas = np.array([2.0069,	1.0000,	1.0044,	1.0033,	1.0034,	1.6741,	0.8370])
betas = np.array([-2320.60,	0.00,	-32.64,	59.77,	-270.82,	1082.34,	-2529.59])

def compute_global_time(frame_indices, alpha, beta = 0):
    """
    Convert frame indices to global timestamps.
    
    frame_indices: List or array of frame indices
    alpha: Time scale factor (frame rate correction)
    beta: Time offset (initial shift)
    
    Returns: Global timestamps
    """
    return frame_indices * alpha + beta