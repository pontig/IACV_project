from typing import List
import numpy as np

class Camera_info:

    comment: List[str]
    K_matrix: np.ndarray
    distCoeff: np.ndarray
    fps: float
    resolution: List[int]
    
    # Constructor
    def __init__(self, comment: List[str], K_matrix: np.ndarray, distCoeff: np.ndarray, fps: float, resolution: List[int]):
        self.comment = comment
        self.K_matrix = K_matrix
        self.distCoeff = distCoeff
        self.fps = fps
        self.resolution = resolution
        
    def __str__(self):
        return f'''Camera_info: 
        comment: {self.comment}
        K_matrix: {self.K_matrix}
        distCoeff: {self.distCoeff}
        fps: {self.fps}
        resolution: {self.resolution}'''
    
    