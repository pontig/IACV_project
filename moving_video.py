import cv2 as cv
import numpy as np
import json
from collections import deque
import matplotlib.pyplot as plt
import time
import sys
import argparse


class DroneTracker:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.trajectory_path = output_path.replace('.mp4', '_trajectory.json')
        self.plot_path = output_path.replace('.mp4', '_trajectory.png')
        self.first_frame = None

        # Drone detection parameters
        self.min_area = 10
        self.max_area = 500
        self.aspect_ratio_range = (0.5, 2.0)
        self.search_y_limit = 0

        # Tracking parameters
        self.max_disappeared = 30
        self.max_distance = 100

        # Storage
        self.trajectory = []
        self.drone_positions = deque(maxlen=50)
        self.frame_count = 0
        self.drone_disappeared_count = 0

        # Background subtractor
        self.bg_subtractor = cv.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True)

    def create_search_mask(self):
        mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        mask[0:self.search_y_limit, :] = 255
        return mask

    def detect_drone_candidates(self, frame, fg_mask):
        self.frame_height, self.frame_width = frame.shape[:2]
        search_mask = self.create_search_mask()
        fg_mask = cv.bitwise_and(fg_mask, search_mask)
        contours, _ = cv.findContours(fg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        candidates = []
        for contour in contours:
            area = cv.contourArea(contour)
            if self.min_area <= area <= self.max_area:
                x, y, w, h = cv.boundingRect(contour)
                aspect_ratio = w / h
                if y < self.search_y_limit:
                    if self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]:
                        cx = x + w // 2
                        cy = y + h // 2
                        candidates.append({
                            'center': (cx, cy),
                            'bbox': (x, y, w, h),
                            'area': area,
                            'contour': contour
                        })
        return candidates

    def find_best_drone_candidate(self, candidates):
        if not candidates:
            return None
        if self.drone_positions:
            last_pos = self.drone_positions[-1]
            min_distance = float('inf')
            best_candidate = None
            for candidate in candidates:
                distance = np.hypot(candidate['center'][0] - last_pos[0], candidate['center'][1] - last_pos[1])
                if distance < min_distance and distance < self.max_distance:
                    min_distance = distance
                    best_candidate = candidate
            if best_candidate:
                return best_candidate
        return max(candidates, key=lambda x: x['area'])

    def smooth_position(self, new_position):
        if len(self.drone_positions) < 3:
            return new_position
        recent_positions = list(self.drone_positions)[-5:]
        avg_x = sum(pos[0] for pos in recent_positions) / len(recent_positions)
        avg_y = sum(pos[1] for pos in recent_positions) / len(recent_positions)
        smooth_x = int(0.7 * new_position[0] + 0.3 * avg_x)
        smooth_y = int(0.7 * new_position[1] + 0.3 * avg_y)
        return (smooth_x, smooth_y)

    def update_tracking_data(self, frame_num, current_time, best_candidate):
        if best_candidate:
            smooth_pos = self.smooth_position(best_candidate['center'])
            self.drone_positions.append(smooth_pos)
            self.trajectory.append({
                'frame': frame_num,
                'time': current_time,
                'x': smooth_pos[0],
                'y': smooth_pos[1],
                'area': best_candidate['area'],
                'status': 'detected'
            })
            self.drone_disappeared_count = 0
            return smooth_pos, best_candidate
        else:
            self.drone_disappeared_count += 1
            if self.drone_disappeared_count <= self.max_disappeared:
                self.trajectory.append({
                    'frame': frame_num,
                    'time': current_time,
                    'x': None,
                    'y': None,
                    'area': None,
                    'status': 'lost'
                })
            return None, None

    def draw_frame_annotations(self, frame, smooth_pos, best_candidate, frame_num, total_frames, current_time):
        self.frame_height = frame.shape[0]
        cv.line(frame, (0, self.search_y_limit), (frame.shape[1], self.search_y_limit), (255, 255, 0), 2)
        cv.putText(frame, "Search Area", (10, self.search_y_limit - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        if best_candidate and smooth_pos:
            x, y, w, h = best_candidate['bbox']
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.circle(frame, smooth_pos, 5, (0, 0, 255), -1)
            cv.putText(frame, "Drone", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif self.drone_disappeared_count <= self.max_disappeared:
            cv.putText(frame, "DRONE LOST", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if len(self.drone_positions) > 1:
            for i in range(1, len(self.drone_positions)):
                cv.line(frame, self.drone_positions[i-1], self.drone_positions[i], (255, 0, 0), 2)
        cv.putText(frame, f"Frame: {frame_num}/{total_frames}", (10, self.frame_height - 50), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv.putText(frame, f"Time: {current_time:.2f}s", (10, self.frame_height - 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame

    def process_video(self):
        cap = cv.VideoCapture(self.input_path)
        if not cap.isOpened():
            print("Error: Cannot open video file.")
            return
        self.frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv.CAP_PROP_FPS)
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(self.output_path, fourcc, fps, (self.frame_width, self.frame_height))

        print(f"Processing {total_frames} frames at {fps} FPS...")
        print(f"Search area limited to Y < {self.search_y_limit} pixels")

        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret: # or self.frame_count > 900:
                break
            self.frame_count += 1
            current_time = self.frame_count / fps
            
            if self.first_frame is None:
                self.first_frame = frame.copy()

            fg_mask = self.bg_subtractor.apply(frame)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_OPEN, kernel)
            fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_CLOSE, kernel)

            candidates = self.detect_drone_candidates(frame, fg_mask)
            best_candidate = self.find_best_drone_candidate(candidates)
            smooth_pos, best_candidate = self.update_tracking_data(self.frame_count, current_time, best_candidate)
            annotated_frame = self.draw_frame_annotations(frame, smooth_pos, best_candidate,
                                                          self.frame_count, total_frames, current_time)
            out.write(annotated_frame)

            if self.frame_count % 100 == 0:
                elapsed_time = time.time() - start_time
                progress = (self.frame_count / total_frames) * 100
                estimated_total = elapsed_time / (self.frame_count / total_frames)
                remaining_time = estimated_total - elapsed_time
                print(f"Progress: {progress:.1f}% - ETA: {remaining_time:.1f}s")

        cap.release()
        out.release()

        processing_time = time.time() - start_time
        self.save_trajectory()
        self.plot_trajectory()

        print(f"Processing complete!")
        print(f"Total processing time: {processing_time:.2f} seconds")
        print(f"Average FPS: {total_frames/processing_time:.2f}")
        print(f"Output video: {self.output_path}")
        print(f"Trajectory data: {self.trajectory_path}")
        print(f"Trajectory plot: {self.plot_path}")

    def save_trajectory(self):
        detected_frames = len([p for p in self.trajectory if p['status'] == 'detected'])
        lost_frames = len([p for p in self.trajectory if p['status'] == 'lost'])
        trajectory_data = {
            'metadata': {
                'total_frames': self.frame_count,
                'detected_frames': detected_frames,
                'lost_frames': lost_frames,
                'search_y_limit': self.search_y_limit
            },
            'trajectory': self.trajectory
        }
        with open(self.trajectory_path, 'w') as f:
            json.dump(trajectory_data, f, indent=2)

    def plot_trajectory(self):
        detected_points = [p for p in self.trajectory if p['status'] == 'detected']
        if not detected_points:
            print("No trajectory points to plot.")
            return
        x_coords = [p['x'] for p in detected_points]
        y_coords = [p['y'] for p in detected_points]
        times = [p['time'] for p in detected_points]

        plt.figure(figsize=(19, 10))
        plt.subplot(2, 2, 1)
        # Show the first frame as background
        if self.first_frame is not None:
            plt.imshow(cv.cvtColor(self.first_frame, cv.COLOR_BGR2RGB))
        # Plot trajectory on top
        plt.plot(x_coords, y_coords, color='lime', linestyle='-')
        plt.scatter(x_coords[0], y_coords[0], color='green', s=100, label='Start')
        plt.scatter(x_coords[-1], y_coords[-1], color='red', s=100, label='End')
        plt.axhline(y=self.search_y_limit, color='aqua', linestyle='--', label='Search Limit')
        plt.gca().invert_yaxis()
        plt.xlim(0, self.frame_width)
        plt.ylim(self.frame_height, 0)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Drone Trajectory')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(times, x_coords, 'r-')
        plt.xlabel('Time (s)')
        plt.ylabel('X Position')
        plt.title('X Position Over Time')

        plt.subplot(2, 2, 4)
        plt.plot(times, y_coords, 'g-')
        plt.axhline(y=self.search_y_limit, color='aqua', linestyle='--')
        plt.gca().invert_yaxis()
        plt.xlabel('Time (s)')
        plt.ylabel('Y Position')
        plt.title('Y Position Over Time')

        plt.subplot(2, 2, 2)
        speeds = [np.hypot(x_coords[i] - x_coords[i-1], y_coords[i] - y_coords[i-1]) /
                  (times[i] - times[i-1]) for i in range(1, len(times)) if times[i] > times[i-1]]
        if speeds:
            plt.plot(times[1:len(speeds)+1], speeds, 'm-')
            plt.xlabel('Time (s)')
            plt.ylabel('Speed (px/s)')
            plt.title('Drone Speed Over Time')

        plt.tight_layout()
        plt.savefig(self.plot_path)
        plt.close()
        
    def process_video_with_optical_flow(self):
        """
        Enhanced drone tracking using optical flow combined with background subtraction.
        Handles drone entering/exiting the frame while maintaining precision.
        """
        cap = cv.VideoCapture(self.input_path)
        if not cap.isOpened():
            print("Error: Cannot open video file.")
            return
        
        self.frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv.CAP_PROP_FPS)
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(self.output_path, fourcc, fps, (self.frame_width, self.frame_height))

        print(f"Processing {total_frames} frames with optical flow at {fps} FPS...")
        print(f"Search area limited to Y < {self.search_y_limit} pixels")

        # Optical flow parameters
        lk_params = dict(winSize=(15, 15),
                        maxLevel=2,
                        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # Feature detection parameters
        feature_params = dict(maxCorners=100,
                            qualityLevel=0.01,
                            minDistance=7,
                            blockSize=7)
        
        # Tracking state variables
        prev_gray = None
        tracking_points = None
        drone_bbox = None
        optical_flow_active = False
        frames_without_detection = 0
        confidence_threshold = 0.7
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            self.frame_count += 1
            current_time = self.frame_count / fps
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            
            if self.first_frame is None:
                self.first_frame = frame.copy()
            
            # Background subtraction for initial detection and validation
            fg_mask = self.bg_subtractor.apply(frame)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_OPEN, kernel)
            fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_CLOSE, kernel)
            
            candidates = self.detect_drone_candidates(frame, fg_mask)
            bg_best_candidate = self.find_best_drone_candidate(candidates)
            
            smooth_pos = None
            best_candidate = None
            tracking_method = "none"
            
            # Optical flow tracking logic
            if optical_flow_active and prev_gray is not None and tracking_points is not None:
                # Calculate optical flow
                new_points, status, error = cv.calcOpticalFlowPyrLK(
                    prev_gray, gray, tracking_points, None, **lk_params)
                
                # Select good points based on status and error
                good_new = new_points[status == 1]
                good_old = tracking_points[status == 1]
                good_error = error[status == 1]
                
                # Filter points by error threshold
                error_mask = good_error.flatten() < 50
                good_new = good_new[error_mask]
                good_old = good_old[error_mask]
                
                if len(good_new) >= 3:  # Minimum points for reliable tracking
                    # Calculate movement vector
                    movement = good_new - good_old
                    median_movement = np.median(movement, axis=0)
                    
                    # Update drone position based on optical flow
                    if drone_bbox is not None:
                        x, y, w, h = drone_bbox
                        new_x = int(x + median_movement[0])
                        new_y = int(y + median_movement[1])
                        
                        # Ensure bbox stays within frame bounds
                        new_x = max(0, min(new_x, self.frame_width - w))
                        new_y = max(0, min(new_y, self.frame_height - h))
                        
                        drone_bbox = (new_x, new_y, w, h)
                        cx = new_x + w // 2
                        cy = new_y + h // 2
                        
                        # Check if drone is still in search area
                        if cy < self.search_y_limit:
                            smooth_pos = self.smooth_position((cx, cy))
                            best_candidate = {
                                'center': smooth_pos,
                                'bbox': drone_bbox,
                                'area': w * h,
                                'contour': None
                            }
                            tracking_method = "optical_flow"
                            frames_without_detection = 0
                            
                            # Update tracking points around new position
                            roi_mask = np.zeros_like(gray)
                            roi_mask[new_y:new_y+h, new_x:new_x+w] = 255
                            new_features = cv.goodFeaturesToTrack(gray, mask=roi_mask, **feature_params)
                            if new_features is not None:
                                tracking_points = new_features
                        else:
                            # Drone moved out of search area
                            optical_flow_active = False
                            tracking_points = None
                            drone_bbox = None
                            frames_without_detection += 1
                    else:
                        optical_flow_active = False
                else:
                    frames_without_detection += 1
                    if frames_without_detection > 5:  # Lost tracking
                        optical_flow_active = False
                        tracking_points = None
                        drone_bbox = None
            
            # Fall back to background subtraction or start new tracking
            if not optical_flow_active or frames_without_detection > 3:
                if bg_best_candidate:
                    # Validate with optical flow confidence if we were tracking
                    confidence = 1.0
                    if optical_flow_active and smooth_pos is not None:
                        # Calculate distance between optical flow and background subtraction results
                        distance = np.hypot(bg_best_candidate['center'][0] - smooth_pos[0],
                                        bg_best_candidate['center'][1] - smooth_pos[1])
                        confidence = max(0, 1.0 - distance / 100.0)
                    
                    # Use background subtraction result
                    if confidence > confidence_threshold or not optical_flow_active:
                        smooth_pos = self.smooth_position(bg_best_candidate['center'])
                        best_candidate = bg_best_candidate
                        tracking_method = "background_subtraction"
                        
                        # Initialize/reinitialize optical flow tracking
                        x, y, w, h = bg_best_candidate['bbox']
                        drone_bbox = (x, y, w, h)
                        
                        # Extract features for optical flow
                        roi_mask = np.zeros_like(gray)
                        roi_mask[y:y+h, x:x+w] = 255
                        tracking_points = cv.goodFeaturesToTrack(gray, mask=roi_mask, **feature_params)
                        
                        if tracking_points is not None and len(tracking_points) >= 3:
                            optical_flow_active = True
                            frames_without_detection = 0
                        else:
                            optical_flow_active = False
                    else:
                        # Trust optical flow over background subtraction
                        tracking_method = "optical_flow_validated"
                else:
                    # No detection from either method
                    frames_without_detection += 1
                    if frames_without_detection > self.max_disappeared:
                        optical_flow_active = False
                        tracking_points = None
                        drone_bbox = None
            
            # Update tracking data
            final_pos, final_candidate = self.update_tracking_data(
                self.frame_count, current_time, best_candidate)
            
            # Enhanced frame annotations
            annotated_frame = self.draw_enhanced_frame_annotations(
                frame, final_pos, final_candidate, self.frame_count, total_frames, 
                current_time, tracking_method, optical_flow_active, tracking_points)
            
            out.write(annotated_frame)
            
            # Progress reporting
            if self.frame_count % 100 == 0:
                elapsed_time = time.time() - start_time
                progress = (self.frame_count / total_frames) * 100
                estimated_total = elapsed_time / (self.frame_count / total_frames)
                remaining_time = estimated_total - elapsed_time
                print(f"Progress: {progress:.1f}% - ETA: {remaining_time:.1f}s - Method: {tracking_method}")
            
            prev_gray = gray.copy()
        
        cap.release()
        out.release()
        
        processing_time = time.time() - start_time
        self.save_trajectory()
        self.plot_trajectory()
        
        print(f"Optical flow processing complete!")
        print(f"Total processing time: {processing_time:.2f} seconds")
        print(f"Average FPS: {total_frames/processing_time:.2f}")
        print(f"Output video: {self.output_path}")
        print(f"Trajectory data: {self.trajectory_path}")
        print(f"Trajectory plot: {self.plot_path}")

    def draw_enhanced_frame_annotations(self, frame, smooth_pos, best_candidate, frame_num, 
                                    total_frames, current_time, tracking_method, 
                                    optical_flow_active, tracking_points):
        """
        Enhanced frame annotations showing optical flow tracking status.
        """
        self.frame_height = frame.shape[0]
        
        # Draw search area limit
        cv.line(frame, (0, self.search_y_limit), (frame.shape[1], self.search_y_limit), (255, 255, 0), 2)
        cv.putText(frame, "Search Area", (10, self.search_y_limit - 10), 
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Draw drone detection and tracking
        if best_candidate and smooth_pos:
            x, y, w, h = best_candidate['bbox']
            
            # Color code based on tracking method
            if tracking_method == "optical_flow":
                color = (0, 255, 255)  # Yellow for optical flow
                method_text = "OF"
            elif tracking_method == "optical_flow_validated":
                color = (0, 200, 255)  # Orange for validated optical flow
                method_text = "OF-V"
            else:
                color = (0, 255, 0)    # Green for background subtraction
                method_text = "BG"
            
            cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv.circle(frame, smooth_pos, 5, (0, 0, 255), -1)
            cv.putText(frame, f"Drone ({method_text})", (x, y - 10), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw optical flow points if active
            if optical_flow_active and tracking_points is not None:
                for point in tracking_points:
                    cv.circle(frame, tuple(point[0].astype(int)), 2, (255, 0, 255), -1)
        
        elif self.drone_disappeared_count <= self.max_disappeared:
            cv.putText(frame, "DRONE LOST", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw trajectory
        if len(self.drone_positions) > 1:
            for i in range(1, len(self.drone_positions)):
                cv.line(frame, self.drone_positions[i-1], self.drone_positions[i], (255, 0, 0), 2)
        
        # Status information
        cv.putText(frame, f"Frame: {frame_num}/{total_frames}", 
                (10, self.frame_height - 70), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv.putText(frame, f"Time: {current_time:.2f}s", 
                (10, self.frame_height - 50), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv.putText(frame, f"Tracking: {tracking_method}", 
                (10, self.frame_height - 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv.putText(frame, f"OF Active: {'Yes' if optical_flow_active else 'No'}", 
                (10, self.frame_height - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Drone tracking with background subtraction and optical flow.")
    parser.add_argument("dataset", type=int, help="Dataset number")
    parser.add_argument("camera", type=int, help="Camera number")
    parser.add_argument("--y_limit", type=int, default=1080, help="Y limit for search area (default: 1080)")
    args = parser.parse_args()

    input_path = f'drone-tracking-datasets/dataset{args.dataset}/cam{args.camera}/cam{args.camera}.mp4'
    output_path = f'plots/dataset{args.dataset}_cam{args.camera}_tracking_output.mp4'
    
    print(f"Got dataset: {args.dataset}, camera: {args.camera}")

    tracker = DroneTracker(input_path, output_path)
    tracker.search_y_limit = args.y_limit
    tracker.process_video()

    tracker_of = DroneTracker(input_path, output_path.replace('.mp4', '_optical_flow.mp4'))
    tracker_of.search_y_limit = args.y_limit
    tracker_of.process_video_with_optical_flow()

    print("Drone tracking completed successfully.")