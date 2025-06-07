import cv2 as cv
from ultralytics import YOLO
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from collections import deque


class YOLODroneTracker:
    def __init__(self, input_path, output_path, model_path="yolov8n.pt"):
        self.input_path = input_path
        self.output_path = output_path
        self.trajectory_path = output_path.replace('.mp4', '_trajectory.json')
        self.plot_path = output_path.replace('.mp4', '_trajectory.png')
        self.first_frame = None
        
        # Load YOLO model
        self.model = YOLO(model_path)
        
        # Tracking parameters
        self.max_disappeared = 30
        self.max_distance = 100
        self.min_confidence = 0.3
        
        # Storage
        self.trajectory = []
        self.drone_positions = deque(maxlen=50)
        self.frame_count = 0
        self.drone_disappeared_count = 0
        
        # Video properties
        self.frame_width = 0
        self.frame_height = 0
        
    def filter_drone_detections(self, results):
        """Filter detections to find the most likely drone."""
        drone_candidates = []
        
        for r in results:
            if len(r.boxes) == 0:
                continue
                
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.model.names[cls_id].lower()
                
                # Filter by confidence and class (adjust class names as needed)
                if conf >= self.min_confidence:
                    # Accept various object classes that might represent drones
                    # You can customize this list based on your specific use case
                    drone_classes = ['person', 'bird', 'airplane', 'kite', 'sports ball', 'frisbee']
                    if any(drone_class in class_name for drone_class in drone_classes) or conf > 0.7:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        area = (x2 - x1) * (y2 - y1)
                        
                        drone_candidates.append({
                            'center': (cx, cy),
                            'bbox': (x1, y1, x2 - x1, y2 - y1),
                            'area': area,
                            'confidence': conf,
                            'class_name': class_name,
                            'class_id': cls_id
                        })
        
        return drone_candidates
    
    def find_best_drone_candidate(self, candidates):
        """Find the best drone candidate based on tracking history and confidence."""
        if not candidates:
            return None
            
        if self.drone_positions:
            # Prioritize candidates close to last known position
            last_pos = self.drone_positions[-1]
            scored_candidates = []
            
            for candidate in candidates:
                distance = np.hypot(candidate['center'][0] - last_pos[0], 
                                  candidate['center'][1] - last_pos[1])
                
                if distance < self.max_distance:
                    # Score based on distance and confidence
                    distance_score = max(0, 1.0 - distance / self.max_distance)
                    confidence_score = candidate['confidence']
                    combined_score = 0.6 * distance_score + 0.4 * confidence_score
                    
                    scored_candidates.append((combined_score, candidate))
            
            if scored_candidates:
                # Return candidate with highest combined score
                return max(scored_candidates, key=lambda x: x[0])[1]
        
        # If no tracking history, return highest confidence candidate
        return max(candidates, key=lambda x: x['confidence'])
    
    def smooth_position(self, new_position):
        """Smooth position using moving average."""
        if len(self.drone_positions) < 3:
            return new_position
            
        recent_positions = list(self.drone_positions)[-5:]
        avg_x = sum(pos[0] for pos in recent_positions) / len(recent_positions)
        avg_y = sum(pos[1] for pos in recent_positions) / len(recent_positions)
        
        # Apply smoothing
        smooth_x = int(0.7 * new_position[0] + 0.3 * avg_x)
        smooth_y = int(0.7 * new_position[1] + 0.3 * avg_y)
        
        return (smooth_x, smooth_y)
    
    def update_tracking_data(self, frame_num, current_time, best_candidate):
        """Update trajectory and tracking data."""
        if best_candidate:
            smooth_pos = self.smooth_position(best_candidate['center'])
            self.drone_positions.append(smooth_pos)
            self.trajectory.append({
                'frame': frame_num,
                'time': current_time,
                'x': smooth_pos[0],
                'y': smooth_pos[1],
                'area': best_candidate['area'],
                'confidence': best_candidate['confidence'],
                'class_name': best_candidate['class_name'],
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
                    'confidence': None,
                    'class_name': None,
                    'status': 'lost'
                })
            return None, None
    
    def draw_frame_annotations(self, frame, smooth_pos, best_candidate, frame_num, total_frames, current_time):
        """Draw annotations on the frame."""
        if best_candidate and smooth_pos:
            x, y, w, h = best_candidate['bbox']
            conf = best_candidate['confidence']
            class_name = best_candidate['class_name']
            
            # Draw bounding box
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.circle(frame, smooth_pos, 5, (0, 0, 255), -1)
            
            # Draw label with confidence
            label = f"{class_name} {conf:.2f}"
            cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        elif self.drone_disappeared_count <= self.max_disappeared:
            cv.putText(frame, "DRONE LOST", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw trajectory
        if len(self.drone_positions) > 1:
            for i in range(1, len(self.drone_positions)):
                cv.line(frame, self.drone_positions[i-1], self.drone_positions[i], (255, 0, 0), 2)
        
        # Draw frame info
        cv.putText(frame, f"Frame: {frame_num}/{total_frames}", 
                  (10, self.frame_height - 50), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv.putText(frame, f"Time: {current_time:.2f}s", 
                  (10, self.frame_height - 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv.putText(frame, "Method: YOLO", 
                  (10, self.frame_height - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def process_video(self, save_video=True, save_frames=False):
        """Process video with YOLO detection and tracking."""
        cap = cv.VideoCapture(self.input_path)
        if not cap.isOpened():
            print("Error: Cannot open video file.")
            return
        
        self.frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv.CAP_PROP_FPS)
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if saving video
        out = None
        if save_video:
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            out = cv.VideoWriter(self.output_path, fourcc, fps, (self.frame_width, self.frame_height))
        
        # Setup frame output directory if saving frames
        if save_frames:
            frames_dir = self.output_path.replace('.mp4', '_frames')
            os.makedirs(frames_dir, exist_ok=True)
        
        print(f"Processing {total_frames} frames with YOLO at {fps} FPS...")
        print(f"Save video: {save_video}, Save frames: {save_frames}")
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            current_time = self.frame_count / fps
            
            if self.first_frame is None:
                self.first_frame = frame.copy()
            
            # Run YOLO detection
            results = self.model(frame, verbose=False)
            
            # Filter and find best drone candidate
            candidates = self.filter_drone_detections(results)
            best_candidate = self.find_best_drone_candidate(candidates)
            
            # Update tracking data
            smooth_pos, final_candidate = self.update_tracking_data(
                self.frame_count, current_time, best_candidate)
            
            # Draw annotations
            annotated_frame = self.draw_frame_annotations(
                frame, smooth_pos, final_candidate, self.frame_count, total_frames, current_time)
            
            # Save video frame if enabled
            if save_video and out is not None:
                out.write(annotated_frame)
            
            # Save individual frame if enabled
            if save_frames:
                frame_path = os.path.join(frames_dir, f"frame_{self.frame_count:05d}.jpg")
                cv.imwrite(frame_path, annotated_frame)
            
            # Progress reporting
            if self.frame_count % 100 == 0:
                elapsed_time = time.time() - start_time
                progress = (self.frame_count / total_frames) * 100
                estimated_total = elapsed_time / (self.frame_count / total_frames)
                remaining_time = estimated_total - elapsed_time
                print(f"Progress: {progress:.1f}% - ETA: {remaining_time:.1f}s")
        
        cap.release()
        if out is not None:
            out.release()
        
        processing_time = time.time() - start_time
        self.save_trajectory()
        self.plot_trajectory()
        
        print(f"YOLO processing complete!")
        print(f"Total processing time: {processing_time:.2f} seconds")
        print(f"Average FPS: {total_frames/processing_time:.2f}")
        if save_video:
            print(f"Output video: {self.output_path}")
        if save_frames:
            print(f"Output frames: {frames_dir}")
        print(f"Trajectory data: {self.trajectory_path}")
        print(f"Trajectory plot: {self.plot_path}")
    
    def save_trajectory(self):
        """Save trajectory data to JSON file."""
        detected_frames = len([p for p in self.trajectory if p['status'] == 'detected'])
        lost_frames = len([p for p in self.trajectory if p['status'] == 'lost'])
        
        # Calculate detection statistics
        confidences = [p['confidence'] for p in self.trajectory if p['confidence'] is not None]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        trajectory_data = {
            'metadata': {
                'total_frames': self.frame_count,
                'detected_frames': detected_frames,
                'lost_frames': lost_frames,
                'detection_rate': detected_frames / self.frame_count if self.frame_count > 0 else 0,
                'average_confidence': avg_confidence,
                'tracking_method': 'YOLO',
                'model_used': str(self.model.model_name) if hasattr(self.model, 'model_name') else 'yolov8n'
            },
            'trajectory': self.trajectory
        }
        
        with open(self.trajectory_path, 'w') as f:
            json.dump(trajectory_data, f, indent=2)
    
    def plot_trajectory(self):
        """Create comprehensive trajectory plots."""
        detected_points = [p for p in self.trajectory if p['status'] == 'detected']
        if not detected_points:
            print("No trajectory points to plot.")
            return
        
        x_coords = [p['x'] for p in detected_points]
        y_coords = [p['y'] for p in detected_points]
        times = [p['time'] for p in detected_points]
        confidences = [p['confidence'] for p in detected_points]
        
        plt.figure(figsize=(19, 10))
        
        # Subplot 1: Trajectory overlay on first frame
        plt.subplot(2, 3, 1)
        if self.first_frame is not None:
            plt.imshow(cv.cvtColor(self.first_frame, cv.COLOR_BGR2RGB))
        
        # Plot trajectory with confidence-based coloring
        scatter = plt.scatter(x_coords, y_coords, c=confidences, cmap='viridis', 
                            s=20, alpha=0.7, label='Trajectory')
        plt.plot(x_coords, y_coords, color='lime', linestyle='-', alpha=0.5)
        plt.scatter(x_coords[0], y_coords[0], color='green', s=100, marker='s', label='Start')
        plt.scatter(x_coords[-1], y_coords[-1], color='red', s=100, marker='s', label='End')
        
        # plt.colorbar(scatter, label='Confidence')
        plt.gca().invert_yaxis()
        plt.xlim(0, self.frame_width)
        plt.ylim(self.frame_height, 0)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('YOLO Drone Trajectory')
        plt.legend()
        
        # Subplot 2: X position over time
        plt.subplot(2, 3, 2)
        plt.plot(times, x_coords, 'r-', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('X Position')
        plt.title('X Position Over Time')
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Y position over time
        plt.subplot(2, 3, 3)
        plt.plot(times, y_coords, 'g-', linewidth=2)
        plt.gca().invert_yaxis()
        plt.xlabel('Time (s)')
        plt.ylabel('Y Position')
        plt.title('Y Position Over Time')
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Speed over time
        plt.subplot(2, 3, 4)
        speeds = []
        for i in range(1, len(times)):
            if times[i] > times[i-1]:
                dx = x_coords[i] - x_coords[i-1]
                dy = y_coords[i] - y_coords[i-1]
                dt = times[i] - times[i-1]
                speed = np.hypot(dx, dy) / dt
                speeds.append(speed)
        
        if speeds:
            plt.plot(times[1:len(speeds)+1], speeds, 'm-', linewidth=2)
            plt.xlabel('Time (s)')
            plt.ylabel('Speed (px/s)')
            plt.title('Drone Speed Over Time')
            plt.grid(True, alpha=0.3)
        
        # Subplot 5: Confidence over time
        plt.subplot(2, 3, 5)
        plt.plot(times, confidences, 'b-', linewidth=2)
        plt.axhline(y=self.min_confidence, color='r', linestyle='--', alpha=0.7, label='Min Confidence')
        plt.xlabel('Time (s)')
        plt.ylabel('Detection Confidence')
        plt.title('YOLO Confidence Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 6: Detection statistics
        plt.subplot(2, 3, 6)
        detection_stats = {
            'Detected': len([p for p in self.trajectory if p['status'] == 'detected']),
            'Lost': len([p for p in self.trajectory if p['status'] == 'lost'])
        }
        
        colors = ['green', 'red']
        plt.pie(detection_stats.values(), labels=detection_stats.keys(), 
               colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Detection Statistics')
        
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO-based drone tracking with comprehensive analysis.")
    parser.add_argument("dataset", type=int, help="Dataset number")
    parser.add_argument("camera", type=int, help="Camera number")
    parser.add_argument("--model", type=str, default="yolov8n.pt", 
                       help="YOLO model path (default: yolov8n.pt)")
    parser.add_argument("--min_confidence", type=float, default=0.3,
                       help="Minimum detection confidence (default: 0.3)")
    parser.add_argument("--save_video", action="store_true", default=True,
                       help="Save annotated video output")
    parser.add_argument("--no_video", action="store_true",
                       help="Don't save annotated video output")
    parser.add_argument("--save_frames", action="store_true",
                       help="Save individual annotated frames")
    
    args = parser.parse_args()
    
    input_path = f'drone-tracking-datasets/dataset{args.dataset}/cam{args.camera}/cam{args.camera}.mp4'
    output_path = f'plots/dataset{args.dataset}_cam{args.camera}_yolo_tracking_output.mp4'
    
    print(f"Processing dataset: {args.dataset}, camera: {args.camera}")
    print(f"Using model: {args.model}")
    print(f"Minimum confidence: {args.min_confidence}")
    
    # Handle video saving flag
    save_video = args.save_video and not args.no_video
    
    tracker = YOLODroneTracker(input_path, output_path, args.model)
    tracker.min_confidence = args.min_confidence
    tracker.process_video(save_video=save_video, save_frames=args.save_frames)
    
    print("YOLO drone tracking completed successfully.")