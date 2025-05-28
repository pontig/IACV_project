import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from scipy.interpolate import make_interp_spline
import numpy as np
import matplotlib.pyplot as plt

class MainNode(Node):
    def __init__(self):
        super().__init__('detections_subscriber')
        
        self.compiled_cameras = 0
        self.possible_values = range(0, 7)

        # Store raw data points for each camera
        self.data_lists = {value: [] for value in self.possible_values}
        
        # Store ALL splines for each camera: list of (spline_x, spline_y, time_points, data_indices)
        self.splines = {value: [] for value in self.possible_values}
        
        # Configuration
        self.min_points_for_spline = 4
        self.max_time_gap = 0.1  # seconds for continuous updates
        self.max_spline_time_span = 0.2  # max time span for creating new splines

        self.detection_sub = self.create_subscription(
            Float32MultiArray,
            '/detections',
            self.listener_callback,
            10
        )
        
    def pltall(self):
        # Plot all splines at the end, one in each figure
        for camera_id, splines in self.splines.items():
            if not splines:
                continue
            
            plt.figure()
            plt.title(f'Camera {camera_id} Splines')
            plt.xlabel('Time (s)')
            plt.ylabel('Position')
            
            for spline_x, spline_y, time_points, _ in splines:
                t = np.linspace(min(time_points), max(time_points), 100)
                x = spline_x(t)
                y = spline_y(t)
                plt.plot(x, -y) # Invert y-axis for image coordinates
                            
            plt.legend()
            plt.grid()
        plt.show()

    def listener_callback(self, msg):
        if len(msg.data) < 4:  # Need at least [time, camera_id, x, y]
            return
            
        element_1 = msg.data[1]
        if element_1 not in self.data_lists:
            self.get_logger().warn(f'Value {element_1} not in possible values')
            return
        
        # if msg.data[0] > 10000:  
        #     self.pltall()
        #     return

        # Create new data point
        new_point = (msg.data[0]/59.940060, msg.data[1], msg.data[2], msg.data[3])
        
        # First detection from this camera
        if len(self.data_lists[element_1]) == 0:
            self.compiled_cameras += 1
            self.get_logger().info(f'New camera detected: {element_1}')
        
        self.data_lists[element_1].append(new_point)
        self._process_new_point(element_1, new_point)

    def _process_new_point(self, camera_id, new_point):
        """Process new point: either update existing spline or create new one"""
        data_points = self.data_lists[camera_id]
        current_time = new_point[0]
        
        # Need minimum points for any spline operations
        if len(data_points) < self.min_points_for_spline:
            return
        
        # Check if we can update the most recent spline
        if self.splines[camera_id]:
            last_spline_info = self.splines[camera_id][-1]
            last_spline_time = last_spline_info[2][-1]  # Last time point of most recent spline
            
            # If new point is close enough in time, update the existing spline
            if current_time - last_spline_time < self.max_time_gap:
                self._update_current_spline(camera_id, data_points)
                return
        
        # Otherwise, try to create a new spline
        self._try_create_new_spline(camera_id, data_points)

    def _update_current_spline(self, camera_id, data_points):
        """Update the most recent spline with all points used by it plus the new point"""
        if not self.splines[camera_id]:
            return
            
        # Get the most recent spline info
        spline_x, spline_y, time_points, data_indices = self.splines[camera_id][-1]
        
        # Add the new point index
        new_data_index = len(data_points) - 1
        updated_indices = data_indices + [new_data_index]
        
        # Get all points for this spline (existing + new)
        spline_points = [data_points[i] for i in updated_indices]
        
        try:
            # Extract coordinates
            t = [p[0] for p in spline_points]
            x = [p[2] for p in spline_points]
            y = [p[3] for p in spline_points]
            
            # Ensure we have unique time points
            if len(set(t)) >= self.min_points_for_spline:
                # Create updated splines
                k = min(3, len(t) - 1)  # Spline degree, max 3 or n-1
                new_spline_x = make_interp_spline(t, x, k=k)
                new_spline_y = make_interp_spline(t, y, k=k)
                
                # Update the most recent spline
                self.splines[camera_id][-1] = (new_spline_x, new_spline_y, t, updated_indices)
                
        except Exception as e:
            self.get_logger().warn(f'Failed to update spline for camera {camera_id}: {e}')
            # If update fails, try to create a new spline instead
            self._try_create_new_spline(camera_id, data_points)

    def _try_create_new_spline(self, camera_id, data_points):
        """Try to create a new spline using recent points"""
        # Use recent points for new spline
        recent_count = min(self.min_points_for_spline, len(data_points))
        recent_points = data_points[-recent_count:]
        recent_indices = list(range(len(data_points) - recent_count, len(data_points)))
        
        # Check if points are close enough in time to form a coherent spline
        time_span = recent_points[-1][0] - recent_points[0][0]
        if time_span > self.max_spline_time_span:
            # Try with just the last few points that are closer in time
            for i in range(len(recent_points) - self.min_points_for_spline + 1):
                subset_points = recent_points[i:]
                subset_indices = recent_indices[i:]
                subset_time_span = subset_points[-1][0] - subset_points[0][0]
                
                if subset_time_span <= self.max_spline_time_span and len(subset_points) >= self.min_points_for_spline:
                    recent_points = subset_points
                    recent_indices = subset_indices
                    break
            else:
                # No suitable subset found
                return
        
        try:
            # Extract coordinates
            t = [p[0] for p in recent_points]
            x = [p[2] for p in recent_points]
            y = [p[3] for p in recent_points]
            
            # Ensure we have enough unique time points
            if len(set(t)) >= self.min_points_for_spline:
                k = min(3, len(t) - 1)  # Spline degree
                spline_x = make_interp_spline(t, x, k=k)
                spline_y = make_interp_spline(t, y, k=k)
                
                # Add new spline to the list
                self.splines[camera_id].append((spline_x, spline_y, t, recent_indices))
                
        except Exception as e:
            self.get_logger().warn(f'Failed to create new spline for camera {camera_id}: {e}')

    def get_interpolated_position(self, camera_id, time):
        """Get interpolated position from the most appropriate spline"""
        if not self.splines[camera_id]:
            return None
            
        # Find the spline that contains this time or is closest to it
        best_spline = None
        best_distance = float('inf')
        
        for spline_x, spline_y, time_points, _ in self.splines[camera_id]:
            min_time, max_time = min(time_points), max(time_points)
            
            if min_time <= time <= max_time:
                # Time is within spline range
                best_spline = (spline_x, spline_y)
                break
            else:
                # Calculate distance to closest endpoint
                distance = min(abs(time - min_time), abs(time - max_time))
                if distance < best_distance:
                    best_distance = distance
                    best_spline = (spline_x, spline_y)
        
        if best_spline is None:
            return None
            
        try:
            spline_x, spline_y = best_spline
            x = float(spline_x(time))
            y = float(spline_y(time))
            return (x, y)
        except (ValueError, IndexError):
            return None

    def get_splines_info(self, camera_id):
        """Get information about all splines for a camera"""
        if camera_id not in self.splines:
            return []
            
        info = []
        for i, (spline_x, spline_y, time_points, data_indices) in enumerate(self.splines[camera_id]):
            info.append({
                'spline_id': i,
                'time_range': (min(time_points), max(time_points)),
                'num_points': len(time_points),
                'data_indices': data_indices
            })
        return info


def main(args=None):
    rclpy.init(args=args)
    node = MainNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()