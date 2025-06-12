#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose


class GridMapNode(Node):
    """
    ROS2 node that publishes a hardcoded occupancy grid map with obstacles.
    
    The node creates a rectangular map with configurable dimensions and resolution,
    and places obstacles at predefined locations.
    """
    
    def __init__(self):
        super().__init__('grid_map_node')
        
        # Declare parameters with default values
        self.declare_parameter('map_length', 6.0)
        self.declare_parameter('map_width', 3.0)
        self.declare_parameter('obstacle_size', 0.41)
        self.declare_parameter('resolution', 0.1)
        
        # Get parameter values
        self.map_length_ = self.get_parameter('map_length').get_parameter_value().double_value
        self.map_width_ = self.get_parameter('map_width').get_parameter_value().double_value
        self.obstacle_size_ = self.get_parameter('obstacle_size').get_parameter_value().double_value
        self.resolution_ = self.get_parameter('resolution').get_parameter_value().double_value
        
        # Create the publisher for occupancy grid
        self.grid_pub_ = self.create_publisher(OccupancyGrid, '/occupancy_map', 10)
        
        # Create a timer that calls publish_grid_map() every 33 milliseconds (~30Hz)
        self.timer_ = self.create_timer(0.033, self.publish_grid_map)
        
        self.get_logger().info(f'Grid Map Node started with map size: {self.map_length_}x{self.map_width_}m, resolution: {self.resolution_}m')
    
    def publish_grid_map(self) -> None:
        """
        Create and publish the occupancy grid map with hardcoded obstacles.
        """
        # Compute the number of cells in the grid
        width_cells = int(self.map_width_ / self.resolution_)
        height_cells = int(self.map_length_ / self.resolution_)
        
        # Prepare the grid message
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = 'map'
        grid_msg.info.resolution = self.resolution_
        grid_msg.info.width = width_cells
        grid_msg.info.height = height_cells
        
        # Set map origin
        grid_msg.info.origin = Pose()
        grid_msg.info.origin.position.x = 0.0
        grid_msg.info.origin.position.y = 0.0
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0
        
        # Initialize the grid data with free space (0)
        grid_msg.data = [0] * (width_cells * height_cells)
        
        # Add obstacles to the grid at hardcoded positions
        self.add_obstacle(grid_msg, 0.762, 2.54)
        self.add_obstacle(grid_msg, 2.794, 3.429)
        self.add_obstacle(grid_msg, 0.762, 4.318)
        
        # Publish the grid map
        self.grid_pub_.publish(grid_msg)
    
    def add_obstacle(self, grid_msg: OccupancyGrid, x: float, y: float) -> None:
        """
        Add a square obstacle to the occupancy grid at the specified position.
        
        Args:
            grid_msg: The occupancy grid message to modify
            x: X position of the obstacle center in meters
            y: Y position of the obstacle center in meters
        """
        width_cells = grid_msg.info.width
        height_cells = grid_msg.info.height
        obs_cells = int(self.obstacle_size_ / self.resolution_)
        obs_x = int(x / self.resolution_)
        obs_y = int(y / self.resolution_)
        
        # Mark cells corresponding to the obstacle with a value of 100 (occupied)
        for i in range(-obs_cells // 2, obs_cells // 2 + 1):
            for j in range(-obs_cells // 2, obs_cells // 2 + 1):
                idx_x = obs_x + i
                idx_y = obs_y + j
                
                # Check bounds before setting the cell value
                if 0 <= idx_x < width_cells and 0 <= idx_y < height_cells:
                    # Convert 2D coordinates to 1D array index
                    array_index = idx_y * width_cells + idx_x
                    grid_msg.data[array_index] = 100  # 100 = occupied


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = GridMapNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
