#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
import math
from .unicycle import Unicycle


class UnicycleRobotNode(Node):
    """
    ROS2 node that simulates a unicycle robot using the Unicycle kinematics model.
    
    Subscribes to cmd_vel commands and publishes the robot's pose.
    """
    
    def __init__(self):
        super().__init__('unicycle_robot_node')
        
        # Declare and retrieve the robot_id parameter (default "robot_1")
        self.declare_parameter('robot_id', 'robot_1')
        self.robot_id = self.get_parameter('robot_id').get_parameter_value().string_value
        
        # Declare and retrieve initial pose parameters (default to 0.0)
        self.declare_parameter('init_x', 0.0)
        self.declare_parameter('init_y', 0.0)
        self.declare_parameter('init_yaw', 0.0)
        init_x = self.get_parameter('init_x').get_parameter_value().double_value
        init_y = self.get_parameter('init_y').get_parameter_value().double_value
        init_yaw = self.get_parameter('init_yaw').get_parameter_value().double_value
        
        # Create the unicycle instance with the initial pose
        self.unicycle = Unicycle(init_x, init_y, init_yaw)
        
        # Construct topic names using robot_id
        cmd_vel_topic = f"{self.robot_id}/cmd_vel"
        pose_topic = f"{self.robot_id}/pose"
        
        # Subscribe to velocity commands on the constructed topic
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            cmd_vel_topic,
            self.cmd_vel_callback,
            10
        )
        
        # Publisher for robot state as PoseStamped on the constructed topic
        self.pose_pub = self.create_publisher(PoseStamped, pose_topic, 10)
        
        # Timer to update simulation at fixed intervals (10ms = 100Hz)
        self.timer = self.create_timer(0.01, self.update_simulation)
        
        self.last_time = self.get_clock().now()
        
        self.get_logger().info(f"Unicycle robot node started for robot: {self.robot_id}")
        self.get_logger().info(f"Initial pose: x={init_x:.2f}, y={init_y:.2f}, yaw={init_yaw:.2f}")
    
    def cmd_vel_callback(self, msg: Twist) -> None:
        """
        Callback to update the unicycle commands from incoming Twist messages.
        
        Args:
            msg: Twist message containing linear and angular velocity commands
        """
        self.unicycle.set_command(msg.linear.x, msg.angular.z)
    
    def update_simulation(self) -> None:
        """
        Update simulation state and publish the robot state as a PoseStamped message.
        """
        # Compute elapsed time
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9  # Convert to seconds
        self.last_time = current_time
        
        # Propagate the unicycle state
        self.unicycle.update(dt)
        state = self.unicycle.get_state()  # [x, y, theta]
        
        # Prepare and publish the PoseStamped message
        pose_msg = PoseStamped()
        pose_msg.header.stamp = current_time.to_msg()
        pose_msg.header.frame_id = "map"
        
        pose_msg.pose.position.x = state[0]
        pose_msg.pose.position.y = state[1]
        pose_msg.pose.position.z = 0.0
        
        # Convert yaw (theta) to a quaternion (rotation around Z-axis)
        half_yaw = state[2] * 0.5
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = math.sin(half_yaw)
        pose_msg.pose.orientation.w = math.cos(half_yaw)
        
        self.pose_pub.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = UnicycleRobotNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
