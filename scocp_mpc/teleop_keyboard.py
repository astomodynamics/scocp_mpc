#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import threading
import sys
import termios
import tty
import select


class TeleopKeyboard(Node):
    def __init__(self):
        super().__init__('teleop_keyboard')
        
        # Declare and retrieve the robot_id parameter (default "robot_1")
        self.declare_parameter('robot_id', 'robot_1')
        self.robot_id = self.get_parameter('robot_id').get_parameter_value().string_value
        
        # Construct the cmd_vel topic based on robot_id
        cmd_vel_topic = f"{self.robot_id}/cmd_vel"
        self.publisher_ = self.create_publisher(Twist, cmd_vel_topic, 10)
        
        # Initialize velocity variables
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.linear_step = 0.1
        self.angular_step = 0.1
        
        # Thread control
        self.exit_requested = False
        
        # Start the keyboard reading thread
        self.keyboard_thread = threading.Thread(target=self.keyboard_loop)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()
        
    def keyboard_loop(self):
        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)
        
        try:
            # Set terminal to raw mode for immediate key press reading
            tty.setraw(sys.stdin.fileno())
            
            print(f"\nTeleop Keyboard Control for robot: {self.robot_id}")
            print("-----------------------------------------")
            print("w: increase forward speed")
            print("s: decrease forward speed (or move backwards)")
            print("a: turn left")
            print("d: turn right")
            print("x: stop")
            print("q: quit")
            print()
            
            while rclpy.ok() and not self.exit_requested:
                # Check if there's input available
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    
                    twist_msg = Twist()
                    
                    if key == 'w':
                        self.linear_vel += self.linear_step
                    elif key == 's':
                        self.linear_vel -= self.linear_step
                    elif key == 'a':
                        self.angular_vel += self.angular_step
                    elif key == 'd':
                        self.angular_vel -= self.angular_step
                    elif key == 'x':
                        self.linear_vel = 0.0
                        self.angular_vel = 0.0
                    elif key == 'q':
                        self.exit_requested = True
                        break
                    else:
                        continue
                    
                    twist_msg.linear.x = self.linear_vel
                    twist_msg.angular.z = self.angular_vel
                    self.publisher_.publish(twist_msg)
                    
                    print(f"Published to {self.robot_id}/cmd_vel: linear = {twist_msg.linear.x:.2f}, angular = {twist_msg.angular.z:.2f}")
        
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    def destroy_node(self):
        self.exit_requested = True
        if self.keyboard_thread.is_alive():
            self.keyboard_thread.join(timeout=1.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = TeleopKeyboard()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
