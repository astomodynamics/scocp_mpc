#!/usr/bin/env python3

import math
from typing import List


class Unicycle:
    """
    A simple unicycle kinematics model.
    
    The model accepts linear and angular velocity commands
    and uses a Runge–Kutta 4 (RK4) integration method to update its state.
    """
    
    def __init__(self, init_x: float = 0.0, init_y: float = 0.0, init_theta: float = 0.0):
        """
        Initialize the unicycle model.
        
        Args:
            init_x: Initial x position (default: 0.0)
            init_y: Initial y position (default: 0.0)
            init_theta: Initial heading angle in radians (default: 0.0)
        """
        # Current state (x, y, heading)
        self.x_ = init_x
        self.y_ = init_y
        self.theta_ = init_theta
        
        # Commanded inputs
        self.linear_cmd_ = 0.0
        self.angular_cmd_ = 0.0
    
    def set_command(self, linear_vel: float, angular_vel: float) -> None:
        """
        Set the commanded linear (m/s) and angular (rad/s) velocities.
        
        Args:
            linear_vel: Linear velocity command in m/s
            angular_vel: Angular velocity command in rad/s
        """
        self.linear_cmd_ = linear_vel
        self.angular_cmd_ = angular_vel
    
    def update(self, dt: float) -> None:
        """
        Propagate the state forward by dt seconds.
        
        Args:
            dt: Time step in seconds
        """
        self._integrate_rk4(dt)
    
    def get_state(self) -> List[float]:
        """
        Return the current state as a list: [x, y, theta].
        
        Returns:
            List containing [x_position, y_position, heading_angle]
        """
        return [self.x_, self.y_, self.theta_]
    
    def _compute_derivatives(self, x: float, y: float, theta: float) -> tuple:
        """
        Compute derivatives given state and commands.
        
        Args:
            x: Current x position
            y: Current y position  
            theta: Current heading angle
            
        Returns:
            Tuple of (dx/dt, dy/dt, dtheta/dt)
        """
        # Unicycle kinematics:
        # dx/dt = v * cos(theta)
        # dy/dt = v * sin(theta)
        # dtheta/dt = omega
        dx = self.linear_cmd_ * math.cos(theta)
        dy = self.linear_cmd_ * math.sin(theta)
        dtheta = self.angular_cmd_
        
        return dx, dy, dtheta
    
    def _integrate_rk4(self, dt: float) -> None:
        """
        Propagate state using RK4 integration.
        
        Args:
            dt: Time step in seconds
        """
        # k1: initial derivatives
        k1_x, k1_y, k1_theta = self._compute_derivatives(self.x_, self.y_, self.theta_)
        
        # k2: derivatives at midpoint using k1
        x_mid = self.x_ + 0.5 * dt * k1_x
        y_mid = self.y_ + 0.5 * dt * k1_y
        theta_mid = self.theta_ + 0.5 * dt * k1_theta
        k2_x, k2_y, k2_theta = self._compute_derivatives(x_mid, y_mid, theta_mid)
        
        # k3: derivatives at midpoint using k2
        x_mid = self.x_ + 0.5 * dt * k2_x
        y_mid = self.y_ + 0.5 * dt * k2_y
        theta_mid = self.theta_ + 0.5 * dt * k2_theta
        k3_x, k3_y, k3_theta = self._compute_derivatives(x_mid, y_mid, theta_mid)
        
        # k4: derivatives at end using k3
        x_end = self.x_ + dt * k3_x
        y_end = self.y_ + dt * k3_y
        theta_end = self.theta_ + dt * k3_theta
        k4_x, k4_y, k4_theta = self._compute_derivatives(x_end, y_end, theta_end)
        
        # Combine increments
        self.x_ += (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
        self.y_ += (dt / 6.0) * (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y)
        self.theta_ += (dt / 6.0) * (k1_theta + 2.0 * k2_theta + 2.0 * k3_theta + k4_theta)
