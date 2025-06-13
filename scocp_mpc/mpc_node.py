#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
import numpy as np
import cvxpy as cp
import sympy as sp
import math
import sys
import os
sys.path.append(os.path.join(".."))
import scocp


class UnicycleMPCNode(Node):
    """
    MPC node for unicycle robot control using SCOCP optimization.
    
    Subscribes to robot pose and goal, publishes velocity commands and predicted path.
    Includes obstacle avoidance and goal tracking capabilities.
    """
    
    def __init__(self):
        super().__init__('unicycle_mpc_node')
        
        # Declare and get parameters
        self.declare_parameter("robot_id", "robot_1")
        self.declare_parameter("horizon", 20)
        self.declare_parameter("dt", 0.1)
        self.declare_parameter("v_max", 1.0)
        self.declare_parameter("omega_max", 2.0)
        self.declare_parameter("trust_region_radius", 1.0)
        
        self.robot_id = self.get_parameter("robot_id").get_parameter_value().string_value
        self.horizon = self.get_parameter("horizon").get_parameter_value().integer_value
        self.dt = self.get_parameter("dt").get_parameter_value().double_value
        self.v_max = self.get_parameter("v_max").get_parameter_value().double_value
        self.omega_max = self.get_parameter("omega_max").get_parameter_value().double_value
        self.trust_region_radius = self.get_parameter("trust_region_radius").get_parameter_value().double_value
        
        # Current state and goal
        self.current_state = np.zeros(3)
        self.goal_state = np.array([2.0, 4.0, math.pi/2])
        
        # Flags
        self.is_current_pose_received = False
        self.is_goal_pose_received = True  # Default goal is set
        
        # Obstacle parameters (from hardcoded map)
        self.obstacle_centers = [(0.762, 2.54), (2.794, 3.429), (0.762, 4.318)]
        self.obstacle_radius = 0.41 * math.sqrt(2) / 2 + 0.1  # Safety margin
        
        # Setup unicycle dynamics and SCOCP problem
        self.setup_dynamics()
        self.setup_scocp()
        
        # ROS2 interfaces
        self.state_sub = self.create_subscription(
            PoseStamped,
            f'/{self.robot_id}/pose',
            self.current_state_callback,
            10
        )
        
        self.goal_sub = self.create_subscription(
            PoseStamped,
            f'/goal_pose',
            self.goal_pose_callback,
            10
        )
        
        self.control_pub = self.create_publisher(
            Twist,
            f'/{self.robot_id}/cmd_vel',
            10
        )
        
        self.local_path_pub = self.create_publisher(
            Path,
            f'/{self.robot_id}/local_path',
            10
        )
        
        # Control timer (matches simulation frequency)
        self.control_timer = self.create_timer(self.dt, self.control_callback)
        
        self.get_logger().info(f'Unicycle MPC Node initialized for robot: {self.robot_id}')
        self.get_logger().info(f'Horizon: {self.horizon}, dt: {self.dt}s, v_max: {self.v_max}m/s')
    
    def setup_dynamics(self):
        """Setup unicycle dynamics using sympy for SCOCP."""
        # Define symbolic variables
        n_x, n_u = 3, 2
        _x = sp.IndexedBase('x')
        _u = sp.IndexedBase('u')
        x = sp.Matrix([_x[i] for i in range(n_x)])
        u = sp.Matrix([_u[i] for i in range(n_u)])
        
        # Unicycle dynamics: [x_dot, y_dot, theta_dot] = [v*cos(theta), v*sin(theta), omega]
        f_sym = sp.Matrix([
            u[0] * sp.cos(x[2]),  # x_dot = v * cos(theta)
            u[0] * sp.sin(x[2]),  # y_dot = v * sin(theta)
            u[1]                  # theta_dot = omega
        ])
        
        # Compute jacobians
        jac_x_sym = f_sym.jacobian(x)
        jac_u_sym = f_sym.jacobian(u)
        
        # Create numerical functions
        self.eval_dynamics = sp.lambdify([x, u], f_sym, modules=[{'atan2': np.arctan2}, 'numpy'])
        self.eval_dfdx = sp.lambdify([x, u], jac_x_sym, modules=[{'atan2': np.arctan2}, 'numpy'])
        self.eval_dfdu = sp.lambdify([x, u], jac_u_sym, modules=[{'atan2': np.arctan2}, 'numpy'])
        
        # Define RHS functions for SCOCP
        self.scipy_rhs_eom = lambda t, x, u: self.eval_dynamics(x, u).flatten()
        
        def scipy_rhs_eom_augmented(t, x, u):
            assert len(x) == n_x + n_x*n_x + n_x*n_u, f"x must be of length {n_x + n_x*n_x + n_x*n_u}, but got {len(x)}"
            dxdt_aug = np.zeros(n_x + n_x*n_x + n_x*n_u)
            dxdt_aug[:n_x] = self.scipy_rhs_eom(t, x[:n_x], u)
            A = self.eval_dfdx(x[:n_x], u)
            B = self.eval_dfdu(x[:n_x], u)
            dxdt_aug[n_x:n_x*(n_x+1)] = (A @ x[n_x:n_x*(n_x+1)].reshape(n_x, n_x)).flatten()
            dxdt_aug[n_x*(n_x+1):] = (np.dot(A, x[n_x*(n_x+1):].reshape(n_x, n_u)) + B).flatten()
            return dxdt_aug
        
        self.scipy_rhs_eom_augmented = scipy_rhs_eom_augmented
        
        # Build integrator object
        self.integrator = scocp.ScipyIntegrator(
            nx=3,
            nu=2,
            rhs=self.scipy_rhs_eom,
            rhs_stm=self.scipy_rhs_eom_augmented,
            impulsive=False,
            args=([0.0, 0.0],),  # Placeholder for 2 control inputs [v, omega]
            method='RK45',
            reltol=1e-8,
            abstol=1e-8
        )
    
    def setup_scocp(self):
        """Setup SCOCP problem formulation."""
        # Create time grid
        self.times = np.linspace(0, self.horizon * self.dt, self.horizon + 1)
        
        # Create SCOCP problem
        self.problem = UnicycleSCOCPProblem(
            self.integrator, 
            self.times, 
            self.trust_region_radius,
            self.goal_state,
            self.obstacle_centers,
            self.obstacle_radius,
            self.v_max,
            self.omega_max
        )
        
        # Create algorithm instance
        self.algo = scocp.SCvxStar(self.problem, tol_opt=1e-4, tol_feas=1e-6)
    
    def current_state_callback(self, msg):
        """Callback for current robot pose."""
        x = msg.pose.position.x
        y = msg.pose.position.y
        
        # Convert quaternion to yaw
        quat = msg.pose.orientation
        yaw = math.atan2(
            2.0 * (quat.w * quat.z + quat.x * quat.y),
            1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        )
        
        self.current_state = np.array([x, y, yaw])
        self.is_current_pose_received = True
    
    def goal_pose_callback(self, msg):
        """Callback for goal pose."""
        x = msg.pose.position.x
        y = msg.pose.position.y
        
        # Convert quaternion to yaw
        quat = msg.pose.orientation
        yaw = math.atan2(
            2.0 * (quat.w * quat.z + quat.x * quat.y),
            1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        )
        
        self.goal_state = np.array([x, y, yaw])
        self.problem.update_goal(self.goal_state)  # Update goal in SCOCP problem
        self.is_goal_pose_received = True
        
        self.get_logger().info(f'Updated goal: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}')
    
    def control_callback(self):
        """Main control loop - solve SCOCP and publish commands."""
        if not self.is_current_pose_received or not self.is_goal_pose_received:
            return
        
        # Check if goal is reached
        goal_error = np.linalg.norm(self.current_state[:2] - self.goal_state[:2])
        if goal_error < 0.1:
            # Publish zero control
            control_msg = Twist()
            self.control_pub.publish(control_msg)
            return
        
        try:
            # Create initial guess trajectory
            xs_guess = np.zeros((self.horizon + 1, 3))
            us_guess = np.zeros((self.horizon, 2))
            
            # Linear interpolation for state guess
            for i in range(3):
                xs_guess[:, i] = np.linspace(self.current_state[i], self.goal_state[i], self.horizon + 1)
            
            # Set current state as initial condition
            xs_guess[0, :] = self.current_state
            
            # Add some small control inputs to avoid singular initial guess
            us_guess[:, 0] = 0.1  # Small forward velocity
            us_guess[:, 1] = 0.01  # Small angular velocity
            
            # Update problem with current state
            self.problem.update_initial_state(self.current_state)
            
            # Debug: Check if states are reasonable
            if np.any(np.isnan(xs_guess)) or np.any(np.isinf(xs_guess)):
                self.get_logger().warn('Initial state guess contains NaN or inf values')
                return
            
            if np.any(np.isnan(us_guess)) or np.any(np.isinf(us_guess)):
                self.get_logger().warn('Initial control guess contains NaN or inf values')
                return
            
            # Solve SCOCP (single iteration for real-time performance)
            solution = self.algo.solve(
                xs_guess,
                us_guess,
                maxiter=3,  # Limited iterations for real-time
                verbose=False
            )
            
            # Check if solution is valid and has required attributes
            if (solution is not None and 
                hasattr(solution, 'status') and 
                hasattr(solution, 'u') and 
                hasattr(solution, 'x') and
                solution.u is not None and 
                solution.x is not None and
                solution.status == 'solved'):
                
                # Extract first control input
                u_opt = solution.u[0, :]
                
                # Publish control command
                control_msg = Twist()
                control_msg.linear.x = float(u_opt[0])
                control_msg.angular.z = float(u_opt[1])
                self.control_pub.publish(control_msg)
                
                # Publish predicted path
                self.publish_predicted_path(solution.x)
                
            else:
                # Publish zero control if optimization failed
                control_msg = Twist()
                self.control_pub.publish(control_msg)
                
                # Log more detailed error information
                if solution is None:
                    self.get_logger().warn('SCOCP optimization returned None solution')
                elif not hasattr(solution, 'status'):
                    self.get_logger().warn('SCOCP solution missing status attribute')
                elif solution.u is None or solution.x is None:
                    self.get_logger().warn('SCOCP solution has None values for u or x')
                else:
                    status = getattr(solution, 'status', 'unknown')
                    self.get_logger().warn(f'SCOCP optimization failed with status: {status}')
                
        except Exception as e:
            self.get_logger().warn(f'SCOCP optimization error: {str(e)}')
            # Publish zero control as fallback
            control_msg = Twist()
            self.control_pub.publish(control_msg)
    
    def publish_predicted_path(self, X_opt):
        """Publish the predicted trajectory as a path."""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"
        
        for i in range(min(self.horizon + 1, X_opt.shape[0])):
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            pose_stamped.pose.position.x = float(X_opt[i, 0])
            pose_stamped.pose.position.y = float(X_opt[i, 1])
            pose_stamped.pose.position.z = 0.0
            
            # Convert yaw to quaternion
            theta = float(X_opt[i, 2])
            pose_stamped.pose.orientation.x = 0.0
            pose_stamped.pose.orientation.y = 0.0
            pose_stamped.pose.orientation.z = math.sin(theta / 2.0)
            pose_stamped.pose.orientation.w = math.cos(theta / 2.0)
            
            path_msg.poses.append(pose_stamped)
        
        self.local_path_pub.publish(path_msg)


class UnicycleSCOCPProblem(scocp.ContinuousControlSCOCP):
    """SCOCP problem formulation for unicycle robot with obstacle avoidance."""
    
    def __init__(self, integrator, times, trust_region_radius, goal_state, 
                 obstacle_centers, obstacle_radius, v_max, omega_max):
        super().__init__(integrator, times, trust_region_radius=trust_region_radius)
        self.goal_state = goal_state.copy()
        self.initial_state = np.zeros(3)
        self.obstacle_centers = obstacle_centers
        self.obstacle_radius = obstacle_radius
        self.v_max = v_max
        self.omega_max = omega_max
        
        # Cost weights
        self.Q = np.diag([10.0, 10.0, 1.0])  # State cost
        self.R = np.diag([0.1, 0.1])         # Control cost
        self.Qf = np.diag([50.0, 50.0, 5.0]) # Terminal cost
    
    def update_goal(self, new_goal):
        """Update the goal state."""
        self.goal_state = new_goal.copy()
    
    def update_initial_state(self, new_initial):
        """Update the initial state."""
        self.initial_state = new_initial.copy()
    
    def evaluate_objective(self, xs, us, vs=None, ys=None):
        """Quadratic objective with state tracking and control effort."""
        N, nx = xs.shape
        _, nu = us.shape
        
        cost = 0.0
        
        # Stage costs
        for k in range(N-1):
            state_error = xs[k, :] - self.goal_state
            cost += 0.5 * state_error.T @ self.Q @ state_error
            cost += 0.5 * us[k, :].T @ self.R @ us[k, :]
        
        # Terminal cost
        terminal_error = xs[-1, :] - self.goal_state
        cost += 0.5 * terminal_error.T @ self.Qf @ terminal_error
        
        return cost
    
    def solve_convex_problem(self, xbar, ubar, vbar=None, ybar=None):
        """Solve the convex subproblem."""
        N, nx = xbar.shape
        _, nu = ubar.shape
        Nseg = N - 1
        
        # Decision variables
        xs = cp.Variable((N, nx), name='state')
        us = cp.Variable((Nseg, nu), name='control')
        xis_dyn = cp.Variable((Nseg, nx), name='xi_dyn')  # Dynamics slack
        
        # Objective with augmented Lagrangian penalty
        penalty = scocp.get_augmented_lagrangian_penalty(self.weight, xis_dyn, self.lmb_dynamics)
        
        # Quadratic cost
        cost = 0
        for k in range(Nseg):
            state_error = xs[k, :] - self.goal_state
            cost += 0.5 * cp.quad_form(state_error, self.Q)
            cost += 0.5 * cp.quad_form(us[k, :], self.R)
        
        # Terminal cost
        terminal_error = xs[-1, :] - self.goal_state
        cost += 0.5 * cp.quad_form(terminal_error, self.Qf)
        
        objective_func = cost + penalty
        
        # Constraints
        constraints = []
        
        # Dynamics constraints
        for i in range(Nseg):
            constraints.append(
                xs[i+1, :] == self.Phi_A[i, :, :] @ xs[i, :] + 
                self.Phi_B[i, :, :] @ us[i, :] + self.Phi_c[i, :] + xis_dyn[i, :]
            )
        
        # Trust region constraints
        for i in range(N):
            constraints.append(xs[i, :] - xbar[i, :] <= self.trust_region_radius)
            constraints.append(xs[i, :] - xbar[i, :] >= -self.trust_region_radius)
        
        # Initial condition
        constraints.append(xs[0, :] == self.initial_state)
        
        # Control bounds
        for i in range(Nseg):
            constraints.append(us[i, 0] <= self.v_max)      # v_max
            constraints.append(us[i, 0] >= -self.v_max)     # v_min
            constraints.append(us[i, 1] <= self.omega_max)  # omega_max
            constraints.append(us[i, 1] >= -self.omega_max) # omega_min
        
        # Obstacle avoidance constraints (linearized)
        for i in range(N):
            for obs_x, obs_y in self.obstacle_centers:
                # Linearize obstacle constraint around reference trajectory
                x_ref = xbar[i, 0]
                y_ref = xbar[i, 1]
                dist_ref = (x_ref - obs_x)**2 + (y_ref - obs_y)**2
                
                if dist_ref > self.obstacle_radius**2:
                    # Linear approximation of constraint
                    grad_x = 2 * (x_ref - obs_x)
                    grad_y = 2 * (y_ref - obs_y)
                    linear_approx = (grad_x * (xs[i, 0] - x_ref) + 
                                   grad_y * (xs[i, 1] - y_ref) + dist_ref)
                    constraints.append(linear_approx >= self.obstacle_radius**2)
        
        # Solve problem
        convex_problem = cp.Problem(cp.Minimize(objective_func), constraints)
        convex_problem.solve(solver=self.solver, verbose=self.verbose_solver)
        self.cp_status = convex_problem.status
        
        return xs.value, us.value, None, None, xis_dyn.value, None, None


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = UnicycleMPCNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
# 