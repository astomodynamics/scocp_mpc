#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Declare launch arguments
    robot_id_arg = DeclareLaunchArgument(
        'robot_id',
        default_value='robot_1',
        description='Robot ID for topic namespacing'
    )
    
    init_x_arg = DeclareLaunchArgument(
        'init_x',
        default_value='0.0',
        description='Initial x position of the robot'
    )
    
    init_y_arg = DeclareLaunchArgument(
        'init_y',
        default_value='0.0',
        description='Initial y position of the robot'
    )
    
    init_yaw_arg = DeclareLaunchArgument(
        'init_yaw',
        default_value='0.0',
        description='Initial yaw angle of the robot (radians)'
    )
    
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to launch RViz'
    )
    
    # Get package directory
    pkg_share = FindPackageShare('scocp_mpc')
    
    # Path to RViz config file
    rviz_config_path = PathJoinSubstitution([
        pkg_share,
        'rviz',
        'unicycle_robot.rviz'
    ])
    
    # Unicycle robot node
    unicycle_robot_node = Node(
        package='scocp_mpc',
        executable='unicycle_robot_node',
        name='unicycle_robot_node',
        parameters=[{
            'robot_id': LaunchConfiguration('robot_id'),
            'init_x': LaunchConfiguration('init_x'),
            'init_y': LaunchConfiguration('init_y'),
            'init_yaw': LaunchConfiguration('init_yaw')
        }],
        output='screen'
    )

    # Hardcoded map node
    hardcoded_map_node = Node(
        package='scocp_mpc',
        executable='hardcoded_map_node',
        name='hardcoded_map_node',
        output='screen'
    )
    
    # Teleop keyboard node
    teleop_keyboard_node = Node(
        package='scocp_mpc',
        executable='teleop_keyboard',
        name='teleop_keyboard',
        parameters=[{
            'robot_id': LaunchConfiguration('robot_id')
        }],
        output='screen',
        prefix='gnome-terminal -- '  # Launch in separate terminal for keyboard input
    )
    
    # RViz node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
        output='screen'
    )
    
    return LaunchDescription([
        robot_id_arg,
        init_x_arg,
        init_y_arg,
        init_yaw_arg,
        use_rviz_arg,
        unicycle_robot_node,
        hardcoded_map_node,
        teleop_keyboard_node,
        rviz_node
    ]) 