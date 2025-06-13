from setuptools import find_packages, setup

package_name = 'scocp_mpc'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/unicycle_simulation.launch.py']),
        ('share/' + package_name + '/rviz', ['rviz/simple_launch.rviz', 'rviz/unicycle_robot.rviz']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='astomodynamics',
    maintainer_email='tomo.sasaki.hiro@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'teleop_keyboard = scocp_mpc.teleop_keyboard:main',
            'unicycle_robot_node = scocp_mpc.unicycle_robot_node:main',
            'hardcoded_map_node = scocp_mpc.hardcoded_map_node:main',
            'mpc_node = scocp_mpc.mpc_node:main',
        ],
    },
)
