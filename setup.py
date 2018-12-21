
## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=['zero_shot_detector'],
    package_dir={'': 'scripts'},
    requires=['std_msgs', 'rospy', 'sensor_msgs', 'geometry_msgs']
)

setup(**setup_args)
