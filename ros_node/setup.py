from setuptools import find_packages, setup

package_name = 'drones'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='elia',
    maintainer_email='elia.pontiggia@mail.polimi.it',
    description='TODO: Package description',
    license='TODO: License declaration',
    # package_dir={'': 'drones'},
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = drones.publisher_member_function:main',
            'listener = drones.subscriber_member_function:main',
            'drones = drones.drones:main',
            'live_detections_generator = drones.live_detections_generator:main',
        ],
    },  

)
