#!/usr/bin/env python
"""
| File: 3_ros2_single_vehicle.py
| Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: This files serves as an example on how to build an app that makes use of the Pegasus API to run a 
simulation with a single vehicle, controlled using the ROS2 backend system. NOTE: this ROS2 interface only works on Ubuntu 20.04LTS
for now. Check the website https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_ros.html#enabling-the-ros-ros-2-bridge-extension
and follow the steps 1, 2 and 3 to make sure that the ROS2 example runs properly
"""

# Imports to start Isaac Sim from this script
import carb
from omni.isaac.kit import SimulationApp

# Start Isaac Sim's simulation environment
# Note: this simulation app must be instantiated right after the SimulationApp import, otherwise the simulator will crash
# as this is the object that will load all the extensions and load the actual simulator.
simulation_app = SimulationApp({"headless": False})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import omni.timeline
from omni.isaac.core.world import World

import omni.isaac.core.utils.prims as prim_utils

# Import the Pegasus API for simulating drones
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS, UNREAL_ENVIRONMENTS
from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
from pegasus.simulator.logic.vehicles.moving_camera import MovingCamera
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.sensors import RGBDCamera

# Auxiliary scipy and numpy modules
from scipy.spatial.transform import Rotation

class PegasusApp:
    """
    A Template class that serves as an example on how to build a simple Isaac Sim standalone App.
    """

    def __init__(self):
        """
        Method that initializes the PegasusApp and is used to setup the simulation environment.
        """

        # Acquire the timeline that will be used to start/stop the simulation
        self.timeline = omni.timeline.get_timeline_interface()

        # Start the Pegasus Interface
        self.pg = PegasusInterface()

        # Acquire the World, .i.e, the singleton that controls that is a one stop shop for setting up physics, 
        # spawning asset primitives, etc.
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # Launch an Unreal Environment World
        self.pg.load_environment(UNREAL_ENVIRONMENTS['Forest'])


        # Add a custom light
        prim_utils.create_prim(
            "/World/Light/DomeLight",
            "DomeLight",
            attributes={
                "intensity": 6500.0
        })

        # self.camera_factory(0, position=[-2.0, -2.0, 2.0], rotation=[90,0,0])
        self.moving_camera_factory(0, position=[-2.0, -2.0, 2.0], rotation=[90,0,0])
        self.world.reset()

        # Auxiliar variable for the timeline callback example
        self.stop_sim = False

    # def moving_camera_factory(self, vehicle_id: int, gap_x_axis: float, position: tuple, rotation: tuple = [0, 0, 0], num_cameras: int = 1):
    #     """Auxiliar method to create multiple moving cameras

    #     Args:
    #         vehicle_id (int): the vehicle ID to define the unique vehicle instance
    #         gap_x_axis (float): the gap to set between vehicle spawns
    #     """

    #     # Create the vehicle
    #     # Try to spawn the selected robot in the world to the specified namespace
    #     config_moving_cam = MovingCameraConfig()
    #     for cam_id in range(num_cameras):
    #         config_moving_cam.sensors += [RGBDCamera(id=cam_id, app=simulation_app)]
    #     config_moving_cam.backends = [ROS2Backend(vehicle_id=vehicle_id)]

    #     if position is None:
    #         MovingCamera(
    #             "/World/moving_camera",
    #             ROBOTS['moving_camera'],
    #             vehicle_id,
    #             [gap_x_axis * vehicle_id, 0.0, 0.07],
    #             Rotation.from_euler("XYZ", rotation, degrees=True).as_quat(),
    #             config=config_moving_cam,
    #         )
    #     else:
    #         MovingCamera(
    #             "/World/moving_camera",
    #             ROBOTS['moving_camera'],
    #             vehicle_id,
    #             position,
    #             Rotation.from_euler("XYZ", rotation, degrees=True).as_quat(),
    #             config=config_moving_cam,
    #         )
    def moving_camera_factory(self, cam_id: int, position: tuple, rotation: tuple = [0,0,0]):
        camera_config = {"position": position,                                    # Meters
                         "orientation": Rotation.from_euler(
                                        "XYZ", rotation, degrees=True).as_quat(), # Quaternion [qx, qy, qz, qw]
                         "focal_length": 24.0,                                    # Pixels
                         "focal_distance": 400.0,                                 # Meters
                         "resolution": [3840, 2160],                              # Pixels
                         "update_rate": 30.0,  
                         "focal_length": 24.0,
                         "focal_distance": 400.0,
                         "clipping_range": [0.0001, 1000000.0],
                         "set_projection_type": "pinhole",
                         "horizonal_aperture": 20.9550,
                         "vertical_aperture": 15.2908,
                         "set_projection_type": "pinhole",                        # pinhole, fisheyeOrthographic, fisheyeEquidistant, fisheyeEquisolid, fisheyePolynomial, fisheyeSpherical
        }
        moving_cam = MovingCamera(id=cam_id, config=camera_config, app=simulation_app)
        moving_cam.spawn()

    def camera_factory(self, cam_id: int, position: tuple, rotation: tuple = [0, 0, 0]):
        camera_config = {"position": position,                                    # Meters
                         "orientation": Rotation.from_euler(
                                        "XYZ", rotation, degrees=True).as_quat(), # Quaternion [qx, qy, qz, qw]
                         "focal_length": 24.0,                                    # Pixels
                         "focal_distance": 400.0,                                 # Meters
                         "resolution": [3840, 2160],                              # Pixels
                         "update_rate": 30.0,  
                         "focal_length": 24.0,
                         "focal_distance": 400.0,
                         "clipping_range": [0.0001, 1000000.0],
                         "set_projection_type": "pinhole",
                         "horizonal_aperture": 20.9550,
                         "vertical_aperture": 15.2908,
                         "set_projection_type": "pinhole",                        # pinhole, fisheyeOrthographic, fisheyeEquidistant, fisheyeEquisolid, fisheyePolynomial, fisheyeSpherical
        }
        camera = RGBDCamera(id=cam_id, config=camera_config, app=simulation_app)
        camera.spawn()

    def run(self):
        """
        Method that implements the application main loop, where the physics steps are executed.
        """

        # Start the simulation
        self.timeline.play()

        # The "infinite" loop
        while simulation_app.is_running() and not self.stop_sim:

            # Update the UI of the app and perform the physics step
            self.world.step(render=True)
        
        # Cleanup and stop
        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()

def main():

    # Instantiate the template app
    pg_app = PegasusApp()

    # Run the application loop
    pg_app.run()

if __name__ == "__main__":
    main()
