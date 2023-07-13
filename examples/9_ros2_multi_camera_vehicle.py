# !/usr/bin/env python
"""
| File: 9_multi_camera_vehicle.py
| Author: Micah Nye (micahn@andrew.cmu.edu)
| License: BSD-3-Clause. Copyright (c) 2023, Micah Nye. All rights reserved.
| Description: This files serves as an example on how to build an app that makes use of the Pegasus API to run a 
simulation with a multiple vehicle, controlled using the ROS2 backend system. These vehicles are capable of being equipped
with multiple cameras, provided the models have the cameras. NOTE: this ROS2 interface only works on Ubuntu 20.04LTS
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

# Extra omni tools
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.debug_draw import _debug_draw


# Import the Pegasus API for simulating drones
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.sensors import RGBDCamera

# Auxiliary  modules
from scipy.spatial.transform import Rotation
import numpy as np
import time as tm

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

        # Add a custom light
        prim_utils.create_prim(
            "/World/Light/DomeLight",
            "DomeLight",
            attributes={
                "intensity": 1000.0
        })

        # Launch one of the worlds provided by NVIDIA
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Warehouse with Forklifts"])

        for i in range(2):
            self.vehicle_factory(i, gap_x_axis=1.0, num_cameras=2)

        # Reset the simulation environment so that all articulations (aka robots) are initialized
        self.world.reset()

        # Auxiliar variable for the timeline callback example
        self.stop_sim = False

    def vehicle_factory(self, vehicle_id: int, gap_x_axis: float, num_cameras: int = 0):
        """Auxiliar method to create multiple multirotor vehicles

        Args:
            vehicle_id (int): the vehicle ID to define the unique vehicle instance
            gap_x_axis (float): the gap to set between vehicle spawns
        """

        # Create the vehicle
        # Try to spawn the selected robot in the world to the specified namespace
        config_multirotor = MultirotorConfig()
        for cam_id in range(num_cameras):
            config_multirotor.sensors += [RGBDCamera(id=cam_id, app=simulation_app)]
        config_multirotor.backends = [ROS2Backend(vehicle_id=vehicle_id)]

        Multirotor(
            "/World/quadrotor",
            ROBOTS['Iris'],
            vehicle_id,
            [gap_x_axis * vehicle_id, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )

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