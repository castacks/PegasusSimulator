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

# Used for adding extra lights to the environment
import omni.isaac.core.utils.prims as prim_utils

from omni.isaac.debug_draw import _debug_draw

# Import the Pegasus API for simulating drones
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
from pegasus.simulator.logic.backends.mavlink_backend import MavlinkBackend, MavlinkBackendConfig
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

# Use os and pathlib for parsing the desired trajectory from a CSV file
import os
from pathlib import Path

# Auxiliary scipy and numpy modules
from scipy.spatial.transform import Rotation
import numpy as np

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

        # Add a custom light with a high-definition HDR surround environment of an exhibition hall,
        # instead of the typical ground plane
        prim_utils.create_prim(
            "/World/Light/DomeLight",
            "DomeLight",
            attributes={
                "intensity": 1000.0
        })
    
        # Get the current directory used to read trajectories and save results
        self.curr_dir = str(Path(os.path.dirname(os.path.realpath(__file__))).resolve())

        # Launch one of the worlds provided by NVIDIA
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Default Environment"])

        # Create the vehicle
        # Try to spawn the selected robot in the world to the specified namespace
        config_multirotor = MultirotorConfig()
        config_multirotor.backends = [ROS2Backend(vehicle_id=0)]#, MavlinkBackend(mavlink_config)]

        Multirotor(
            "/World/quadrotor1",
            ROBOTS['Iris'],
            0,
            [2.3, -1.5, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )

        # Set the camera to a nice position so that we can see the 2 drones almost touching each other
        self.pg.set_viewport_camera([7.53, -1.6, 4.96], [0.0, 3.3, 7.0])

        # Read the trajectories and plot them inside isaac sim
        trajectory = np.flip(np.genfromtxt(self.curr_dir + "/trajectories/baf_slow_x_flipped.csv", delimiter=','), axis=0)
        num_samples2,_ = trajectory.shape

        # Draw the lines of the desired trajectory in Isaac Sim with the same color as the output plots for the paper
        draw = _debug_draw.acquire_debug_draw_interface()
        point_list_2 = [(trajectory[i,1], trajectory[i,2], trajectory[i,3]) for i in range(num_samples2)]
        draw.draw_lines_spline(point_list_2, (255/255, 0, 0, 1), 5, False)

        # Reset the simulation environment so that all articulations (aka robots) are initialized
        self.world.reset()

    def run(self):
        """
        Method that implements the application main loop, where the physics steps are executed.
        """

        # Start the simulation
        self.timeline.play()

        # The "infinite" loop
        while simulation_app.is_running():

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
