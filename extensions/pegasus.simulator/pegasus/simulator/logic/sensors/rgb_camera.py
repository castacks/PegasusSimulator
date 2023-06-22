"""
| File: rgb_camera.py
| Author: Micah Nye (micahn@andrew.cmu.edu)
| License: BSD-3-Clause. Copyright (c) 2023, Micah Nye. All rights reserved.
| Description: Simulates an RGB camera using the Omnigraph framework provided in Isaacsim
"""
__all__ = ["RGBCamera"]

import carb
import sys

import omni
from omni.isaac.core.utils import stage
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.kit.viewport.utility import get_active_viewport, get_active_viewport_and_window
import omni.graph.core as og
from omni.isaac.core.utils.prims import set_targets

from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.sensors import Sensor
import numpy as np

class RGBCamera(Sensor):
    """The class that implements the Camera sensor. This class inherits the base class Sensor.
    """
    def __init__(self, id=0, config={"update_rate": 30}, app=None):
        """Initialize the Camera class

        Args:
            id (int): The id of the camera
            config (dict): A Dictionary that contains all the parameters for configuring the IMU - it can be empty or only have some of the parameters used by the IMU.

        Examples:
            The dictionary default parameters are

            >>> {"position": [0.0, 0.0, 0.0],         # Meters
            >>>  "orientation": [0.0, 0.0, 0.0, 1.0], # Quaternion [qx, qy, qz, qw]
            >>>  "focal_length": 250.0,               # Pixels
            >>>  "resolution": [640, 480],            # Pixels
            >>>  "set_projection_type": "pinhole",    # pinhole, fisheyeOrthographic, fisheyeEquidistant, fisheyeEquisolid, fisheyePolynomial, fisheyeSpherical
            >>>  "update_rate": 60.0}                 # Hz
        """

        # Initialize the Super class "object" attribute
        # update_rate not necessary
        super().__init__(sensor_type="RGBCamera", update_rate=config.get("update_rate"))
        self._simulator_app = app

        # Save the id of the sensor
        self._id = id

        # Reference to the actual camera object. This is set when the camera is initialized
        self._vehicle = None                # The vehicle this sensor is associated with
        self.camera = None

        # Set the position of the camera relative to the vehicle
        self._position = np.array(config.get("position", [0.0, 0.0, 0.0]))
        self._orientation = np.array(config.get("orientation", [0.0, 0.0, 0.0, 1.0]))  # Quaternion [qx, qy, qz, qw]

        # Set the default camera parameters if making a new camera, not already on the vehicle. TODO: Add making new camera support
        self._focal_length = config.get("focal_length", 24.0)
        self._focal_distance = config.get("focal_distance", 400.0)
        self._clipping_range = config.get("clipping_range", [0.0001, 1000000.0])
        self._resolution = config.get("resolution", [640, 480])
        self._set_projection_type = config.get("set_projection_type", "pinhole")
        self._horizonal_aperture = config.get("horizontal_aperture", 20.9550)
        self._vertical_aperture = config.get("vertical_aperture", 15.2908)

        # Save the current state of the camera sensor
        self._state = {
            "id": self._id,
            "position": np.array([0.0, 0.0, 0.0]),
            "orientation": np.array([0.0, 0.0, 0.0, 1.0]),
            "frame_num": 0,
            "frame": None,
        }

    def initialize(self, origin_lat, origin_lon, origin_alt, vehicle=None):
        """Method that initializes the action graph of the camera. It also initalizes the sensor latitude, longitude and
        altitude attributes as well as the vehicle that the sensor is attached to.
        
        Args:
            origin_lat (float): NOT USED BY THIS SENSOR
            origin_lon (float): NOT USED BY THIS SENSOR
            origin_alt (float): NOT USED BY THIS SENSOR
            vehicle (Vehicle): The vehicle that this sensor is attached to.
        """

        self._vehicle = vehicle

        # Set the prim_path for the camera. If the vehicle has one, no need to create a new prim from scratch
        camera_prim_path = self._vehicle.prim_path + "/body/camera" + str(self._id)
        if not is_prim_path_valid(camera_prim_path):
            carb.log_error(f"Could not find camera at prim_path {camera_prim_path}. Not generating the ROS2 publisher")
            return
        ros_camera_graph_path = self._vehicle.prim_path  + "/body/ROS2_Camera" + str(self._id)

        self._simulator_app.update()

        # Creating an on-demand push graph with cameraHelper nodes to generate ROS image publishers
        keys = og.Controller.Keys
        (ros_camera_graph, _, _, _) = og.Controller.edit(
            {
                "graph_path": ros_camera_graph_path,
                "evaluator_name": "push",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
            },
            {
                keys.CREATE_NODES: [
                    ("OnTick", "omni.graph.action.OnTick"),
                    ("createViewport", "omni.isaac.core_nodes.IsaacCreateViewport"),
                    ("getRenderProduct", "omni.isaac.core_nodes.IsaacGetViewportRenderProduct"),
                    ("setCamera", "omni.isaac.core_nodes.IsaacSetCameraOnRenderProduct"),
                    ("cameraHelperRgb", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                    ("cameraHelperInfo", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                    ("cameraHelperDepth", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                ],
                keys.CONNECT: [
                    ("OnTick.outputs:tick", "createViewport.inputs:execIn"),
                    ("createViewport.outputs:execOut", "getRenderProduct.inputs:execIn"),
                    ("createViewport.outputs:viewport", "getRenderProduct.inputs:viewport"),
                    ("getRenderProduct.outputs:execOut", "setCamera.inputs:execIn"),
                    ("getRenderProduct.outputs:renderProductPath", "setCamera.inputs:renderProductPath"),
                    ("setCamera.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
                    ("setCamera.outputs:execOut", "cameraHelperInfo.inputs:execIn"),
                    ("setCamera.outputs:execOut", "cameraHelperDepth.inputs:execIn"),
                    ("getRenderProduct.outputs:renderProductPath", "cameraHelperRgb.inputs:renderProductPath"),
                    ("getRenderProduct.outputs:renderProductPath", "cameraHelperInfo.inputs:renderProductPath"),
                    ("getRenderProduct.outputs:renderProductPath", "cameraHelperDepth.inputs:renderProductPath"),
                ],
                keys.SET_VALUES: [
                    ("createViewport.inputs:viewportId", self._id),
                    ("createViewport.inputs:name", self._vehicle._stage_prefix + "/camera" + str(self._id)),
                    ("cameraHelperRgb.inputs:frameId", "sim_camera" + str(self._id)),
                    ("cameraHelperRgb.inputs:topicName", "vehicle" + str(self._vehicle._vehicle_id) + "/camera" + str(self._id) + "/image"),
                    ("cameraHelperRgb.inputs:type", "rgb"),
                    ("cameraHelperInfo.inputs:frameId", "sim_camera" + str(self._id)),
                    ("cameraHelperInfo.inputs:topicName", "vehicle" + str(self._vehicle._vehicle_id) + "/camera" + str(self._id) + "/camera_info"),
                    ("cameraHelperInfo.inputs:type", "camera_info")
                ],
            },
        )

        set_targets(
            prim=stage.get_current_stage().GetPrimAtPath(ros_camera_graph_path + "/setCamera"),
            attribute="inputs:cameraPrim",
            target_prim_paths=[camera_prim_path]
        )

        # Run the ROS Camera graph once to generate ROS image publishers in SDGPipeline
        og.Controller.evaluate_sync(ros_camera_graph)

        self._simulator_app.update()

    @property
    def state(self):
        """
        (dict) The 'state' of the sensor, i.e. the data produced by the sensor at any given point in time
        """
        return self._state

    @Sensor.update_at_rate
    def update(self, state: State, dt: float):
        """

        Args:
            state (State): The current state of the vehicle. UNUSED IN THIS SENSOR
            dt (float): The time elapsed between the previous and current function calls (s). UNUSED IN THIS SENSOR
        Returns:
            None
        """

        return None