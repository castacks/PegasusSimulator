import numpy as np
from omni.isaac.core.utils.extensions import disable_extension, enable_extension
disable_extension("omni.isaac.ros_bridge")
enable_extension("omni.isaac.ros2_bridge")

# ROS
import rclpy
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped

# PEGASUS
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.sensors import RGBDCamera

class MovingCamera(RGBDCamera):
    def __init__(self, id=0, config={"update_rate": 30}, ros="ROS2", app=None):

        # Init world
        self._world = PegasusInterface().world

        # Init camera prim
        super().__init__(id=id, config=config, ros=ros, app=app)

        # Start the actual ROS2 setup here
        self._namespace = "camera"+str(id)
        self.node = rclpy.create_node(node_name="node",namespace=self._namespace)
        self.pose_sub = self.node.create_subscription(PoseStamped, "pose_topic", self.pose_callback, 10)

    def pose_callback(self, msg: PoseStamped):
        print(f"I received a message in MovingCamera! {msg.pose.position} {msg.pose.orientation}")
        position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        orientation = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        super().set_pose(position, orientation)

    def spawn(self):
        """
        Spawn in ROS-controlled Camera and add physics callback to use as ROS update
        """
        super().spawn()
        self._world.add_physics_callback(super().stage_prefix + "/update", self.update)

    def update(self, dt: float):
        """
        Method that when implemented, should be used to update the state of ROS and the information being sent/received
        from the communication interface. This method will be called by the simulation on every physics step
        """
        print("update")
        # In this case, do nothing as we are sending messages as soon as new data arrives from the sensors and state
        # and updating the reference for the thrusters as soon as receiving from ROS2 topics
        # Just poll for new ROS 2 messages in a non-blocking way
        rclpy.spin_once(self.node, timeout_sec=0)
