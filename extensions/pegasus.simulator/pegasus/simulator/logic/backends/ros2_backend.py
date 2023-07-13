"""
| File: ros2_backend.py
| Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
| Description: File that implements the ROS2 Backend for communication/control with/of the vehicle simulation through ROS2 topics
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
"""
import carb
# import cv2
import numpy as np
from omni.isaac.core.utils.extensions import disable_extension, enable_extension

# Perform some checks, because Isaac Sim some times does not play nice when using ROS/ROS2
disable_extension("omni.isaac.ros_bridge")
enable_extension("omni.isaac.ros2_bridge")

# Inform the user that now we are actually import the ROS2 dependencies 
# Note: we are performing the imports here to make sure that ROS2 extension was load correctly
import rclpy
from std_msgs.msg import Float64
from sensor_msgs.msg import Imu, MagneticField, NavSatFix, NavSatStatus, Image
from geometry_msgs.msg import PoseStamped, TwistStamped, AccelStamped, PoseWithCovariance, TwistWithCovariance
from nav_msgs.msg import Odometry
# from pegasus_msgs.msg import Control
# from omni.isaac.ros2_bridge.omni.isaac.rclpy.pegasus_msgs import Control

import omni.kit.app
from pegasus.simulator.logic.backends.backend import Backend

is_init = False
if not is_init:
    rclpy.init()
    is_init = True

class ROS2Backend(Backend):

    def __init__(self, vehicle_id: int, num_rotors=4):

        # Save the configurations for this backend
        self._id = vehicle_id
        self._num_rotors = num_rotors
        self._namespace = "vehicle"+str(vehicle_id)

        # Start the actual ROS2 setup here
        self.node = rclpy.create_node(node_name="node",namespace=self._namespace)

        # Create publishers for the state of the vehicle in ENU
        self.pose_pub = self.node.create_publisher(PoseStamped,"state/pose", 10)
        self.twist_pub = self.node.create_publisher(TwistStamped, "state/twist", 10)
        self.odom_pub = self.node.create_publisher(Odometry, "state/odom", 10)
        self.twist_inertial_pub = self.node.create_publisher(TwistStamped, "state/twist_inertial", 10)
        self.accel_pub = self.node.create_publisher(AccelStamped, "state/accel", 10)

        # Create publishers for some sensor data
        self.imu_pub = self.node.create_publisher(Imu, "sensors/imu", 10)
        self.mag_pub = self.node.create_publisher(MagneticField, "sensors/mag", 10)
        self.gps_pub = self.node.create_publisher(NavSatFix, "sensors/gps", 10)
        self.gps_vel_pub = self.node.create_publisher(TwistStamped, "sensors/gps_twist", 10)

        # TODO: Investigate camera data publishing from traditional interface for standard node format
        self.camera_pubs = {}

        # self.control_sub = self.node.create_subscription(Control, self._namespace+"/command", self.control_callback, 10)
        # Subscribe to vector of floats with the target angular velocities to control the vehicle
        # This is not ideal, but we need to reach out to NVIDIA so that they can improve the ROS2 support with custom messages
        # The current setup as it is.... its a pain!!!!
        self.control_sub0 = self.node.create_subscription(Float64, "command0", self.control_callback0, 10)
        self.control_sub1 = self.node.create_subscription(Float64, "command1", self.control_callback1, 10)
        self.control_sub2 = self.node.create_subscription(Float64, "command2", self.control_callback2, 10)
        self.control_sub3 = self.node.create_subscription(Float64, "command3", self.control_callback3, 10)

        # Setup zero input reference for the thrusters
        self.intermediate_input_ref = [0.0 for i in range(self._num_rotors)]
        self.rec0 = self.rec1 = self.rec2 = self.rec3 = False
        self.input_ref = [0.0 for i in range(self._num_rotors)]

    def update_state(self, state):
        """
        Method that when implemented, should handle the receival of the state of the vehicle using this callback
        """

        pose = PoseStamped()
        twist = TwistStamped()
        twist_inertial = TwistStamped()
        odom = Odometry()
        accel = AccelStamped()

        pose.header.frame_id = "world"
        twist.header.frame_id = "base_link"
        twist_inertial.header.frame_id = "world"
        odom.header.frame_id = "world"
        accel.header.frame_id = "world"

        # Fill the position and attitude of the vehicle in ENU
        pose.pose.position.x = state.position[0]
        pose.pose.position.y = state.position[1]
        pose.pose.position.z = state.position[2]

        pose.pose.orientation.x = state.attitude[0]
        pose.pose.orientation.y = state.attitude[1]
        pose.pose.orientation.z = state.attitude[2]
        pose.pose.orientation.w = state.attitude[3]

        # Fill the linear and angular velocities in the body frame of the vehicle
        twist.twist.linear.x = state.linear_body_velocity[0]
        twist.twist.linear.y = state.linear_body_velocity[1]
        twist.twist.linear.z = state.linear_body_velocity[2]

        twist.twist.angular.x = state.angular_velocity[0]
        twist.twist.angular.y = state.angular_velocity[1]
        twist.twist.angular.z = state.angular_velocity[2]

        # Fill the linear velocity of the vehicle in the inertial frame
        twist_inertial.twist.linear.x = state.linear_velocity[0]
        twist_inertial.twist.linear.y = state.linear_velocity[1]
        twist_inertial.twist.linear.z = state.linear_velocity[2]

        # Update odometry proper
        odom.pose.pose = pose.pose
        odom.twist.twist.linear = twist_inertial.twist.linear
        odom.twist.twist.angular = twist.twist.angular

        # Fill the linear acceleration in the inertial frame
        accel.accel.linear.x = state.linear_acceleration[0]
        accel.accel.linear.y = state.linear_acceleration[1]
        accel.accel.linear.z = state.linear_acceleration[2]

        # Update the header
        pose.header.stamp = self.node.get_clock().now().to_msg()
        twist.header.stamp = pose.header.stamp
        twist_inertial.header.stamp = pose.header.stamp
        odom.header.stamp = pose.header.stamp
        accel.header.stamp = pose.header.stamp

        # Publish the messages containing the state of the vehicle
        self.pose_pub.publish(pose)
        self.twist_pub.publish(twist)
        self.twist_inertial_pub.publish(twist_inertial)
        self.odom_pub.publish(odom)
        self.accel_pub.publish(accel)

    def control_callback(self, msg):
        u_1 = msg.thrust
        tau = np.array([msg.moment1, msg.moment2, msg.moment3])
        self.input_ref = self.vehicle.force_and_torques_to_velocities(u_1, tau)
    
    # GROSS
    def control_callback0(self, msg):
        self.intermediate_input_ref[0]=float(msg.data)
        self.rec0 = True
    def control_callback1(self, msg):
        self.intermediate_input_ref[1]=float(msg.data)
        self.rec1 = True
    def control_callback2(self, msg):
        self.intermediate_input_ref[2]=float(msg.data)
        self.rec2 = True
    def control_callback3(self, msg):
        self.intermediate_input_ref[3]=float(msg.data)
        self.rec3 = True

    def rotor_callback(self, ros_msg: Float64, rotor_id):
        # Update the reference for the rotor of the vehicle
        self.input_ref[rotor_id] = float(ros_msg.data)

    def update_sensor(self, sensor_type: str, data):
        """
        Method that when implemented, should handle the receival of sensor data
        """

        if sensor_type == "IMU":
            self.update_imu_data(data)
        elif sensor_type == "GPS":
            self.update_gps_data(data)
        elif sensor_type == "Magnetometer":
            self.update_mag_data(data)
        elif sensor_type == "Barometer":        # TODO - create a topic for the barometer later on
            pass

    def update_imu_data(self, data):

        msg = Imu()

        # Update the header
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "base_link_frd"
        
        # Update the angular velocity (NED + FRD)
        msg.angular_velocity.x = data["angular_velocity"][0]
        msg.angular_velocity.y = data["angular_velocity"][1]
        msg.angular_velocity.z = data["angular_velocity"][2]
        
        # Update the linear acceleration (NED)
        msg.linear_acceleration.x = data["linear_acceleration"][0]
        msg.linear_acceleration.y = data["linear_acceleration"][1]
        msg.linear_acceleration.z = data["linear_acceleration"][2]

        # Publish the message with the current imu state
        self.imu_pub.publish(msg)

    def update_gps_data(self, data):

        msg = NavSatFix()
        msg_vel = TwistStamped()

        # Update the headers
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "world_ned"
        msg_vel.header.stamp = msg.header.stamp
        msg_vel.header.frame_id = msg.header.frame_id

        # Update the status of the GPS
        status_msg = NavSatStatus()
        status_msg.status = 0 # unaugmented fix position
        status_msg.service = 1 # GPS service
        msg.status = status_msg

        # Update the latitude, longitude and altitude
        msg.latitude = data["latitude"]
        msg.longitude = data["longitude"]
        msg.altitude = data["altitude"]

        # Update the velocity of the vehicle measured by the GPS in the inertial frame (NED)
        msg_vel.twist.linear.x = data["velocity_north"]
        msg_vel.twist.linear.y = data["velocity_east"]
        msg_vel.twist.linear.z = data["velocity_down"]

        # Publish the message with the current GPS state
        self.gps_pub.publish(msg)
        self.gps_vel_pub.publish(msg_vel)

    def update_mag_data(self, data):
        
        msg = MagneticField()

        # Update the headers
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "base_link_frd"

        msg.magnetic_field.x = data["magnetic_field"][0]
        msg.magnetic_field.y = data["magnetic_field"][1]
        msg.magnetic_field.z = data["magnetic_field"][2]

        # Publish the message with the current magnetic data
        self.mag_pub.publish(msg)

    def input_reference(self):
        """
        Method that is used to return the latest target angular velocities to be applied to the vehicle

        Returns:
            A list with the target angular velocities for each individual rotor of the vehicle
        """
        print("Calling input_reference")
        if self.rec0 and self.rec1 and self.rec2 and self.rec3:
            self.input_ref = self.vehicle.force_and_torques_to_velocities(self.intermediate_input_ref[0], self.intermediate_input_ref[1:])
            print(f"We've received everything! intermediate input ref {self.intermediate_input_ref[0], self.intermediate_input_ref[1:]}")
            self.rec0 = self.rec1 = self.rec2 = self.rec3 = False

        print(f"self.input_ref {self.input_ref}")
        return self.input_ref

    def update(self, dt: float):
        """
        Method that when implemented, should be used to update the state of the backend and the information being sent/received
        from the communication interface. This method will be called by the simulation on every physics step
        """

        # In this case, do nothing as we are sending messages as soon as new data arrives from the sensors and state
        # and updating the reference for the thrusters as soon as receiving from ROS2 topics
        # Just poll for new ROS 2 messages in a non-blocking way
        rclpy.spin_once(self.node, timeout_sec=0)

    def start(self):
        """
        Method that when implemented should handle the begining of the simulation of vehicle
        """
        # Reset the reference for the thrusters
        self.input_ref = [0.0 for i in range(self._num_rotors)]

    def stop(self):
        """
        Method that when implemented should handle the stopping of the simulation of vehicle
        """
        # Reset the reference for the thrusters
        self.input_ref = [0.0 for i in range(self._num_rotors)]

    def reset(self):
        """
        Method that when implemented, should handle the reset of the vehicle simulation to its original state
        """
        # Reset the reference for the thrusters
        self.input_ref = [0.0 for i in range(self._num_rotors)]

    def check_ros_extension(self):
        """
        Method that checks which ROS extension is installed.
        """

        # Get the handle for the extension manager
        extension_manager = omni.kit.app.get_app().get_extension_manager()

        version = ""

        if self._ext_manager.is_extension_enabled("omni.isaac.ros_bridge"):
            version = "ros"
        elif self._ext_manager.is_extension_enabled("omni.isaac.ros2_bridge"):
            version = "ros2"
        else:
            carb.log_warn("Neither extension 'omni.isaac.ros_bridge' nor 'omni.isaac.ros2_bridge' is enabled")
