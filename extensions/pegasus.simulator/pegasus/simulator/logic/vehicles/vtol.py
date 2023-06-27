"""
| File: vtol.py
| Author: Mohammadreza Mousaei (mmousaei@andrew.cmu.edu)
| License: BSD-3-Clause. Copyright (c) 2023. All rights reserved.
| Description: Definition of the VTOL class which is used as the base for standard vehicles.
"""

import numpy as np

# The vehicle interface
from pegasus.simulator.logic.vehicles.vehicle import Vehicle

# Mavlink interface
from pegasus.simulator.logic.backends.mavlink_backend_vtol import MavlinkBackendVTOL

# Sensors and dynamics setup
from pegasus.simulator.logic.dynamics import LinearDrag
from pegasus.simulator.logic.dynamics import Lift
from pegasus.simulator.logic.thrusters import VtolActuations
from pegasus.simulator.logic.sensors import Barometer, IMU, Magnetometer, GPS, Airspeed
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
import subprocess
import json
from scipy.spatial.transform import Rotation as R

from pegasus.simulator.logic.vehicles.utils.user_input import UserInput

# Use os and pathlib for reading config files
import os
from pathlib import Path

class VTOLConfig:
    """
    A data class that is used for configuring a VTOL
    """

    def __init__(self):
        """
        Initialization of the VTOLConfig class
        """

        # Stage prefix of the vehicle when spawning in the world
        self.stage_prefix = "vtol"

        # The USD file that describes the visual aspect of the vehicle (and some properties such as mass and moments of inertia)
        self.usd_file = ""

        # The default thrust curve for a quadrotor and dynamics relating to drag
        self.actuations = VtolActuations()

        self.drag = LinearDrag([0.08, 0.0, 0.0])
        self.lift = Lift(1.2)
        

        # The default sensors for a quadrotor
        self.sensors = [Barometer(), IMU(), Magnetometer(), GPS(), Airspeed()]

        # The backends for actually sending commands to the vehicle. By default use mavlink (with default mavlink configurations)
        # [Can be None as well, if we do not desired to use PX4 with this simulated vehicle]. It can also be a ROS2 backend
        # or your own custom Backend implementation!
        self.backends = [MavlinkBackendVTOL()]

        


class VTOL(Vehicle):
    """VTOL class - It defines a base interface for creating a vtol
    """
    def __init__(
        self,
        # Simulation specific configurations
        stage_prefix: str = "vtol",
        usd_file: str = "",
        vehicle_id: int = 0,
        # Spawning pose of the vehicle
        init_pos=[0.0, 0.0, 0.07],
        init_orientation=[0.0, 0.0, 0.0, 1.0],
        config=VTOLConfig(),
    ):
        """Initializes the vtol object

        Args:
            stage_prefix (str): The name the vehicle will present in the simulator when spawned. Defaults to "quadrotor".
            usd_file (str): The USD file that describes the looks and shape of the vehicle. Defaults to "".
            vehicle_id (int): The id to be used for the vehicle. Defaults to 0.
            init_pos (list): The initial position of the vehicle in the inertial frame (in ENU convention). Defaults to [0.0, 0.0, 0.07].
            init_orientation (list): The initial orientation of the vehicle in quaternion [qx, qy, qz, qw]. Defaults to [0.0, 0.0, 0.0, 1.0].
            config (_type_, optional): _description_. Defaults to VTOLConfig().
        """

        # 1. Initiate the Vehicle object itself
        super().__init__(stage_prefix, usd_file, init_pos, init_orientation)

        self.pegasus_interface = PegasusInterface()    

        # 2. Initialize all the vehicle sensors
        self._sensors = config.sensors
        for sensor in self._sensors:
            sensor.initialize(self.pegasus_interface.latitude, self.pegasus_interface.longitude, self.pegasus_interface.altitude)

        self.controller_type = self.pegasus_interface.controller_type # (0 for Rhino joystick/throttle, 1 for Logitech gamepad)
        self.user_input = UserInput(self.controller_type)

        # Add callbacks to the physics engine to update each sensor at every timestep
        # and let the sensor decide depending on its internal update rate whether to generate new data
        self._world.add_physics_callback(self._stage_prefix + "/Sensors", self.update_sensors)

        # 3. Setup the dynamics of the system
        # Get the thrust curve of the vehicle from the configuration
        self._thrusters = config.actuations
        self._drag = config.drag
        self._lift = config.lift

        # 4. Save the backend interface (if given in the configuration of the multirotor)
        # and initialize them
        self._backends = config.backends
        for backend in self._backends:
            backend.initialize(self)

        # Add a callbacks for the
        self._world.add_physics_callback(self._stage_prefix + "/mav_state", self.update_sim_state)

        self.curr_dir = str(Path(os.path.dirname(os.path.realpath(__file__))).resolve())

    def update_sensors(self, dt: float):
        """Callback that is called at every physics steps and will call the sensor.update method to generate new
        sensor data. For each data that the sensor generates, the backend.update_sensor method will also be called for
        every backend. For example, if new data is generated for an IMU and we have a MavlinkBackend, then the update_sensor
        method will be called for that backend so that this data can latter be sent thorugh mavlink.

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """

        # Call the update method for the sensor to update its values internally (if applicable)
        for sensor in self._sensors:
            sensor_data = sensor.update(self._state, dt)

            # If some data was updated and we have a mavlink backend or ros backend (or other), then just update it
            if sensor_data is not None:
                for backend in self._backends:
                    backend.update_sensor(sensor.sensor_type, sensor_data)

    def update_sim_state(self, dt: float):
        """
        Callback that is used to "send" the current state for each backend being used to control the vehicle. This callback
        is called on every physics step.

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """
        for backend in self._backends:
            backend.update_state(self._state)

    def start(self):
        """
        Intializes the communication with all the backends. This method is invoked automatically when the simulation starts
        """
        for backend in self._backends:
            backend.start()
        self.plotter = subprocess.Popen(['python3', self.curr_dir+'/utils/realtime_plotter.py'], stdin=subprocess.PIPE)

    def stop(self):
        """
        Signal all the backends that the simulation has stopped. This method is invoked automatically when the simulation stops
        """
        for backend in self._backends:
            backend.stop()
        self.plotter.terminate()
        self.plotter.wait()

    def update(self, dt: float):
        """
        Method that computes and applies the forces to the vehicle in simulation based on the motor speed. 
        This method must be implemented by a class that inherits this type. This callback
        is called on every physics step.

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """

        # Get the articulation root of the vehicle
        articulation = self._world.dc_interface.get_articulation(self._stage_prefix)
        # Get the desired angular velocities for each rotor from the first backend (can be mavlink or other) expressed in rad/s
        # if len(self._backends) != 0:
        #     desired_rotor_velocities = self._backends[0].input_reference()
        # else:
        #     desired_rotor_velocities = [0.0 for i in range(self._thrusters._num_rotors)]

        desired_rotor_velocities = self.user_input.get_input_reference()


        """
        input_reference=[
        r1
        r2
        r3
        r4
        pusher
        al
        ar
        elevator
        rudder
        ]
        """

        # print("desired_rotor_vel = ", len(desired_rotor_velocities))
        # Input the desired rotor velocities in the thruster model
        self._thrusters.set_input_reference(desired_rotor_velocities)

        # Get the desired forces to apply to the vehicle


        
        forces, _, roll_moment, pitch_moment, yaw_moment = self._thrusters.update(self._state, dt)
        # print("force z = ", forces)
        # print("yaw_moment = ", yaw_moment)
        # print("roll_moment = ", roll_moment)
        # print("pitch_moment = ", pitch_moment)

        print(f"d : {desired_rotor_velocities}")
        print(f"f : {forces}")


        # Apply force to each rotor
        for i in range(4):

            # Apply the force in Z on the rotor frame
            self.apply_force([0.0, 0.0, forces[i]], body_part="/rotor" + str(i))

            # Generate the rotating propeller visual effect
            self.handle_propeller_visual(i, forces[i], articulation)

        
        self.apply_force([forces[4], 0.0, 0.0], body_part="/body")
        self.apply_force([0.0, 0.0, forces[5]], body_part="/a_l")
        self.apply_force([0.0, 0.0, forces[6]], body_part="/a_r")
        self.apply_force([0.0, 0.0, forces[7]], body_part="/elevator")
        self.handle_propeller_visual(4, forces[4], articulation)
        self.handle_surface_visual(5, self._thrusters._input_reference[5], articulation)
        self.handle_surface_visual(6, -self._thrusters._input_reference[6], articulation)
        self.handle_surface_visual(7, self._thrusters._input_reference[7], articulation)
        self.handle_surface_visual(8, self._thrusters._input_reference[8], articulation)
        # Apply the torque to the body frame of the vehicle that corresponds to the rolling moment
        # self.apply_torque([roll_moment, pitch_moment, yaw_moment], "/body")
        # self.apply_torque([roll_moment, pitch_moment, 0], "/body")
        # self.apply_torque([0, pitch_moment, yaw_moment], "/body")
        # self.apply_torque([0, 0, 0], "/body")
        # self.apply_torque([0.0, 0.0, yaw_moment], "/body")


        rot = R.from_quat(self._state.attitude)

        # Get the Euler angles (roll, pitch, yaw)
        euler_angles = rot.as_euler('xyz', degrees=True)
        # Compute the total linear drag force to apply to the vehicle's body frame
        drag = self._drag.update(self._state, euler_angles[1]+8, dt)
        lift = self._lift.update(self._state, euler_angles[1]+8, dt)

        

        plots_data = [
            # {"label": "Drag X",       "value": drag[0]},
            # {"label": "Drag Y",       "value": drag[1]},
            # {"label": "Drag Z",       "value": drag[2]},
            # {"label": "Pitch",       "value": euler_angles[1]},
            {"label": "Roll",       "value": euler_angles[0]},
            {"label": "Pitch",       "value": euler_angles[1]},
            {"label": "Yaw",       "value": euler_angles[2]},
            {"label": "Lift",       "value": lift[2]},
            {"label": "Airspeed",   "value": self._state.linear_body_velocity[0]},
            {"label": "Pusher",   "value": self._thrusters.force[4]},
            # {"label": "Pusher Command",   "value": self._thrusters._input_reference[4]},
            # {"label": "Altitude",   "value": self._state.position[2]},
            {"label": "Elevator",   "value": self._thrusters._input_reference[7]},
            # {"label": "Pitch Force",   "value": forces[7]},
            {"label": "Pitch Moment",   "value": pitch_moment},
            {"label": "Aileron",   "value": self._thrusters._input_reference[5]},
            # {"label": "Roll Force",   "value": forces[5]},
            {"label": "Roll Moment",   "value": roll_moment},
            {"label": "Rudder Command",   "value": self._thrusters._input_reference[8]},
            {"label": "Yaw Moment",   "value": yaw_moment},
            # {"label": "Roll Force Right",   "value": forces[6]},
            # {"label": "Aileron Right Command",   "value": self._thrusters._input_reference[6]},
            
            # {"label": "MC Rotor 1",   "value": self._thrusters.force[0]},
            # {"label": "MC Rotor 2",   "value": self._thrusters.force[1]},
            # {"label": "MC Rotor 3",   "value": self._thrusters.force[2]},
            # {"label": "MC Rotor 4",   "value": self._thrusters._input_reference[3]},
        ]

        # Send each plot data to the qt plot script
        for plot_data in plots_data:
            try:
                message = json.dumps(plot_data)
                self.plotter.stdin.write((message + '\n').encode())
                self.plotter.stdin.flush()
            except BrokenPipeError:
                print(BrokenPipeError)

        self.apply_force(drag, body_part="/body")
        self.apply_force(lift, body_part="/body")

        # Call the update methods in all backends
        for backend in self._backends:
            backend.update(dt)

    def handle_propeller_visual(self, rotor_number, force: float, articulation):
        """
        Auxiliar method used to set the joint velocity of each rotor (for animation purposes) based on the 
        amount of force being applied on each joint

        Args:
            rotor_number (int): The number of the rotor to generate the rotation animation
            force (float): The force that is being applied on that rotor
            articulation (_type_): The articulation group the joints of the rotors belong to
        """

        # Rotate the joint to yield the visual of a rotor spinning (for animation purposes only)
        joint = self._world.dc_interface.find_articulation_dof(articulation, "joint" + str(rotor_number))

        # Spinning when armed but not applying force
        
        # if(rotor_number == 4):
        if 0.0 < force < 1:
            self._world.dc_interface.set_dof_velocity(joint, 0 * self._thrusters.rot_dir[rotor_number])
        else:
            self._world.dc_interface.set_dof_velocity(joint, 10 * self._thrusters.rot_dir[rotor_number])
        # else:
        #     if 0.0 < force < 0.1:
        #         self._world.dc_interface.set_dof_velocity(joint, 5 * self._thrusters.rot_dir[rotor_number])
        #     else:
        #         self._world.dc_interface.set_dof_velocity(joint, 100 * self._thrusters.rot_dir[rotor_number])
            
        # Not spinning
        if(force == 0):
            self._world.dc_interface.set_dof_velocity(joint, 0)
    
    def handle_surface_visual(self, joint_number, force: float, articulation):
        """
        Auxiliar method used to set the joint velocity of each rotor (for animation purposes) based on the 
        amount of force being applied on each joint

        Args:
            rotor_number (int): The number of the rotor to generate the rotation animation
            force (float): The force that is being applied on that rotor
            articulation (_type_): The articulation group the joints of the rotors belong to
        """

        # Rotate the joint to yield the visual of a rotor spinning (for animation purposes only)
        joint = self._world.dc_interface.find_articulation_dof(articulation, "joint" + str(joint_number))

        # Spinning when armed but not applying force
        if 0.0 < force < 0.1:
            self._world.dc_interface.set_dof_velocity(joint, 0)
        # Spinning when armed and applying force
        elif 0.1 <= force:
            self._world.dc_interface.set_dof_velocity(joint, 100)
        # Not spinning
        else:
            self._world.dc_interface.set_dof_velocity(joint, 0)

    def force_and_torques_to_velocities(self, force: float, torque: np.ndarray):
        """
        Auxiliar method used to get the target angular velocities for each rotor, given the total desired thrust [N] and
        torque [Nm] to be applied in the multirotor's body frame.

        Note: This method assumes a quadratic thrust curve. This method will be improved in a future update,
        and a general thrust allocation scheme will be adopted. For now, it is made to work with multirotors directly.

        Args:
            force (np.ndarray): A vector of the force to be applied in the body frame of the vehicle [N]
            torque (np.ndarray): A vector of the torque to be applied in the body frame of the vehicle [Nm]

        Returns:
            list: A list of angular velocities [rad/s] to apply in reach rotor to accomplish suchs forces and torques
        """
        # print("force = ", force)
        # print("torque = ", torque)
        # Get the body frame of the vehicle
        rb = self._world.dc_interface.get_rigid_body(self._stage_prefix + "/body")

        # Get the rotors of the vehicle
        rotors = [self._world.dc_interface.get_rigid_body(self._stage_prefix + "/rotor" + str(i)) for i in range(self._thrusters._num_rotors)]

        # Get the relative position of the rotors with respect to the body frame of the vehicle (ignoring the orientation for now)
        relative_poses = self._world.dc_interface.get_relative_body_poses(rb, rotors)

        # Define the alocation matrix
        aloc_matrix = np.zeros((4, self._thrusters._num_rotors))
        
        # Define the first line of the matrix (T [N])
        aloc_matrix[0, :] = np.array(self._thrusters._rotor_constant)                                           

        # Define the second and third lines of the matrix (\tau_x [Nm] and \tau_y [Nm])
        aloc_matrix[1, :] = np.array([relative_poses[i].p[1] * self._thrusters._rotor_constant[i] for i in range(self._thrusters._num_rotors)])
        aloc_matrix[2, :] = np.array([-relative_poses[i].p[0] * self._thrusters._rotor_constant[i] for i in range(self._thrusters._num_rotors)])

        # Define the forth line of the matrix (\tau_z [Nm])
        aloc_matrix[3, :] = np.array([self._thrusters._yaw_moment_coefficient[i] * self._thrusters._rot_dir[i] for i in range(self._thrusters._num_rotors)])

        # Compute the inverse allocation matrix, so that we can get the angular velocities (squared) from the total thrust and torques
        aloc_inv = np.linalg.pinv(aloc_matrix)

        # Compute the target angular velocities (squared)
        squared_ang_vel = aloc_inv @ np.array([force, torque[0], torque[1], torque[2]])

        # Making sure that there is no negative value on the target squared angular velocities
        squared_ang_vel[squared_ang_vel < 0] = 0.0

        # ------------------------------------------------------------------------------------------------
        # Saturate the inputs while preserving their relation to each other, by performing a normalization
        # ------------------------------------------------------------------------------------------------
        max_thrust_vel_squared = np.power(self._thrusters.max_rotor_velocity[0], 2)
        max_val = np.max(squared_ang_vel)

        if max_val >= max_thrust_vel_squared:
            normalize = np.maximum(max_val / max_thrust_vel_squared, 1.0)

            squared_ang_vel = squared_ang_vel / normalize

        # Compute the angular velocities for each rotor in [rad/s]
        ang_vel = np.sqrt(squared_ang_vel)

        return ang_vel
