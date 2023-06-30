"""
| File: vtol_actuations.py
| Author: Mohammadreza Mousaei
| Descriptio: File that implements a quadratic thrust curve for rotors
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
"""
import numpy as np
from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.thrusters.thrust_curve import ThrustCurve
import os
from pathlib import Path

class VtolActuations(ThrustCurve):
    """Class that implements the dynamics of rotors that can be described by a quadratic thrust curve
    """
    def __init__(self, config={}):
        """_summary_

        Args:
            config (dict): A Dictionary that contains all the parameters for configuring the QuadraticThrustCurve - it can be empty or only have some of the parameters used by the QuadraticThrustCurve.
        
        Examples:
            The dictionary default parameters are

            >>> {"num_rotors": 4,
            >>>  "rotor_constant": [5.84e-6, 5.84e-6, 5.84e-6, 5.84e-6],
            >>>  "yaw_moment_coefficient": [1e-6, 1e-6, 1e-6, 1e-6],
            >>>  "rot_dir": [-1, -1, 1, 1],
            >>>  "min_rotor_velocity": [0, 0, 0, 0],                      # rad/s
            >>>  "max_rotor_velocity": [1100, 1100, 1100, 1100],          # rad/s
            >>> }
        """

        # Get the total number of rotors to simulate
        self._num_rotors = config.get("num_rotors", 5)
        self._num_surfaces = config.get("num_surfaces", 4)

        # The rotor constant used for computing the total thrust produced by the rotor: T = rotor_constant * omega^2
        # kt = 8.54858e-6 * 4000
        # km = 1e-6 * 4000
        # kt = 2e-05 * 7/5 * 7/5
        # kt_p = 8.54858e-06 
        # km = 1e-06 * 7/5 * 7/5

        # kt = 2e-05 * 4150.37012 / 5  * 7/5* 7/5
        kt = 0.18315984515978836
        # kt_p = 2e-05 * 4150.37012 / 5  * 7/5* 7/5
        kt_p = 0.18315984515978836 * 3
        km = 1e-06 * 4150.37012 / 5 * 7/5* 7/5

        self._rotor_constant = config.get("rotor_constant", [kt, kt, kt, kt, kt_p])
        assert len(self._rotor_constant) == self._num_rotors

        # The rotor constant used for computing the total torque generated about the vehicle Z-axis
        self._yaw_moment_coefficient = config.get("yaw_moment_coefficient", [km, km, km, km, km])
        assert len(self._yaw_moment_coefficient) == self._num_rotors

        # Save the rotor direction of rotation
        self._rot_dir = config.get("rot_dir", [-1, -1, 1, 1, 1])
        assert len(self._rot_dir) == self._num_rotors

        # Values for the minimum and maximum rotor velocity in rad/s
        self.min_rotor_velocity = config.get("min_rotor_velocity", [0, 0, 0, 0, 0])
        assert len(self.min_rotor_velocity) == self._num_rotors
        mx_v = 300.
        self.max_rotor_velocity = config.get("max_rotor_velocity", [mx_v, mx_v, mx_v, mx_v, mx_v])
        assert len(self.max_rotor_velocity) == self._num_rotors

        # The actual speed references to apply to the vehicle rotor joints
        self._input_reference = [0.0 for i in range(self._num_rotors)]

        # The actual velocity that each rotor is spinning at
        self._velocity = [0.0 for i in range(self._num_rotors)]

        # The actual force that each rotor is generating
        self._force = [0.0 for i in range(self._num_rotors+self._num_surfaces)]

        # The actual rolling moment that is generated on the body frame of the vehicle
        self._yaw_moment = 0.0

        self._rudder_coef = 0.00001
        self._aileron_coef = 0.001
        self._elevator_coef = 0.0000001
        self.curr_dir = str(Path(os.path.dirname(os.path.realpath(__file__))).resolve())


    def set_input_reference(self, input_reference):
        """
        Receives as input a list of target angular velocities of each rotor in rad/s
        """

        # The target angular velocity of the rotor
        self._input_reference = input_reference
        # print("input ref = ", input_reference)

    def update(self, state: State, dt: float):
        """
        Note: the state and dt variables are not used in this implementation, but left
        to add support to other rotor models where the total thrust is dependent on
        states such as vehicle linear velocity

        Args:
            state (State): The current state of the vehicle.
            dt (float): The time elapsed between the previous and current function calls (s).
        """

        yaw_moment = 0.0

        # Compute the actual force to apply to the rotors and the rolling moment contribution
        for i in range(self._num_rotors):

            # Set the actual velocity that each rotor is spinning at (instantaneous model - no delay introduced)
            # Only apply clipping of the input reference

            # print("vel size = ", len(self._velocity))
            # print("_input_reference size = ", len(self._input_reference))
            # print("min_rotor_velocity size = ", len(self.min_rotor_velocity))
            # print("max_rotor_velocity size = ", len(self.max_rotor_velocity))
            self._velocity[i] = np.maximum(
                self.min_rotor_velocity[i], np.minimum(self._input_reference[i], self.max_rotor_velocity[i])
            )

            # Set the force using a quadratic thrust curve
            self._force[i] = self._rotor_constant[i] * np.power(self._velocity[i], 2)

            # Compute the rolling moment coefficient
            if(i != 4):
                yaw_moment += self._yaw_moment_coefficient[i] * np.power(self._velocity[i], 2.0) * self._rot_dir[i]
        
        # Update the rolling moment variable
        self._yaw_moment = yaw_moment

        body_vel = state.linear_body_velocity
        # self._yaw_moment += self._rudder_coef * (self._input_reference[8]-100) * body_vel[0]**2
        # with open(self.curr_dir+'/aileron_coef.txt', 'r') as f:
        #     content = f.read()
        # self._aileron_coef = float(content)
        # with open(self.curr_dir+'/pusher_const.txt', 'r') as f:
        #     content = f.read()
        # self._rotor_constant[4] = float(content)
        # with open(self.curr_dir+'/elev_coef.txt', 'r') as f:
        #     content = f.read()
        # self._elevator_coef = float(content)
        # with open(self.curr_dir+'/rudd_coef.txt', 'r') as f:
        #     content = f.read()
        # self._rudder_coef = float(content)
        
        self._roll_moment = self._aileron_coef * (self._input_reference[5]-100) * (body_vel[0]**2)
        self._pitch_moment = self._elevator_coef * (self._input_reference[7]-100) * (body_vel[0]**2)

        # self.force[5] = self._aileron_coef * (self._input_reference[5]-100) * (body_vel[0]**2)
        # self.force[6] = self._aileron_coef * (self._input_reference[6]-100) * (body_vel[0]**2)
        # self.force[7] = self._elevator_coef * (self._input_reference[7]-100) * (body_vel[0]**2)

        # with open('/home/honda/Documents/pitch_moment.txt', 'r') as f:
        #     content = f.read()
        # self._pitch_moment = float(content)

        # print(f"self._input_reference   {self._input_reference}")
        # print(f"self._force             {self._force}")
        # print(f"self._velocity          {self._velocity}")
        # print(f"self._roll_moment       {self._roll_moment}")
        # print(f"self._pitch_moment      {self._pitch_moment}")
        # print(f"self._yaw_moment        {self._yaw_moment}")

        # Return the forces and velocities on each rotor and total torque applied on the body frame
        return self._force, self._velocity, self._roll_moment, self._pitch_moment, self._yaw_moment

    @property
    def force(self):
        """The force to apply to each rotor of the vehicle at any given time instant

        Returns:
            list: A list of forces (in Newton N) to apply to each rotor of the vehicle (on its Z-axis) at any given time instant
        """
        return self._force

    @property
    def velocity(self):
        """The velocity at which each rotor of the vehicle should be rotating at any given time instant

        Returns:
            list: A list of angular velocities (in rad/s) of each rotor (about its Z-axis) at any given time instant
        """
        return self._velocity

    @property
    def yaw_moment(self):
        """The total rolling moment being generated on the body frame of the vehicle by the rotating propellers

        Returns:
            float: The total rolling moment to apply to the vehicle body frame (Torque about the Z-axis) in Nm
        """
        return self._yaw_moment

    @property
    def rot_dir(self):
        """The direction of rotation of each rotor of the vehicle

        Returns:
            list(int): A list with the rotation direction of each rotor (-1 is counter-clockwise and 1 for clockwise)
        """
        return self._rot_dir
