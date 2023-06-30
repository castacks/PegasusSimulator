import numpy as np

from pegasus.simulator.logic.vehicles.utils.usb_drivers import ThrottleState, StickState, PedalsState, GamepadState
import pegasus.simulator.logic.vehicles.utils.quadrotor_dynamics as quadrotor_dynamics
from pegasus.simulator.logic.vehicles.utils.quad_control import Control

import time

import pickle

class UserInput():
    def __init__(self, controller_type = 0, px4_control_surface_bias = 0):

        self.controller_type = controller_type

        self.throttle = ThrottleState()
        self.stick = StickState()
        self.pedals = PedalsState()
        self.gamepad = GamepadState()

        self.quad = quadrotor_dynamics.quad(0.1, 1e100)
        self.curr_throttle_proportion = 0. 

        self.control_surface_bias = px4_control_surface_bias

        self.quad_control = Control()

        self.prev_time = time.time()
        with open("/home/honda/workspace/PegasusSimulator/extensions/pegasus.simulator/pegasus/simulator/logic/vehicles/utils/parameters/quad_pid_params.pkl", 'rb') as f:
            self.Kp_attitude,\
            self.Ki_attitude,\
            self.Kd_attitude,\
            self.Kp_rate,\
            self.Ki_rate,\
            self.Kd_rate,\
            self.Kp_altitude,\
            self.Ki_altitude,\
            self.Kd_altitude,\
            self.Kp_velocity,\
            self.Ki_velocity,\
            self.Kd_velocity,\
            self.motor_constant = pickle.load(f)

    def check_reload_control(self):
        with open("/home/honda/workspace/PegasusSimulator/extensions/pegasus.simulator/pegasus/simulator/logic/vehicles/utils/parameters/quad_pid_params.pkl", 'rb') as f:
            self.Kp_attitude,\
            self.Ki_attitude,\
            self.Kd_attitude,\
            self.Kp_rate,\
            self.Ki_rate,\
            self.Kd_rate,\
            self.Kp_altitude,\
            self.Ki_altitude,\
            self.Kd_altitude,\
            self.Kp_velocity,\
            self.Ki_velocity,\
            self.Kd_velocity,\
            self.motor_constant = pickle.load(f)
        
            if (self.quad_control.Kp_attitude == self.Kp_attitude).all() and\
            (self.quad_control.Ki_attitude == self.Ki_attitude).all() and\
            (self.quad_control.Kd_attitude == self.Kd_attitude).all() and\
            (self.quad_control.Kp_rate == self.Kp_rate).all() and\
            (self.quad_control.Ki_rate == self.Ki_rate).all() and\
            (self.quad_control.Kd_rate == self.Kd_rate).all() and\
            self.quad_control.Kp_altitude == self.Kp_altitude and\
            self.quad_control.Ki_altitude == self.Ki_altitude and\
            self.quad_control.Kd_altitude == self.Kd_altitude and\
            (self.quad_control.Kp_velocity == self.Kp_velocity).all() and\
            (self.quad_control.Ki_velocity == self.Ki_velocity).all() and\
            (self.quad_control.Kd_velocity == self.Kd_velocity).all() and\
            self.quad_control.motor_constant == self.motor_constant:
                return
            else:
                self.quad_control = Control()

    def get_input_reference(self, state, euler_angles):

        input_reference = [0.0 for i in range(9)]

        if self.controller_type == 0:
            self.stick.refresh()
            self.throttle.refresh()
            self.pedals.refresh()

            thrust_max  = 300.
            thrust_add_max = 50.

            mode = "velocity_altitude"

            if mode == "rotor_vels":
                
                input_reference = [
                    self.throttle.left * thrust_max - self.stick.x*thrust_add_max - self.stick.y*thrust_add_max, # front right
                    self.throttle.left * thrust_max + self.stick.x*thrust_add_max + self.stick.y*thrust_add_max, # back left
                    self.throttle.left * thrust_max + self.stick.x*thrust_add_max - self.stick.y*thrust_add_max, # front left
                    self.throttle.left * thrust_max - self.stick.x*thrust_add_max + self.stick.y*thrust_add_max, # back right
                    self.throttle.right, # ~3000 rpm in rad/s
                    self.stick.x,
                    -self.stick.x,
                    self.stick.y,
                    self.stick.z
                ]
        
            elif mode == 'velocity_altitude':
                
                new_time = time.time()

                self.check_reload_control()

                rotor_vels = self.quad_control.control(
                    state,
                    euler_angles,
                    10,#state.position[2] + self.throttle.left * 2,
                    euler_angles[2] + self.stick.z * 0.3,
                    np.array([self.stick.y*20., -self.stick.x*20.]),
                    new_time - self.prev_time
                )
                self.prev_time = new_time

                input_reference = [
                    rotor_vels[0],
                    rotor_vels[1],
                    rotor_vels[2],
                    rotor_vels[3],
                    self.throttle.right, # ~3000 rpm in rad/s
                    self.stick.x,
                    -self.stick.x,
                    self.stick.y,
                    self.stick.z
                ]


        elif self.controller_type == 1:
            self.gamepad.refresh()

            thrust_max  = 50000
            thrust_add_max = 1000

            if self.gamepad.b7:
                self.curr_throttle_proportion -= 0.01
            elif self.gamepad.b8:
                self.curr_throttle_proportion += 0.01
            print(self.curr_throttle_proportion)

            input_reference = [
                self.gamepad.left_y * thrust_max - self.gamepad.right_x*thrust_add_max - self.gamepad.right_y*thrust_add_max, # front right
                self.gamepad.left_y * thrust_max + self.gamepad.right_x*thrust_add_max + self.gamepad.right_y*thrust_add_max, # back left
                self.gamepad.left_y * thrust_max + self.gamepad.right_x*thrust_add_max - self.gamepad.right_y*thrust_add_max, # front left
                self.gamepad.left_y * thrust_max - self.gamepad.right_x*thrust_add_max + self.gamepad.right_y*thrust_add_max, # back right
                self.curr_throttle_proportion*250*3, # ~2500 rpm in rad/s (*3)
                self.gamepad.right_x,
                -self.gamepad.right_x,
                self.gamepad.right_y,
                0 # rudder
            ]


        input_reference[5] += self.control_surface_bias
        input_reference[6] += self.control_surface_bias
        input_reference[7] += self.control_surface_bias
        input_reference[8] += self.control_surface_bias
        return input_reference