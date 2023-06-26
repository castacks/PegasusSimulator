import numpy as np

from pegasus.simulator.logic.vehicles.utils.usb_drivers import ThrottleState, StickState, PedalsState, GamepadState
import pegasus.simulator.logic.vehicles.utils.quadrotor_dynamics as quadrotor_dynamics


class UserInput():
    def __init__(self, controller_type = 0):

        self.controller_type = controller_type

        self.throttle = ThrottleState()
        self.stick = StickState()
        self.pedals = PedalsState()
        self.gamepad = GamepadState()

        self.quad = quadrotor_dynamics.quad(0.1, 1e100)
        self.curr_throttle_proportion = 0. 

    def get_input_reference(self):

        input_reference = [0.0 for i in range(9)]

        if self.controller_type == 0:
            self.stick.refresh()
            self.throttle.refresh()
            self.pedals.refresh()
            thrust = self.throttle.left * 0.5
            moments = np.array([
                [0.0 if abs(self.stick.x) < 0.1 else min(self.stick.x, 0.9)*0.005],
                [0.0 if abs(self.stick.y) < 0.1 else min(self.stick.y, 0.9)*0.005],
                [0.0 if abs(self.stick.z) < 0.1 else min(self.stick.z, 0.9)*0.005]
            ])
            u, w, F_new, M_new = self.quad.f2w(thrust, moments)

            w_mul = 6.

            thrust_max  = 50000
            thrust_add_max = 1000

            input_reference = [
                self.throttle.left * thrust_max - self.stick.x*thrust_add_max - self.stick.y*thrust_add_max, # front right
                self.throttle.left * thrust_max + self.stick.x*thrust_add_max + self.stick.y*thrust_add_max, # back left
                self.throttle.left * thrust_max + self.stick.x*thrust_add_max - self.stick.y*thrust_add_max, # front left
                self.throttle.left * thrust_max - self.stick.x*thrust_add_max + self.stick.y*thrust_add_max, # back right
                self.throttle.right*250*3, # ~2500 rpm in rad/s (*3)
                self.stick.x,
                -self.stick.x,
                self.stick.y,
                self.stick.z
            ]
        
        elif self.controller_type == 1 or True:
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
                self.stick.x,
                -self.stick.x,
                self.stick.y,
                self.stick.z
            ]

        return input_reference