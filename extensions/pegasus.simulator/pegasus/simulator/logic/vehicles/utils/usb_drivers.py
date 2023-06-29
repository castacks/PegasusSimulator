#!/usr/bin/env python3
"""
| File: usb_drivers.py
| Author: Ian Higgins (ihiggins@andrew.cmu.edu)
| Description: Driver for Saitek Joystick, Throttle, and Pedals
| License: BSD-3-Clause. Copyright (c) 2023. All rights reserved.
"""
import numpy as np
import evdev

import matplotlib.pyplot as plt

class Device(evdev.device.InputDevice):
    def __init__(self, device_path = None, device_type = None):
        if device_path is None:
            device_path = self.get_device_path(device_type)
        if device_path is not None:
            super().__init__(device_path)
        self.device_path = device_path

    def get_device_path(self, device_type):
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        for d in devices:
            # print(d.name)
            if device_type == 'stick' and d.name == "Mad Catz Saitek Pro Flight X-56 Rhino Stick":
                return d
            elif device_type == 'throttle' and d.name == "Mad Catz Saitek Pro Flight X-56 Rhino Throttle":
                return d
            elif device_type == 'pedals' and d.name == "Saitek Saitek Pro Flight Rudder Pedals":
                return d
            elif device_type == 'gamepad' and d.name == "Logitech Logitech Dual Action":
                return d
        
    def get_newest_events(self):
        events_tmp = []
        try:
            for event in self.read():
                events_tmp.append(event)
        except BlockingIOError:
            return []
        events_tmp.reverse()
        events = []
        for e in events_tmp:
            too_late = False
            for et in events:
                if e.code==et.code and e.type==et.type:
                    too_late = True
                    break
            if not too_late:
                events.append(e)
        events.reverse()
        return events

class UsbState():
    def __init__(self):
        pass
    def scale(self, raw_val, max_val):
        return float(raw_val / max_val)
    def scale_inverted(self, raw_val, max_val):
        return float((max_val - raw_val) / max_val)
    def scale_center(self, raw_val, max_val):
        return float((raw_val - max_val/2) / (max_val/2))
    def scale_center_inverted(self, raw_val, max_val):
        return float((max_val/2 - raw_val) / (max_val/2))

class ThrottleState(UsbState):
    def __init__(self, device_path = None):
        super().__init__()
        
        self.device = Device(device_type='throttle', device_path=device_path)

        self._throttle_max = 2**10 - 1

        self._left_raw = 0
        self._right_raw = 0

        self.left = 0.
        self.right = 0.
        
        self.SLD = 0
    
    def refresh(self):
        if self.device.device_path is not None:
            events = self.device.get_newest_events()

            for event in events:
                # print(event)
                if event.code == 0 and event.type == 3:
                    self._left_raw = event.value
                    self.left = self.scale_inverted(event.value, self._throttle_max)
                elif event.code == 1 and event.type == 3:
                    self._right_raw = event.value
                    self.right = self.scale_inverted(event.value, self._throttle_max)
                elif event.code == 720 and event.type == 1:
                    self.SLD = event.value

class StickState(UsbState):
    def __init__(self, device_path = None):
        super().__init__()

        self.device = Device(device_type='stick', device_path=device_path)

        self._stick_max = 2**16 - 1
        self._twist_max = 2**12 - 1

        self._right_raw = self._stick_max//2        # center
        self._back_raw = self._stick_max//2         # center
        self._clockwise_raw = self._twist_max//2    # center

        self.x = 0. # center
        self.y = 0. # center
        self.z = 0. # center
    
    def refresh(self):
        if self.device.device_path is not None:
            events = self.device.get_newest_events()
            for event in events:
                # print(event)
                if event.code == 0 and event.type == 3:
                    self._right_raw = event.value
                    self.x = self.scale_center(event.value, self._stick_max)
                elif event.code == 1 and event.type == 3:
                    self._back_raw = event.value
                    self.y = self.scale_center_inverted(event.value, self._stick_max)
                elif event.code == 5 and event.type == 3:
                    self._clockwise_raw = event.value
                    self.z = self.scale_center_inverted(event.value, self._twist_max)

class PedalsState(UsbState):
    def __init__(self, device_path = None):
        super().__init__()
        
        self.device = Device(device_type='pedals', device_path=device_path)

        self._right_max = 2**9 - 1
        self._brake_max = 2**7 - 1

        self._pedals_raw = self._right_max//2   # center
        self._brake_right_raw = 0               # max up
        self._brake_left_raw  = 0               # max up
    
        self.right = 0.         # center
        self.brake_right = 0.   # unpressed
        self.brake_left  = 0.   # unpressed

    def refresh(self):
        if self.device.device_path is not None:
            events = self.device.get_newest_events()
            for event in events:
                # print(event)

                if event.code == 5 and event.type == 3:
                    self._pedals_raw = event.value
                    self.right = self.scale_center(event.value, self._right_max)
                elif event.code == 0 and event.type == 3:
                    self._brake_left_raw = event.value
                    self.brake_left = self.scale(event.value, self._brake_max)
                elif event.code == 1 and event.type == 3:
                    self._brake_right_raw = event.value
                    self.brake_right = self.scale(event.value, self._brake_max)

class GamepadState(UsbState):
    def __init__(self, device_path = None):
        super().__init__()
        
        self.device = Device(device_type='gamepad', device_path=device_path)

        self._stick_max = 2**8 - 1

        self.left_x_raw  = self._stick_max // 2 # center
        self.left_y_raw  = self._stick_max // 2 # center
        self.right_x_raw = self._stick_max // 2 # center
        self.right_y_raw = self._stick_max // 2 # center

        self.left_x  = 0. # center
        self.left_y  = 0. # center
        self.right_x = 0. # center
        self.right_y = 0. # center

        self.b1 = 0
        self.b2 = 0
        self.b3 = 0
        self.b4 = 0
        self.b5 = 0
        self.b6 = 0
        self.b7 = 0
        self.b8 = 0
        self.b9 = 0
        self.b10 = 0

        # d-pad (-1 is left, 1 is right; 1 is up, -1 is down)
        self.dx = 0
        self.dy = 0

    def refresh(self):
        if self.device.device_path is not None:
            events = self.device.get_newest_events()
            for event in events:
                print(event)

                # left stick
                if event.code == 0 and event.type == 3:
                    self.left_x_raw = event.value
                    self.left_x = self.scale_center(event.value, self._stick_max)
                elif event.code == 1 and event.type == 3:
                    self.left_y_raw = event.value
                    self.left_y = self.scale_center_inverted(event.value, self._stick_max)

                # right stick
                elif event.code == 2 and event.type == 3:
                    self.right_x_raw = event.value
                    self.right_x = self.scale_center(event.value, self._stick_max)
                elif event.code == 5 and event.type == 3:
                    self.right_y_raw = event.value
                    self.right_y = self.scale_center_inverted(event.value, self._stick_max)

                # buttons
                elif event.code == 288 and event.type == 1:
                    self.b1 = event.value
                elif event.code == 289 and event.type == 1:
                    self.b2 = event.value
                elif event.code == 290 and event.type == 1:
                    self.b3 = event.value
                elif event.code == 291 and event.type == 1:
                    self.b4 = event.value
                elif event.code == 292 and event.type == 1:
                    self.b5 = event.value
                elif event.code == 293 and event.type == 1:
                    self.b6 = event.value
                elif event.code == 294 and event.type == 1:
                    self.b7 = event.value
                elif event.code == 295 and event.type == 1:
                    self.b8 = event.value
                elif event.code == 296 and event.type == 1:
                    self.b9 = event.value
                elif event.code == 297 and event.type == 1:
                    self.b10= event.value
                
                # d-pad
                elif event.code == 16 and event.type == 3:
                    self.dx = event.value
                elif event.code == 17 and event.type == 3:
                    self.dy = -event.value


def main():
    import time
    throttle = ThrottleState()
    stick = StickState()
    pedals = PedalsState()
    gamepad = GamepadState()

    fig, ax = plt.subplots(figsize=(6, 6))

    while True:
        stick.refresh()
        throttle.refresh()
        pedals.refresh()
        gamepad.refresh()

        time.sleep(.1)
        # print()
        # print('x',stick.x)
        # print('y',stick.y)
        # print('z',stick.z)
        # print('left',throttle.left)
        # print('right',throttle.right)
        # print(f'{pedals.brake_left=}')
        # print(f'{pedals.brake_right=}')
        # print(f'{pedals.right=}')
        # ax.cla()
        # ax.plot(stick.x,stick.y, 'yx')
        # ax.plot(0,0, 'g.')
        # ax.set_xlim([-2.5,2.5])
        # ax.set_ylim([-2.5,2.5])
        # plt.pause(0.0001)


if __name__ == '__main__':
    main()