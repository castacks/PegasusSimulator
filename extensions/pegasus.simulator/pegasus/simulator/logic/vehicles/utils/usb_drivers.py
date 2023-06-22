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
        super().__init__(device_path)

    def get_device_path(self, device_type):
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        for d in devices:
            if device_type == 'stick' and d.name == "Mad Catz Saitek Pro Flight X-56 Rhino Stick":
                return d
            elif device_type == 'throttle' and d.name == "Mad Catz Saitek Pro Flight X-56 Rhino Throttle":
                return d
            elif device_type == 'pedals' and d.name == "Saitek Saitek Pro Flight Rudder Pedals":
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
        return float((raw_val - max_val//2) / (max_val//2))
    def scale_center_inverted(self, raw_val, max_val):
        return float((max_val//2 - raw_val) / (max_val//2))

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
                print(self.SLD)


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


def main():
    import time
    throttle = ThrottleState()
    stick = StickState()
    pedals = PedalsState()

    fig, ax = plt.subplots(figsize=(6, 6))

    while True:
        stick.refresh()
        throttle.refresh()
        pedals.refresh()

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