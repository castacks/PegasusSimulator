#!/usr/bin/env python3
import pickle
import numpy as np

import os
from pathlib import Path


Kp_attitude = np.array([2.0, 2.0, 2.0])
Ki_attitude = np.array([0.0, 0.0, 0.0])
Kd_attitude = np.array([0.0, 0.0, 0.0])

Kp_rate = np.array([1.0, 1.0, 1.0])
Ki_rate = np.array([0.0, 0.0, 0.0])
Kd_rate = np.array([0.0, 0.0, 0.0])

Kp_altitude = 500000.0
Ki_altitude = 0.0
Kd_altitude = 0.0

Kp_velocity = np.array([1.0, 1.0])
Ki_velocity = np.array([0.0, 0.0])
Kd_velocity = np.array([0.0, 0.0])

motor_constant = 0.18315984515978836

curr_dir = str(Path(os.path.dirname(os.path.realpath(__file__))).resolve())

with open(curr_dir + "/../../extensions/pegasus.simulator/pegasus/simulator/logic/vehicles/utils/parameters/quad_pid_params.pkl","wb") as f:
    pickle.dump((Kp_attitude,
        Ki_attitude,
        Kd_attitude,
        Kp_rate,
        Ki_rate,
        Kd_rate,
        Kp_altitude,
        Ki_altitude,
        Kd_altitude,
        Kp_velocity,
        Ki_velocity,
        Kd_velocity,
        motor_constant), f)