import numpy as np
import time
import pickle

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.prev_error = 0
    
    def control(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

class QuadrotorController:
    def __init__(self, attitude_PID, rate_PID, altitude_PID, velocity_PID, arm_length, motor_constant):
        self.attitude_PID = attitude_PID
        self.rate_PID = rate_PID
        self.altitude_PID = altitude_PID
        self.velocity_PID = velocity_PID
        self.arm_length = arm_length
        self.motor_constant = motor_constant
    
    def control(self, desired_position, desired_velocity, current_position, current_velocity, current_attitude, current_rate, dt):
        # Altitude control (Z)
        altitude_error = desired_position[2] - current_position[2]
        desired_thrust = self.altitude_PID.control(altitude_error, dt)
        
        # Lateral velocity control (X, Y)
        velocity_error = desired_velocity[:2] - current_velocity[:2]
        desired_tilt_angles = self.velocity_PID.control(velocity_error, dt)
        desired_attitude = np.hstack([desired_tilt_angles, desired_position[3]])  # roll, pitch, yaw
        
        # Attitude control
        attitude_error = desired_attitude - current_attitude
        desired_rate = self.attitude_PID.control(attitude_error, dt)
        
        # Rate control
        rate_error = desired_rate - current_rate
        torques = self.rate_PID.control(rate_error, dt)
        
        # Convert torques and thrust to rotor angular velocities
        inputs = np.array([
            [1, 1, 1, 1       ],
            [-self.arm_length, self.arm_length, self.arm_length, -self.arm_length],
            [-self.arm_length, self.arm_length, -self.arm_length, self.arm_length],
            [-self.motor_constant, -self.motor_constant, self.motor_constant, self.motor_constant]
        ])

        torques_thrust = np.concatenate(([desired_thrust], torques))
        angular_velocities_sq = np.linalg.inv(inputs) @ torques_thrust
        
        return np.sqrt(np.maximum(angular_velocities_sq, 0))
    
class Control:

    def __init__(self):
        
        # Parameters
        # self.Kp_attitude = np.array([2.0, 2.0, 2.0])
        # self.Ki_attitude = np.array([0.0, 0.0, 0.0])
        # self.Kd_attitude = np.array([0.0, 0.0, 0.0])

        # self.Kp_rate = np.array([1.0, 1.0, 1.0])
        # self.Ki_rate = np.array([0.0, 0.0, 0.0])
        # self.Kd_rate = np.array([0.0, 0.0, 0.0])

        # self.Kp_altitude = 2.0
        # self.Ki_altitude = 0.0
        # self.Kd_altitude = 0.0

        # self.Kp_velocity = np.array([1.0, 1.0])
        # self.Ki_velocity = np.array([0.0, 0.0])
        # self.Kd_velocity = np.array([0.0, 0.0])

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


        self.arm_length = 2.  # meters
        

        # Create controllers
        self.attitude_PID = PIDController(self.Kp_attitude, self.Ki_attitude, self.Kd_attitude)
        self.rate_PID = PIDController(self.Kp_rate, self.Ki_rate, self.Kd_rate)
        self.altitude_PID = PIDController(self.Kp_altitude, self.Ki_altitude, self.Kd_altitude)
        self.velocity_PID = PIDController(self.Kp_velocity, self.Ki_velocity, self.Kd_velocity)

        self.quadrotor_controller = QuadrotorController(self.attitude_PID, self.rate_PID, self.altitude_PID, self.velocity_PID, self.arm_length, self.motor_constant)


    def control(self, state, euler_angles, desired_altitude, desired_yaw, desired_velocity, dt):

        # State 
        current_position = np.array([state.position[0], state.position[1], state.position[2], euler_angles[2]])  # x, y, z, yaw
        current_velocity = np.array([state.linear_body_velocity[0], state.linear_body_velocity[1], state.linear_body_velocity[2]])  # x, y, z
        current_attitude = np.array([euler_angles[0], euler_angles[1], euler_angles[2]])  # roll, pitch, yaw (radians)
        current_rate = np.array([state.angular_velocity[0], state.angular_velocity[1], state.angular_velocity[2]])  # roll rate, pitch rate, yaw rate (radians/sec)

        # Controls
        desired_position = np.array([0.0, 0.0, desired_altitude, desired_yaw])  # x, y, z, yaw

        # Call the control method to get the rotor angular velocities
        rotor_angular_velocities = self.quadrotor_controller.control(
            desired_position, desired_velocity, current_position, current_velocity, current_attitude, current_rate, dt)
        
        return rotor_angular_velocities