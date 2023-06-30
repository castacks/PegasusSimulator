import numpy as np
from .parameters.uav_parameters import UAVParams
from .utility.rotations import Quaternion2Rotation, Quaternion2Euler
from .utility.rotations import Euler2Rotation



class UAVDynamics:

    def __init__(self, uav_params):

        assert isinstance(uav_params, UAVParams)

        self.uav = uav_params

        self.state = [
            self.uav.px0,
            self.uav.py0,
            self.uav.pz0,
            self.uav.u0,
            self.uav.v0,
            self.uav.w0,
            self.uav.e0,
            self.uav.e1,
            self.uav.e2,
            self.uav.e3,
            self.uav.p0,
            self.uav.q0,
            self.uav.r0
        ]

        self.wind = np.array([[0.0], [0.0], [0.0]])
        self.forces = np.array([[0.0], [0.0], [0.0]])

        self.v_air = self.uav.u0
        self.alpha = 0
        self.beta = 0

        self.t_gps = 1e6
        self.gps_eta_x = 0.0
        self.gps_eta_y = 0.0
        self.gps_eta_z = 0.0
        
    def _forces_moments(self, delta):



        v_air = self.state[3:6]
        ur = v_air[0]
        vr = v_air[1]
        wr = v_air[2]

        self.v_air = np.sqrt(ur ** 2 + vr ** 2 + wr ** 2)

        if ur == 0:
            self.alpha = np.sign(wr) * np.pi / 2
        else:
            self.alpha = np.arctan(wr / ur)

        tmp = np.sqrt(ur ** 2 + wr ** 2)
        if tmp == 0:
            self.beta = np.sign(vr) * np.pi / 2
        else:
            self.beta = np.arcsin(vr / tmp)




        p = self.state[10]
        q = self.state[11]
        r = self.state[12]

        delta_e = delta[0]
        delta_a = delta[1]
        delta_r = delta[2]
        delta_t = delta[3]

        rot = Quaternion2Rotation(self.state[6:10])

        fx = 0
        fy = 0
        fz = 0

        qbar = 0.5 * self.uav.rho * self.v_air ** 2
        c_alpha = np.cos(self.alpha)
        s_alpha = np.sin(self.alpha)

        p_ndim = p * self.uav.b / (2 * self.v_air)
        q_ndim = q * self.uav.c / (2 * self.v_air)
        r_ndim = r * self.uav.b / (2 * self.v_air)

        tmp1 = np.exp(-self.uav.M * (self.alpha - self.uav.alpha0))
        tmp2 = np.exp(self.uav.M * (self.alpha + self.uav.alpha0))
        sigma = (1 + tmp1 + tmp2) / ((1 + tmp1) * (1 + tmp2))

        cl = ((1 - sigma) * (self.uav.C_L_0 + self.uav.C_L_alpha * self.alpha)
              + sigma * 2 * np.sign(self.alpha) * s_alpha ** 2 * c_alpha)

        cd = (self.uav.C_D_p
              + ((self.uav.C_L_0 + self.uav.C_L_alpha * self.alpha) ** 2
                 / (np.pi * self.uav.e * self.uav.AR)))

        f_lift = qbar * self.uav.S_wing * (
                cl
                + self.uav.C_L_q * q_ndim
                + self.uav.C_L_delta_e * delta_e
        )

        f_drag = qbar * self.uav.S_wing * (
                cd
                + self.uav.C_D_q * q_ndim
                + self.uav.C_D_delta_e * delta_e
        )

        fx = fx - c_alpha * f_drag + s_alpha * f_lift
        fz = fz - s_alpha * f_drag - c_alpha * f_lift

        fy = fy + qbar * self.uav.S_wing * (
                self.uav.C_Y_0
                + self.uav.C_Y_beta * self.beta
                + self.uav.C_Y_p * p_ndim
                + self.uav.C_Y_r * r_ndim
                + self.uav.C_Y_delta_a * delta_a
                + self.uav.C_Y_delta_r * delta_r
        )

        My = qbar * self.uav.S_wing * self.uav.c * (
                self.uav.C_m_0
                + self.uav.C_m_alpha * self.alpha
                + self.uav.C_m_q * q_ndim
                + self.uav.C_m_delta_e * delta_e
        )

        Mx = qbar * self.uav.S_wing * self.uav.b * (
                self.uav.C_ell_0
                + self.uav.C_ell_beta * self.beta
                + self.uav.C_ell_p * p_ndim
                + self.uav.C_ell_r * r_ndim
                + self.uav.C_ell_delta_a * delta_a
                + self.uav.C_ell_delta_r * delta_r
        )

        Mz = qbar * self.uav.S_wing * self.uav.b * (
                self.uav.C_n_0
                + self.uav.C_n_beta * self.beta
                + self.uav.C_n_p * p_ndim
                + self.uav.C_n_r * r_ndim
                + self.uav.C_n_delta_a * delta_a
                + self.uav.C_n_delta_r * delta_r
        )

        p_thrust, p_torque = self._motor_thrust_torque(self.v_air, delta_t)
        fx += p_thrust
        Mx += -p_torque

        self.forces[0] = fx
        self.forces[1] = fy
        self.forces[2] = fz

        return fx, fy, fz, Mx, My, Mz

    def _motor_thrust_torque(self, va, delta_t):

        v_in = self.uav.V_max * delta_t

        a = (self.uav.C_Q0 * self.uav.rho
             * np.power(self.uav.D_prop, 5) / ((2. * np.pi) ** 2))

        b = ((self.uav.C_Q1 * self.uav.rho
              * np.power(self.uav.D_prop, 4) / (2. * np.pi)) * va
             + self.uav.KQ ** 2 / self.uav.R_motor)

        c = (self.uav.C_Q2 * self.uav.rho
             * np.power(self.uav.D_prop, 3) * va ** 2
             - (self.uav.KQ / self.uav.R_motor) * v_in
             + self.uav.KQ * self.uav.i0)

        omega = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        J = 2 * np.pi * self.v_air / (omega * self.uav.D_prop)

        C_T = self.uav.C_T2 * J ** 2 + self.uav.C_T1 * J + self.uav.C_T0
        C_Q = self.uav.C_Q2 * J ** 2 + self.uav.C_Q1 * J + self.uav.C_Q0

        n = omega / (2 * np.pi)
        thrust = self.uav.rho * n ** 2 * np.power(self.uav.D_prop, 4) * C_T
        torque = self.uav.rho * n ** 2 * np.power(self.uav.D_prop, 5) * C_Q
        
        return thrust, 0.#torque

