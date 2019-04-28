import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import scipy


class QuadRotorEnv(gym.Env):

    def __init__(self, m, L, r, prop_diameter, prop_pitch, b=0.0245):
        self.max_speed = 9000
        self.min_speed = 0
        self.max_angular_speed = np.pi/6
        self.min_x = 0
        self.max_x = 20
        self.min_y = 0
        self.max_y = 20
        self.min_z = 0
        self.max_z = 20
        self.min_angle = -np.pi/6
        self.max_angle = np.pi/6
        self.g = 9.81
        self.m = m
        self.L = L
        self.prop_diameter = prop_diameter
        self.prop_pitch = prop_pitch

        self.action_space = spaces.Box(low=self.min_speed, high=self.max_speed,
                                       shape=(4,), dtype=np.float32)

        """
        high = np.array([
                   self.x_threshold * 2,
                   np.finfo(np.float32).max,
                   self.theta_threshold_radians * 2,
                   np.finfo(np.float32).max])
        """
        # TODO: fix observation_space bound - @nimbus state[]
        self.observation_space = spaces.Box(low=-1000, high=1000,
                                            shape=(12,), dtype=np.float32)

        self.motor1 = Propeller(self.prop_diameter, self.prop_pitch)
        self.motor2 = Propeller(self.prop_diameter, self.prop_pitch)
        self.motor3 = Propeller(self.prop_diameter, self.prop_pitch)
        self.motor4 = Propeller(self.prop_diameter, self.prop_pitch)

        # moment of Inertia
        Ixx = (2*self.m*self.r**2)/5 + 2*self.m*(self.L**2)
        Iyy = Ixx
        Izz = (2*self.m*self.r**2)/5 + 4*self.m*(self.L**2)
        self.In = np.array([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])
        self.invI = np.linalg.inv(self.In)

        self.ode = scipy.integrate.ode(self._state_dot).set_integrator('vode', nsteps=500, method='bdf')

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.motor1.set_speed(action[0])
        self.motor2.set_speed(action[1])
        self.motor3.set_speed(action[2])
        self.motor4.set_speed(action[3])

        self.state = self._get_new_state()
        done = self._crashed() or self._reach_target()
        reward = self.reward()

        return self.state, reward, done, {}

    def reward(self):
        if self.crashed():
            return -5
        d = np.sqrt((self.state[0] - self.target[0])**2 + \
                    (self.state[1] - self.target[1])**2 + \
                    (self.state[2] - self.target[2])**2)
        dist_reward = self.reward_list[(np.digitize(d, self.reward_bin)-1)]
        pose_reward = -((abs(self.state[3:6])/self.max_speed).sum() + \
                        (abs(self.state[6:8])/np.pi).sum() + \
                        (abs(self.state[9:12])/self.max_angular_speed).sum())

        reward = 0.4*dist_reward + 0.3*pose_reward

        if self.reach_target():
            return reward + 1

        return reward

    def reset(self):
        # random spawn target
        self.target = np.zeros((3,))
        self.target[0] = np.random.uniform(low=self.min_x, high=self.max_x)
        self.target[1] = np.random.uniform(low=self.min_y, high=self.max_y)
        self.target[2] = np.random.uniform(low=self.min_z, high=self.max_z)

        # random spawn agent
        self.state = np.zeros(self.observation_space.shape)
        self.state[0] = np.random.uniform(low=self.min_x, high=self.max_x)
        self.state[1] = np.random.uniform(low=self.min_y, high=self.max_y)
        self.state[2] = np.random.uniform(low=self.min_z, high=self.max_z)
        self.state[6:9] = np.random.uniform(low=self.min_angle, high=self.max_angle, size=3)

        # reset propeller
        self.motor1.reset()
        self.motor2.reset()
        self.motor3.reset()
        self.motor4.reset()

        return np.array(self.state)

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def _rotation_matrix(self, angles):
        ct = np.cos(angles[0])
        cp = np.cos(angles[1])
        cg = np.cos(angles[2])
        st = np.sin(angles[0])
        sp = np.sin(angles[1])
        sg = np.sin(angles[2])
        R_x = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
        R_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        R_z = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
        R = np.dot(R_z, np.dot(R_y, R_x))
        return R

    def _normalize(self, angle):
        return ((angle + np.pi) % (2*np.pi) - np.pi)

    def _state_dot(self):
        state_dot = np.zeros(12)
        # The velocities(t+1 x_dots equal the t x_dots)
        state_dot[0] = self.state[3]
        state_dot[1] = self.state[4]
        state_dot[2] = self.state[5]
        # The acceleration
        x_dotdot = np.array([0, 0, -self.m*self.g]) + \
            np.dot(self._rotation_matrix(self.state[6:9]),
                   np.array([0, 0, (self.motor1.thrust + self.motor2.thrust \
                            + self.motor3.thrust + self.motor4.thrust)]))/self.m

        state_dot[3] = x_dotdot[0]
        state_dot[4] = x_dotdot[1]
        state_dot[5] = x_dotdot[2]
        # The angular rates(t+1 theta_dots equal the t theta_dots)
        state_dot[6] = self.state[9]
        state_dot[7] = self.state[10]
        state_dot[8] = self.state[11]
        # The angular accelerations
        omega = self.state[9:12]
        tau = np.array([self.L*(self.motor1.thrust-self.motor3.thrust),
                        self.L*(self.motor2.thrust-self.motor4.thrust),
                        self.b*(self.motor1.thrust-self.motor2.thrust + \
                                self.motor3.thrust-self.motor4.thrust)])

        omega_dot = np.dot(self.invI, (tau - np.cross(omega, np.dot(self.I, omega))))
        state_dot[9] = omega_dot[0]
        state_dot[10] = omega_dot[1]
        state_dot[11] = omega_dot[2]
        return state_dot

    def _get_new_state(self, dt=0.01):

        self.ode.set_initial_value(self.state, 0)
        self.state = self.ode.integrate(self.ode.t + dt)
        self.state[6:9] = self.normalize(self.state[6:9])
        self.state[2] = max(0, self.state[2])

        return self.state

    def _crashed(self):
        # check if agent is crashed
        # in another word, touch the ground with large velocity
        if (self.state[2] < (self.L + 0.05)):
            return True
        else:
            return False

    def _reach_target(self):
        # check if agent reach target postion
        d = np.sqrt((self.state[0] - self.target[0])**2 + \
                    (self.state[1] - self.target[1])**2 + \
                    (self.state[2] - self.target[2])**2)
        if d < 0.05:
            return True
        else:
            return False


class Propeller:
    def __init__(self, diameter, pitch, thrust_unit='N'):
        self.d = diameter
        self.pitch = pitch
        self.thrust_unit = thrust_unit
        self.reset()

    def set_speed(self, speed):
        self.speed = speed
        self.update_thrust()

    def update_thrust(self):
        self.thrust = 4.392e-8*self.speed*(self.d**3.5)/(np.sqrt(self.pitch))
        self.thrust = self.thrust*(4.23e-4 * self.speed * self.pitch)
        if self.thrust_unit == 'kg':
            self.thrust_unit = self.thrust*0.101972

    def reset(self):
        self.speed = 0
        self.thrust = 0
