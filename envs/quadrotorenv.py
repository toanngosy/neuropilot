import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D
import sys


class QuadRotorEnv_v0(gym.Env):

    def __init__(self, m=1.2, L=0.5, r=0.2, prop_diameter=10, prop_pitch=4.5):
        self.max_speed = 1
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
        self.b = 0.0245
        self.m = m
        self.L = L
        self.r = r
        self.prop_diameter = prop_diameter
        self.prop_pitch = prop_pitch

        self.action_space = spaces.Box(low=self.min_speed, high=self.max_speed,
                                       shape=(4,), dtype=np.float32)

        obs_high = np.array([
                   self.max_x, self.max_y, self.max_z,
                   np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                   np.pi/2, np.pi/2, np.pi/2,
                   np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max])
        obs_low = np.array([
                   self.min_x, self.min_y, self.min_z,
                   np.finfo(np.float32).min, np.finfo(np.float32).min, np.finfo(np.float32).min,
                   -np.pi/2, -np.pi/2, -np.pi/2,
                   np.finfo(np.float32).min, np.finfo(np.float32).min, np.finfo(np.float32).min])

        # TODO: fix observation_space bound - @nimbus state[]
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

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

        # set solver
        self.ode = integrate.ode(self._state_dot).set_integrator('vode', nsteps=500, method='bdf')

        # set random seed
        self.seed()

        # reset env
        self.reset()

        # set viewer
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.motor1.set_speed(action[0])
        self.motor2.set_speed(action[1])
        self.motor3.set_speed(action[2])
        self.motor4.set_speed(action[3])

        self.state = self._get_new_state()
        done = self._crashed_box() or self._reach_target()
        reward = self.reward()

        # increase num_step
        self.num_step += 1
        if done is True:
            print(self.num_step)
        return self.state, reward, done, {}

    def reward(self):
        if self._crashed():
            return -1
        #print(self.state)
        old_dist = self.d
        self.d = np.sqrt((self.state[0] - self.target[0])**2 + \
                    (self.state[1] - self.target[1])**2 + \
                    (self.state[2] - self.target[2])**2)
        #self.reward_bin = np.array([0, d/3, 2*d/3, d])
        #self.reward_list = np.array([-0.75, -0.5, 0, 0.5])
        #dist_reward = self.reward_list[(np.digitize(d, self.reward_bin)-1)]
        dist = self.d
        pose_reward = -(np.mean(abs(self.state[6:8]))/(np.pi/2))

        #time_reward = -0.01*self.num_step
        dist_reward = -(dist - old_dist)
        reward = (100*dist_reward + pose_reward)/10
        #reward = dist_reward
        #print(reward, dist_reward, pose_reward)
        if self._reach_target():
            return 100

        return reward

    def reset(self):
        # random spawn target
        self.target = np.zeros((3,))
        #self.target[0] = np.random.uniform(low=self.min_x, high=self.max_x)
        #self.target[1] = np.random.uniform(low=self.min_y, high=self.max_y)
        #self.target[2] = np.random.uniform(low=self.min_z, high=self.max_z)
        self.target = np.array([10,10,10])

        # random spawn agent
        self.state = np.zeros(self.observation_space.shape)
        #self.state[0] = np.random.uniform(low=self.min_x, high=self.max_x)
        #self.state[1] = np.random.uniform(low=self.min_y, high=self.max_y)
        #self.state[2] = np.random.uniform(low=self.min_z, high=self.max_z)
        self.state[0] = 15
        self.state[1] = 15
        self.state[2] = 15
        self.state[6:9] = np.random.uniform(low=self.min_angle, high=self.max_angle, size=3)

        # reset propeller
        self.motor1.reset()
        self.motor2.reset()
        self.motor3.reset()
        self.motor4.reset()

        # reset no. step
        self.num_step = 0

        self.d = np.sqrt((self.state[0] - self.target[0])**2 + \
                    (self.state[1] - self.target[1])**2 + \
                    (self.state[2] - self.target[2])**2)
        return np.array(self.state)

    # TODO: current state: matplotlib, todo: upward indicator. openGL? Unity?
    def render(self, mode='human'):

        if self.viewer is None:
            #self.viewer, self.ax = plt.subplots(figsize=(40,10), ncols=4, nrows=1)
            self.viewer = plt.figure(figsize=(40,40))
            # set up 2D postion view
            # XY PLAN view
            self.ax1 = self.viewer.add_subplot(222)
            self.ax1.set_xlim([self.min_x-5, self.max_x+5])
            self.ax1.set_xlabel('X')
            self.ax1.set_ylim([self.min_y-5, self.max_y+5])
            self.ax1.set_ylabel('Y')
            self.ax1.set_title("XY plan view")
            self.agentplotxy, = self.ax1.plot([], [], marker='o', color='blue', markersize=6, antialiased=False)
            self.targetplotxy, = self.ax1.plot([], [],  marker='o', color='red', markersize=6, antialiased=False)
            # XZ PLAN view
            self.ax2 = self.viewer.add_subplot(223)
            self.ax2.set_xlim([self.min_x-5, self.max_x+5])
            self.ax2.set_xlabel('X')
            self.ax2.set_ylim([self.min_z-5, self.max_z+5])
            self.ax2.set_ylabel('Z')
            self.ax2.set_title("XZ plan view")
            self.agentplotxz, = self.ax2.plot([], [], marker='o', color='blue', markersize=6, antialiased=False)
            self.targetplotxz, = self.ax2.plot([], [],  marker='o', color='red', markersize=6, antialiased=False)

            # ZY PLAN view
            self.ax3 = self.viewer.add_subplot(224)
            self.ax3.set_xlim([self.min_y-5, self.max_y+5])
            self.ax3.set_xlabel('Y')
            self.ax3.set_ylim([self.min_z-5, self.max_z+5])
            self.ax3.set_ylabel('Z')
            self.ax3.set_title("YZ plan view")
            self.agentplotyz, = self.ax3.plot([], [], marker='o', color='blue', markersize=6, antialiased=False)
            self.targetplotyz, = self.ax3.plot([], [],  marker='o', color='red', markersize=6, antialiased=False)



            #set up 3D view
            self.ax3D = self.viewer.add_subplot(221, projection='3d')
            self.ax3D.set_xlim3d([self.min_x, self.max_x])
            self.ax3D.set_xlabel('X')
            self.ax3D.set_ylim3d([self.min_y, self.max_y])
            self.ax3D.set_ylabel('Y')
            self.ax3D.set_zlim3d([self.min_z, self.max_z])
            self.ax3D.set_zlabel('Z')
            self.ax3D.set_title('Quadrotor Simulation')
            self.l1,  = self.ax3D.plot([], [], [], color='blue', linewidth=3, antialiased=False)
            self.l2,  = self.ax3D.plot([], [], [], color='blue', linewidth=3, antialiased=False)
            self.hub, = self.ax3D.plot([], [], [], marker='^', color='blue', markersize=6, antialiased=False)
            self.target3Dplot, = self.ax3D.plot([], [], [], marker='o', color='red', markersize=6, antialiased=False)
            # Add manual control
            self.viewer.canvas.mpl_connect('key_press_event', self._keypress_routine)

        #Update 3D model 
        R = self._rotation_matrix(self.state[6:9])
        L = self.L
        points = np.array([[-L, 0, 0], [L, 0, 0], [0, -L, 0], [0, L, 0], [0, 0, 0], [0, 0, 0]]).T
        points = np.dot(R, points)
        points[0, :] += self.state[0]
        points[1, :] += self.state[1]
        points[2, :] += self.state[2]

        self.l1.set_data(points[0, 0:2], points[1, 0:2])
        self.l1.set_3d_properties(points[2, 0:2])

        self.l2.set_data(points[0, 2:4], points[1, 2:4])
        self.l2.set_3d_properties(points[2, 2:4])

        self.hub.set_data(points[0, 5], points[1, 5])
        self.hub.set_3d_properties(points[2, 5])
        
        self.target3Dplot.set_data(self.target[0], self.target[1])
        self.target3Dplot.set_3d_properties(self.target[2])


        #Update XY view 

        self.agentplotxy.set_data([self.state[0]], [self.state[1]])
        self.targetplotxy.set_data([self.target[0]], [self.target[1]])

        #Update XZ view
        self.agentplotxz.set_data([self.state[0]], [self.state[2]])
        self.targetplotxz.set_data([self.target[0]], [self.target[2]])

        #Update YZ view
        self.agentplotyz.set_data([self.state[1]], [self.state[2]])
        self.targetplotyz.set_data([self.target[1]], [self.target[2]])
        # TODO: add hub indicator

        plt.pause(5)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

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
                   np.array([0, 0, (self.motor1.thrust + self.motor2.thrust
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
                        self.b*(self.motor1.thrust-self.motor2.thrust +
                                self.motor3.thrust-self.motor4.thrust)])

        omega_dot = np.dot(self.invI, (tau - np.cross(omega, np.dot(self.In, omega))))
        state_dot[9] = omega_dot[0]
        state_dot[10] = omega_dot[1]
        state_dot[11] = omega_dot[2]
        return state_dot

    def _get_new_state(self, dt=0.01):

        self.ode.set_initial_value(self.state, 0)
        self.state = self.ode.integrate(self.ode.t + dt)
        self.state[6:9] = self._normalize(self.state[6:9])
        self.state[2] = max(0, self.state[2])

        return self.state

    #Since we used negative reward function, we removed crash mechanism to prevent agent from suicide 
    '''
    def _crashed_box(self):
        # check if agent is out of the box
        if ((self.state[0] > self.max_x or self.state[0] < self.min_x)
            or (self.state[1] > self.max_y or self.state[1] < self.min_y)
            or (self.state[2] > self.max_z or self.state[2] < self.min_z)):
            return True
        else:
            return False

    def _crashed(self):
        # check if agent is crashed
        # in another word, touch the ground with large velocity
        if (self.state[2] < (self.L + 0.05)):
            return True
        else:
            return False
    '''
    def _reach_target(self):
        # check if agent reach target postion
        d = np.linalg.norm(self.target - self.state[0:3])
        if d < 0.5:
            print("reach!")
            return True
        else:
            return False

    def _keypress_routine(self, event):
        sys.stdout.flush()
        if event.key == 'w':
            y = list(self.ax.get_ylim3d())
            y[0] += 1
            y[1] += 1
            self.ax.set_ylim3d(y)

        if event.key == 'x':
            y = list(self.ax.get_ylim3d())
            y[0] -= 1
            y[1] -= 1
            self.ax.set_ylim3d(y)

        if event.key == 'd':
            x = list(self.ax.get_xlim3d())
            x[0] += 1
            x[1] += 1
            self.ax.set_xlim3d(x)

        if event.key == 'a':
            x = list(self.ax.get_xlim3d())
            x[0] -= 1
            x[1] -= 1
            self.ax.set_xlim3d(x)


class QuadRotorEnv_v1(gym.Env):

    def __init__(self, m=1.2, L=0.5, r=0.2, prop_diameter=10, prop_pitch=4.5):
        self.max_speed = 1
        self.min_speed = 0
        self.max_angular_speed = np.pi/6
        self.min_x = 0
        self.max_x = 5
        self.min_y = 0
        self.max_y = 5
        self.min_z = 0
        self.max_z = 5
        self.min_angle = -np.pi/6
        self.max_angle = np.pi/6
        self.g = 9.81
        self.b = 0.0245
        self.m = m
        self.L = L
        self.r = r
        self.prop_diameter = prop_diameter
        self.prop_pitch = prop_pitch

        self.target = None
        self.total_reward = 0

        self.action_space = spaces.Box(low=self.min_speed, high=self.max_speed,
                                       shape=(4,), dtype=np.float32)

        obs_high = np.array([
                   self.max_x, self.max_y, self.max_z,
                   np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                   np.pi/2, np.pi/2, np.pi/2,
                   np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max])
        obs_low = np.array([
                   self.min_x, self.min_y, self.min_z,
                   np.finfo(np.float32).min, np.finfo(np.float32).min, np.finfo(np.float32).min,
                   -np.pi/2, -np.pi/2, -np.pi/2,
                   np.finfo(np.float32).min, np.finfo(np.float32).min, np.finfo(np.float32).min])

        # TODO: fix observation_space bound - @nimbus state[]
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

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

        # set solver
        self.ode = integrate.ode(self._state_dot).set_integrator('vode', nsteps=500, method='bdf')

        # set random seed
        self.seed()

        # reset env
        self.reset()

        # set viewer
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.motor1.set_speed(action[0])
        self.motor2.set_speed(action[1])
        self.motor3.set_speed(action[2])
        self.motor4.set_speed(action[3])

        self.state = self._get_new_state()
        done = self.num_step > 3000 or self._crashed_box()
        reward = self.reward()
        self.total_reward += reward
        # increase num_step
        self.num_step += 1
        if done is True:
            print(self.num_step, self._crashed_box(), self.target, self.total_reward, self.state[0:3])
            self.total_reward = 0
        return self.state, reward, done, {}

    def reward(self):
        if self._crashed_box():
            return -100
        #print(self.state)
        dist_reward = - np.sqrt((self.state[0] - self.target[0])**2 + \
                      (self.state[1] - self.target[1])**2 + \
                      (self.state[2] - self.target[2])**2)
        #self.reward_bin = np.array([0, d/3, 2*d/3, d])
        #self.reward_list = np.array([-0.75, -0.5, 0, 0.5])
        #dist_reward = self.reward_list[(np.digitize(d, self.reward_bin)-1)]

        pose_reward = 1/((np.mean(abs(self.state[6:8]))/(np.pi))+0.5)

        time_reward = -0.01*self.num_step
        reward = dist_reward/10 + pose_reward + time_reward

        if self._reach_target():
            self.target[0] = np.random.uniform(low=self.min_x, high=self.max_x)
            self.target[1] = np.random.uniform(low=self.min_y, high=self.max_y)
            self.target[2] = np.random.uniform(low=self.min_z, high=self.max_z)
            return 100

        return reward

    def reset(self):
        # random spawn target
        if self.target is None:
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

        self.num_step = 0

        self.d = np.sqrt((self.state[0] - self.target[0])**2 + \
                    (self.state[1] - self.target[1])**2 + \
                    (self.state[2] - self.target[2])**2)

        return np.array(self.state)

    # TODO: current state: matplotlib, todo: upward indicator. openGL? Unity?
    def render(self, mode='human'):

        if self.viewer is None:
            self.viewer = plt.figure()
            self.ax = Axes3D.Axes3D(self.viewer)
            self.ax.set_xlim3d([self.min_x, self.max_x])
            self.ax.set_xlabel('X')
            self.ax.set_ylim3d([self.min_y, self.max_y])
            self.ax.set_ylabel('Y')
            self.ax.set_zlim3d([self.min_z, self.max_z])
            self.ax.set_zlabel('Z')
            self.ax.set_title('Quadrotor Simulation')
            self.l1,  = self.ax.plot([], [], [], color='blue', linewidth=3, antialiased=False)
            self.l2,  = self.ax.plot([], [], [], color='blue', linewidth=3, antialiased=False)
            self.hub, = self.ax.plot([], [], [], marker='^', color='blue', markersize=6, antialiased=False)

            #self.upward_indicator = Arrow3D([], [], [], mutation_scale=20, lw=3, arrowstyle="-|>", color="r")
            #self.ax.add_artist(self.upward_indicator)

            # plot target
            #print(self.target)
            self.ax.plot([self.target[0]], [self.target[1]], [self.target[2]], marker='o', color='red', markersize=6, antialiased=False)

            # add manual control
            self.viewer.canvas.mpl_connect('key_press_event', self._keypress_routine)

        R = self._rotation_matrix(self.state[3:6])
        L = self.L
        points = np.array([[-L, 0, 0], [L, 0, 0], [0, -L, 0], [0, L, 0], [0, 0, 0], [0, 0, 0]]).T
        points = np.dot(R, points)
        points[0, :] += self.state[0]
        points[1, :] += self.state[1]
        points[2, :] += self.state[2]

        self.l1.set_data(points[0, 0:2], points[1, 0:2])
        self.l1.set_3d_properties(points[2, 0:2])
        self.l2.set_data(points[0, 2:4], points[1, 2:4])
        self.l2.set_3d_properties(points[2, 2:4])
        self.hub.set_data(points[0, 5], points[1, 5])
        self.hub.set_3d_properties(points[2, 5])

        # TODO: add hub indicator

        plt.pause(1e-12)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

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
                   np.array([0, 0, (self.motor1.thrust + self.motor2.thrust
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
                        self.b*(self.motor1.thrust-self.motor2.thrust +
                                self.motor3.thrust-self.motor4.thrust)])

        omega_dot = np.dot(self.invI, (tau - np.cross(omega, np.dot(self.In, omega))))
        state_dot[9] = omega_dot[0]
        state_dot[10] = omega_dot[1]
        state_dot[11] = omega_dot[2]
        return state_dot

    def _get_new_state(self, dt=0.01):

        self.ode.set_initial_value(self.state, 0)
        self.state = self.ode.integrate(self.ode.t + dt)
        self.state[6:9] = self._normalize(self.state[6:9])
        self.state[2] = max(0, self.state[2])

        return self.state

    def _crashed_box(self):
        # check if agent is out of the box
        if ((self.state[0] > self.max_x or self.state[0] < self.min_x)
            or (self.state[1] > self.max_y or self.state[1] < self.min_y)
            or (self.state[2] > self.max_z or self.state[2] < self.min_z)):
            return True
        else:
            return False

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
        if d < 0.5:
            print("reach!")
            return True
        else:
            return False

    def _keypress_routine(self, event):
        sys.stdout.flush()
        if event.key == 'w':
            y = list(self.ax.get_ylim3d())
            y[0] += 1
            y[1] += 1
            self.ax.set_ylim3d(y)

        if event.key == 'x':
            y = list(self.ax.get_ylim3d())
            y[0] -= 1
            y[1] -= 1
            self.ax.set_ylim3d(y)

        if event.key == 'd':
            x = list(self.ax.get_xlim3d())
            x[0] += 1
            x[1] += 1
            self.ax.set_xlim3d(x)

        if event.key == 'a':
            x = list(self.ax.get_xlim3d())
            x[0] -= 1
            x[1] -= 1
            self.ax.set_xlim3d(x)


# support class
class Propeller:
    def __init__(self, diameter, pitch, thrust_unit='N', min_speed = 0, max_speed = 15000):
        self.d = diameter
        self.pitch = pitch
        self.thrust_unit = thrust_unit
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.reset()

    def set_speed(self, normalized_speed):

        self.speed = self.min_speed + (self.max_speed - self.min_speed)*normalized_speed
        self.update_thrust()

    def update_thrust(self):
        self.thrust = 4.392e-8*self.speed*(self.d**3.5)/(np.sqrt(self.pitch))
        self.thrust = self.thrust*(4.23e-4 * self.speed * self.pitch)
        if self.thrust_unit == 'kg':
            self.thrust_unit = self.thrust*0.101972

    def reset(self):
        self.speed = 0
        self.thrust = 0
