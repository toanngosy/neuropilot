import numpy as np 
import math
import scipy.integrate
import time
import datetime
import threading
import sys
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D


MAX_SPEED = 0
MIN_SPEED = 9000

MAX_ANGULAR_SPEED = np.pi/6

MIN_X = 0
MAX_X = 20
MIN_Y = 0
MAX_Y = 20
MIN_Z = 0
MAX_Z = 20

MIN_ANGLE = -np.pi/6
MAX_ANGLE =  np.pi/6


class Propeller():
	def __init__(self, diameter,pitch, thrust_unit = 'N'):
		self.d = diameter
		self.pitch = pitch
		self.thrust_unit = thrust_unit
		self.speed = 0
		self.thrust = 0

	def set_speed(self, speed):
		self.speed = speed
		self.update_thrust()

	def update_thrust(self):
		self.thrust = 4.392e-8*self.speed*(self.d**3.5)/(math.sqrt(self.pitch))
		self.thrust = self.thrust*(4.23e-4 * self.speed * self.pitch)
		if self.thrust_unit == 'kg':
			self.thrust_unit = self.thrust*0.101972


class Quadrotor():

	def __init__(self, param, g = 9.81 , b= 0.0245):
		self.g = g
		self.b = b
		self.clock = 0

		self.m = param['weight']
		self.r = param['r']
		self.L = param['L']

		self.thread_object = None
		self.ode =  scipy.integrate.ode(self.state_dot).set_integrator('vode',nsteps=500,method='bdf')
		self.time = datetime.datetime.now()

		self.state = np.zeros(12)
		self.state[0:3] = param['position']
		self.state[6:9] = param['orientation']
		self.target = param['target']

		# create reward_bin to calculate reward from each step
		# reward_bin will divice distance form spawn location to target location into 3 bins
		# current postion will yeild difference reward depending on which bin it is in

		d = math.sqrt( (self.target[0] - self.state[0])**2 +  (self.target[0] - self.state[0])**2 + (self.target[0] - self.state[0])**2 )
		self.reward_bin  = np.array([0,d/3,2*d/3,d])
		self.reward_list = np.array([-0.75,-0.5,0, 0.5])

		self.motor1 = Propeller(param['prop_diameter'], param['prop_pitch'])
		self.motor2 = Propeller(param['prop_diameter'], param['prop_pitch'])
		self.motor3 = Propeller(param['prop_diameter'], param['prop_pitch'])
		self.motor4 = Propeller(param['prop_diameter'], param['prop_pitch'])

		
		self.done   = False

		Ixx = (2*self.m*self.r**2)/5 + 2*self.m*(self.L**2)
		Iyy = Ixx
		Izz = (2*self.m*self.r**2)/5 + 4*self.m*(self.L**2)
		self.I = np.array([[Ixx,0,0],[0,Iyy,0],[0,0,Izz]])
		self.invI = np.linalg.inv(self.I)

		self.run = True

		#Init render_unit()
		self.fig = plt.figure()
		self.ax  = Axes3D.Axes3D(self.fig)
		self.ax.set_xlim3d([MIN_X , MAX_X])
		self.ax.set_xlabel('X')
		self.ax.set_ylim3d([MIN_Y , MAX_Y])
		self.ax.set_ylabel('Y')
		self.ax.set_zlim3d([MIN_Z , MAX_Z])
		self.ax.set_zlabel('Z')
		self.ax.set_title('Quadrotor Simulation')
		self.init_plot()
		self.fig.canvas.mpl_connect('key_press_event', self.keypress_routine)

	def init_plot(self):
		self.l1,  = self.ax.plot([],[],[],color = 'blue', linewidth = 3, antialiased = False)
		self.l2,  = self.ax.plot([],[],[],color = 'blue', linewidth = 3, antialiased = False)
		self.hub, = self.ax.plot([],[],[], marker = 'o', color = 'blue', markersize = 6, antialiased = False)

	def rotation_matrix(self, angles):
		ct = math.cos(angles[0])
		cp = math.cos(angles[1])
		cg = math.cos(angles[2])
		st = math.sin(angles[0])
		sp = math.sin(angles[1])
		sg = math.sin(angles[2])
		R_x = np.array([[1,0,0],[0,ct,-st],[0,st,ct]])
		R_y = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
		R_z = np.array([[cg,-sg,0],[sg,cg,0],[0,0,1]])
		R = np.dot(R_z, np.dot( R_y, R_x ))
		return R


	def render_update(self):
		R = self.rotation_matrix(self.state[3:6])
		L = self.L
		points = np.array([[-L,0,0], [L,0,0],[0,-L,0], [0,L,0], [0,0,0], [0,0,0] ]).T
		points = np.dot(R,points)
		points[0,:] += self.state[0]
		points[1,:] += self.state[1]
		points[2,:] += self.state[2]
		self.l1.set_data(points[0,0:2], points[1,0:2])
		self.l1.set_3d_properties(points[2,0:2])
		self.l2.set_data(points[0,2:4], points[1,2:4])
		self.l2.set_3d_properties(points[2,2:4])
		self.hub.set_data(points[0,5], points[1,5])
		self.hub.set_3d_properties(points[2,5])
		plt.pause(1e-12)

	def normalize(self,angle):
		return ((angle + np.pi)%(2*np.pi) -np.pi)


	def state_dot(self):
		state_dot = np.zeros(12)
		# The velocities(t+1 x_dots equal the t x_dots)
		state_dot[0] = self.state[3]
		state_dot[1] = self.state[4]
		state_dot[2] = self.state[5]
		# The acceleration
		x_dotdot = np.array([0,0,-self.m*self.g]) + np.dot(self.rotation_matrix(self.state[6:9]),np.array([0,0,(self.motor1.thrust + self.motor2.thrust + self.motor3.thrust + self.motor4.thrust)]))/self.m
		state_dot[3] = x_dotdot[0]
		state_dot[4] = x_dotdot[1]
		state_dot[5] = x_dotdot[2]
		# The angular rates(t+1 theta_dots equal the t theta_dots)
		state_dot[6] = self.state[9]
		state_dot[7] = self.state[10]
		state_dot[8] = self.state[11]
		#The angular accelerations
		omega = self.state[9:12]
		tau = np.array([self.L*(self.motor1.thrust-self.motor3.thrust), self.L*(self.motor2.thrust-self.motor4.thrust), self.b*(self.motor1.thrust-self.motor2.thrust+self.motor3.thrust-self.motor4.thrust)])
		omega_dot = np.dot(self.invI, (tau - np.cross(omega, np.dot(self.I,omega))))
		state_dot[9] = omega_dot[0]
		state_dot[10] = omega_dot[1]
		state_dot[11] = omega_dot[2]
		return state_dot

	def get_new_state(self,dt):

		self.ode.set_initial_value(self.state,0)
		self.state = self.ode.integrate(self.ode.t + dt)
		self.state[6:9] = self.normalize(self.state[6:9])
		self.state[2] = max(0,self.state[2])
		self.clock += 1

	def get_reward(self):
		if self.crashed():
			return -5


		d = math.sqrt(( self.state[0] - self.target[0])**2 + (self.state[1] - self.target[1])**2 +  (self.state[2] - self.target[2])**2) 
		dist_reward = self.reward_list[(np.digitize(d,self.reward_bin)-1)]
		time_reward = -0.01*self.clock
		pose_reward = -((abs(self.state[3:6])/MAX_SPEED).sum() + (abs(self.state[6:8])/np.pi).sum() + (abs(self.state[9:12])/MAX_ANGULAR_SPEED).sum())
		
		reward = 0.4*dist_reward + 0.3*time_reward + 0.3*pose_reward

		if self.reach_target():
			return reward + 1

		return reward

	def reach_target(self):
		#check if agent reach target postion
		d = math.sqrt(( self.state[0] - self.target[0])**2 + (self.state[1] - self.target[1])**2 +  (self.state[2] - self.target[2])**2) 
		if d < 0.05:
			self.done = True
			return True
		else:
			return False


	def crashed(self):
		#check if agent is crashed
		#in another word, touch the ground with large velocity
		if (self.state[2] < (self.L + 0.05)):
			self.done = True
			return True
		else: 
			return False

	def step(self,action):
		self.motor1.set_speed(action[0])
		self.motor2.set_speed(action[1])
		self.motor3.set_speed(action[2])
		self.motor4.set_speed(action[3])

		self.get_new_state(0.01)
		
		return (self.state , self.get_reward(), self.done)

	def sample_action(self):
		return np.random.uniform(MIN_SPEED,MAX_SPEED,size = 4)

	def reset(self):
		#init new reward_bin
		self.state = np.zeros(self.state.shape)
		# recalculate reward bins
		d = math.sqrt( (self.target[0] - self.state[0])**2 +  (self.target[0] - self.state[0])**2 + (self.target[0] - self.state[0])**2 )
		self.reward_bin  = np.array([0,d/3,2*d/3,d])
		# reset done flat
		self.done = False

	def random_spawn(self):
		#init new postion 

		self.state = np.zeros(self.state.shape)
		self.state[0] = np.random.uniform(MIN_X,MAX_X)
		self.state[1] = np.random.uniform(MIN_X,MAX_X)
		self.state[2] = np.random.uniform(MIN_X,MAX_X)

		

		self.state[6:9] = np.random.uniform(MIN_ANGLE,MAX_ANGLE,size = 3)
		#init new reward_bin
		d = math.sqrt( (self.target[0] - self.state[0])**2 +  (self.target[0] - self.state[0])**2 + (self.target[0] - self.state[0])**2 )
		self.reward_bin  = np.array([0,d/3,2*d/3,d])
		#reset done flat
		self.done = False


	def keypress_routine(self,event):
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



if __name__ == "__main__":


	PARAM = {'position':[1,0,4], 
			 'orientation':[0,0,0],
			 'L':0.5,'r':0.2,
			 'prop_diameter':10, 
			 'prop_pitch': 4.5,
			 'weight':1.2,
			 'target': [10,10,10]}

	quad = Quadrotor(PARAM)

	while not quad.done:
		#quad.random_spawn()
		quad.step(quad.sample_action())
		quad.render_update()
		if quad.clock >10000:
			quad.done = True
