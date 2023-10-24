from .System import System
# from .System cimport System
from gym import Env
from gym.spaces import Box
import numpy as np
import time


class CustomEnv(Env):
    def __init__(self, time_step = 0.001, time_limit = 1):
        self.system = System()

        self.action_space = Box(low = -30, high = 30, shape = (3,))  # Discrete((2*30)**3/deg_step)

        self.observation_space = Box(low=np.array([-1,-1,-1,-1]), high=np.array([1,1,1,1]))

        self.quat = np.array(self.system.get_output())
        self.motor_position = np.array(self.system.get_motor_position())

        self.done = False

        self.time = 0
        self.time_step = time_step
        self.time_limit = time_limit

        self.update_func = None

    def step(self, action):
        if self.update_func is not None:
            self.update_func(self.time_step)

        self.motor_position = self.process_action(action)
        self.system.feed_input(np.radians(self.motor_position[0]), np.radians(self.motor_position[1]), np.radians(self.motor_position[2]))
        self.system.update(self.time_step)
        self.quat = np.array(self.system.get_output())
        self.time += self.time_step

        self.done = self.is_done()

        # TODO implement reward (reward_func)
        reward, reward_info = self.reward_function()

        self.post_processing()

        # Set placeholder for info
        info = {"time_length": self.time, "reward_info": reward_info}

        # Return step information
        return self.quat, reward, self.done, info

    def get_system_angular_velocity(self):
        return np.array(self.system.get_eye_angular_velocity())

    def process_action(self, action):
        return action

    def post_processing(self):
        pass

    def change_time_step(self, value):
        self.time_step = value

    def reset(self):
        self.time = 0
        self.done = False
        self.system.reset()
        self.quat = np.array(self.system.get_output())
        self.motor_position = np.array(self.system.get_motor_position())

        return self.quat

    def reward_function(self):
        return 0, []

    def is_done(self):
        return self.time_limit < self.time