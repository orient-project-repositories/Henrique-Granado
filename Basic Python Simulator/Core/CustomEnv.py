from .System import *
from .Renderer import *
from gym import Env
from gym.spaces import Box
import numpy as np
import time


class CustomEnv(Env):
    def __init__(self, time_step = 0.001, time_limit = 1):
        self.system = System()

        self.action_space = Box(low = -30, high = 30, shape = (3,), dtype = np.float32)

        self.observation_space = Box(low=np.array([0,-1,-1,-1]), high=np.array([1,1,1,1]), dtype=np.float32)

        self.state = self.system.get_output()
        self.motor_position = self.system.get_motor_position()

        self.done = False

        self.time_passed = 0
        self.time_step = time_step
        self.time_limit = time_limit

        self.viewer = None

        self.update_func = None

    def step(self, action):
        if self.update_func is not None:
            self.update_func(self.time_step)

        self.motor_position = self.process_action(action)
        self.system.feed_input(self.motor_position)
        self.system.update(self.time_step)
        self.state = self.system.get_output()
        self.time_passed += self.time_step

        self.done = self.is_done()

        reward = self.reward_function()

        # Set placeholder for info
        info = {"time_length": self.time_passed}

        # Return step information
        return self.state, reward, self.done, info

    def process_action(self, action):
        return action

    def change_time_step(self, value):
        self.time_step = value

    def render(self, mode='human'):
        screen_width = 864
        screen_height = 580

        if self.viewer is None:
            import Renderer
            self.viewer = Renderer.Viewer(screen_width, screen_height)

            self.viewer.add_geom(Floor())
            self.viewer.add_geom(self.system)

            self.update_func = self.viewer.get_update_func()

        if self.state is None:
            return None

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def reset(self):
        self.time_passed = 0
        self.done = False
        self.system.reset()
        self.state = self.system.get_output()
        self.motor_position = self.system.get_motor_position()

        return self.state

    def reward_function(self):
        return 0

    def is_done(self):
        return self.time_limit <= self.time_passed


if __name__ == "__main__":  # testing the environment -> movement of the horizontal motor of 15 degrees at time step of 100
    env = CustomEnv()

    step0 = [0 for i in range(0,1000)]
    step1 = [0 if i < 100 else 15 for i in range(0, 1000)]
    step2 = [0 for i in range(0, 1000)]
    action = np.transpose([step0, step1, step2])
    start = time.time()
    for i in range(0,1000):
        env.render()
        n_state, reward, done, info = env.step(action[i])
    end = time.time()
    print(end-start)