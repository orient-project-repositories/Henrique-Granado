from .System import *
from .Renderer import *
from gym import Env
from gym.spaces import Box
import numpy as np
import time


class CustomEnv(Env):
    def __init__(self, time_step = 0.001, time_limit = 1):
        self.system = System()

        self.action_space = Box(low = -30, high = 30, shape = (3,))  # Discrete((2*30)**3/deg_step)

        self.observation_space = Box(low=np.array([0,-1,-1,-1]), high=np.array([1,1,1,1]))

        self.quat = self.system.get_output()
        self.motor_position = self.system.get_motor_position()

        self.done = False

        self.time = 0
        self.time_step = time_step
        self.time_limit = time_limit

        self.viewer = None

        self.update_func = None

    def step(self, action):
        if self.update_func is not None:
            self.update_func(self.time_step)

        self.motor_position = self.process_action(action)
        self.system.feed_input(np.radians(self.motor_position))
        self.system.update(self.time_step)
        self.quat = self.system.get_output()
        self.time += self.time_step

        self.done = self.is_done()

        # TODO implement reward (reward_func)
        reward, reward_info = self.reward_function()

        self.post_processing()

        # Set placeholder for info
        info = {"time_length": self.time, "reward_info": reward_info}

        # Return step information
        return self.quat, reward, self.done, info

    def process_action(self, action):
        return action

    def post_processing(self):
        pass

    def change_time_step(self, value):
        self.time_step = value

    def render(self, mode='human'):
        screen_width = 864
        screen_height = 580

        if self.viewer is None:
            import Core.Renderer as Renderer
            self.viewer = Renderer.Viewer(screen_width, screen_height)

            self.viewer.add_geom(Floor())
            self.viewer.add_geom(self.system)

            self.update_func = self.viewer.get_update_func()

        if self.quat is None:
            return None

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def reset(self):
        self.time = 0
        self.done = False
        self.system.reset()
        self.quat = self.system.get_output()
        self.motor_position = self.system.get_motor_position()

        return self.quat

    def reward_function(self):
        return 0, []

    def is_done(self):
        return self.time_limit < self.time


if __name__ == "__main__":
    env = CustomEnv()
    print(env.action_space.shape)
    # episodes = 10
    # step0 = [0 for i in range(0,1000)]
    # step1 = [0 if i < 100 else 15 for i in range(0, 1000)]
    # step2 = [0 for i in range(0, 1000)]
    # action = np.transpose([step0, step1, step2])
    # start = time.time()
    # for i in range(0,1000):
    #     env.render()
    #     n_state, reward, done, info = env.step(action[i])
    #     # print(n_state)
    #     # time.sleep(0.1)
    #     # if i%100 == 0:
    #     #     print("Stuff")
    #     # print(action[i])
    # end = time.time()
    # print(end-start)