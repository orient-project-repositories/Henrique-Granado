from gym.spaces import Box
from Core_pygletless import CustomEnv
import numpy as np
import quaternion
from utils import Pulse, get_quaternion_angle, calculate_accuracy, generate_desired_angle, normalize_


class OpenLoopNoiselessEnv(CustomEnv):
    def __init__(self, weights = [1,1,1,1], n_actions = 3, n_motors = 1, *args, **kwargs): # setting up the environment for open-loop training
        super().__init__(*args, **kwargs)
        self.n_actions = n_actions  # per motor
        self.n_motors = n_motors # number of motors that are used (1 - only horizontal, 2 - the two vertical, 3 - all 3 motors)
        self.motors_in_use = [n_motors != 1, n_motors != 2, n_motors != 1]  # if true it means it is going to be used

        if self.n_actions != 3:
            raise NotImplementedError("Invalid number of actions")

        if self.n_motors not in [1,2]:  # this environment hasn't been used for more than 2 motors because it is slow training. The cython environment has been used istead
            raise NotImplementedError("Number of motors is invalid")
        as_l = []  # lowest values for the action space
        as_h = []  # highest values for the action space
        if self.n_motors == 1:
            as_l += [15, -45, 5]
            as_h += [115, 45, 25]
        elif self.n_motors == 2:
            as_l = [15, -25, 2]*2
            as_h = [115, 25, 22]*2

        as_l = np.array(as_l)
        as_h = np.array(as_h)

        self.observation_space = Box(low=np.array([-40,-40,-40,-40]), high=np.array([40,40,40,40]))  # (yaw, pitch, desired yaw, desired pitch)
        self.action_space = Box(low = as_l, high = as_h)

        self.min_angle = np.radians(3)  # minimum angle difference between initial and desired orientation
        self.max_angle = np.radians(60)  # maximum angle difference between initial and desired orientation
        self.quaternion_bounds = np.array([0.139, 0.5, 0.342])  # if eye orientation goes beyond this bounds position is reset

        self.quat_list = [self.quat]  # history of orientations that the has had
        self.prev_motor_position = self.motor_position  # previous motor position (for the 3 motors)

        self.desired_quat = None

        self.weight = np.array(weights)  # weights of each of the rewards
        self.norm_weight = np.linalg.norm(self.weight)

        self.settling_time = np.inf

        self.pulse_function = [Pulse(),Pulse(),Pulse()]  # initiating pulse-step functions for each of the motors

        self.premove_dictionary = {}  # this is helpful for when we want to start a movement that is not at origin

    def change_action_space(self, low, high): #make sure these are np.arrays()
        self.action_space = Box(high = high, low= low)

    def change_observation_space(self, low, high): #make sure these are np.arrays()
        self.observation_space = Box(low=low, high=high)

    def get_stringed_weights(self):
        return "{}\t{}\t{}\t{}".format(self.weight[0], self.weight[1], self.weight[2], self.weight[3])

    def change_max_angle(self, value):
        self.max_angle = np.radians(value)

    def is_done(self):  # check if movement is done
        # eye stopped and motor close to final position
        motor_diff = abs(np.subtract(self.motor_position,self.prev_motor_position))
        try:
            quat_cond = (self.quat == self.quat_list[-3:]).all()
        except IndexError:
            quat_cond = False
        cond = ((motor_diff < 0.001).all() and quat_cond)
        return (cond or self.time_limit < self.time) and self.time > 0.05

    def process_action(self, action_, preprocess):  # uses the action_ values to define the pulse functions
        action_ = np.array(action_)
        if preprocess:  # if action_ have been normalized
            as_l = self.action_space.low
            as_h = self.action_space.high
            action_ = (as_h-as_l)/2*action_+(as_h+as_l)/2

        split_action = action_.reshape(len(action_)//self.n_actions, self.n_actions)
        index = 0
        for i, (val, pf) in enumerate(zip(self.motors_in_use, self.pulse_function)):
            if val:
                pf.change_values(split_action[index], self.motor_position[i])
            else:
                pf.change_values([1, self.motor_position[i],1], self.motor_position[i])
            index += val

    def step(self, action, preprocess = True):
        if self.update_func is not None:
            self.update_func(self.time_step)

        self.process_action(action, preprocess)
        for _ in range(int(self.time_limit//self.time_step)): # because this open-loop the whole movement is done (defined within a certain timeframe)
            self.time += self.time_step
            self.prev_motor_position = self.motor_position
            self.motor_position = np.clip([x.tau(self.time) for x in self.pulse_function], -90, 90)  # updates motor position
            self.system.feed_input(np.radians(self.motor_position))  # feed s new motor position to system
            self.system.update(self.time_step)  # updates system
            self.quat = self.system.get_output()  # get new orientation after time step
            self.quat_list.append(self.quat)  # append to list
            if self.is_done():  # checks if movement is done
                break
        self.system.stop()  # forces system to stop moving
        self.done = True

        self.motor_position = [x.tau(np.inf) for x in self.pulse_function]  # updates motor position to be the end one (at infinity)
        self.prev_motor_position = self.motor_position

        reward, reward_info, settling_quat, settling_time = self.reward_function()
        # Set placeholder for info
        info = {"time_length": self.time, "reward_info": reward_info, "orientation_history": self.quat_list, "settling_quat": settling_quat, "settling_time":settling_time}

        # Return step information
        return self.get_state(), reward, self.done, info

    def reward_function(self):  # calculates the reward value of the action applied to the system
        initial_quat = self.quat_list[0]
        final_quat = self.quat
        amplitude = get_quaternion_angle(initial_quat, final_quat)  # of the movement
        lower_band = abs(0.99*amplitude)  # to cvalculate settling time at 1%
        upper_band = abs(1.01*amplitude)  # to calculate settling time at 1%

        settling_quat = self.quat
        settling_time = len(self.quat_list)*self.time_step
        set_time_found = False
        max_amp = amplitude
        for i, quat in enumerate(reversed(self.quat_list)):  # gets the setttling time at 1% (from lower_band and upper_band)
            this_amp = get_quaternion_angle(quat,initial_quat)
            if not (lower_band<this_amp<upper_band) and not set_time_found:
                set_time_found = True
            if not set_time_found:
                settling_time = (len(self.quat_list) - i + 1) * self.time_step
                settling_quat = quat
            if max_amp<this_amp:
                max_amp = this_amp

        acc = -calculate_accuracy(self.desired_quat, settling_quat)
        nrg = -np.sum([x.energy_used() for x in self.pulse_function])  # by the 3 motors
        duration = -(1-1/(1+0.6*settling_time))  # duration cost function from Shadmeer
        os = max(max_amp/max(amplitude,10**-12),1)-1
        os = -os*(os >= 10**-4)
        this_reward = (self.weight[0]*acc+self.weight[1]*nrg+self.weight[2]*duration+self.weight[3]*os)/self.norm_weight

        return this_reward, [acc, nrg, duration, os], settling_quat, settling_time

    def change_weights(self, new_weights):
        self.weight = np.array(new_weights)
        self.norm_weight = np.linalg.norm(self.weight)

    def is_out_of_bounds(self):  # check if system is out of set bounds
        motor_is_out = np.any(self.motor_position < self.action_space.low[1]) or np.any(self.action_space.high[1] < self.motor_position)  # motor orientation is out of action space
        orien = np.degrees(quaternion.as_rotation_vector(quaternion.from_float_array(self.quat)))  # orientation of the eye in rotation vector
        eye_is_out = np.any(orien[1:]<self.observation_space.low[:2]) or np.any(self.observation_space.high[:2]<orien[1:])  # eye orientation is out of the observation space range
        return motor_is_out or eye_is_out

    def reset(self):  # resets environment
        self.time = 0
        self.done = False
        self.system.reset()
        self.quat = self.system.get_output()
        self.motor_position = self.system.get_motor_position()
        self.prev_motor_position = self.motor_position

        self.desired_quat = self.quat

        return self.get_state()

    def select_reset(self, desired_orientation, initial_motor_position):  # resets the system with teh specifications below
        """
        :param desired_orientation:
            'r' --> random orientation (yaw or pitch)
            [yaw, pitch] :-> [float, float]
            angle :-> (float) --> value for yaw or pitch (depending if n_motors is 1 or 2)
        :param initial_motor_position:
            'o' --> origin
            's' --> start from current orientation if not out of bounds
            f2 :-> float --> horizontal motor only
            [f2] :-> [float] --> horizontal motor only
            [f1,f3] :-> [float, float] --> vertical motor only
            [f1,f2,f3] :-> [float, float, float] --> all motors
        :return: [current_yaw, current_pitch, desired_yaw, desored_pitch]
        """
        # while np.linalg.norm(self.get_system_angular_velocity()) != 0:
        #     self.system.update(self.time_step)

        self.system.update(self.time_step)
        if np.linalg.norm(self.get_system_angular_velocity()) != 0:
            self.reset()

        if isinstance(initial_motor_position, str):
            if initial_motor_position == 'o' or self.is_out_of_bounds():
                self.reset()
        elif isinstance(initial_motor_position, float) or isinstance(initial_motor_position, int):
            self.reset()
            if initial_motor_position != 0:
                self.premove_motors([0,initial_motor_position, 0])
        elif len(initial_motor_position) == self.n_motors:
            self.reset()
            f1 = initial_motor_position[0]*(self.n_motors != 1)
            f2 = initial_motor_position[int(self.n_motors == 3)]*(self.n_motors != 2)
            f3 = initial_motor_position[int((self.n_motors > 1)+(self.n_motors > 2))]*(self.n_motors != 1)
            if any([f1,f2,f3]):
                self.premove_motors([f1,f2,f3])
        else:
            raise NotImplementedError("Misimplementation regarding initial_motor_position")

        self.prev_motor_position = self.motor_position
        self.quat = self.system.get_output()

        if isinstance(desired_orientation, str) or isinstance(desired_orientation, float) or isinstance(desired_orientation, int):
            self.desired_quat = self.quat
            while not self.min_angle < get_quaternion_angle(self.quat, self.desired_quat) < self.max_angle:
                desired_angles = np.array([0,
                                           generate_desired_angle(desired_orientation, self.observation_space.low[2], self.observation_space.high[2])*(self.n_motors != 2),
                                           generate_desired_angle(desired_orientation, self.observation_space.low[3], self.observation_space.high[3])*(self.n_motors != 1)])
                self.desired_quat = quaternion.as_float_array(quaternion.from_rotation_vector(desired_angles))
        elif len(desired_orientation) == 2 and all([isinstance(x, float) for x in desired_orientation]):
            desired_angles = np.array([0,
                                       generate_desired_angle(desired_orientation[0], self.observation_space.low[2], self.observation_space.high[2]),
                                       generate_desired_angle(desired_orientation[1], self.observation_space.low[3], self.observation_space.high[3]) ])
            self.desired_quat = quaternion.as_float_array(quaternion.from_rotation_vector(desired_angles))
        elif len(desired_orientation) == 4:
            self.desired_quat = desired_orientation
        else:
            raise NotImplementedError("Invalid value for desired_orientation: {}".format(desired_orientation))

        self.quat_list = [self.quat]
        self.time = 0
        self.done = False
        return self.get_state()

    def premove_motors(self, desired_f):  # if movement wants to start at rest with a certain motor orientation.
        self.system.feed_input(np.radians(desired_f))
        if str(desired_f) in self.premove_dictionary.keys():
            self.system.change_eye_quaternion(self.premove_dictionary[str(desired_f)])
        else:
            for _ in range(10):
                self.system.update(self.time_step)
            while np.linalg.norm(self.get_system_angular_velocity()) != 0:
                self.system.update(self.time_step)
            self.premove_dictionary[str(desired_f)] = self.system.get_output()

        self.motor_position = np.degrees(self.system.get_motor_position())

    def get_desired_quat(self):  # gets desired orientation in quaternions
        return self.desired_quat

    def get_quat(self):  # gets current orientation in quaternions
        return self.quat

    def get_raw_state(self):  # gets the state (yaw, pitch, desired yaw, desired pitch)
        return np.degrees(np.concatenate((quaternion.as_rotation_vector(quaternion.from_float_array(self.quat))[1:],
                                          quaternion.as_rotation_vector(quaternion.from_float_array(self.desired_quat))[1:])))

    def get_state(self):  # gets normalized state from observation space
        return normalize_(self.get_raw_state(), self.observation_space.low, self.observation_space.high)
