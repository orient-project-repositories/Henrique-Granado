from gym.spaces import Box
from Core_Cython.CustomEnv import CustomEnv
from Core_pygletless.System import System
import numpy as np
import quaternion
from utils import Pulse, get_quaternion_angle, calculate_accuracy, generate_desired_angle, normalize_


class CythonOpenLoopNoiselessEnv(CustomEnv):
    def __init__(self, weights=[1,1,1,1], n_actions=3, n_motors=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_actions = n_actions  # per motor
        self.n_motors = n_motors
        self.motors_in_use = [n_motors != 1, n_motors != 2, n_motors != 1]

        if self.n_actions not in [3]:
            raise NotImplementedError("Invalid number of actions")

        if self.n_motors not in [1,2,3]:
            raise NotImplementedError("Number of motors is invalid")
        as_l = []
        as_h = []
        if self.n_motors == 1:
            as_l += [15, -45, 5]
            as_h += [115, 45, 25]
        elif self.n_motors == 2:
            as_l = [15, -25, 2]*2
            as_h = [115, 25, 22]*2

        as_l = np.array(as_l)
        as_h = np.array(as_h)

        self.observation_space = Box(low=np.array([-40,-40,-40,-40]), high=np.array([40,40,40,40]))
        self.action_space = Box(low = as_l, high = as_h)

        self.min_angle = np.radians(3)
        self.max_angle = np.radians(60)
        self.quaternion_bounds = np.array([0.139, 0.5, 0.342])  # [0.241, 0.442, 0.402] <--- sys_id is not done such that it reaches these limits
        # pitch bound decreased from .402 because of VHM

        self.quat_list = [self.quat]
        self.prev_motor_position = self.motor_position

        self.desired_quat = None

        self.weight = None
        self.norm_weight = None
        self.change_weights(weights)

        self.settling_time = np.inf

        self.pulse_function = [Pulse(),Pulse(),Pulse()]

        self.premove_dictionary = {}

        self.max_amplitude = 0

        if len(self.weight) == 5:  # TODO this part (make get_force stuff in Cython)
            self.real_system = System()  # used for the force reward

    def change_action_space(self, low, high): #make sure these are np.arrays()
        self.action_space = Box(high = high, low= low)

    def change_observation_space(self, low, high): #make sure these are np.arrays()
        self.observation_space = Box(low=low, high=high)

    def get_stringed_weights(self):
        return "{}\t{}\t{}\t{}".format(self.weight[0], self.weight[1], self.weight[2], self.weight[3])

    def change_max_angle(self, value):
        self.max_angle = np.radians(value)

    def is_done(self):
        # eye stopped and motor close to final position
        motor_diff = abs(np.subtract(self.motor_position,self.prev_motor_position))
        try:
            quat_cond = (self.quat_list[-1] == self.quat_list[-3:]).all()
        except IndexError:
            quat_cond = False
        cond = ((motor_diff < 0.001).all() and quat_cond)
        return (cond or self.time_limit < self.time) and self.time > 0.05

    def process_action(self, action_, preprocess):
        action_ = np.array(action_)
        if preprocess:
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

    def step(self, action, preprocess = True, skip = None):
        if self.update_func is not None:
            self.update_func(self.time_step)

        self.process_action(action, preprocess)
        self.max_amplitude = 0
        for _ in range(int(self.time_limit//self.time_step)):
            self.time += self.time_step
            self.prev_motor_position = self.motor_position
            self.motor_position = np.clip([x.tau(self.time) for x in self.pulse_function], -90, 90)
            self.system.feed_input(np.radians(self.motor_position[0]), np.radians(self.motor_position[1]), np.radians(self.motor_position[2]))
            self.system.update(self.time_step)
            self.quat = np.array(self.system.get_output())
            this_amp = get_quaternion_angle(self.quat, self.quat_list[0])
            if self.max_amplitude<this_amp:
                self.max_amplitude = this_amp
            self.quat_list.append(self.quat)
            if self.is_done():
                break
        angular_velocity = self.system.get_eye_angular_velocity()
        self.system.stop()
        self.done = True

        self.motor_position = [x.tau(np.inf) for x in self.pulse_function]
        self.prev_motor_position = self.motor_position

        reward, reward_info, settling_quat, settling_time = self.reward_function()
        # Set placeholder for info
        info = {"time_length": self.time, "reward_info": reward_info, "orientation_history": self.get_quat_list, "settling_quat": settling_quat, "settling_time":settling_time, "last_angular_velocity": angular_velocity}
        # Return step information
        return self.get_state(), reward, self.done, info

    def reward_function(self):
        initial_quat = self.quat_list[0]
        final_quat = self.quat
        amplitude = get_quaternion_angle(initial_quat, final_quat)
        error_band = 0.05
        lower_band = abs((1-error_band)*amplitude)
        upper_band = abs((1+error_band)*amplitude)

        settling_quat = self.quat
        settling_time = len(self.quat_list)*self.time_step
        for i, quat in enumerate(reversed(self.quat_list)):
            this_amp = get_quaternion_angle(quat,initial_quat)
            if not (lower_band<this_amp<upper_band):
                break
            settling_time = (len(self.quat_list) - i + 1) * self.time_step
            settling_quat = quat

        acc = -calculate_accuracy(self.desired_quat, settling_quat*(self.reward_mode == 1)+self.quat*(self.reward_mode != 1))  # get_quaternion_distance(self.desired_quat, settling_quat)
        nrg = -np.sum([x.energy_used() for x in self.pulse_function])
        duration = -(1-1/(1+0.6*settling_time))
        os = max((self.max_amplitude-amplitude)/max(amplitude,10**-12),0)
        os = -os * (os >= 10 ** -6)
        force = None
        if len(self.weight) == 4:  # check update_reward_mode
            this_reward = (self.weight[0]*acc+self.weight[1]*nrg+self.weight[2]*duration+self.weight[3]*os)/self.norm_weight
        elif len(self.weight) == 5:
            force = -self.get_force()
            this_reward = (self.weight[0] * acc + self.weight[1] * nrg + self.weight[2] * duration + self.weight[3] * os + self.weight[4] * force) / self.norm_weight

        return this_reward, [acc, nrg, duration, os]+[force]*(len(self.weight) == 5), settling_quat, settling_time

    def get_force(self):  # calculates the force reward
        # feed_input -> change_eye_quaternion -> update elastics -> get_force_magnitude
        total_force = 0
        step = np.radians(self.motor_position)
        self.real_system.feed_input(step)
        final_quat = self.quat
        self.real_system.change_eye_quaternion(final_quat)
        for i, (muscle1, muscle2) in enumerate(self.real_system.get_muscle_pairs()):  # calculates the torque difference between the two elastics at each motor
            muscle1.update(0)
            muscle2.update(0)
            aux_torque = 0
            for m, multiplier in zip([muscle1, muscle2], [1, -1]):
                aux_torque += multiplier*m.get_force_magnitude()
            total_force += aux_torque**2  # it's squared to penalize higher torques

        return total_force

    def change_weights(self, new_weights):
        self.weight = np.array(new_weights)
        self.norm_weight = np.linalg.norm(self.weight)

    def is_out_of_bounds(self):
        motor_is_out = np.any(self.motor_position < self.action_space.low[1]) or np.any(self.action_space.high[1] < self.motor_position)
        orien = np.degrees(quaternion.as_rotation_vector(quaternion.from_float_array(self.quat)))
        eye_is_out = np.any(orien[1:]<self.observation_space.low[:2]) or np.any(self.observation_space.high[:2]<orien[1:])
        return motor_is_out or eye_is_out

    def reset(self):
        self.time = 0
        self.done = False
        self.system.reset()
        self.quat = np.array(self.system.get_output())
        self.motor_position = np.degrees(np.array(self.system.get_motor_position()))
        self.prev_motor_position = np.array(self.motor_position)

        self.desired_quat = self.quat

        self.max_amplitude = 0

        return self.get_state()

    def select_reset(self, desired_orientation, initial_motor_position):
        """
        :param desired_orientation:
            'r' --> random orientation (pitch or yaw)
            [yaw, pitch] :-> [float, float]
            angle :-> (float) --> value for pitch or yaw
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
        self.quat = np.array(self.system.get_output())

        if isinstance(desired_orientation, str) or isinstance(desired_orientation, float) or isinstance(desired_orientation, int):
            self.desired_quat = self.quat
            while not self.min_angle < get_quaternion_angle(self.quat, self.desired_quat) < self.max_angle:
                desired_angles = np.array([0,
                                           generate_desired_angle(desired_orientation, self.observation_space.low[-2], self.observation_space.high[-2])*(self.n_motors != 2),
                                           generate_desired_angle(desired_orientation, self.observation_space.low[-1], self.observation_space.high[-1])*(self.n_motors != 1)])
                self.desired_quat = quaternion.as_float_array(quaternion.from_rotation_vector(desired_angles))
        elif len(desired_orientation) == 2 and all([isinstance(x, float) or isinstance(x, np.int32) or isinstance(x, int)  for x in desired_orientation]):
            desired_angles = np.array([0,
                                       generate_desired_angle(desired_orientation[0], self.observation_space.low[-2], self.observation_space.high[-2]),
                                       generate_desired_angle(desired_orientation[1], self.observation_space.low[-1], self.observation_space.high[-1]) ])
            self.desired_quat = quaternion.as_float_array(quaternion.from_rotation_vector(desired_angles))
        elif len(desired_orientation) == 4:
            self.desired_quat = desired_orientation
        else:
            raise NotImplementedError("Invalid value for desired_orientation: {}".format(desired_orientation))

        self.quat_list = [self.quat]
        self.time = 0
        self.done = False
        self.max_amplitude = 0
        return self.get_state()

    def premove_motors(self, desired_f):
        # self.reset()
        self.system.feed_input(np.radians(desired_f[0]), np.radians(desired_f[1]), np.radians(desired_f[2]))
        if str(desired_f) in self.premove_dictionary.keys():
            q0, qx, qy, qz = self.premove_dictionary[str(desired_f)]
            self.system.change_eye_quaternion(q0, qx, qy, qz)
        else:
            for _ in range(10):
                self.system.update(self.time_step)
            while np.linalg.norm(self.get_system_angular_velocity()) != 0:
                self.system.update(self.time_step)
            self.premove_dictionary[str(desired_f)] = list(self.system.get_output())

        self.motor_position = np.degrees(np.array(self.system.get_motor_position()))

    def get_desired_quat(self):
        return self.desired_quat

    def get_quat(self):
        return self.quat

    def get_raw_state(self):
        return np.degrees(np.concatenate((quaternion.as_rotation_vector(quaternion.from_float_array(self.quat))[1:],
                                          quaternion.as_rotation_vector(quaternion.from_float_array(self.desired_quat))[1:])))

    def get_state(self):
        return normalize_(self.get_raw_state(), self.observation_space.low, self.observation_space.high)

    def get_quat_list(self):
        return [i*self.time_step for i in range(len(self.quat_list))], self.quat_list