import numpy as np
import quaternion


def matrix_multiplication(matrix_list): # multiplies a list of arrays
    I = np.identity(matrix_list[0].shape[0])
    new_matrix = I
    for matrix in matrix_list:
        new_matrix = np.matmul(new_matrix, matrix)
    return new_matrix


def get_quaternion_distance(point1, point2):  # https://math.stackexchange.com/questions/90081/quaternion-distance
    return max(1-np.dot(point1, point2)**2,0)


def get_quaternion_angle(q1, q2):  # https://math.stackexchange.com/questions/90081/quaternion-distance
    return np.arccos(min(2*np.dot(q1,q2)**2-1,1))


def calculate_accuracy(qd,qa):  # torsional component is unnecessary
    rot_d = quaternion.as_rotation_vector(quaternion.from_float_array(qd))
    rot_a = quaternion.as_rotation_vector(quaternion.from_float_array(qa))
    return np.sum((rot_a[1:]-rot_d[1:])**2)


def generate_desired_angle(angle, min_a, max_a):
    a = angle
    if angle == 'r':
        a = np.random.uniform(min_a,max_a)

    return np.radians(a)


class Pulse:
    """Pulse = A*t^2*exp(-B*t)"""
    def __init__(self, b=0, d=0, f=0, k0=1):
        self.A = (d-f)*b**3/k0
        self.B = b
        self.D = d
        self.F = f
        self.k0 = k0

    def pulse_signal(self, t):
        a, b = self.A, self.B
        return a*t**2*np.exp(-b*t)

    def tau(self, t): # pulse-step signal
        if t == np.inf:
            return self.D
        a,b,f,k0 = self.A, self.B, self.F, self.k0
        return a/b*(k0/b**2 - (k0/b**2 + k0/b*t + (k0-b)/2*t**2) * np.exp(-b*t))+f

    def d_tau(self, t): # derivative of pulse-step signal
        a,b,k0 = self.A, self.B, self.k0
        return a*((k0-b)/2*t + 1) * t * np.exp(-b*t)

    def dd_tau(self, t): # second derivative of pulse-step signal
        a,b,k0 = self.A, self.B, self.k0
        return a*(-b*(k0-b)/2*t**2+(k0-2*b)*t+1)*np.exp(-b*t)

    def max_speed_time(self): # time at which d_tau is a maximum
        a,b,k0 = self.A, self.B, self.k0
        return (2*b-k0 - np.sqrt(2*b**2-2*b*k0+k0**2))/(b**2-b*k0)

    def max_speed(self):# maximum speed
        t = self.max_speed_time()
        return self.d_tau(t)

    def change_values(self, actions, f): # method to change the value of the parameters
        b,d,k0 = actions
        self.A = (d-f)*b**3/(k0)
        self.B = b
        self.D = d
        self.F = f
        self.k0 = k0

    def energy_used(self): # enrgy spent by the system form time zero to infinity (integration of (d_tau)^2 from 0 to inf)
        a,b,k0, d, f = self.A, self.B, self.k0, self.D, self.F
        value = (b**2+3*k0**2)/4*(d-f)**2*b/(4*k0**2)/4
        return value

    def get_parameters(self):
        return self.A, self.B, self.D, self.F, self.k0


def get_environment(control_type, noise, number_of_motors, time_limit = 2, extra_info = {}):  # use this function to obtained the desired environment given the parameters
    print("Creating {} Environment with{} noise using {} motor{}".format(control_type, "out"*(not noise), number_of_motors, "s"*(number_of_motors>1)))
    extra_info_copy = extra_info.copy()

    if control_type == "OpenLoop" and not noise:
        from Environments import OpenLoopNoiselessEnv
        if number_of_motors in [1,2]:
            env = OpenLoopNoiselessEnv(n_motors = number_of_motors, time_limit = time_limit)
        else:
            raise NotImplementedError("Number of motors hasn't been implemented yet")

    elif (control_type == "CythonOpenLoop" or control_type[:-1] == "CythonOpenLoop") and not noise:
        if control_type == "CythonOpenLoop":  # in case of addition of other environments that inherit from CythonOpenLoop Environment (control_type could be "CythonnOpenLoop2" to refer to them)
            from Environments import CythonOpenLoopNoiselessEnv as CyEnv

        if "weights" in [key for key in extra_info_copy.keys()]:
            aux_weights = extra_info_copy["weights"]
            del extra_info_copy["weights"]
        else:
            aux_weights = [1, 1, 1, 1]

        if number_of_motors in [1,2,3]:
            env = CyEnv(n_motors = number_of_motors, time_limit = time_limit, weights=aux_weights)
        else:
            raise NotImplementedError("Number of motors hasn't been implemented yet")

    for key, val in extra_info_copy.items():
        if key == "search_bounds":  # [low, high]
            env.change_action_space(val[0], val[1])
        elif key == "observation_bounds":  # [low, high]
            env.change_observation_space(val[0], val[1])
        elif key == "weights":
            env.change_weights(val)
        elif key == "n_actions":
            env.change_n_actions(val)
        elif key == "reward_mode":
            env.change_reward_mode(val)
        else:
            raise ValueError("This key doesn't represent anything: {}".format(key))

    print("Environment created!")
    return env


def open_info(folder):  # this function gets the information from an info file of training to a dictionary (might be useful for analysis post training)
    info_dictionary = {}
    n_actions = None
    n_hidden_layers = [2,2,2]
    import os
    if os.path.exists(folder+"/checkpoint"):
        os.remove(folder+"/checkpoint")
    with open(folder+"/info",'r') as file:
        for line in file:
            if line[:len("angle")] == "angle":
                info_dictionary["angle"] = line[len("angle")+1:-1]
            elif line[:len("initial_angle")] == "initial_angle":
                info_dictionary["initial_angle"] = line[len("initial_angle")+1:-1]
            elif line[:7] == "lambdas":
                info_dictionary["lambdas"] = eval(line[len("lambdas")+1:-1])
            elif line[:len("high")] == "high":
                info_dictionary["high"] = np.array(eval(line[len("high")+1:-1]))
            elif line[:len("low")] == "low":
                info_dictionary["low"] = np.array(eval(line[len("low")+1:-1]))
            elif line[:len("hidden_layers")] == "hidden_layers":
                info_dictionary["hidden_layers"] = eval(line[len("hidden_layers")+1:-1])
            elif line[:len("optimizer")] == "optimizer":
                info_dictionary["optimizer"] = line[len("optimizer")+1:-1]
            elif line[:len("alpha")] == "alpha":
                info_dictionary["alpha"] = float(line[len("alpha")+1:-1])
            elif line[:len("beta")] == "beta":
                info_dictionary["beta"] = float(line[len("beta")+1:-1])
            elif line[:len("actor_q1_q2")] == "actor_q1_q2":
                n_hidden_layers = eval(line[len("actor_q1_q2")+1:-1])
            elif line[:len("number_of_motors")] == "number_of_motors":
                info_dictionary["number_of_motors"] = int(line[len("number_of_motors")+1:-1])
            elif line[:len("control_type")] == "control_type":
                info_dictionary["control_type"] = line[len("control_type")+1:-1]
            elif line[:len("noise")] == "noise":
                info_dictionary["noise"] = eval(line[len("noise")+1:-1])
            elif line[:len("time_limit")] == "time_limit":
                info_dictionary["time_limit"] = float(line[len("time_limit")+1:-1])
            elif line[:len("obs_space_low")] == "obs_space_low":
                info_dictionary["observation_space_low"] = np.array(eval(line[len("obs_space_low")+1:-1]))
            elif line[:len("obs_space_high")] == "obs_space_high":
                info_dictionary["observation_space_high"] = np.array(eval(line[len("obs_space_high")+1:-1]))
            elif line[:len("n_actions")] == "n_actions":
                info_dictionary["n_actions"] = eval(line[len("n_actions")+1:-1])
            elif line[:len("batch_size")] == "batch_size":
                info_dictionary["batch_size"] = int(line[len("batch_size")+1:-1])
            elif line[:len("number")] == "number":
                info_dictionary["suffix"] = line[len("number")+1:-1]

    def convert_hidden_layer_info(h_l, n_hl):  # separates the hidden layers sizes into a list with 3 lists containing the size of the hidden layers for the actor network, for the critic 1 and for the critic 2
        aux = []
        aux2 = []
        for i, n in enumerate(n_hl):
            aux += [i]*n
            aux2.append([])

        for aux_, hl_s in zip(aux, h_l):
            aux2[aux_].append(hl_s)

        return aux2
    info_dictionary["hidden_layers"] = convert_hidden_layer_info(info_dictionary["hidden_layers"], n_hidden_layers)

    if n_actions is None:
        info_dictionary["n_actions"] = [[1,1,1]]*info_dictionary["number_of_motors"]

    return info_dictionary


def normalize_(x, low, high):  # for states and actions
    return (x-(high+low)/2)/((high-low)/2)


def unnormalize_(x, low, high):  # for states and actions
    return (high-low)/2*x+(high+low)/2