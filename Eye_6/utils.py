import numpy as np
import quaternion


def matrix_multiplication(matrix_list): # multiplies a list of arrays
    I = np.identity(matrix_list[0].shape[0])
    new_matrix = I
    for matrix in matrix_list:
        new_matrix = np.matmul(new_matrix, matrix)
    return new_matrix


def get_quaternion_distance(point1, point2):
    return max(1-np.dot(point1, point2)**2,0)


def get_quaternion_angle(q1, q2):
    return np.arccos(min(2*np.dot(q1,q2)**2-1,1))


def calculate_accuracy(qd,qa):
    rot_d = quaternion.as_rotation_vector(quaternion.from_float_array(qd))
    rot_a = quaternion.as_rotation_vector(quaternion.from_float_array(qa))
    return np.sum((rot_a[1:]-rot_d[1:])**2)


def generate_desired_angle(angle, min_a, max_a):
    a = angle
    if angle == 'r':
        a = 0
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

    def tau(self, t):
        if t == np.inf:
            return self.D
        a,b,f,k0 = self.A, self.B, self.F, self.k0
        return a/b*(k0/b**2 - (k0/b**2 + k0/b*t + (k0-b)/2*t**2) * np.exp(-b*t))+f

    def d_tau(self, t):
        a,b,k0 = self.A, self.B, self.k0
        return a*((k0-b)/2*t + 1) * t * np.exp(-b*t)

    def dd_tau(self, t):
        a,b,k0 = self.A, self.B, self.k0
        return a*(-b*(k0-b)/2*t**2+(k0-2*b)*t+1)*np.exp(-b*t)

    def max_speed_time(self):
        a,b,k0 = self.A, self.B, self.k0
        return (2*b-k0 - np.sqrt(2*b**2-2*b*k0+k0**2))/(b**2-b*k0)

    def max_speed(self):
        t = self.max_speed_time()
        return self.d_tau(t)

    def change_values(self, actions, f):
        b,d,k0 = actions
        self.A = (d-f)*b**3/(k0)
        self.B = b
        self.D = d
        self.F = f
        self.k0 = k0

    def energy_used(self):
        a,b,k0, d, f = self.A, self.B, self.k0, self.D, self.F
        value = (b**2+3*k0**2)/4*(d-f)**2*b/(4*k0**2)/4
        return value

    def get_parameters(self):
        return self.A, self.B, self.D, self.F, self.k0


def get_environment(control_type, noise, number_of_motors, time_limit = 2, extra_info = {}):  # search_bounds = None, file_name = None):
    print("Creating {} Environment with{} noise using {} motor{}".format(control_type, "out"*(not noise), number_of_motors, "s"*(number_of_motors>1)))
    if control_type == "OpenLoop" and not noise:
        from Environments import OpenLoopNoiselessEnv
        if number_of_motors in [1,2]:
            env = OpenLoopNoiselessEnv(n_motors = number_of_motors, time_limit = time_limit)
        else:
            raise NotImplementedError("Number of motors hasn't been implemented yet")

    elif control_type == "LinearOpenLoop" and not noise:
        from Environments import LinearOpenLoopNoiselessH
        if number_of_motors in [1]:
            env = LinearOpenLoopNoiselessH()
        else:
            raise NotImplementedError("Number of motors hasn't been implemented yet (LowPass): {}".format(number_of_motors))

    elif control_type == "LookupOpenLoop" and not noise:
        from Environments import LookupEnvironment
        file_name = extra_info["file_name"]
        lut_method = extra_info["lu_method"]
        if number_of_motors in [1]:
            env = LookupEnvironment(file_name, lu_method=lut_method)
        else:
            raise NotImplementedError("This number of motors hasn't been implemented yer (Lookup): {}".format(number_of_motors))
    elif control_type == "CythonOpenLoop" and not noise:
        from Environments import CythonOpenLoopNoiselessEnv
        if number_of_motors in [1,2,3]:
            env = CythonOpenLoopNoiselessEnv(n_motors = number_of_motors, time_limit = time_limit)
        else:
            raise NotImplementedError("Number of motors hasn't been implemented yet")
    elif control_type == "LowPassOpenLoop" and not noise:
        from Environments import LowPassOpenLoopEnv
        if "n_actions" in [key for key in extra_info.keys()]:
            aux_nactions = extra_info["n_actions"]
            del extra_info["n_actions"]
        else:
            aux_nactions = [[1, 1, 1]]*number_of_motors
        if "search_bounds" in [key for key in extra_info.keys()]:
            aux_search_bounds = extra_info["search_bounds"]
            del extra_info["search_bounds"]
        else:
            aux_search_bounds = None
        if "weights" in [key for key in extra_info.keys()]:
            aux_weights = extra_info["weights"]
            del extra_info["weights"]
        else:
            aux_weights = [1, 1, 1, 1]
        if "reward_mode" in [key for key in extra_info.keys()]:
            aux_reward_mode = extra_info["reward_mode"]
            del extra_info["reward_mode"]
        else:
            aux_reward_mode = 2 + (len(aux_weights) == 5)
        env = LowPassOpenLoopEnv(n_motors = number_of_motors, n_actions = aux_nactions, search_bounds = aux_search_bounds, weights = aux_weights, reward_mode = aux_reward_mode)
    else:
        raise NotImplementedError("Type of environment hasn't been implemented: {} (noise = {})".format(control_type, noise))

    for key, val in extra_info.items():
        if key == "search_bounds":  # [low, high]
            env.change_action_space(val[0], val[1])
        elif key == "observation_bounds":  # [low, high]
            env.change_observation_space(val[0], val[1])
        elif key == "weights":
            env.change_weights(val)
        elif key == "n_actions":
            env.change_n_actions(val)
        elif key == "file_name" or key == "lu_method":
            pass
        elif key == "logged_b":
            env.change_b_is_logged(val)
        else:
            raise ValueError("This key doesn't represent anything: {}".format(key))

    print("Environment created!")
    return env


def calc_rew(weight, individual_rewards):
    return -np.log10(-np.dot(weight, individual_rewards)/np.linalg.norm(weight))


def prestore_replaypool(agent, weight, search_bounds, control_type, noise, n_motors, initial_orientation, desired_orientation):
    if n_motors == 3:
        raise NotImplementedError("Invalid number of motors: {}".format(n_motors))
    if isinstance(desired_orientation, list) or isinstance(initial_orientation, list):
        raise NotImplementedError("desired orientation or initial orientation are lists")

    low = search_bounds[0]
    high = search_bounds[1]

    filename = "ReplayPool_"+control_type+"Noiseless"*(not noise)+"_"+str(n_motors)+"_v2"
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append([eval(x) for x in line.rstrip().split("\t")])

    print(len(data))
    for i, (state, action, ind_rew, done, new_state, set_quat) in enumerate(data):
        cond1 = (initial_orientation == 'o' or initial_orientation == 0) and not state[n_motors == 2] == 0  # it's suppose to start at origin but doesn't
        cond2 = not (state[n_motors == 1] == 0 and state[(n_motors == 1)+2] == 0)  # the type of motion doesn't reflect the initial and desired orientation (horizontal motion doesn't have a initial and desired pitch of 0)
        cond3 = not (np.all(low<=action) and np.all(action<=high))  # at least one of the actions is out of search bounds
        if cond1 or cond2 or cond3:
            continue
        des_yaw = 0
        des_pitch = 0
        if n_motors != 2:
            if desired_orientation == 'r':
                des_yaw = np.random.uniform(-45, 45)
            else:
                des_yaw = desired_orientation
        if n_motors != 1:
            if desired_orientation == 'r':
                des_pitch = np.random.uniform(-45, 45)
            else:
                des_pitch = desired_orientation

        des_ori = [0,des_yaw, des_pitch]
        des_quat = quaternion.as_float_array(quaternion.from_rotation_vector(np.radians(des_ori)))
        r_acc = -calculate_accuracy(des_quat, set_quat)

        fin_reward = [r_acc]+ind_rew[1:]

        state_ = state[:2]+[des_yaw, des_pitch]
        new_state_ = new_state[:2]+[des_yaw, des_pitch]

        agent.remember(np.array(state_)/180, normalize_(action, low, high), calc_rew(weight, fin_reward), np.array(new_state_)/180, done)
        # print(i, state)


def first_order_linear_horizontal_sysid(final_d, initial_f):
    from Environments import SysIDEnv
    sysid = SysIDEnv()
    sysid.reset()
    data = [sysid.premove([0,initial_f,0])]
    done = False
    while not done:
        state, _, done, _ = sysid.step([0,final_d,0])
        data.append(state)
    yaw = np.degrees(np.array([quaternion.as_rotation_vector(quaternion.from_float_array(x))[1] for x in data]))
    time_ = np.array([x*sysid.time_step for x in range(len(yaw))])
    y_f = yaw[-1]
    y_i = yaw[0]
    w0 = -np.sum(np.log((yaw[1:-3]-y_f)/(y_i-y_f)) * time_[1:-3]) / np.sum(time_[1:-3] ** 2)
    return (y_f-y_i)/(final_d-initial_f), w0, data


def first_order_linear_vertical_sysid(final_d, initial_f):
    from Environments import SysIDEnv
    sysid = SysIDEnv()
    sysid.time_limit = 5
    sysid.reset()
    data = [sysid.premove([initial_f,0,initial_f])]
    done = False
    while not done:
        state, _, done, _ = sysid.step([final_d,0,final_d])
        data.append(state)
    pitch = np.degrees(np.array([quaternion.as_rotation_vector(quaternion.from_float_array(x))[2] for x in data]))
    time_ = np.array([x*sysid.time_step for x in range(len(pitch))])
    p_f = pitch[-1]
    p_i = pitch[0]
    w0 = -np.sum(np.log((pitch[1:-3]-p_f)/(p_i-p_f)) * time_[1:-3]) / np.sum(time_[1:-3] ** 2)
    return (p_f-p_i)/(final_d-initial_f), w0, data


def graph(x,y, title = "", xlabel = "", ylabel = "", ax = None, show = True):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(x,y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if show:
        plt.show()


def open_info(folder):
    angle, initial_angle, lamb, high, low, = None, None, None, None, None
    hidden_layers, last_best, optimizer, alpha, beta = None, None, None, None, None
    n_motors, control_type, noise, time_limit = None, None, None, None
    obs_low, obs_high, n_actions = None, None, None
    n_hidden_layers = [2,2,2]
    import os
    if os.path.exists(folder+"/checkpoint"):
        os.remove(folder+"/checkpoint")
    with open(folder+"/info",'r') as file:
        for line in file:
            if line[:len("angle")] == "angle":
                angle = line[len("angle")+1:-1]
            elif line[:len("initial_angle")] == "initial_angle":
                initial_angle = line[len("initial_angle")+1:-1]
            elif line[:7] == "lambdas":
                lamb = eval(line[8:-1])
            elif line[:len("high")] == "high":
                high = np.array(eval(line[len("high")+1:-1]))
            elif line[:len("low")] == "low":
                low = np.array(eval(line[len("low")+1:-1]))
            elif line[:len("hidden_layers")] == "hidden_layers":
                hidden_layers = eval(line[len("hidden_layers")+1:-1])
            elif line[:len("episode_of_best_model")] == "episode_of_best_model":
                last_best = int(line[len("episode_of_best_model")+1:-1])
            elif line[:len("optimizer")] == "optimizer":
                optimizer = line[len("optimizer")+1:-1]
            elif line[:len("alpha")] == "alpha":
                alpha = float(line[len("alpha")+1:-1])
            elif line[:len("beta")] == "beta":
                beta = float(line[len("beta")+1:-1])
            elif line[:len("actor_q1_q2")] == "actor_q1_q2":
                n_hidden_layers = eval(line[len("actor_q1_q2")+1:-1])
            elif line[:len("number_of_motors")] == "number_of_motors":
                n_motors = int(line[len("number_of_motors")+1:-1])
            elif line[:len("control_type")] == "control_type":
                control_type = line[len("control_type")+1:-1]
            elif line[:len("noise")] == "noise":
                noise = eval(line[len("noise")+1:-1])
            elif line[:len("time_limit")] == "time_limit":
                time_limit = float(line[len("time_limit")+1:-1])
            elif line[:len("obs_space_low")] == "obs_space_low":
                obs_low = np.array(eval(line[len("obs_space_low")+1:-1]))
            elif line[:len("obs_space_high")] == "obs_space_high":
                obs_high = np.array(eval(line[len("obs_space_high")+1:-1]))
            elif line[:len("n_actions")] == "n_actions":
                n_actions = eval(line[len("n_actions")+1:-1])
    def convert_hidden_layer_info(h_l, n_hl):
        aux = []
        aux2 = []
        for i, n in enumerate(n_hl):
            aux += [i]*n
            aux2.append([])
        # print(aux)
        # print(aux2)
        for aux_, hl_s in zip(aux, h_l):
            # print(aux_, hl_s)
            aux2[aux_].append(hl_s)
        # print(aux2)
        return aux2
    hidden_layers = convert_hidden_layer_info(hidden_layers, n_hidden_layers)

    if n_actions is None:
        n_actions = [[1,1,1]]*n_motors

    return angle, initial_angle, lamb, high, low, hidden_layers, last_best, optimizer, alpha, beta, n_motors, control_type, noise, time_limit, obs_low, obs_high, n_actions


def newtons_method(fun, d_fun, x0, error=0.001, y_val=0, max_iterations = 5, x_lower_bound = -np.inf, x_upper_bound = np.inf):
    iter = 0
    try:
        x_next = x0.copy()
    except AttributeError:
        x_next = float(x0)
    y_next= fun(x_next)-y_val
    while (np.abs(y_next)>error).any() and iter<max_iterations:
        # print(x_next, fun(x_next), y_val, d_fun(x_next))
        deriv = d_fun(x_next)
        if deriv == 0:
            return np.inf, fun(np.inf)
        x_next -= y_next/deriv
        if x_next < x_lower_bound or x_upper_bound < x_next:
            return np.inf, fun(np.inf)
        y_next= fun(x_next)-y_val
        iter += 1
    if iter == max_iterations:
        return np.inf, fun(np.inf)
    return x_next, y_next+y_val


def bisection_method(fun, x_low, x_high, error=0.001, y_val=0):
    y_low = fun(x_low)-y_val
    y_high = fun(x_high)-y_val
    if y_low*y_high>0:
        return bisection_method(fun, x_low, x_high*3, error= error, y_val=y_val)
        raise ValueError("Initial guesses for x_low and x_high are mistaken")
    y_next = y_high
    x_next = x_high
    while abs(y_next)>error:
        x_next = (x_low+x_high)/2
        y_next = fun(x_next)-y_val
        if y_low*y_next<0:
            x_high = x_next
        elif y_low*y_next>0:
            x_low = x_next
            y_low = y_next
        else:
            break
    return x_next, y_next+y_val


def ternary_search(fun, x_low, x_high, error = 0.001):
    while x_high - x_low>error:
        m1 = x_low+(x_high-x_low)/3
        m2 = x_high - (x_high - x_low) / 3
        y1 = fun(m1)
        y2 = fun(m2)
        cond = (y1<y2)
        n_cond = (not cond)

        x_low = x_low*n_cond + m1*cond
        x_high = x_high*cond + m2*n_cond
    return (x_low+x_high)/2


def golden_section_search(f, a, b, error = 0.001):
    """Golden-section search.

        Given a function f with a single local minimum in
        the interval [a,b], gss returns a subset interval
        [c,d] that contains the minimum with d-c <= tol.
    """

    invphi = 0.618033988749894903
    invphi2 = 0.381966011250105208

    (a, b) = (min(a, b), max(a, b))
    h = b - a
    if h <= error:
        return (a, b)

    # Required steps to achieve tolerance
    n = int(np.ceil(np.log(error / h) / np.log(invphi)))

    c = a + invphi2 * h
    d = a + invphi * h
    yc = f(c)
    yd = f(d)

    for k in range(n - 1):
        if yd < yc:  # yc < yd to find the minimum
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            yc = f(c)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            yd = f(d)
    if yc > yd:
        return (a+d)/2
    else:
        return (c+b)/2


def horizontal_line_maximum_search(fun, d_fun, x_low, x_high, error = 10**-6, y_error = 10**-9, depth = np.inf):
    """
    Will find a maximum for a function in an interval [x_low, x_high] such that:
    Conditions:
    f(x_low)<f(x_high) and
    f(x_high) is a local minimum and
    or
    f(x_low) and f(x_high) are local minima


    :param fun:
    :param d_fun:
    :param x_low:
    :param x_high:
    :param error:
    :return:
    """
    if depth <= 0:
        return []
    x_final = x_high
    y_final = fun(x_high)
    y_high = y_final
    y_low = fun(x_low)
    points = []
    while x_high - x_low > error:
        x_next = (x_low+x_high)/2
        y_next = fun(x_next)
        dy_next = d_fun(x_next)
        dy_next = np.sign(dy_next)
        # print("f({:.6f})={:.6f}, f({:.6f})={:.6f}, f({:.6f})={:.6f}, f'({:.6f})={:.6e}".format(x_low, y_low, x_next, y_next, x_high, y_high, x_next, dy_next))
        if y_high < y_next and dy_next <= 0:
            x_high = x_next
            y_high = y_next
        elif y_next < y_low or dy_next >= 0:
            x_low = x_next
            y_low = y_next
        elif y_error > abs(y_high-y_next):
            x_high = x_next
            y_high = y_next
        elif y_error > abs(y_low-y_next):
            x_low = x_next
            y_low = y_next
        else:
            depth -= 1
            points += horizontal_line_maximum_search(fun, d_fun, x_low, x_next, error = error, depth = depth)
            x_low = x_next
            y_low = y_next
    x_max = (x_low + x_high) / 2
    points.append(x_max)

    y_max = fun(x_max)
    x_new = (x_max+x_final)/2
    y_new = fun(x_new)
    dy_new = np.sign(d_fun(x_new))
    depth -= 1
    if y_new<y_final or dy_new>0:
        points += horizontal_line_maximum_search(fun, d_fun, x_new, x_final, error = error, depth = depth)
    elif y_max<y_new:
        points += horizontal_line_maximum_search(fun, d_fun, x_max, x_new, error = error, depth = depth)

    return points


def normalize_(x, low, high):
    return (x-(high+low)/2)/((high-low)/2)


def unnormalize_(x, low, high):
    return (high-low)/2*x+(high+low)/2


def generate_state(des_orie, ini_orie, n_motors, min_angle = -40, max_angle = 40):
    ini_part = None
    end_part = None
    if isinstance(ini_orie, str):
        if ini_orie == 'o':
            ini_part = [0,0]
        else:
            raise NotImplementedError("Invalid initial_orientation value: {}".format(ini_orie))
    elif isinstance(ini_orie, float) or isinstance(ini_orie, int):
        ini_part = np.array([ini_orie, ini_orie])*np.array([n_motors != 2, n_motors != 1])
    else:
        raise NotImplementedError("Invalid initial_orientation value: {}".format(ini_orie))

    if isinstance(des_orie, str):
        if des_orie == 'r':
            end_part = np.random.uniform(min_angle, max_angle, 2)*np.array([n_motors != 2, n_motors != 1])
        else:
            raise NotImplementedError("Invalid desired_orientation value: {}".format(des_orie))
    elif isinstance(des_orie, float) or isinstance(des_orie, int):
        end_part = np.array([des_orie, des_orie])*np.array([n_motors != 2, n_motors != 1])
    else:
        raise NotImplementedError("Invalid desired_orientation value: {}".format(des_orie))

    return np.concatenate((ini_part,end_part))