import tensorflow as tf
import threading
from queue import Queue
import signal
import time
import numpy as np
from utils import generate_desired_angle, normalize_, unnormalize_


def main(env_info, sac_info, training_info, use_gpu, with_thread=True):
    control_type, noise, number_of_motors, weights, time_limit, search_bounds, n_actions = env_info
    alpha, beta, epsilon, batch_size, buffer_size, hidden_layers, optimizer, update_q, update_alpha, start_using_actor, target_entropy_scale = sac_info
    initial_orientation, desired_orientation, folder_, load, max_episodes, rep = training_info

    # FILE NAMES
    score_file_name = "/score"  # file that will have the score obtained after each episode (currently disabled)
    average_score_file_name = "/average_score"  # file that will have the average score obtained after some amount of episodes episodes (currently 1000)
    info_file_name = "/info"  # file that will have the specifications of each training
    checkpoint_file = "/checkpt_params"  # file that will get the output of the Actor for a certain set of inputs
    loss_file_name = "/actor_c1_c2_alpha_loss"  # file that will have the loss values of the networks after each episode (currently disabled)
    average_loss_file_name = "/average_actor_c1_c2_alpha_loss"  # file that will have the average loss values of the networks after some amount of episodes (currently 1000)
    alpha_file = "/alpha_file"  # file that will have the entropy value after each episode (disabled)

    mean_of = 1000

    import os
    if not os.path.exists(folder_): # create folder where the trainings will be placed if it doesn't exist
        os.mkdir(folder_)

    folder = folder_+control_type+"_"+str(rep)
    if not os.path.exists(folder):  # creates folder for this training if it doesn't exist
        os.mkdir(folder)

    if use_gpu:  # if value is true and device as a GPU configured for training set it to be used
        physical_devices = tf.config.list_physical_devices('GPU')
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device,True)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"


    # ----------------- Threaad -----------------
    # Setup of thread that prints more information during training

    mess = None
    q = Queue()
    ex_q = Queue()
    def thread_func(q, exit_q):
        message = None
        while exit_q.empty():
            message = input()
            if message in ['q', 'p', 'c', 't', 's','m', 'j','f','l','i','d', 'h', 'u', 'n','a', 'r']: # acceptable inputs
                q.put(message)
            elif message == 'sleep':
                print("Thread is sleeping...")
                time.sleep(10)
                print("Thread woke up!")
            if message in ['q', 't']: # to end training (q- quit) or delete it (t - trash)
                while exit_q.empty():
                    pass
                return
        return
    threadx = threading.Thread(target=thread_func, args=(q, ex_q))
    threadx.setDaemon(True)
    # --------------------------------------------
    # -------------Signals ----------------------

    def signal_handler(*args):
        q.put("q")

    # signal.signal(signal.SIGKILL, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler) # if process is forcefully terminated it will save training and close

    # -------------------------------------------
    from utils import get_environment
    env = get_environment(control_type, noise, number_of_motors, time_limit=time_limit, extra_info={"search_bounds": search_bounds, "file_name": lookup_file, "weights": weights, "lu_method": lu_method, "logged_b": b_is_logged, "n_actions": n_actions, "reward_mode": reward_mode})  # get desired environment

    from SAC import Agent
    agent = Agent(input_dims = env.observation_space.shape, n_actions = env.action_space.shape[0], batch_size= batch_size, alpha=alpha, beta=beta, epsilon = epsilon,
                  chkpt_dir=folder, optimizer=optimizer, max_size=buffer_size, hidden_layers=hidden_layers, update_q = update_q, update_alpha = update_alpha,
                  start_using_actor=start_using_actor, target_entropy_scale=target_entropy_scale)  # initiate agent
    start_episode = 0
    if load:  # in case we are loading an existing neural networks
        agent.load_models()   # load weights into agent
        if os.path.getsize(folder+score_file_name)>0:   # find starting episode if score file was used
            with open(folder+score_file_name, 'r') as file:
                for _ in file:
                    start_episode += 1
        else:  # find starting episode if score file wasn't used. use average files instead to estimate current episode
            aux_prev = -mean_of
            with open(folder+average_score_file_name, 'r') as file:
                for line in file:
                    aux_this = int(line.rstrip().split()[0])
                    start_episode = aux_this+(aux_this-aux_prev)
                    aux_prev = aux_this
        if os.path.exists(folder + "/checkpoint"):
            os.remove(folder + "/checkpoint")
    else:  # create all files to store information
        with open(folder+score_file_name, 'w') as _:
            pass
        with open(folder+average_score_file_name, 'w') as _:
            pass
        with open(folder+checkpoint_file, 'w') as file:
            file.write("episode\t[desired_angle,B,D,k0]\t...\n")
        if os.path.exists(folder+"/bad"):
            os.remove(folder+"/bad")
        with open(folder+loss_file_name, 'w') as _:
            pass
        with open(folder+average_loss_file_name, 'w') as _:
            pass
        with open(folder+alpha_file, 'w') as _:
            pass

    # -------- exporting info to file -------------
    print("Exporting Training Information to File")
    with open(folder + info_file_name, 'w') as file:  # creates info file with the information below
        file.write("{}:\n".format(env.pulse_function[0].__doc__))
        file.write("initial_angle\t{}\n".format(initial_orientation))
        file.write("angle\t{}\n".format(desired_orientation))
        file.write("alpha\t{}\n".format(alpha))
        file.write("beta\t{}\n".format(beta))
        file.write("epsilon\t{}\n".format(epsilon))
        file.write("batch_size\t{}\n".format(batch_size))
        file.write("optimizer\t{}\n".format(optimizer))
        file.write("buffer_size\t{}\n".format(buffer_size))
        file.write("update_q\t{}\n".format(update_q))
        file.write("update_alpha\t{}\n".format(update_alpha))
        file.write("number\t{}\n".format(rep))
        file.write("lambdas\t{}\n".format(list(env.weight)))
        file.write("reward_mode\t{}\n".format(env.reward_mode))
        file.write("high\t{}\n".format(list(env.action_space.high)))
        file.write("low\t{}\n".format(list(env.action_space.low)))
        file.write("n_actions\t{}\n".format(n_actions))
        file.write("actor_q1_q2\t{}\n".format([len(hidden_layers[0]), len(hidden_layers[1]), len(hidden_layers[2])]))
        file.write("hidden_layers\t{}\n".format(list(np.concatenate(hidden_layers).flat)))
        file.write("number_of_motors\t{}\n".format(number_of_motors))
        file.write("control_type\t{}\n".format(control_type))
        file.write("noise\t{}\n".format(noise))
        file.write("time_limit\t{}\n".format(time_limit))
        file.write("update_q\t{}\nupdate_alpha\t{}\nstart_using_actor\t{}\n".format(update_q, update_alpha, start_using_actor))
        file.write("obs_space_low\t{}\n".format(list(env.observation_space.low)))
        file.write("obs_space_high\t{}\n".format(list(env.observation_space.high)))
    # ---------------------------------------------
    if with_thread:  # if manual updates are wanted during training start thread
        threadx.start()

    start_time = time.time()  # for estimation of time remaining
    score_history = []  # to record and consenquently store data
    action_history = []  # to record and consenquently store data
    loss_history = []  # to record and consenquently store data
    print("Starting Training...({})".format(rep))
    for i in range(start_episode+1, max_episodes+batch_size*int(load)+1):  # episode loop
        this_loop_i = i
        # reset the environment state and parameters to start training
        env.select_reset(desired_orientation, initial_orientation)
        done = False
        score = 0
        while not done:  # start training
            state = env.get_state()
            action = agent.choose_action(state)  # get action from agent
            if True in tf.math.is_nan(action):  # in case neural networks diverge
                print("episode {}: action_1 is bad. Exiting...".format(i))
                with open(folder+"/bad",'w'):  # creates bad file if this occurs
                    pass
                mess = 'bad'
                break
            next_state, reward, done, info = env.step(action)  # apply action to environment

            agent.remember(state, action, -np.log10(-reward), next_state, done)  # store results
            # agent.remember(state, action, reward, next_state, done)
            score += reward
            c1_loss, c2_loss, actor_loss, alpha_loss = agent.learn()  # update Neural Networks
            if None not in [c1_loss, c2_loss, actor_loss, alpha_loss]:
                loss_history.append([actor_loss, c1_loss, c2_loss, alpha_loss])
        if mess == 'bad':
            break

        score_history.append(score)
        # append last action applied. since the training is in open-loop is the only action applied to state
        action_history.append(list((env.action_space.high-env.action_space.low)/2*np.array(action)+(env.action_space.high+env.action_space.low)/2))

        if i%100 == 0 and i<1000:   # reset the environment state
            avg_score = np.mean(score_history[-100:])
            print('episode ', i, 'avg_score %.5f' % avg_score)
        if i%1000==0 and i>=1000:   # reset the environment state
            avg_score = np.mean(score_history[-100:])
            print('episode ', i, 'avg_score %.5f' % avg_score)

            # with open(folder+alpha_file, "a") as file:
            #     file.write("{}\t{}\n".format(i, agent.alpha))

        if i%5000 == 0:  # save model every n epsides (5000 now)
            agent.save_models()
        if i%5000 == 0:  # store information in files about the training every n episodes (5000)
            with open(folder+checkpoint_file, 'a') as file:  # store a screeenshot of the output of the actor for a set of inputs
                if desired_orientation == 'r':  # if the desired orientation is randomized
                    desired_yaw = [-30, -20, -10, 0, 10, 20, 30]
                    desired_pitch = [-30, -20, -10, 0, 10, 20, 30]
                    if number_of_motors == 2:
                        desired_yaw = [0]
                    elif number_of_motors == 1:
                        desired_pitch = [0]

                    desired_list = []
                    for d_y in desired_yaw:
                        for d_p in desired_pitch:
                            if d_y == d_p == 0:
                                continue
                            desired_list.append([d_y, d_p])
                else:  # if desired orientation is specified
                    desired_list = [[desired_orientation*(number_of_motors != 2), desired_orientation*(number_of_motors != 1)]]

                action_list = []
                for desired_ang in desired_list:
                    state_chk = np.array([0, 0, desired_ang[0]*(number_of_motors != 2), desired_ang[1]*(number_of_motors != 1)])
                    mu_chk, _ = agent.get_actor_output(normalize_(state_chk, env.observation_space.low, env.observation_space.high))
                    action_chk = unnormalize_(np.tanh(mu_chk), env.action_space.low, env.action_space.high)
                    action_list.append([desired_ang]+list(action_chk[0]))
                file.write(str(i))
                for ac in action_list:
                    file.write("\t[{}".format(ac[0]))
                    for aux in ac[1:]:
                        file.write(",{:.2f}".format(aux))
                    file.write("]")
                file.write("\n")

        if i%500000 == 0 and i>0 and len(score_history) > 0 and len(loss_history)>0 and len(action_history)>0:  # store score and losses to files every n episodes (500000 now)
            # print("Exporting Score to File")  # export every score value to a file. one per line. (disabled because it uses a lot of memory for long trainings)
            # with open(folder + score_file_name, 'a') as file:
            #     for j, (score_, action_) in enumerate(zip(score_history, action_history)):
            #         if j % (len(score_history) // 10) == 0:
            #             print("{}%", int(100 * j / len(score_history)))
            #         file.write("{}\t{}\n".format(action_, score_))

            print("Exporting Average Score to File")
            with open(folder + average_score_file_name, 'a') as file:  # stores a mean of the score value every mean_of episodes
                for j in range(0, len(score_history), mean_of):
                    if j + mean_of <= len(score_history):
                        file.write("{}\t{}\n".format(i-len(score_history)+j, np.mean(score_history[j:j+mean_of])))

            score_history = []
            action_history = []

            # print("Exporting Losses to File")  # stores the loss values of the networks
            # # loss_history = np.array(loss_history)
            # with open(folder + loss_file_name, 'a') as file:
            #     for j, (actor_loss_, c1_loss_, c2_loss_, alpha_loss_) in enumerate(loss_history):
            #         if j % (len(loss_history) // 10) == 0:
            #             print("{}%", int(100 * j / len(loss_history)))
            #         file.write("{}\t{}\t{}\t{}\n".format(actor_loss_, c1_loss_, c2_loss_, alpha_loss_))

            print("Exporting Average Losses to File")
            with open(folder + average_loss_file_name, 'a') as file:  # stores the mean of the loss values of the networks
                for j in range(0, len(loss_history), mean_of):
                    if j + mean_of <= len(loss_history):
                        average_actor_loss, average_c1_loss, average_c2_loss, average_alpha_loss = np.mean([[x[0], x[1], x[2], x[3]] for x in loss_history[j:j+mean_of]], axis = 0)
                        file.write("{}\t{}\t{}\t{}\t{}\n".format(i-len(loss_history)+j, average_actor_loss, average_c1_loss, average_c2_loss, average_alpha_loss))

            loss_history = []

        if not q.empty():  # if it receives input from user during training
            mess = q.get()
            if mess == 'p':  # pause training
                print("Paused")
                while mess not in ['c','t','q']:
                    mess = q.get()
                    pass
                print("Continuing...")
            if mess in ['q','j']:
                pass
            if mess in ['q', 't']:  # stop training
                print("Exiting...")
                break
            if mess == 'i':  # print episode and current average score
                avg_score = np.mean(score_history[-100:])
                print('episode ', i, 'avg_score %.5f' % avg_score)
            if mess == 'm':  # print the current output of the actor for a certain set of inputs
                if desired_orientation == 'r':
                    desired_list = [-30,-20,-10,-5,5,10,15,20,25,30]
                else:
                    desired_list = [desired_orientation]
                print()
                print("low = ", env.action_space.low)
                print("high = ", env.action_space.high)
                for desired_ang in desired_list:
                    state_monitor = np.array([0, 0, desired_ang*(number_of_motors != 2), desired_ang*(number_of_motors != 1)])
                    mu_monitor, _ = agent.get_actor_output(normalize_(state_monitor, env.observation_space.low, env.observation_space.high))
                    action_monitor = unnormalize_(np.tanh(mu_monitor), env.action_space.low, env.action_space.high)
                    string_to_print = "["
                    for aux in action_monitor[0]:
                        string_to_print += "{:.2f}, ".format(aux)
                    string_to_print = string_to_print[:-2]+"]"
                    print(desired_ang,"-> state = ({}) average_inputs: [B,D,k0] =".format(list(state_monitor)), string_to_print)
                print()

            if mess == 'f':  # estimates the time remaining of training
                time_remaining = (time.time()-start_time)/i*(max_episodes-i)
                print("\ntime remaining: {} hours for this environment\n".format(time_remaining))

            if mess == 'a':  # prints the results from applying the current actions obtained from applying a certain set of inputs to the actor
                if desired_orientation == 'r':
                    desired_list = [-30,-20,-10,-5,5,10,15,20,25,30]
                else:
                    desired_list = [desired_orientation]
                print()
                print("low = ", env.action_space.low)
                print("high = ", env.action_space.high)
                for desired_ang in desired_list:
                    state_monitor = np.array([0, 0, desired_ang*(number_of_motors != 2), desired_ang*(number_of_motors != 1)])
                    mu_monitor, _ = agent.get_actor_output(normalize_(state_monitor, env.observation_space.low, env.observation_space.high))
                    action_monitor = unnormalize_(np.tanh(mu_monitor), env.action_space.low, env.action_space.high)
                    env.select_reset(desired_ang, 'o')
                    done = False
                    score = 0
                    while not done:
                        action = action_monitor[0]
                        next_state, reward, done, info = env.step(action, preprocess = False)
                        score += reward
                    print(desired_ang, list(action), info["reward_info"], score)
                print()
            if mess == 'r':  # prints the results of  applying the same action to the environment 10 times
                if desired_orientation == 'r':
                    desired_r_angle = 30
                else:
                    desired_r_angle = desired_orientation
                print()
                print("low = ", env.action_space.low)
                print("high = ", env.action_space.high)
                for _ in range(10):
                    state_random = env.select_reset(desired_r_angle, 'o')
                    action_random = agent.choose_action(state_random)
                    done = False
                    score = 0
                    while not done:
                        _, reward, done, info = env.step(action_random)
                        score += reward
                    print(desired_r_angle, list(env.get_raw_state()), list(np.array(unnormalize_(action_random, env.action_space.low, env.action_space.high))), info["reward_info"], score)
                print()
    if mess not in ['t','bad']:  # saves model if mess (message) is not of the delete training type
        agent.save_models()

    # print("Exporting Score to File")  # store score
    # with open(folder+score_file_name,'a') as file:
    #     for i, (score_, action_) in enumerate(zip(score_history,action_history)):
    #         if i%(len(score_history)//10) == 0:
    #             print("{}%",int(100*i/len(score_history)))
    #         file.write("{}\t{}\n".format(action_, score_))

    print("Exporting Average Score to File")  # store average score
    with open(folder + average_score_file_name, 'a') as file:
        for j in range(0, len(score_history), mean_of):
            if j + mean_of <= len(score_history):
                file.write("{}\t{}\n".format(this_loop_i-len(score_history)+j, np.mean(score_history[j:j+mean_of])))

    # print("Exporting Losses to File")  # store score
    # # loss_history = np.array(loss_history)
    # with open(folder+loss_file_name, 'a') as file:
    #     for i, (actor_loss_, c1_loss_, c2_loss_, alpha_loss_) in enumerate(loss_history):
    #         if i%(len(loss_history)//10) == 0:
    #             print("{}%",int(100*i/len(loss_history)))
    #         file.write("{}\t{}\t{}\t{}\n".format(actor_loss_, c1_loss_, c2_loss_, alpha_loss_))

    print("Exporting Average Losses to File")
    with open(folder + average_loss_file_name, 'a') as file:  # store average loss values
        for j in range(0, len(loss_history), mean_of):
            if j+mean_of<=len(loss_history):
                average_actor_loss, average_c1_loss, average_c2_loss, average_alpha_loss = np.mean([[x[0], x[1], x[2], x[3]] for x in loss_history[j:j+mean_of]], axis = 0)
                file.write("{}\t{}\t{}\t{}\t{}\n".format(this_loop_i-len(score_history)+j, average_actor_loss, average_c1_loss, average_c2_loss, average_alpha_loss))


    ex_q.put("Exit")  # exits thread
    # threadx.join()


if __name__ == "__main__":
    CONTROL_TYPE = "CythonOpenLoop"
    NOISE = False
    N_MOTORS = 3
    if N_MOTORS == 1:
        WEIGHTS = [9608177.57470925, 1, 755460.0846412841, 2962997.77056245]  # (1 motor) B = 60, D = 20, F = 0, k0 = 16.52
    elif N_MOTORS == 2:
        WEIGHTS = [33, 1, 4.86825024727549497e+6, 100, 4.65830844833322917e+5]
    elif N_MOTORS == 3:
        WEIGHTS = [30,1,3.47e+6,0.001,3.04e+6] # LP B = 40 ...

    TIME_LIMIT = 20
    # Search Bounds
    SEARCH_BOUNDS = [[],[]]
    horizontal = [[1, -30, 1], [150, 30, 25]]
    vertical = [[1, -30, 1],[150, 30, 25]]
    for i, x in enumerate([N_MOTORS != 1, N_MOTORS != 2, N_MOTORS != 1]):
        if i == 1 and x:
            for j in range(2):
                SEARCH_BOUNDS[j] += horizontal[j]
        elif i != 1 and x:
            for j in range(2):
                SEARCH_BOUNDS[j] += vertical[j]
    for i in range(2):
        SEARCH_BOUNDS[i] = np.array(SEARCH_BOUNDS[i])
    # SEARCH_BOUNDS = [np.array([15, -40, 5]),np.array([115, 40, 25])] if N_MOTORS == 1 else [np.array([2, -70, 1]*2),np.array([250, 70, 30]*2)]# [low, high]
    N_ACTIONS = [[1,1,1],[1,1,1],[1,1,1]]  # each list inside is for one motor. if one of those values is 0 it should be fixed
    ENVIRONMENT_INFO = (CONTROL_TYPE, NOISE, N_MOTORS, WEIGHTS, TIME_LIMIT, SEARCH_BOUNDS, N_ACTIONS)

    ALPHA = 0.0001
    BETA =  0.0001
    EPSILON = 0.01
    BATCH_SIZE = 128 #2*8192
    BUFFER_SIZE = 1000000
    HIDDEN_LAYERS = [[8,8],[16,8,8],[16,8,8]] # [[8,8,6],[6,6,6],[6,6,6]]  # [[16,16,16],[16,8,8],[16,8,8]]
    OPTIMIZER = "Adam"
    UPDATE_Q = True
    UPDATE_ALPHA = False
    START_USING_ACTOR = 0
    TARGET_ENTROPY_SCALE = 1
    SAC_INFO = (ALPHA, BETA, EPSILON, BATCH_SIZE, BUFFER_SIZE, HIDDEN_LAYERS, OPTIMIZER, UPDATE_Q, UPDATE_ALPHA, START_USING_ACTOR, TARGET_ENTROPY_SCALE)

    INITIAL_ORIENTATION = 's'
    DESIRED_ORIENTATION = 'r'
    LOAD = False
    MAX_EPISODES = 250 # 100*10*10**5
    REP = "test"
    FOLDER = "TrainedPolicies/"
    TRAINING_INFO = (INITIAL_ORIENTATION, DESIRED_ORIENTATION, FOLDER, LOAD, MAX_EPISODES, REP)

    USE_GPU = True

    WITH_THREAD = True

    def process_system_inputs(args, env_info, sac_info, training_info, use_gpu, with_thread):
        control_type, noise, n_motors, weights, time_limit, search_bounds, n_actions = env_info
        alpha, beta, epsilon, batch_size, buffer_size, hidden_layers, optimizer, update_q, update_alpha, start_using_actor, target_entropy_scale = sac_info
        initial_orientation, desired_orientation, folder_, load, max_episodes, rep = training_info
        use_gpu_ = use_gpu
        with_thread_ = with_thread
        if len(args)%2 != 0:
            raise IndexError("System Arguments don't come in pairs: {}".format(args))
        print()
        args = np.reshape(args, (-1,2))
        for (arg1, arg2) in args:
            if arg1 == "-gpu":
                use_gpu_ = bool(eval(arg2))
                print("Changing gpu to:", use_gpu_)
            elif arg1 == "-rep":
                rep = arg2
                print("Changing rep to:", rep)
            elif arg1 == "-optimizer":
                optimizer = arg2
                print("Changing optimizer to:", optimizer)
            elif arg1 == "-load":
                load = bool(eval(arg2))
                eval("print('Changing {} to:', {})".format(arg1[1:], arg1[1:]))
            elif arg1 == "-start_using_actor":
                start_using_actor = int(arg2)
                eval("print('Changing {} to:', {})".format(arg1[1:], arg1[1:]))
            elif arg1 == "-batch_size":
                batch_size = int(arg2)
                eval("print('Changing {} to:', {})".format(arg1[1:], arg1[1:]))
            elif arg1 == "-alpha":
                alpha = float(arg2)
                eval("print('Changing {} to:', {})".format(arg1[1:], arg1[1:]))
            elif arg1 == "-beta":
                beta = float(arg2)
                eval("print('Changing {} to:', {})".format(arg1[1:], arg1[1:]))
            elif arg1 == "-epsilon":
                epsilon = float(arg2)
                eval("print('Changing {} to:', {})".format(arg1[1:], arg1[1:]))
            elif arg1 == "-target_entropy_scale":
                target_entropy_scale = float(arg2)
                eval("print('Changing {} to:', {})".format(arg1[1:], arg1[1:]))
            elif arg1 == "-update_alpha":
                update_alpha = bool(eval(arg2))
                eval("print('Changing {} to:', {})".format(arg1[1:], arg1[1:]))
            elif arg1 == "-update_q":
                update_q = bool(eval(arg2))
                eval("print('Changing {} to:', {})".format(arg1[1:], arg1[1:]))
            elif arg1 == "-hidden_layers":
                aux = eval(arg2)
                if len(aux) == 3 and all([isinstance(x, list) for x in aux]):
                    hidden_layers = aux
                    eval("print('Changing {} to:', {})".format(arg1[1:], arg1[1:]))
                else:
                    raise ValueError("Argument of -hidden_layers is invalid")
            elif arg1 == "-control_type":
                control_type = arg2
                eval("print('Changing {} to:', {})".format(arg1[1:], arg1[1:]))
            elif arg1 == "-max_episodes":
                max_episodes = int(arg2)
                eval("print('Changing {} to:', {})".format(arg1[1:], arg1[1:]))
            elif arg1 == "-n_actions":
                n_actions = eval(arg2)
                eval("print('Changing {} to:', {})".format(arg1[1:], arg1[1:]))
            elif arg1 == "-weights":
                weights = eval(arg2)
                eval("print('Changing {} to:', {})".format(arg1[1:], arg1[1:]))
            elif arg1 == "-buffer_size":
                buffer_size = int(arg2)
                eval("print('Changing {} to:', {})".format(arg1[1:], arg1[1:]))
            elif arg1 == "-desired_orientation":
                if arg2 == 'r':
                    desired_orientation = 'r'
                else:
                    desired_orientation = int(arg2)
                eval("print('Changing {} to:', {})".format(arg1[1:], arg1[1:]))
            elif arg1 == "-initial_orientation":
                if arg2 in ["s", "o"]:
                    initial_orientation = arg2
                else:
                    raise ValueError("Invalid Value  for Initial Orientation: {}".format(arg2))
                eval("print('Changing {} to:', {})".format(arg1[1:], arg1[1:]))
            elif arg1 == "-search_bounds":
                aux = eval(arg2)
                search_bounds = [np.array(aux[0]), np.array(aux[1])]
                eval("print('Changing {} to:', {})".format(arg1[1:], arg1[1:]))
            elif arg1 == "-n_motors":
                n_motors = int(arg2)
                eval("print('Changing {} to:', {})".format(arg1[1:], arg1[1:]))
            elif arg1 == "-thread":
                with_thread_ = bool(eval(arg2))
                eval("print('Changing {} to:', {})".format(arg1[1:], arg1[1:]))
            elif arg1 == "-noise":
                noise = bool(eval(arg2))
            elif arg1 == "time_limit":
                time_limit = float(arg2)
            elif arg1 == "-folder":
                folder_ = arg2
            else:
                raise ValueError("Unrecognized argument: {}".format(arg1))
        print()
        env_info_ = (control_type, noise, n_motors, weights, time_limit, search_bounds, n_actions)
        sac_info_ = (alpha, beta, epsilon, batch_size, buffer_size, hidden_layers, optimizer, update_q, update_alpha, start_using_actor, target_entropy_scale)
        training_info_ = (initial_orientation, desired_orientation, folder_, load, max_episodes, rep)
        return env_info_, sac_info_, training_info_, use_gpu_, with_thread_

    import sys
    ENVIRONMENT_INFO, SAC_INFO, TRAINING_INFO, USE_GPU, WITH_THREAD = process_system_inputs(sys.argv[1:], ENVIRONMENT_INFO, SAC_INFO, TRAINING_INFO, USE_GPU, WITH_THREAD)

    # ask = input("Satisfied with the changes?")
    ask = 1
    if ask not in ["0", "q"]:
        main(ENVIRONMENT_INFO, SAC_INFO, TRAINING_INFO, USE_GPU, WITH_THREAD)
        print("Training has finished")
