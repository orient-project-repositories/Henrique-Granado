import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
from SAC.buffer import ReplayBuffer
from SAC.networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(self, alpha=0.0003, beta=0.0003, epsilon=0.0003, input_dims=[8], gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
                 batch_size=256, chkpt_dir='tmp/sac', optimizer="Adam", hidden_layers=[[16, 16], [16, 16], [16, 16]],
                 update_q=True, update_alpha=True, start_using_actor=0, target_entropy_scale=1.0):
        self.n_actions = n_actions

        self.gamma = gamma
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size

        self.actor = ActorNetwork(n_actions=n_actions, name='actor', fc_dims = hidden_layers[0], chkpt_dir = chkpt_dir)
        self.q1 = CriticNetwork(name='q1', fc_dims = hidden_layers[1], chkpt_dir = chkpt_dir)
        self.q2 = CriticNetwork(name='q2', fc_dims = hidden_layers[2], chkpt_dir = chkpt_dir)
        self.target_q1 = CriticNetwork(name='target_q1', fc_dims = hidden_layers[1], chkpt_dir = chkpt_dir)
        self.target_q2 = CriticNetwork(name='target_q2', fc_dims = hidden_layers[2], chkpt_dir = chkpt_dir)

        self.log_alpha = tf.Variable(0, dtype = tf.float32)
        self.alpha = tf.Variable(0, dtype = tf.float32)
        if update_alpha:
            self.alpha = tfp.util.DeferredTensor(self.log_alpha, tf.exp)
        self.target_entropy = -tf.constant(n_actions*target_entropy_scale, dtype=tf.float32)
        self.gamma = gamma
        self.polyak = 1-tau

        self.update_q = update_q
        self.update_alpha_ = update_alpha

        self.start_using_actor = start_using_actor

        if optimizer[:7] == "RMSprop":
            mom = float(optimizer[7:])
            self.actor_optimizer = tf.keras.optimizers.RMSprop(alpha, momentum = mom)
            self.critic1_optimizer = tf.keras.optimizers.RMSprop(beta, momentum = mom)
            self.critic2_optimizer = tf.keras.optimizers.RMSprop(beta, momentum = mom)
            self.alpha_optimizer = tf.keras.optimizers.RMSprop(epsilon, momentum = mom)
        elif optimizer == "SGD":
            self.actor_optimizer = tf.keras.optimizers.SGD(alpha)
            self.critic1_optimizer = tf.keras.optimizers.SGD(beta)
            self.critic2_optimizer = tf.keras.optimizers.SGD(beta)
            self.alpha_optimizer = tf.keras.optimizers.SGD(epsilon)
        else:
            self.actor_optimizer = tf.keras.optimizers.Adam(alpha)
            self.critic1_optimizer = tf.keras.optimizers.Adam(beta)
            self.critic2_optimizer = tf.keras.optimizers.Adam(beta)
            self.alpha_optimizer = tf.keras.optimizers.Adam(epsilon)

    def get_actor_output(self, observation):
        state = tf.convert_to_tensor([observation])
        mu, sigma = self.actor(state)
        return mu, sigma

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])

        if self.start_using_actor <= self.memory.mem_cntr:
            actions, _ = self.actor.sample_normal(state)
        else:
            actions = [np.tanh(np.random.uniform(-2,2,self.n_actions))]

        return actions[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def best_save_models(self):
        #print('... saving best models ...')
        self.actor.save_weights(self.actor.best_file)
        self.q1.save_weights(self.q1.best_file)
        self.q2.save_weights(self.q2.best_file)
        self.target_q1.save_weights(self.target_q1.best_file)
        self.target_q2.save_weights(self.target_q2.best_file)

    def save_models(self):
        #print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.q1.save_weights(self.q1.checkpoint_file)
        self.q2.save_weights(self.q2.checkpoint_file)
        self.target_q1.save_weights(self.target_q1.checkpoint_file)
        self.target_q2.save_weights(self.target_q2.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.q1.load_weights(self.q1.checkpoint_file)
        self.q2.load_weights(self.q2.checkpoint_file)
        self.target_q1.load_weights(self.target_q1.checkpoint_file)
        self.target_q2.load_weights(self.target_q2.checkpoint_file)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size or self.memory.mem_cntr<self.start_using_actor:
            # print("Can't learn yet: memory_counter = {}, buffer_size = {}".format(self.memory.mem_cntr, self.batch_size))
            return None, None, None, None

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        current_states = tf.convert_to_tensor(state, dtype=tf.float32)
        next_states = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(np.transpose([reward]), dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        done = tf.convert_to_tensor(np.transpose([done]), dtype=tf.float32)

        critic1_loss, critic2_loss = (0, 0)
        if self.update_q:
            critic1_loss, critic2_loss = self.update_q_network(current_states, actions, rewards, next_states, done)

        # Update policy network weights
        actor_loss = self.update_actor_network(current_states)

        alpha_loss = 0
        if self.update_alpha_:
            alpha_loss = self.update_alpha(current_states)

        # Update target Q network weights
        # self.update_weights()

        #if self.epoch_step % 10 == 0:
        #    self.alpha = max(0.1, 0.9**(1+self.epoch_step/10000))
        #    print("alpha: ", self.alpha, 1+self.epoch_step/10000)

        return float(critic1_loss), float(critic2_loss), float(actor_loss), float(alpha_loss)

    def update_q_network(self, current_states, actions, rewards, next_states, dones):
        alpha = tf.convert_to_tensor(self.alpha)
        with tf.GradientTape() as tape1:
            # Get Q value estimates, action used here is from the replay buffer
            q1 = self.q1(current_states, actions)

            # Sample actions from actor for next state
            next_action, next_log_pi = self.actor.sample_normal(next_states)

            # Get Q value estimates deom tarfet Q network
            q1_target = self.target_q1(next_states, next_action)
            q2_target = self.target_q2(next_states, next_action)

            # Apply the clipped double Q trick
            # Get the minimum Q value of the 2 target networks
            min_q_target = tf.minimum(q1_target, q2_target)

            # Add the entropy term to get soft Q target
            soft_q_target = min_q_target - alpha * next_log_pi
            #soft_q_target = min_q_target - next_log_pi
            y = tf.stop_gradient(rewards + self.gamma * (1-dones) * soft_q_target)

            critic1_loss = tf.reduce_mean((q1 - y)**2)

        with tf.GradientTape() as tape2:
            # Get Q value estimates, action used here is from the replay buffer
            q2 = self.q2(current_states, actions)

            # Sample actions from actor for next state
            next_action, next_log_pi = self.actor.sample_normal(next_states)

            # Get Q value estimates deom tarfet Q network
            q1_target = self.target_q1(next_states, next_action)
            q2_target = self.target_q2(next_states, next_action)

            # Apply the clipped double Q trick
            # Get the minimum Q value of the 2 target networks
            min_q_target = tf.minimum(q1_target, q2_target)

            # Add the entropy term to get soft Q target
            soft_q_target = min_q_target - alpha * next_log_pi
            #soft_q_target = min_q_target - next_log_pi
            y = tf.stop_gradient(rewards + self.gamma * (1-dones) * soft_q_target)

            critic2_loss = tf.reduce_mean((q2 - y)**2)

        grads1 = tape1.gradient(critic1_loss, self.q1.trainable_variables)
        self.critic1_optimizer.apply_gradients(zip(grads1,
                                                   self.q1.trainable_variables))

        grads2 = tape2.gradient(critic2_loss, self.q2.trainable_variables)
        self.critic2_optimizer.apply_gradients(zip(grads2, self.q2.trainable_variables))

        return critic1_loss, critic2_loss

    def update_actor_network(self, current_states):
        alpha = tf.convert_to_tensor(self.alpha)
        with tf.GradientTape() as tape:
            # Sample actions from the policy for current states
            new_action, log_probs = self.actor.sample_normal(current_states)

            # Get Q value estimates from target Q network
            q1 = self.q1(current_states, new_action)
            q2 = self.q2(current_states, new_action)

            # Apply the clipped double Q trick
            # Get the minimum Q value of the 2 target networks
            min_q = tf.minimum(q1, q2)

            soft_q = min_q - alpha * log_probs
            # soft_q = min_q - log_probs

            actor_loss = -tf.reduce_mean(soft_q)

        variables = self.actor.trainable_variables
        actor_network_gradient = tape.gradient(actor_loss, variables)
        self.actor_optimizer.apply_gradients(zip(actor_network_gradient, variables))

        return actor_loss

    def update_alpha(self, current_states):
        # alpha = tf.convert_to_tensor(self.alpha)
        with tf.GradientTape() as tape:
            # Sample actions from the policy for current states
            pi_a, log_pi_a = self.actor.sample_normal(current_states)

            alpha_loss = tf.reduce_mean(-self.alpha*(log_pi_a + self.target_entropy))

        variables = [self.log_alpha]
        grads = tape.gradient(alpha_loss, variables)
        self.alpha_optimizer.apply_gradients(zip(grads, variables))

        return alpha_loss

    def update_weights(self):
        for theta_target, theta in zip(self.target_q1.trainable_variables, self.q1.trainable_variables):
            theta_target = self.polyak * theta_target + (1 - self.polyak) * theta

        for theta_target, theta in zip(self.target_q2.trainable_variables, self.q2.trainable_variables):
            theta_target = self.polyak * theta_target + (1 - self.polyak) * theta

