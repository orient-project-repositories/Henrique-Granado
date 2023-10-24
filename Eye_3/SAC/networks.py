import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense


class CriticNetwork(keras.Model):
    def __init__(self,fc_dims = [256, 256], name='critic', chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.best_dir = chkpt_dir+"/best"
        self.best_file = os.path.join(self.best_dir, name+'_sac')

        #self.fc1 = Dense(fc1_dims, activation='relu')
        #self.fc2 = Dense(fc2_dims, activation='relu')
        #self.q = Dense(1, activation=None)

        self.fc_layer_list = []
        for dims in fc_dims:
            self.fc_layer_list.append(Dense(dims, activation = 'relu'))
        self.q = Dense(1, activation=None)


    def call(self, state, action):
        state_action = tf.concat([state, action], axis=1)
        x = state_action
        for fc_layer in self.fc_layer_list:
            x = fc_layer(x)
        q = self.q(x)

        return q

    @property
    def trainable_variables(self):
        tv = []
        for fc_layer in self.fc_layer_list:
            tv += fc_layer.trainable_variables
        tv += self.q.trainable_variables
        return tv


class ActorNetwork(keras.Model):
    def __init__(self, fc_dims=[256, 256], n_actions=2, name='actor', chkpt_dir='tmp/sac'):
        super().__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_sac')
        self.noise = 1e-10

        self.best_dir = chkpt_dir+"/best"
        self.best_file = os.path.join(self.best_dir, name+'_sac')

        self.fc_layer_list = []
        for dims in fc_dims:
            self.fc_layer_list.append(Dense(dims, activation = 'relu'))
        self.mu = Dense(n_actions, activation=None)
        self.sigma = Dense(n_actions, activation=None)
    def call(self, state):
        x = state
        for fc_layer in self.fc_layer_list:
            x = fc_layer(x)

        mu = self.mu(x)

        # Standard deviation is bounded by a constraint of being non-negative
        # therefore we produce log stdev as output which can be [-inf, inf]
        log_sigma = self.sigma(x)
        sigma = tf.exp(log_sigma)

        return mu, sigma

    def sample_normal(self, state):
        mu, sigma = self.call(state)
        probabilities = tfp.distributions.Normal(mu, sigma)

        action_ = probabilities.sample()

        action = tf.math.tanh(action_)
        log_pi_ = probabilities.log_prob(action)
        log_pi = log_pi_ - tf.math.reduce_sum(tf.math.log(1-action**2+self.noise), axis=1, keepdims=True)

        return action, log_pi

    @property
    def trainable_variables(self):
        tv = []
        for fc_layer in self.fc_layer_list:
            tv += fc_layer.trainable_variables
        tv += self.mu.trainable_variables+self.sigma.trainable_variables
        return tv