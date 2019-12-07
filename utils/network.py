'''
## Network ##
# Defines the D4PG Value (critic) and Policy (Actor) networks - with and without batch norm
@author: Mark Sinton (msinto93@gmail.com)
Modified by: Junhyeok Ahn (junhyeokahn91@utexas.edu) and Mihir Vedantam (vedantam.mihir@utexas.edu)
'''

import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dense

from params import train_params
from utils.l2_projection import _l2_project

class Critic(tf.keras.Model):

    def __init__(self, state_dims, action_dims, dense1_size, dense2_size, final_layer_init, num_atoms, v_min, v_max, name='critic'):
        super().__init__(name=name)
        state_dims = np.prod(state_dims)
        action_dims = np.prod(action_dims)

        self.dense1 = Dense(dense1_size, activation=tf.keras.activations.relu, kernel_initializer=tf.random_uniform_initializer((-1/tf.sqrt(tf.cast(state_dims, tf.float32))), 1/tf.sqrt(tf.cast(state_dims, tf.float32))),
                            bias_initializer=tf.random_uniform_initializer((-1/tf.sqrt(tf.cast(state_dims, tf.float32))), 1/tf.sqrt(tf.cast(state_dims, tf.float32))), name='D1')

        self.dense2a = Dense(dense2_size, kernel_initializer=tf.random_uniform_initializer((-1/tf.sqrt(tf.cast(dense1_size+action_dims, tf.float32))), 1/tf.sqrt(tf.cast(dense1_size+action_dims, tf.float32))),
                            bias_initializer=tf.random_uniform_initializer((-1/tf.sqrt(tf.cast(dense1_size+action_dims, tf.float32))), 1/tf.sqrt(tf.cast(dense1_size+action_dims, tf.float32))), name='D2a')

        self.dense2b = Dense(dense2_size, kernel_initializer=tf.random_uniform_initializer((-1/tf.sqrt(tf.cast(dense1_size+action_dims, tf.float32))), 1/tf.sqrt(tf.cast(dense1_size+action_dims, tf.float32))),
                            bias_initializer=tf.random_uniform_initializer((-1/tf.sqrt(tf.cast(dense1_size+action_dims, tf.float32))), 1/tf.sqrt(tf.cast(dense1_size+action_dims, tf.float32))), name='D2b')

        self.dense3 = Dense(num_atoms, kernel_initializer=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init),
                                   bias_initializer=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init), name='D3')

        # self.dense1 = Dense(dense1_size, activation=tf.keras.activations.relu, kernel_initializer=tf.random_uniform_initializer(1, 1), bias_initializer=tf.random_uniform_initializer(1, 1), name='D1')

        # self.dense2a = Dense(dense2_size, kernel_initializer=tf.random_uniform_initializer(1, 1), bias_initializer=tf.random_uniform_initializer(1, 1), name='D2a')

        # self.dense2b = Dense(dense2_size, kernel_initializer=tf.random_uniform_initializer(1, 1), bias_initializer=tf.random_uniform_initializer(1, 1), name='D2b')

        # self.dense3 = Dense(num_atoms, kernel_initializer=tf.random_uniform_initializer(1, 1), bias_initializer=tf.random_uniform_initializer(1, 1), name='D3')

        self.z_atoms = tf.linspace(v_min, v_max, num_atoms)

        self.optimizer = tf.keras.optimizers.Adam(train_params.CRITIC_LEARNING_RATE)

        with tf.device("/cpu:0"):
            self(tf.constant(np.zeros(shape=(1,)+(state_dims,), dtype=np.float32)), tf.constant(np.zeros(shape=(1,)+(action_dims,), dtype=np.float32)))

    @tf.function
    def call(self, state, action):
        d1 = self.dense1(state)
        d2a = self.dense2a(d1)
        d2b = self.dense2b(action)
        d2 = tf.nn.relu(d2a + d2b)
        output_logits = self.dense3(d2)
        output_probs = tf.nn.softmax(output_logits)
        return output_logits, output_probs

    @tf.function
    def train(self, states_batch, actions_batch, target_Z_atoms, target_Z_dist, weights_batch):
        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(self.trainable_variables)
            output_logits, _ = self.call(states_batch, actions_batch)
            target_Z_projected = _l2_project(target_Z_atoms, target_Z_dist, self.z_atoms)
            td_error = tf.nn.softmax_cross_entropy_with_logits(logits=output_logits, labels=tf.stop_gradient(target_Z_projected))
            weighted_loss = td_error * weights_batch
            mean_loss = tf.reduce_mean(weighted_loss)
            l2_reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_variables if 'kernel' in v.name]) * train_params.CRITIC_L2_LAMBDA
            total_loss = mean_loss + l2_reg_loss
        critic_grads = g.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(critic_grads, self.trainable_variables))
        return td_error, total_loss

    @tf.function
    def get_action_grads(self, states_batch, actor_actions):
        # Compute gradients of critic's value output distribution wrt actions
        with tf.GradientTape() as g:
            g.watch(actor_actions)
            _, output_probs = self.call(states_batch, actor_actions)
        expanded_atoms = np.repeat(np.expand_dims(self.z_atoms, axis=0), train_params.BATCH_SIZE, axis=0)
        action_grads = g.gradient(output_probs, actor_actions, expanded_atoms)
        return action_grads


class Actor(tf.keras.Model):

    def __init__(self, state_dims, action_dims, action_bound_low, action_bound_high, dense1_size, dense2_size, final_layer_init, name='actor'):
        super().__init__(name=name)

        state_dims = np.prod(state_dims)       #Used to calculate the fan_in of the state layer (e.g. if state_dims is (3,2) fan_in should equal 6)
        action_dims = np.prod(action_dims)
        self.action_bound_low = action_bound_low
        self.action_bound_high = action_bound_high

        self.dense1 = Dense(dense1_size, activation=tf.keras.activations.relu, kernel_initializer=tf.random_uniform_initializer((-1/tf.sqrt(tf.cast(state_dims, tf.float32))), 1/tf.sqrt(tf.cast(state_dims, tf.float32))),
                            bias_initializer=tf.random_uniform_initializer((-1/tf.sqrt(tf.cast(state_dims, tf.float32))), 1/tf.sqrt(tf.cast(state_dims, tf.float32))), name='D1')

        self.dense2 = Dense(dense2_size, activation=tf.keras.activations.relu, kernel_initializer=tf.random_uniform_initializer((-1/tf.sqrt(tf.cast(dense1_size, tf.float32))), 1/tf.sqrt(tf.cast(dense1_size, tf.float32))),
                            bias_initializer=tf.random_uniform_initializer((-1/tf.sqrt(tf.cast(dense1_size, tf.float32))), 1/tf.sqrt(tf.cast(dense1_size, tf.float32))), name='D2')

        self.dense3 = Dense(action_dims, activation=tf.keras.activations.tanh, kernel_initializer=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init),
                            bias_initializer=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init), name='output')

        # self.dense1 = Dense(dense1_size, activation=tf.keras.activations.relu, kernel_initializer=tf.random_uniform_initializer(1, 1), bias_initializer=tf.random_uniform_initializer(1, 1), name='D1')

        # self.dense2 = Dense(dense2_size, activation=tf.keras.activations.relu, kernel_initializer=tf.random_uniform_initializer(1, 1), bias_initializer=tf.random_uniform_initializer(1, 1), name='D2')

        # self.dense3 = Dense(action_dims, activation=tf.keras.activations.tanh, kernel_initializer=tf.random_uniform_initializer(1, 1), bias_initializer=tf.random_uniform_initializer(1, 1), name='D3')

        self.optimizer = tf.keras.optimizers.Adam(train_params.CRITIC_LEARNING_RATE)

        # this initialize the weights
        with tf.device("/cpu:0"):
            self(tf.constant(np.zeros(shape=(1,)+(state_dims,), dtype=np.float32)))

    @tf.function
    def call(self, state):
        d1 = self.dense1(state)
        d2 = self.dense2(d1)
        d3 = self.dense3(d2)
        output = tf.multiply(0.5, tf.multiply(d3, (self.action_bound_high-self.action_bound_low)) + (self.action_bound_high+self.action_bound_low))
        return output

    @tf.function
    def train(self, states_batch, action_grads):
        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(self.trainable_variables)
            actor_actions = self.call(states_batch)
        actor_grads = g.gradient(actor_actions, self.trainable_variables, -action_grads)
        actor_grads_scaled = list(map(lambda x: tf.divide(x, train_params.BATCH_SIZE), actor_grads))
        self.optimizer.apply_gradients(zip(actor_grads_scaled, self.trainable_variables))
        return None

'''
class Critic_BN:
    def __init__(self, state, action, state_dims, action_dims, dense1_size, dense2_size, final_layer_init, num_atoms, v_min, v_max, is_training=False, name='critic'):
        # state - State input to pass through the network
        # action - Action input for which the Z distribution should be predicted

        self.state = state
        self.action = action
        self.state_dims = np.prod(state_dims)       #Used to calculate the fan_in of the state layer (e.g. if state_dims is (3,2) fan_in should equal 6)
        self.action_dims = np.prod(action_dims)
        self.is_training = is_training
        self.name = name


        with tf.variable_scope(self.name):
            self.input_norm = batchnorm(self.state, self.is_training, name='input_norm')

            self.dense1_mul = dense(self.input_norm, dense1_size, kernel_initializer=tf.random_uniform_initializer((-1/tf.sqrt(tf.cast(self.state_dims))), 1/tf.sqrt(tf.cast(self.state_dims))),
                                bias_initializer=tf.random_uniform_initializer((-1/tf.sqrt(tf.cast(self.state_dims))), 1/tf.sqrt(tf.cast(self.state_dims))), name='dense1')

            self.dense1_bn = batchnorm(self.dense1_mul, self.is_training, name='dense1')

            self.dense1 = relu(self.dense1_bn, name='dense1')

            #Merge first dense layer with action input to get second dense layer
            self.dense2a = dense(self.dense1, dense2_size, kernel_initializer=tf.random_uniform_initializer((-1/tf.sqrt(tf.cast(dense1_size+self.action_dims))), 1/tf.sqrt(tf.cast(dense1_size+self.action_dims))),
                                bias_initializer=tf.random_uniform_initializer((-1/tf.sqrt(tf.cast(dense1_size+self.action_dims))), 1/tf.sqrt(tf.cast(dense1_size+self.action_dims))), name='dense2a')

            self.dense2b = dense(self.action, dense2_size, kernel_initializer=tf.random_uniform_initializer((-1/tf.sqrt(tf.cast(dense1_size+self.action_dims))), 1/tf.sqrt(tf.cast(dense1_size+self.action_dims))),
                                bias_initializer=tf.random_uniform_initializer((-1/tf.sqrt(tf.cast(dense1_size+self.action_dims))), 1/tf.sqrt(tf.cast(dense1_size+self.action_dims))), name='dense2b')

            self.dense2 = relu(self.dense2a + self.dense2b, name='dense2')

            self.output_logits = dense(self.dense2, num_atoms, kernel_initializer=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init),
                                       bias_initializer=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init), name='output_logits')

            self.output_probs = softmax(self.output_logits, name='output_probs')


            self.network_params = tf.trainable_variables(name=self.name)
            self.bn_params = [v for v in tf.global_variables(name=self.name) if 'batch_normalization/moving' in v.name]


            self.z_atoms = tf.lin_space(v_min, v_max, num_atoms)

            self.Q_val = tf.reduce_sum(self.z_atoms * self.output_probs) # the Q value is the mean of the categorical output Z-distribution

            self.action_grads = tf.gradients(self.output_probs, self.action, self.z_atoms) # gradient of mean of output Z-distribution wrt action input - used to train actor network, weighing the grads by z_values gives the mean across the output distribution

    def train_step(self, target_Z_dist, target_Z_atoms, IS_weights, learn_rate, l2_lambda):
        # target_Z_dist - target Z distribution for next state
        # target_Z_atoms - atom values of target network with Bellman update applied

        with tf.variable_scope(self.name):
            with tf.variable_scope('train'):
                self.optimizer = tf.train.AdamOptimizer(learn_rate)

                # Project the target distribution onto the bounds of the original network
                target_Z_projected = _l2_project(target_Z_atoms, target_Z_dist, self.z_atoms)

                self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.output_logits, labels=tf.stop_gradient(target_Z_projected))
                self.weighted_loss = self.loss * IS_weights
                self.mean_loss = tf.reduce_mean(self.weighted_loss)

                self.l2_reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.network_params if 'kernel' in v.name]) * l2_lambda
                self.total_loss = self.mean_loss + self.l2_reg_loss

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, self.name) # Ensure batch norm moving means and variances are updated every training step
                with tf.control_dependencies(update_ops):
                    train_step = self.optimizer.minimize(self.total_loss, var_list=self.network_params)

                return train_step


class Actor_BN:
    def __init__(self, state, state_dims, action_dims, action_bound_low, action_bound_high, dense1_size, dense2_size, final_layer_init, is_training=False, name='actor'):
        # state - State input to pass through the network
        # action_bounds - Network will output in range [-1,1]. Multiply this by action_bound to get output within desired boundaries of action space

        self.state = state
        self.state_dims = np.prod(state_dims)       #Used to calculate the fan_in of the state layer (e.g. if state_dims is (3,2) fan_in should equal 6)
        self.action_dims = np.prod(action_dims)
        self.action_bound_low = action_bound_low
        self.action_bound_high = action_bound_high
        self.is_training = is_training
        self.name = name

        with tf.variable_scope(self.name):

            self.input_norm = batchnorm(self.state, self.is_training, name='input_norm')

            self.dense1_mul = dense(self.input_norm, dense1_size, kernel_initializer=tf.random_uniform_initializer((-1/tf.sqrt(tf.cast(self.state_dims))), 1/tf.sqrt(tf.cast(self.state_dims))),
                                bias_initializer=tf.random_uniform_initializer((-1/tf.sqrt(tf.cast(self.state_dims))), 1/tf.sqrt(tf.cast(self.state_dims))), name='dense1')

            self.dense1_bn = batchnorm(self.dense1_mul, self.is_training, name='dense1')

            self.dense1 = relu(self.dense1_bn, name='dense1')

            self.dense2_mul = dense(self.dense1, dense2_size, kernel_initializer=tf.random_uniform_initializer((-1/tf.sqrt(tf.cast(dense1_size))), 1/tf.sqrt(tf.cast(dense1_size))),
                                bias_initializer=tf.random_uniform_initializer((-1/tf.sqrt(tf.cast(dense1_size))), 1/tf.sqrt(tf.cast(dense1_size))), name='dense2')

            self.dense2_bn = batchnorm(self.dense2_mul, self.is_training, name='dense2')

            self.dense2 = relu(self.dense2_bn, name='dense2')

            self.output_mul = dense(self.dense2, self.action_dims, kernel_initializer=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init),
                                bias_initializer=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init), name='output')

            self.output_tanh = tanh(self.output_mul, name='output')

            # Scale tanh output to lower and upper action bounds
            self.output = tf.multiply(0.5, tf.multiply(self.output_tanh, (self.action_bound_high-self.action_bound_low)) + (self.action_bound_high+self.action_bound_low))


            self.network_params = tf.trainable_variables(name=self.name)
            self.bn_params = [v for v in tf.global_variables(name=self.name) if 'batch_normalization/moving' in v.name]

    def train_step(self, action_grads, learn_rate, batch_size):
        # action_grads - gradient of value output wrt action from critic network

        with tf.variable_scope(self.name):
            with tf.variable_scope('train'):

                self.optimizer = tf.train.AdamOptimizer(learn_rate)
                self.grads = tf.gradients(self.output, self.network_params, -action_grads)
                self.grads_scaled = list(map(lambda x: tf.divide(x, batch_size), self.grads)) # tf.gradients sums over the batch dimension here, must therefore divide by batch_size to get mean gradients

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, self.name) # Ensure batch norm moving means and variances are updated every training step
                with tf.control_dependencies(update_ops):
                    train_step = self.optimizer.apply_gradients(zip(self.grads_scaled, self.network_params))

                return train_step
'''



