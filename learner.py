'''
## Learner ##
# Learner class - this trains the D4PG network on experiences sampled (by priority) from the PER buffer
@author: Mark Sinton (msinto93@gmail.com)
Modified by: Junhyeok Ahn (junhyeokahn91@utexas.edu) and Mihir Vedantam (vedantam.mihir@utexas.edu)
'''

import os
import sys
import tensorflow as tf
import numpy as np
from tqdm import trange

from params import train_params
from utils.env_wrapper import PendulumWrapper, LunarLanderContinuousWrapper, BipedalWalkerWrapper
from utils.network import Actor, Critic
from utils.l2_projection import _l2_project
from utils.misc_utils import compute_avg_return

class Learner:
    def __init__(self, PER_memory, run_agent_event, stop_agent_event):
        print("Initialising learner... \n\n")

        self.PER_memory = PER_memory
        self.run_agent_event = run_agent_event
        self.stop_agent_event = stop_agent_event

        self.critic_optimizer = tf.keras.optimizers.Adam(train_params.CRITIC_LEARNING_RATE)
        self.actor_optimizer = tf.keras.optimizers.Adam(train_params.ACTOR_LEARNING_RATE)

        if train_params.ENV == 'Pendulum-v0':
            self.eval_env = PendulumWrapper()
        elif train_params.ENV == 'LunarLanderContinuous-v2':
            self.eval_env = LunarLanderContinuousWrapper()
        elif train_params.ENV == 'BipedalWalker-v2':
            self.eval_env = BipedalWalkerWrapper()
        elif train_params.ENV == 'BipedalWalkerHardcore-v2':
            self.eval_env = BipedalWalkerWrapper(hardcore=True)
        else:
            raise Exception('Chosen environment does not have an environment wrapper defined. Please choose an environment with an environment wrapper defined, or create a wrapper for this environment in utils.env_wrapper.py')


    def build_network(self):
        # Create value (critic) network + target network
        if train_params.USE_BATCH_NORM:
            pass # for now
            # self.critic_net = Critic_BN(self.state_ph, self.action_ph, train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, train_params.NUM_ATOMS, train_params.V_MIN, train_params.V_MAX, is_training=True, scope='learner_critic_main')
            # self.critic_target_net = Critic_BN(self.state_ph, self.action_ph, train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, train_params.NUM_ATOMS, train_params.V_MIN, train_params.V_MAX, is_training=True, scope='learner_critic_target')
        else:
            self.critic_net = Critic(train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, train_params.NUM_ATOMS, train_params.V_MIN, train_params.V_MAX, name='critic')
            self.critic_net_params = self.critic_net.trainable_variables
            self.critic_target_net = Critic(train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, train_params.NUM_ATOMS, train_params.V_MIN, train_params.V_MAX, name='critic_target')
            self.critic_target_net_params = self.critic_target_net.trainable_variables

        # Create policy (actor) network + target network
        if train_params.USE_BATCH_NORM:
            pass # for now
            # self.actor_net = Actor_BN(self.state_ph, train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.ACTION_BOUND_LOW, train_params.ACTION_BOUND_HIGH, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, is_training=True, scope='learner_actor_main')
            # self.actor_target_net = Actor_BN(self.state_ph, train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.ACTION_BOUND_LOW, train_params.ACTION_BOUND_HIGH, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, is_training=True, scope='learner_actor_target')
        else:
            self.actor_net = Actor(train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.ACTION_BOUND_LOW, train_params.ACTION_BOUND_HIGH, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, name='actor')
            self.actor_net_params = self.actor_net.trainable_variables
            self.actor_target_net = Actor(train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.ACTION_BOUND_LOW, train_params.ACTION_BOUND_HIGH, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, name='actor_target')
            self.actor_target_net_params = self.actor_target_net.trainable_variables

    def target_network_update(self, tau):
        network_params = self.actor_net_params + self.critic_net_params
        target_network_params = self.actor_target_net_params + self.critic_target_net_params
        for from_var,to_var in zip(network_params, target_network_params):
            to_var.assign((tf.multiply(from_var, tau) + tf.multiply(to_var, 1. - tau)))

    def initialise_vars(self):
        # Load ckpt file if given, otherwise initialise variables and hard copy to target networks
        if train_params.INITIAL_ACTOR_MODEL is not None:
            self.actor_net.load_weights(train_params.INITIAL_ACTOR_MODEL)
            self.critic_net.load_weights(train_params.INITIAL_CRITIC_MODEL)
        else:
            self.start_step = 0
            # Perform hard copy (tau=1.0) of initial params to target networks
            self.target_network_update(1.0)

    # @tf.function
    def run(self):
        # Sample batches of experiences from replay memory and train learner networks

        # Initialise beta to start value
        priority_beta = train_params.PRIORITY_BETA_START
        beta_increment = (train_params.PRIORITY_BETA_END - train_params.PRIORITY_BETA_START) / train_params.NUM_STEPS_TRAIN

        avg_return = compute_avg_return(self.eval_env, self.actor_net, train_params.MAX_EP_LENGTH)

        # Can only train when we have at least batch_size num of samples in replay memory
        while len(self.PER_memory) <= train_params.BATCH_SIZE:
            sys.stdout.write('\rPopulating replay memory up to batch_size samples...')
            sys.stdout.flush()

        t = trange(self.start_step+1, train_params.NUM_STEPS_TRAIN+1, desc='ML')
        for train_step in t:
            # Get minibatch
            minibatch = self.PER_memory.sample(train_params.BATCH_SIZE, priority_beta)

            states_batch = minibatch[0].astype(np.float32)
            actions_batch = minibatch[1].astype(np.float32)
            rewards_batch = minibatch[2].astype(np.float32)
            next_states_batch = minibatch[3].astype(np.float32)
            terminals_batch = minibatch[4]
            gammas_batch = minibatch[5].astype(np.float32)
            weights_batch = minibatch[6].astype(np.float32)
            idx_batch = minibatch[7]

            # ==================================================================
            # Critic training step
            # ==================================================================
            # Predict actions for next states by passing next states through policy target network
            future_action = self.actor_target_net(next_states_batch)
            # Predict future Z distribution by passing next states and actions through value target network, also get target network's Z-atom values
            _, target_Z_dist = self.critic_target_net(next_states_batch, future_action)
            target_Z_atoms = self.critic_target_net.z_atoms
            # Create batch of target network's Z-atoms
            target_Z_atoms = np.repeat(np.expand_dims(target_Z_atoms, axis=0), train_params.BATCH_SIZE, axis=0)
            # Value of terminal states is 0 by definition
            target_Z_atoms[terminals_batch, :] = 0.0
            # Apply Bellman update to each atom
            target_Z_atoms = np.expand_dims(rewards_batch, axis=1) + (target_Z_atoms*np.expand_dims(gammas_batch, axis=1))
            # Train critic
            with tf.GradientTape(watch_accessed_variables=False) as g:
                g.watch(self.critic_net_params)
                output_logits, _ = self.critic_net(states_batch, actions_batch)
                target_Z_projected = _l2_project(target_Z_atoms, target_Z_dist, self.critic_net.z_atoms)
                td_error = tf.nn.softmax_cross_entropy_with_logits(logits=output_logits, labels=tf.stop_gradient(target_Z_projected))
                weighted_loss = td_error * weights_batch
                mean_loss = tf.reduce_mean(weighted_loss)
                l2_reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.critic_net_params if 'kernel' in v.name]) * train_params.CRITIC_L2_LAMBDA
                total_loss = mean_loss + l2_reg_loss
            critic_grads = g.gradient(total_loss, self.critic_net_params)
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic_net_params))
            # Use critic TD errors to update sample priorities
            # self.PER_memory.update_priorities(idx_batch, (np.abs(td_error.eval(session=tf.compat.v1.Session()))+train_params.PRIORITY_EPSILON))
            self.PER_memory.update_priorities(idx_batch, (np.abs(td_error.numpy())+train_params.PRIORITY_EPSILON))

            # ==================================================================
            # Actor training step
            # ==================================================================
            # Get policy network's action outputs for selected states
            actor_actions = self.actor_net(states_batch)
            # Compute gradients of critic's value output distribution wrt actions
            with tf.GradientTape() as g:
                g.watch(actor_actions)
                _, output_probs = self.critic_net(states_batch, actor_actions)
            action_grads = g.gradient(output_probs, actor_actions, self.critic_net.z_atoms)
            # Train actor
            with tf.GradientTape(watch_accessed_variables=False) as g:
                g.watch(self.actor_net_params)
                actor_actions = self.actor_net(states_batch)
            actor_grads = g.gradient(actor_actions, self.actor_net_params, -action_grads)
            actor_grads_scaled = list(map(lambda x: tf.divide(x, train_params.BATCH_SIZE), actor_grads))
            self.actor_optimizer.apply_gradients(zip(actor_grads_scaled, self.actor_net_params))

            # Update target networks
            self.target_network_update(train_params.TAU)

            actor_actions = self.actor_net(states_batch)
            # Increment beta value at end of every step
            priority_beta += beta_increment

            # Periodically check capacity of replay mem and remove samples (by FIFO process) above this capacity
            if train_step % train_params.REPLAY_MEM_REMOVE_STEP == 0:
                if len(self.PER_memory) > train_params.REPLAY_MEM_SIZE:
                    # Prevent agent from adding new experiences to replay memory while learner removes samples
                    self.run_agent_event.clear()
                    samples_to_remove = len(self.PER_memory) - train_params.REPLAY_MEM_SIZE
                    self.PER_memory.remove(samples_to_remove)
                    # Allow agent to continue adding experiences to replay memory
                    self.run_agent_event.set()

            if train_step % train_params.PRINTOUT_STEP == 0:
                t.set_description('ML (loss=%g)' % total_loss)

            if train_step % train_params.EVALUATE_SAVE_MODEL_STEP == 0:
                ## TODO : compare time of model.save vs model.save_weights
                __import__('ipdb').set_trace()
                self.actor_net.save_weights(train_params.LOG_DIR + '/eval/actor_%d' % train_step)
                self.critic_net.save_weights(train_params.LOG_DIR + '/eval/critic_%d' % train_step)
                avg_return = compute_avg_return(self.eval_env, self.actor_net, train_params.MAX_EP_LENGTH)
                # print('[Evaluation]: Average return at training step {} is {}'.format(train_step, avg_return))

        # Stop the agents
        self.stop_agent_event.set()
