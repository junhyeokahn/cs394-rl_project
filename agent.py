'''
## Agent ##
# Agent class - the agent explores the environment, collecting experiences and adding them to the PER buffer. Can also be used to test/run a trained network in the environment.
@author: Mark Sinton (msinto93@gmail.com)
Modified by: Junhyeok Ahn (junhyeokahn91@utexas.edu) and Mihir Vedantam (vedantam.mihir@utexas.edu)
'''

import os
import sys
import tensorflow as tf
import numpy as np
import scipy.stats as ss
from collections import deque
import cv2
import imageio

from params import train_params
from utils.network import Actor
from utils.env_wrapper import PendulumWrapper, LunarLanderContinuousWrapper, BipedalWalkerWrapper

class Agent:

    # @tf.function
    def __init__(self, env, seed, learner_policy_params, n_agent=0):
        self.n_agent = n_agent

        # Create environment
        if env == 'Pendulum-v0':
            self.env_wrapper = PendulumWrapper()
        elif env == 'LunarLanderContinuous-v2':
            self.env_wrapper = LunarLanderContinuousWrapper()
        elif env == 'BipedalWalker-v2':
            self.env_wrapper = BipedalWalkerWrapper()
        elif env == 'BipedalWalkerHardcore-v2':
            self.env_wrapper = BipedalWalkerWrapper(hardcore=True)
        else:
            raise Exception('Chosen environment does not have an environment wrapper defined. Please choose an environment with an environment wrapper defined, or create a wrapper for this environment in utils.env_wrapper.py')
        self.env_wrapper.set_random_seed(seed*(n_agent+1))

        self.learner_policy_params = learner_policy_params

    # @tf.function
    def build_network(self, training):

        name = ('actor_agent_%02d'%self.n_agent)
        # Create policy (actor) network
        if train_params.USE_BATCH_NORM:
            pass
            # self.actor_net = Actor_BN(self.state_ph, train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.ACTION_BOUND_LOW, train_params.ACTION_BOUND_HIGH, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, is_training=False, scope=var_scope)
            # self.agent_policy_params = self.actor_net.network_params + self.actor_net.bn_params
        else:
            self.actor_net = Actor(train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.ACTION_BOUND_LOW, train_params.ACTION_BOUND_HIGH, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, name)
            self.actor_net_params = self.actor_net.trainable_variables

    # @tf.function
    def update_network(self):
        from_vars = self.learner_policy_params
        to_vars = self.actor_net_params

        for from_var,to_var in zip(from_vars,to_vars):
            to_var.assign(from_var)

    def build_summaries(self, logdir):
        # Create summary writer to write summaries to disk
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        self.summary_writer = tf.summary.create_file_writer(logdir)

    def run(self, PER_memory, gaussian_noise, run_agent_event, stop_agent_event):
        # Continuously run agent in environment to collect experiences and add to replay memory

        # Initialise deque buffer to store experiences for N-step returns
        self.exp_buffer = deque()

        # Perform initial copy of params from learner to agent
        self.update_network()

        # Initially set threading event to allow agent to run until told otherwise
        run_agent_event.set()

        num_eps = 0

        while not stop_agent_event.is_set():
            num_eps += 1
            # Reset environment and experience buffer
            state = self.env_wrapper.reset()
            state = self.env_wrapper.normalise_state(state)
            self.exp_buffer.clear()

            num_steps = 0
            episode_reward = 0
            ep_done = False

            while not ep_done:
                num_steps += 1
                ## Take action and store experience
                if train_params.RENDER:
                    self.env_wrapper.render()
                action = self.actor_net(np.expand_dims(state.astype(np.float32), 0))[0]
                action += (gaussian_noise() * train_params.NOISE_DECAY**num_eps)
                next_state, reward, terminal = self.env_wrapper.step(action)

                episode_reward += reward

                next_state = self.env_wrapper.normalise_state(next_state)
                reward = self.env_wrapper.normalise_reward(reward)

                self.exp_buffer.append((state, action, reward))

                # We need at least N steps in the experience buffer before we can compute Bellman rewards and add an N-step experience to replay memory
                if len(self.exp_buffer) >= train_params.N_STEP_RETURNS:
                    state_0, action_0, reward_0 = self.exp_buffer.popleft()
                    discounted_reward = reward_0
                    gamma = train_params.DISCOUNT_RATE
                    for (_, _, r_i) in self.exp_buffer:
                        discounted_reward += r_i * gamma
                        gamma *= train_params.DISCOUNT_RATE

                    # If learner is requesting a pause (to remove samples from PER), wait before adding more samples
                    run_agent_event.wait()
                    PER_memory.add(state_0, action_0, discounted_reward, next_state, terminal, gamma)

                state = next_state

                if terminal or num_steps == train_params.MAX_EP_LENGTH:
                    # Log total episode reward
                    with self.summary_writer.as_default():
                        tf.summary.scalar("Episode Reward", episode_reward, step=num_eps)
                    self.summary_writer.flush()
                    # Compute Bellman rewards and add experiences to replay memory for the last N-1 experiences still remaining in the experience buffer
                    while len(self.exp_buffer) != 0:
                        state_0, action_0, reward_0 = self.exp_buffer.popleft()
                        discounted_reward = reward_0
                        gamma = train_params.DISCOUNT_RATE
                        for (_, _, r_i) in self.exp_buffer:
                            discounted_reward += r_i * gamma
                            gamma *= train_params.DISCOUNT_RATE

                        # If learner is requesting a pause (to remove samples from PER), wait before adding more samples
                        run_agent_event.wait()
                        PER_memory.add(state_0, action_0, discounted_reward, next_state, terminal, gamma)

                    # Start next episode
                    ep_done = True

            # Update agent networks with learner params every 'update_agent_ep' episodes
            if num_eps % train_params.UPDATE_AGENT_EP == 0:
                self.update_network()

        self.env_wrapper.close()

    def play(self):
        pass
        ## TODO
        '''
        # Play a saved ckpt of actor network in the environment, visualise performance on screen and save a GIF (optional)

        def load_ckpt(ckpt_dir, ckpt_file):
            # Load ckpt given by ckpt_file, or else load latest ckpt in ckpt_dir
            loader = tf.train.Saver()
            if ckpt_file is not None:
                ckpt = ckpt_dir + '/' + ckpt_file
            else:
                ckpt = tf.train.latest_checkpoint(ckpt_dir)

            loader.restore(self.sess, ckpt)
            sys.stdout.write('%s restored.\n\n' % ckpt)
            sys.stdout.flush()

            ckpt_split = ckpt.split('-')
            self.train_ep = ckpt_split[-1]

        # Load ckpt from ckpt_dir
        load_ckpt(play_params.CKPT_DIR, play_params.CKPT_FILE)

        # Create record directory
        if not os.path.exists(play_params.RECORD_DIR):
            os.makedirs(play_params.RECORD_DIR)

        for ep in range(1, play_params.NUM_EPS_PLAY+1):
            state = self.env_wrapper.reset()
            state = self.env_wrapper.normalise_state(state)
            step = 0
            ep_done = False

            while not ep_done:
                frame = self.env_wrapper.render()
                if play_params.RECORD_DIR is not None:
                    filepath = play_params.RECORD_DIR + '/Ep%03d_Step%04d.jpg' % (ep, step)
                    cv2.imwrite(filepath, frame)
                action = self.sess.run(self.actor_net.output, {self.state_ph:np.expand_dims(state, 0)})[0]     # Add batch dimension to single state input, and remove batch dimension from single action output
                state, _, terminal = self.env_wrapper.step(action)
                state = self.env_wrapper.normalise_state(state)

                step += 1

                # Episode can finish either by reaching terminal state or max episode steps
                if terminal or step == play_params.MAX_EP_LENGTH:
                    ep_done = True

        # Convert saved frames to gif
        if play_params.RECORD_DIR is not None:
            images = []
            for file in sorted(os.listdir(play_params.RECORD_DIR)):
                # Load image
                filename = play_params.RECORD_DIR + '/' + file
                im = cv2.imread(filename)
                images.append(im)
                # Delete static image once loaded
                os.remove(filename)

            # Save as gif
            imageio.mimsave(play_params.RECORD_DIR + '/%s.gif' % play_params.ENV, images, duration=0.01)

        self.env_wrapper.close()
        '''

