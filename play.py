'''
## Test ##
# Test a trained D4PG network.
Written by: Junhyeok Ahn (junhyeokahn91@utexas.edu) and Mihir Vedantam (vedantam.mihir@utexas.edu)
'''
import os

import tensorflow as tf
import numpy as np
import cv2
import imageio
from tqdm import tqdm

from utils.env_wrapper import PendulumWrapper, LunarLanderContinuousWrapper, BipedalWalkerWrapper
from params import train_params, play_params
from utils.network import Actor, Critic


def play():

    if play_params.ENV == 'Pendulum-v0':
        play_env = PendulumWrapper()
    elif play_params.ENV == 'LunarLanderContinuous-v2':
        play_env = LunarLanderContinuousWrapper()
    elif play_params.ENV == 'BipedalWalker-v2':
        play_env = BipedalWalkerWrapper()
    elif play_params.ENV == 'BipedalWalkerHardcore-v2':
        play_env = BipedalWalkerWrapper(hardcore=True)
    else:
        raise Exception('Chosen environment does not have an environment wrapper defined. Please choose an environment with an environment wrapper defined, or create a wrapper for this environment in utils.env_wrapper.py')


    actor_net = Actor(play_params.STATE_DIMS, play_params.ACTION_DIMS, play_params.ACTION_BOUND_LOW, play_params.ACTION_BOUND_HIGH, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, name='actor_play')
    critic_net = Critic(play_params.STATE_DIMS, play_params.ACTION_DIMS, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, train_params.NUM_ATOMS, train_params.V_MIN, train_params.V_MAX, name='critic_play')

    actor_net.load_weights(play_params.ACTOR_MODEL_DIR)
    critic_net.load_weights(play_params.CRITIC_MODEL_DIR)

    if not os.path.exists(play_params.RECORD_DIR):
        os.makedirs(play_params.RECORD_DIR)

    for ep in tqdm(range(1, play_params.NUM_EPS_PLAY+1), desc='playing'):
        state = play_env.reset()
        state = play_env.normalise_state(state)
        step = 0
        ep_done = False

        while not ep_done:
            frame = play_env.render()
            if play_params.RECORD_DIR is not None:
                filepath = play_params.RECORD_DIR + '/Ep%03d_Step%04d.jpg' % (ep, step)
                cv2.imwrite(filepath, frame)
            action = actor_net(np.expand_dims(state.astype(np.float32), 0))[0]
            state, _, terminal = play_env.step(action)
            state = play_env.normalise_state(state)

            step += 1

            # Episode can finish either by reaching terminal state or max episode steps
            if terminal or step == play_params.MAX_EP_LENGTH:
                ep_done = True

    # Convert saved frames to gif
    exit()
    if play_params.RECORD_DIR is not None:
        images = []
        for file in tqdm(sorted(os.listdir(play_params.RECORD_DIR)), desc='converting to gif'):
            # Load image
            filename = play_params.RECORD_DIR + '/' + file
            im = cv2.imread(filename)
            images.append(im)
            # Delete static image once loaded
            os.remove(filename)

        # Save as gif
        print("Saving to ", play_params.RECORD_DIR)
        imageio.mimsave(play_params.RECORD_DIR + '/%s.gif' % play_params.ENV, images[:-1], duration=0.01)

    play_env.close()

if  __name__ == '__main__':
    play()
