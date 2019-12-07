import glob
import os
import numpy as np
import tensorflow as tf

def get_latest_run_id(save_path, dir_name):
    """
    returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.
    :return: (int) latest run number
    """
    max_run_id = 0
    for path in glob.glob(save_path + "/{}_[0-9]*".format(dir_name)):
        file_name = path.split("/")[-1]
        ext = file_name.split("_")[-1]
        if dir_name == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id

def get_log_dir(env, algo):
    log_dir = os.getcwd() + '/data/' + env + '/'
    latest_run_id = get_latest_run_id(log_dir, algo)
    log_dir = os.path.join(log_dir, "{}_{}".format(algo, latest_run_id+1))
    if os.path.exists(log_dir):
        raise ValueError('Log directory is already exists')
    return log_dir

def compute_avg_return(environment, policy, max_epi_len=10000, num_episodes=5):

  rewards = [] 
  for _ in range(num_episodes):

    state = environment.reset()
    state = environment.normalise_state(state)
    ep_reward = 0
    step = 0
    ep_done = False

    while not ep_done:
        action = policy(np.expand_dims(state.astype(np.float32), 0))[0]
        state, reward, terminal = environment.step(action)
        state = environment.normalise_state(state)

        ep_reward += reward
        step += 1

        # Episode can finish either by reaching terminal state or max episode steps
        if terminal or step == max_epi_len:
            rewards.append(ep_reward)
            ep_done = True  
  mean_ep_reward = np.mean(rewards)
  return mean_ep_reward

def scalar_summary(writer, name, value, step):
    with writer.as_default():
        tf.summary.scalar(name, value, step=step)
    writer.flush()
