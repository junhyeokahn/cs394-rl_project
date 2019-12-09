from __future__ import absolute_import, division, print_function

import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import PIL.Image
import pyglet
import pyvirtualdisplay
import numpy as np

#display = pyvirtualdisplay.Display(visible=0, size=(600, 400)).start()

import tensorflow as tf

from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.policies import py_tf_policy
from tf_agents.policies import greedy_policy
from tf_agents.policies import policy_saver


tf.compat.v1.enable_v2_behavior()

num_iterations = 125000 # @param {type:"integer"}

initial_collect_steps = 1000  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}

log_interval = 500000000  # @param {type:"integer"}

num_eval_episodes = 5  # @param {type:"integer"}
eval_interval = 10000 # @param {type:"integer"}

env_name = 'BipedalWalker-v2'

train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

eval_py_env.reset()
# eval_py_env.render()

if env_name == 'LunarLanderContinuous-v2':
    num_iterations = 700000
    fc_layer_params = (400,300)
    critic_fc_layer_params = (400,300)
    critic_obs_layer_params = None
    target_update_tau=0.001
    target_update_period=5
    actor_learning_rate = 0.000025  # @param {type:"number"}
    critic_learning_rate = 0.00025
elif env_name == 'Pendulum-v0':
    num_iterations = 125000
    fc_layer_params = (32,32)
    critic_fc_layer_params = (64,)
    critic_obs_layer_params = (32,)
    target_update_tau=0.001
    target_update_period=5
    actor_learning_rate = 0.0002  # @param {type:"number"}
    critic_learning_rate = 0.002
elif env_name == 'BipedalWalker-v2':
    num_iterations = 700000
    fc_layer_params = (400,300)
    critic_fc_layer_params = (400,300)
    critic_obs_layer_params = None
    target_update_tau=0.001
    target_update_period=5
    actor_learning_rate = 0.000025  # @param {type:"number"}
    critic_learning_rate = 0.00025

actor_net = actor_network.ActorNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params
)

critic_net_input_specs = (train_env.observation_spec(),
                          train_env.action_spec())

critic_net = critic_network.CriticNetwork(
    critic_net_input_specs,
    observation_fc_layer_params=critic_obs_layer_params,
    action_fc_layer_params=None,
    joint_fc_layer_params=critic_fc_layer_params,
)

global_step = tf.compat.v1.train.get_or_create_global_step()
tf_agent = ddpg_agent.DdpgAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    actor_network=actor_net,
    critic_network=critic_net,
    actor_optimizer=tf.compat.v1.train.AdamOptimizer(
        learning_rate=actor_learning_rate),
    critic_optimizer=tf.compat.v1.train.AdamOptimizer(
        learning_rate=critic_learning_rate),
    train_step_counter = global_step,
    gamma = 0.995,
    td_errors_loss_fn=tf.keras.losses.MSE,
    target_update_tau=target_update_tau,
    target_update_period=target_update_period
)
tf_agent.initialize()

eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy

def compute_avg_return(environment, policy, num_episodes=5):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]

print(compute_avg_return(eval_env, eval_policy, num_eval_episodes))

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    tf_agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)

initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=initial_collect_steps)
initial_collect_driver.run()

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)

iterator = iter(dataset)

collect_driver = dynamic_step_driver.DynamicStepDriver(
    train_env,
    collect_policy,
    observers=[replay_buffer.add_batch],
    num_steps=collect_steps_per_iteration)

tf_agent.train = common.function(tf_agent.train)
collect_driver.run = common.function(collect_driver.run)

# Reset the train step
tf_agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, eval_policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):

  # Collect a few steps using collect_policy and save to the replay buffer.
  for _ in range(collect_steps_per_iteration):
    collect_driver.run()

  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = tf_agent.train(experience)

  step = tf_agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss.loss))

  if step % eval_interval == 0:
    avg_return = compute_avg_return(eval_env, eval_policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)

iterations = range(0, num_iterations + 1, eval_interval)
plt.plot(iterations, returns)
plt.ylabel('Average Return')
plt.xlabel('Iterations')
plt.savefig('lander.png')

model_dir = "~/ddpgmodel2/"

policy_saver.PolicySaver(eval_policy).save(model_dir + 'eval_policy')
np.savetxt('result.out', returns, delimiter=',')

num_episodes = 5
video_filename = 'lander.mp4'

with imageio.get_writer(video_filename, fps=60) as video:
  for _ in range(num_episodes):
    time_step = eval_env.reset()
    video.append_data(eval_py_env.render())
    while not time_step.is_last():
      action_step = tf_agent.policy.action(time_step)
      time_step = eval_env.step(action_step.action)
      video.append_data(eval_py_env.render())
