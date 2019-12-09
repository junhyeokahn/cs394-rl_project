import os
import time
from absl import app
from absl import flags
from absl import logging
import threading

import tensorflow as tf
import tqdm

from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.policies import policy_saver

from params import TrainingParameters
from params import EvaluationParameters
from networks import distributional_critic_network
from d4pg import d4pg_agent
from utils.misc_utils import get_log_dir, run_background

def train_eval():
    # ==========================================================================
    # Setup Logging
    # ==========================================================================
    log_dir = get_log_dir(TrainingParameters.ENV, TrainingParameters.ALGO)
    train_summary_writer = tf.compat.v2.summary.create_file_writer(
        log_dir + '/train', flush_millis=TrainingParameters.LOG_FLUSH_STEP * 1000)
    train_summary_writer.set_as_default()

    eval_summary_writer = tf.compat.v2.summary.create_file_writer(
        log_dir + '/eval', flush_millis=EvaluationParameters.LOG_FLUSH_STEP * 1000)
    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=EvaluationParameters.NUM_EVAL_EPISODE),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=EvaluationParameters.NUM_EVAL_EPISODE)
    ]

    global_step = tf.compat.v1.train.get_or_create_global_step()
    with tf.compat.v2.summary.record_if(
          lambda: tf.math.equal(global_step % TrainingParameters.SUMMARY_INTERVAL, 0)):
        # ======================================================================
        # Create Parallel Environment
        # ======================================================================
        if TrainingParameters.NUM_AGENTS > 1:
          tf_env = tf_py_environment.TFPyEnvironment(
              parallel_py_environment.ParallelPyEnvironment(
                  [lambda: TrainingParameters.ENV_LOAD_FN(TrainingParameters.ENV)] * TrainingParameters.NUM_AGENTS))
        else:
          tf_env = tf_py_environment.TFPyEnvironment(
                  TrainingParameters.ENV_LOAD_FN(TrainingParameters.ENV))
        eval_tf_env = tf_py_environment.TFPyEnvironment(TrainingParameters.ENV_LOAD_FN(TrainingParameters.ENV))

        # ======================================================================
        # Create Actor Network
        # ======================================================================
        actor_net = actor_network.ActorNetwork(
            tf_env.time_step_spec().observation,
            tf_env.action_spec(),
            fc_layer_params=TrainingParameters.ACTOR_LAYERS,
            activation_fn=TrainingParameters.ACTOR_ACTIVATION
        )

        # ======================================================================
        # Create Critic Network and Agent
        # ======================================================================

        critic_input_tensor_spec = (tf_env.time_step_spec().observation,
            tf_env.action_spec())
        if TrainingParameters.ALGO == 'D4PG':
            critic_net = distributional_critic_network.DistributionalCriticNetwork(
                critic_input_tensor_spec,
                num_atoms=TrainingParameters.NUM_ATOMS,
                observation_fc_layer_params=TrainingParameters.CRITIC_OBSERVATION_LAYERS,
                action_fc_layer_params=TrainingParameters.CRITIC_ACTION_LAYERS,
                joint_fc_layer_params=TrainingParameters.CRITIC_JOINT_LAYERS,
                activation_fn=TrainingParameters.CRITIC_ACTIVATION
            )
            tf_agent = d4pg_agent.D4pgAgent(
                tf_env.time_step_spec(),
                tf_env.action_spec(),
                actor_network=actor_net,
                critic_network=critic_net,
                min_v=TrainingParameters.V_MIN,
                max_v=TrainingParameters.V_MAX,
                n_step_return=TrainingParameters.N_STEP_RETURN,
                actor_optimizer=tf.keras.optimizers.Adam(
                    learning_rate=TrainingParameters.ACTOR_LEARNING_RATE),
                actor_l2_lambda=TrainingParameters.ACTOR_L2_LAMBDA,
                critic_optimizer=tf.keras.optimizers.Adam(
                    learning_rate=TrainingParameters.CRITIC_LEARNING_RATE),
                critic_l2_lambda=TrainingParameters.CRITIC_L2_LAMBDA,
                ou_stddev=TrainingParameters.OU_STDDEV,
                ou_damping=TrainingParameters.OU_DAMPING,
                target_update_tau=TrainingParameters.TAU,
                target_update_period=TrainingParameters.TARGET_UPDATE_PERIOD,
                dqda_clipping=None,
                td_errors_loss_fn=TrainingParameters.TD_ERROR_LOSS_FN,
                gamma=TrainingParameters.DISCOUNT_RATE,
                reward_scale_factor=TrainingParameters.REWARD_SCALE_FACTOR,
                gradient_clipping=None,
                debug_summaries=True,
                summarize_grads_and_vars=True,
                train_step_counter=global_step)

        elif TrainingParameters.ALGO == 'DDPG':
            critic_net = critic_network.CriticNetwork(
                critic_input_tensor_spec,
                observation_fc_layer_params=TrainingParameters.CRITIC_OBSERVATION_LAYERS,
                action_fc_layer_params=TrainingParameters.CRITIC_ACTION_LAYERS,
                joint_fc_layer_params=TrainingParameters.CRITIC_JOINT_LAYERS,
            )
            tf_agent = ddpg_agent.DdpgAgent(
                tf_env.time_step_spec(),
                tf_env.action_spec(),
                actor_network=actor_net,
                critic_network=critic_net,
                actor_optimizer=tf.keras.optimizers.Adam(
                    learning_rate=TrainingParameters.ACTOR_LEARNING_RATE),
                critic_optimizer=tf.keras.optimizers.Adam(
                    learning_rate=TrainingParameters.CRITIC_LEARNING_RATE),
                train_step_counter = global_step,
                gamma = TrainingParameters.DISCOUNT_RATE,
                td_errors_loss_fn=TrainingParameters.TD_ERROR_LOSS_FN,
                target_update_tau=TrainingParameters.TAU,
                target_update_period=TrainingParameters.TARGET_UPDATE_PERIOD
            )
        else:
            raise ValueError('Received Wrong ALGO in params.py')

        tf_agent.initialize()

        # ======================================================================
        # Setup Replay Buffer, Trajectory Collector, and Training Operator
        # ======================================================================
        train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(batch_size=TrainingParameters.NUM_AGENTS),
            tf_metrics.AverageEpisodeLengthMetric(batch_size=TrainingParameters.NUM_AGENTS),
        ]

        collect_policy = tf_agent.collect_policy

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            tf_agent.collect_data_spec,
            batch_size=tf_env.batch_size,
            max_length=TrainingParameters.REPLAY_MEM_SIZE)

        initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
            tf_env,
            collect_policy,
            observers=[replay_buffer.add_batch],
            num_steps=TrainingParameters.INITIAL_REPLAY_MEM_SIZE)

        collect_driver = dynamic_step_driver.DynamicStepDriver(
            tf_env,
            collect_policy,
            observers=[replay_buffer.add_batch] + train_metrics,
            num_steps=TrainingParameters.NUM_DATA_COLLECT)

        # Dataset generates trajectories with shape [B x (N_STEP_RETURN+1) x...]
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=TrainingParameters.BATCH_SIZE,
            num_steps=TrainingParameters.N_STEP_RETURN + 1).prefetch(tf.data.experimental.AUTOTUNE)

        dataset_iterator = iter(dataset)
        def train_step():
            experience, _ = next(dataset_iterator)
            return tf_agent.train(experience)

        if TrainingParameters.USE_TF_FUNCTION:
          initial_collect_driver.run = common.function(initial_collect_driver.run)
          collect_driver.run = common.function(collect_driver.run)
          tf_agent.train = common.function(tf_agent.train)
          train_step = common.function(train_step)

        # ======================================================================
        # Collecting Data
        # ======================================================================
        initial_collect_driver.run()

        stop_threading_event = threading.Event()
        threads = []
        threads.append(threading.Thread(target=run_background, args=(collect_driver.run, stop_threading_event)))

        for thread in threads:
            thread.start()

        # ======================================================================
        # Initial Policy Evaluation
        # ======================================================================
        eval_policy = tf_agent.policy
        results = metric_utils.eager_compute(
            eval_metrics,
            eval_tf_env,
            eval_policy,
            num_episodes=EvaluationParameters.NUM_EVAL_EPISODE,
            train_step=global_step,
            summary_writer=eval_summary_writer,
            summary_prefix='Metrics',
        )
        metric_utils.log_metrics(eval_metrics)

        # ======================================================================
        # Train the Agent
        # ======================================================================
        timed_at_step = global_step.numpy()
        time_acc = 0
        for _ in range(TrainingParameters.NUM_ITERATION):
            start_time = time.time()
            for _ in range(TrainingParameters.NUM_STEP_PER_ITERATION):
              train_loss = train_step()
            time_acc += time.time() - start_time

            if global_step.numpy() % TrainingParameters.LOG_INTERVAL == 0:
              logging.info('step = %d, loss = %f', global_step.numpy(),
                           train_loss.loss)
              steps_per_sec = (global_step.numpy() - timed_at_step) / time_acc
              logging.info('%.3f steps/sec', steps_per_sec)
              tf.compat.v2.summary.scalar(
                  name='global_steps_per_sec', data=steps_per_sec, step=global_step)
              timed_at_step = global_step.numpy()
              time_acc = 0

            for train_metric in train_metrics:
              train_metric.tf_summaries(
                  train_step=global_step, step_metrics=train_metrics[:2])

            if global_step.numpy() % EvaluationParameters.EVAL_INTERVAL == 0:
              results = metric_utils.eager_compute(
                  eval_metrics,
                  eval_tf_env,
                  eval_policy,
                  num_episodes=EvaluationParameters.NUM_EVAL_EPISODE,
                  train_step=global_step,
                  summary_writer=eval_summary_writer,
                  summary_prefix='Metrics',
              )
              metric_utils.log_metrics(eval_metrics)
              policy_saver.PolicySaver(eval_policy).save(log_dir + '/eval/policy_%d' % global_step)

        stop_threading_event.set()


if __name__ == "__main__":
    tf.compat.v1.enable_v2_behavior()
    logging.set_verbosity(logging.INFO)

    train_eval()

    print("Training Finished!")
