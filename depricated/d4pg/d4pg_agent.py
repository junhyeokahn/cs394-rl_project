"""A D4PG Agent.
Implements the Distributed Distributional Deep Deterministic Policy Gradient (D4PG) algorithm from
"Continuous control with deep reinforcement learning" -
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gin
import tensorflow as tf

from tf_agents.agents import tf_agent
from tf_agents.policies import actor_policy
from tf_agents.policies import ou_noise_policy
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import eager_utils
from tf_agents.utils import nest_utils

class D4pgInfo(collections.namedtuple(
    'D4pgInfo', ('critic_loss'))):
  pass ##?????


@gin.configurable
class D4pgAgent(tf_agent.TFAgent):
  """A D4PG Agent."""

  def __init__(self,
               time_step_spec,
               action_spec,
               actor_network,
               critic_network,
               min_v,
               max_v,
               n_step_return=1,
               actor_optimizer=None,
               actor_l2_lambda=0.,
               critic_optimizer=None,
               critic_l2_lambda=0.,
               ou_stddev=0.2,
               ou_damping=0.15,
               target_actor_network=None,
               target_critic_network=None,
               target_update_tau=1.0,
               target_update_period=1,
               dqda_clipping=None,
               td_errors_loss_fn=None,
               gamma=1.0,
               reward_scale_factor=1.0,
               gradient_clipping=None,
               debug_summaries=False,
               summarize_grads_and_vars=False,
               train_step_counter=None,
               name=None):
    """Creates a D4PG Agent.
    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      actor_network: A tf_agents.network.Network to be used by the agent. The
        network will be called with call(observation, step_type[, policy_state])
        and should return (action, new_state).
      critic_network: A tf_agents.network.Network to be used by the agent. The
        network will be called with call((observation, action), step_type[,
        policy_state]) and should return (q_value, new_state).
      actor_optimizer: The optimizer to use for the actor network.
      critic_optimizer: The optimizer to use for the critic network.
      ou_stddev: Standard deviation for the Ornstein-Uhlenbeck (OU) noise added
        in the default collect policy.
      ou_damping: Damping factor for the OU noise added in the default collect
        policy.
      target_actor_network: (Optional.)  A `tf_agents.network.Network` to be
        used as the actor target network during Q learning.  Every
        `target_update_period` train steps, the weights from `actor_network` are
        copied (possibly withsmoothing via `target_update_tau`) to `
        target_q_network`.
        If `target_actor_network` is not provided, it is created by making a
        copy of `actor_network`, which initializes a new network with the same
        structure and its own layers and weights.
        Performing a `Network.copy` does not work when the network instance
        already has trainable parameters (e.g., has already been built, or
        when the network is sharing layers with another).  In these cases, it is
        up to you to build a copy having weights that are not
        shared with the original `actor_network`, so that this can be used as a
        target network.  If you provide a `target_actor_network` that shares any
        weights with `actor_network`, a warning will be logged but no exception
        is thrown.
      target_critic_network: (Optional.) Similar network as target_actor_network
         but for the critic_network. See documentation for target_actor_network.
      target_update_tau: Factor for soft update of the target networks.
      target_update_period: Period for soft update of the target networks.
      dqda_clipping: when computing the actor loss, clips the gradient dqda
        element-wise between [-dqda_clipping, dqda_clipping]. Does not perform
        clipping if dqda_clipping == 0.
      td_errors_loss_fn:  A function for computing the TD errors loss. If None,
        a default value of elementwise huber_loss is used.
      gamma: A discount factor for future rewards.
      reward_scale_factor: Multiplicative scale for the reward.
      gradient_clipping: Norm length to clip gradients.
      debug_summaries: A bool to gather debug summaries.
      summarize_grads_and_vars: If True, gradient and network variable summaries
        will be written during training.
      train_step_counter: An optional counter to increment every time the train
        op is run.  Defaults to the global_step.
      name: The name of this agent. All variables in this module will fall
        under that name. Defaults to the class name.
    """
    tf.Module.__init__(self, name=name)

    self._actor_network = actor_network
    actor_network.create_variables()
    if target_actor_network:
      target_actor_network.create_variables()
    self._target_actor_network = common.maybe_copy_target_network_with_checks(
        self._actor_network, target_actor_network, 'TargetActorNetwork')
    self._critic_network = critic_network
    critic_network.create_variables()
    if target_critic_network:
      target_critic_network.create_variables()
    self._target_critic_network = common.maybe_copy_target_network_with_checks(
        self._critic_network, target_critic_network, 'TargetCriticNetwork')

    if critic_network.num_atoms != self._target_critic_network.num_atoms:
        raise ValueError('Ciritc network and target critic network has different number of atoms!')
    self._num_atoms = critic_network.num_atoms

    min_v = tf.convert_to_tensor(min_v, dtype_hint=tf.float32)
    max_v = tf.convert_to_tensor(max_v, dtype_hint=tf.float32)
    self._support = tf.linspace(min_v, max_v, self._num_atoms)

    self._n_step_return = n_step_return

    self._actor_optimizer = actor_optimizer
    self._actor_l2_lambda = actor_l2_lambda
    self._critic_optimizer = critic_optimizer
    self._critic_l2_lambda = critic_l2_lambda

    self._ou_stddev = ou_stddev
    self._ou_damping = ou_damping
    self._target_update_tau = target_update_tau
    self._target_update_period = target_update_period
    self._dqda_clipping = dqda_clipping
    self._td_errors_loss_fn = (
        td_errors_loss_fn or common.element_wise_huber_loss)
    self._gamma = gamma
    self._reward_scale_factor = reward_scale_factor
    self._gradient_clipping = gradient_clipping

    self._update_target = self._get_target_updater(
        target_update_tau, target_update_period)

    policy = actor_policy.ActorPolicy(
        time_step_spec=time_step_spec, action_spec=action_spec,
        actor_network=self._actor_network, clip=True)
    collect_policy = actor_policy.ActorPolicy(
        time_step_spec=time_step_spec, action_spec=action_spec,
        actor_network=self._actor_network, clip=False)
    collect_policy = ou_noise_policy.OUNoisePolicy(
        collect_policy,
        ou_stddev=self._ou_stddev,
        ou_damping=self._ou_damping,
        clip=True)

    super(D4pgAgent, self).__init__(
        time_step_spec,
        action_spec,
        policy,
        collect_policy,
        train_sequence_length=(n_step_return+1),
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step_counter)

  def _initialize(self):
    common.soft_variables_update(
        self._critic_network.variables,
        self._target_critic_network.variables,
        tau=1.0)
    common.soft_variables_update(
        self._actor_network.variables,
        self._target_actor_network.variables,
        tau=1.0)

  def _get_target_updater(self, tau=1.0, period=1):
    """Performs a soft update of the target network parameters.
    For each weight w_s in the original network, and its corresponding
    weight w_t in the target network, a soft update is:
    w_t = (1- tau) x w_t + tau x ws
    Args:
      tau: A float scalar in [0, 1]. Default `tau=1.0` means hard update.
      period: Step interval at which the target networks are updated.
    Returns:
      An operation that performs a soft update of the target network parameters.
    """
    with tf.name_scope('get_target_updater'):
      def update():
        """Update target network."""
        # TODO(b/124381161): What about observation normalizer variables?
        critic_update = common.soft_variables_update(
            self._critic_network.variables,
            self._target_critic_network.variables,
            tau,
            tau_non_trainable=1.0)
        actor_update = common.soft_variables_update(
            self._actor_network.variables,
            self._target_actor_network.variables,
            tau,
            tau_non_trainable=1.0)
        return tf.group(critic_update, actor_update)

      return common.Periodically(update, period, 'periodic_update_targets')

  def _experience_to_transitions(self, experience):
    transitions = trajectory.to_transition(experience)

    # Remove time dim if we are not using a recurrent network.
    if not self._actor_network.state_spec:
      transitions = tf.nest.map_structure(lambda x: tf.squeeze(x, [1]),
                                          transitions)

    time_steps, policy_steps, next_time_steps = transitions
    actions = policy_steps.action
    return time_steps, actions, next_time_steps

  def _train(self, experience, weights=None):
    time_steps, actions, next_time_steps = self._experience_to_transitions(
        experience)

    # TODO(b/124382524): Apply a loss mask or filter boundary transitions.
    trainable_critic_variables = self._critic_network.trainable_variables
    with tf.GradientTape(watch_accessed_variables=False) as g:
      assert trainable_critic_variables, ('No trainable critic variables to optimize.')
      g.watch(trainable_critic_variables)
      critic_loss = self.critic_loss(experience, weights=weights)
    tf.debugging.check_numerics(critic_loss, 'Critic loss is inf or nan.')
    critic_grads = g.gradient(critic_loss, trainable_critic_variables)
    self._apply_gradients(critic_grads, trainable_critic_variables, self._critic_optimizer)

    trainable_actor_variables = self._actor_network.trainable_variables
    with tf.GradientTape() as gg:
      assert trainable_critic_variables, ('No trainable actor variables to optimize.')
      gg.watch(trainable_actor_variables)
      with tf.GradientTape() as ggg:
        actor_output = self._actor_network(time_steps.observation, time_steps.step_type)
        _, distribution, _ = self._critic_network((time_steps.observation, actor_output[0]), time_steps.step_type)
        action_grads = ggg.gradient(distribution, actor_output[0], self._support)
    actor_grads = gg.gradient(actor_output[0], trainable_actor_variables, -action_grads)
    batch_size = time_steps.observation.shape[0] or tf.shape(time_steps.observation)[0]
    actor_grads_scaled = list(map(lambda x: tf.divide(x, batch_size), actor_grads))
    self._apply_gradients(actor_grads_scaled, trainable_actor_variables, self._actor_optimizer)

    self.train_step_counter.assign_add(1)
    self._update_target()

    # TODO(b/124382360): Compute per element TD loss and return in loss_info.
    total_loss = critic_loss
    return tf_agent.LossInfo(total_loss,
                             D4pgInfo(critic_loss))

  def _apply_gradients(self, gradients, variables, optimizer):
    # Tuple is used for py3, where zip is a generator producing values once.
    grads_and_vars = tuple(zip(gradients, variables))
    if self._gradient_clipping is not None:
      grads_and_vars = eager_utils.clip_gradient_norms(grads_and_vars,
                                                       self._gradient_clipping)

    if self._summarize_grads_and_vars:
      eager_utils.add_variables_summaries(grads_and_vars,
                                          self.train_step_counter)
      eager_utils.add_gradients_summaries(grads_and_vars,
                                          self.train_step_counter)

    optimizer.apply_gradients(grads_and_vars)

  def critic_loss(self, experience, weights=None):
    # Check that `experience` includes two outer dimensions [B, T, ...]. This
    # method requires a time dimension to compute the loss properly.
    self._check_trajectory_dimensions(experience)

    if self._n_step_return == 1:
      time_steps, actions, next_time_steps = self._experience_to_transitions(
          experience)
    else:
      # To compute n-step returns, we need the first time steps, the first
      # actions, and the last time steps. Therefore we extract the first and
      # last transitions from our Trajectory.
      first_two_steps = tf.nest.map_structure(lambda x: x[:, :2], experience)
      last_two_steps = tf.nest.map_structure(lambda x: x[:, -2:], experience)
      time_steps, actions, _ = self._experience_to_transitions(first_two_steps)
      _, _, next_time_steps = self._experience_to_transitions(last_two_steps)

    with tf.name_scope('critic_loss'):
      tf.nest.assert_same_structure(actions, self.action_spec)
      tf.nest.assert_same_structure(time_steps, self.time_step_spec)
      tf.nest.assert_same_structure(next_time_steps, self.time_step_spec)

      target_actions, _ = self._target_actor_network(next_time_steps.observation, next_time_steps.step_type)
      target_critic_network_input = (next_time_steps.observation, target_actions)
      _, next_distribution, _ = self._target_critic_network(target_critic_network_input, next_time_steps.step_type)

      batch_size = next_distribution.shape[0] or tf.shape(next_distribution)[0]
      tiled_support = tf.tile(self._support, [batch_size])
      tiled_support = tf.reshape(tiled_support, [batch_size, self._num_atoms])

      if self._n_step_return == 1:
        discount = next_time_steps.discount
        if discount.shape.ndims == 1:
          # We expect discount to have a shape of [batch_size], while
          # tiled_support will have a shape of [batch_size, num_atoms]. To
          # multiply these, we add a second dimension of 1 to the discount.
          discount = discount[:, None]
        next_value_term = tf.multiply(discount,
                                      tiled_support,
                                      name='next_value_term')

        reward = next_time_steps.reward
        if reward.shape.ndims == 1:
          # See the explanation above.
          reward = reward[:, None]
        reward_term = tf.multiply(self._reward_scale_factor,
                                  reward,
                                  name='reward_term')

        target_support = tf.add(reward_term, self._gamma * next_value_term,
                                name='target_support')
      # TODO : This is not correct when n > 2
      else:
        # When computing discounted return, we need to throw out the last time
        # index of both reward and discount, which are filled with dummy values
        # to match the dimensions of the observation.
        rewards = self._reward_scale_factor * experience.reward[:, :-1]
        discounts = self._gamma * experience.discount[:, :-1]

        # TODO(b/134618876): Properly handle Trajectories that include episode
        # boundaries with nonzero discount.

        discounted_returns = value_ops.discounted_return(
            rewards=rewards,
            discounts=discounts,
            final_value=tf.zeros([batch_size], dtype=discounts.dtype),
            time_major=False,
            provide_all_returns=False)

        # Convert discounted_returns from [batch_size] to [batch_size, 1]
        discounted_returns = discounted_returns[:, None]

        final_value_discount = tf.reduce_prod(discounts, axis=1)
        final_value_discount = final_value_discount[:, None]

        # Save the values of discounted_returns and final_value_discount in
        # order to check them in unit tests.
        self._discounted_returns = discounted_returns
        self._final_value_discount = final_value_discount

        target_support = tf.add(discounted_returns,
                                final_value_discount * tiled_support,
                                name='target_support')

      target_distribution = tf.stop_gradient(self._project_distribution(
          target_support, next_distribution, self._support))

      logits, distribution, _ = self._critic_network((time_steps.observation, actions),
                                             time_steps.step_type)

      cross_entropy_loss = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(target_distribution),
                                                  logits=logits))
      l2_reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in self._critic_network.trainable_variables if 'kernel' in v.name]) * self._critic_l2_lambda

      critic_loss = cross_entropy_loss + l2_reg_loss

      with tf.name_scope('Losses/'):
        tf.compat.v2.summary.scalar('critic_loss',
                                    critic_loss,
                                    step=self.train_step_counter)

      if self._debug_summaries:
        distribution_errors = target_distribution - distribution
        with tf.name_scope('distribution_errors'):
          common.generate_tensor_summaries(
              'distribution_errors', distribution_errors,
              step=self.train_step_counter)
          tf.compat.v2.summary.scalar(
              'mean', tf.reduce_mean(distribution_errors),
              step=self.train_step_counter)
          tf.compat.v2.summary.scalar(
              'mean_abs', tf.reduce_mean(tf.abs(distribution_errors)),
              step=self.train_step_counter)
          tf.compat.v2.summary.scalar(
              'max', tf.reduce_max(distribution_errors),
              step=self.train_step_counter)
          tf.compat.v2.summary.scalar(
              'min', tf.reduce_min(distribution_errors),
              step=self.train_step_counter)
        with tf.name_scope('target_distribution'):
          common.generate_tensor_summaries(
              'target_distribution', target_distribution,
              step=self.train_step_counter)

      return critic_loss

  # The following method is copied from the Dopamine codebase with permission
  # (https://github.com/google/dopamine). Thanks to Marc Bellemare and also to
  # Pablo Castro, who wrote the original version of this method.
  def _project_distribution(self, supports, weights, target_support,
                            validate_args=False):
    """Projects a batch of (support, weights) onto target_support.
    Based on equation (7) in (Bellemare et al., 2017):
      https://arxiv.org/abs/1707.06887
    In the rest of the comments we will refer to this equation simply as Eq7.
    This code is not easy to digest, so we will use a running example to clarify
    what is going on, with the following sample inputs:
      * supports =       [[0, 2, 4, 6, 8],
                          [1, 3, 4, 5, 6]]
      * weights =        [[0.1, 0.6, 0.1, 0.1, 0.1],
                          [0.1, 0.2, 0.5, 0.1, 0.1]]
      * target_support = [4, 5, 6, 7, 8]
    In the code below, comments preceded with 'Ex:' will be referencing the above
    values.
    Args:
      supports: Tensor of shape (batch_size, num_dims) defining supports for the
        distribution.
      weights: Tensor of shape (batch_size, num_dims) defining weights on the
        original support points. Although for the CategoricalDQN agent these
        weights are probabilities, it is not required that they are.
      target_support: Tensor of shape (num_dims) defining support of the projected
        distribution. The values must be monotonically increasing. Vmin and Vmax
        will be inferred from the first and last elements of this tensor,
        respectively. The values in this tensor must be equally spaced.
      validate_args: Whether we will verify the contents of the
        target_support parameter.
    Returns:
      A Tensor of shape (batch_size, num_dims) with the projection of a batch of
      (support, weights) onto target_support.
    Raises:
      ValueError: If target_support has no dimensions, or if shapes of supports,
        weights, and target_support are incompatible.
    """
    target_support_deltas = target_support[1:] - target_support[:-1]
    # delta_z = `\Delta z` in Eq7.
    delta_z = target_support_deltas[0]
    validate_deps = []
    supports.shape.assert_is_compatible_with(weights.shape)
    supports[0].shape.assert_is_compatible_with(target_support.shape)
    target_support.shape.assert_has_rank(1)
    if validate_args:
      # Assert that supports and weights have the same shapes.
      validate_deps.append(
          tf.Assert(
              tf.reduce_all(tf.equal(tf.shape(supports), tf.shape(weights))),
              [supports, weights]))
      # Assert that elements of supports and target_support have the same shape.
      validate_deps.append(
          tf.Assert(
              tf.reduce_all(
                  tf.equal(tf.shape(supports)[1], tf.shape(target_support))),
              [supports, target_support]))
      # Assert that target_support has a single dimension.
      validate_deps.append(
          tf.Assert(
              tf.equal(tf.size(tf.shape(target_support)), 1), [target_support]))
      # Assert that the target_support is monotonically increasing.
      validate_deps.append(
          tf.Assert(tf.reduce_all(target_support_deltas > 0), [target_support]))
      # Assert that the values in target_support are equally spaced.
      validate_deps.append(
          tf.Assert(
              tf.reduce_all(tf.equal(target_support_deltas, delta_z)),
              [target_support]))

    with tf.control_dependencies(validate_deps):
      # Ex: `v_min, v_max = 4, 8`.
      v_min, v_max = target_support[0], target_support[-1]
      # Ex: `batch_size = 2`.
      batch_size = tf.shape(supports)[0]
      # `N` in Eq7.
      # Ex: `num_dims = 5`.
      num_dims = tf.shape(target_support)[0]
      # clipped_support = `[\hat{T}_{z_j}]^{V_max}_{V_min}` in Eq7.
      # Ex: `clipped_support = [[[ 4.  4.  4.  6.  8.]]
      #                         [[ 4.  4.  4.  5.  6.]]]`.
      clipped_support = tf.clip_by_value(supports, v_min, v_max)[:, None, :]
      # Ex: `tiled_support = [[[[ 4.  4.  4.  6.  8.]
      #                         [ 4.  4.  4.  6.  8.]
      #                         [ 4.  4.  4.  6.  8.]
      #                         [ 4.  4.  4.  6.  8.]
      #                         [ 4.  4.  4.  6.  8.]]
      #                        [[ 4.  4.  4.  5.  6.]
      #                         [ 4.  4.  4.  5.  6.]
      #                         [ 4.  4.  4.  5.  6.]
      #                         [ 4.  4.  4.  5.  6.]
      #                         [ 4.  4.  4.  5.  6.]]]]`.
      tiled_support = tf.tile([clipped_support], [1, 1, num_dims, 1])
      # Ex: `reshaped_target_support = [[[ 4.]
      #                                  [ 5.]
      #                                  [ 6.]
      #                                  [ 7.]
      #                                  [ 8.]]
      #                                 [[ 4.]
      #                                  [ 5.]
      #                                  [ 6.]
      #                                  [ 7.]
      #                                  [ 8.]]]`.
      reshaped_target_support = tf.tile(target_support[:, None], [batch_size, 1])
      reshaped_target_support = tf.reshape(reshaped_target_support,
                                           [batch_size, num_dims, 1])
      # numerator = `|clipped_support - z_i|` in Eq7.
      # Ex: `numerator = [[[[ 0.  0.  0.  2.  4.]
      #                     [ 1.  1.  1.  1.  3.]
      #                     [ 2.  2.  2.  0.  2.]
      #                     [ 3.  3.  3.  1.  1.]
      #                     [ 4.  4.  4.  2.  0.]]
      #                    [[ 0.  0.  0.  1.  2.]
      #                     [ 1.  1.  1.  0.  1.]
      #                     [ 2.  2.  2.  1.  0.]
      #                     [ 3.  3.  3.  2.  1.]
      #                     [ 4.  4.  4.  3.  2.]]]]`.
      numerator = tf.abs(tiled_support - reshaped_target_support)
      quotient = 1 - (numerator / delta_z)
      # clipped_quotient = `[1 - numerator / (\Delta z)]_0^1` in Eq7.
      # Ex: `clipped_quotient = [[[[ 1.  1.  1.  0.  0.]
      #                            [ 0.  0.  0.  0.  0.]
      #                            [ 0.  0.  0.  1.  0.]
      #                            [ 0.  0.  0.  0.  0.]
      #                            [ 0.  0.  0.  0.  1.]]
      #                           [[ 1.  1.  1.  0.  0.]
      #                            [ 0.  0.  0.  1.  0.]
      #                            [ 0.  0.  0.  0.  1.]
      #                            [ 0.  0.  0.  0.  0.]
      #                            [ 0.  0.  0.  0.  0.]]]]`.
      clipped_quotient = tf.clip_by_value(quotient, 0, 1)
      # Ex: `weights = [[ 0.1  0.6  0.1  0.1  0.1]
      #                 [ 0.1  0.2  0.5  0.1  0.1]]`.
      weights = weights[:, None, :]
      # inner_prod = `\sum_{j=0}^{N-1} clipped_quotient * p_j(x', \pi(x'))`
      # in Eq7.
      # Ex: `inner_prod = [[[[ 0.1  0.6  0.1  0.  0. ]
      #                      [ 0.   0.   0.   0.  0. ]
      #                      [ 0.   0.   0.   0.1 0. ]
      #                      [ 0.   0.   0.   0.  0. ]
      #                      [ 0.   0.   0.   0.  0.1]]
      #                     [[ 0.1  0.2  0.5  0.  0. ]
      #                      [ 0.   0.   0.   0.1 0. ]
      #                      [ 0.   0.   0.   0.  0.1]
      #                      [ 0.   0.   0.   0.  0. ]
      #                      [ 0.   0.   0.   0.  0. ]]]]`.
      inner_prod = clipped_quotient * weights
      # Ex: `projection = [[ 0.8 0.0 0.1 0.0 0.1]
      #                    [ 0.8 0.1 0.1 0.0 0.0]]`.
      projection = tf.reduce_sum(inner_prod, 3)
      projection = tf.reshape(projection, [batch_size, num_dims])
      return projection
