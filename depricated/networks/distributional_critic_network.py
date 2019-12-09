import tensorflow as tf

from tf_agents.networks import encoding_network
from tf_agents.networks import network
from tf_agents.networks import utils

class DistributionalCriticNetwork(network.Network):
  """Creates a distributional critic network."""

  def __init__(self,
               input_tensor_spec,
               num_atoms,
               observation_conv_layer_params=None,
               observation_fc_layer_params=None,
               observation_dropout_layer_params=None,
               action_fc_layer_params=None,
               action_dropout_layer_params=None,
               joint_fc_layer_params=None,
               joint_dropout_layer_params=None,
               activation_fn=tf.nn.relu,
               preprocessing_layers=None,
               preprocessing_combiner=None,
               conv_layer_params=None,
               kernel_initializer=None,
               name='DistributionalCriticNetwork'):
    """Creates an instance of `DistributionalCriticNetwork`.

    Args:
      input_tensor_spec: A tuple of (observation, action) each a nest of
        `tensor_spec.TensorSpec` representing the inputs.
      observation_conv_layer_params: Optional list of convolution layer
        parameters for observations, where each item is a length-three tuple
        indicating (num_units, kernel_size, stride).
      observation_fc_layer_params: Optional list of fully connected parameters
        for observations, where each item is the number of units in the layer.
      observation_dropout_layer_params: Optional list of dropout layer
        parameters, each item is the fraction of input units to drop or a
        dictionary of parameters according to the keras.Dropout documentation.
        The additional parameter `permanent', if set to True, allows to apply
        dropout at inference for approximated Bayesian inference. The dropout
        layers are interleaved with the fully connected layers; there is a
        dropout layer after each fully connected layer, except if the entry in
        the list is None. This list must have the same length of
        observation_fc_layer_params, or be None.
      action_fc_layer_params: Optional list of fully connected parameters for
        actions, where each item is the number of units in the layer.
      action_dropout_layer_params: Optional list of dropout layer parameters,
        each item is the fraction of input units to drop or a dictionary of
        parameters according to the keras.Dropout documentation. The additional
        parameter `permanent', if set to True, allows to apply dropout at
        inference for approximated Bayesian inference. The dropout layers are
        interleaved with the fully connected layers; there is a dropout layer
        after each fully connected layer, except if the entry in the list is
        None. This list must have the same length of action_fc_layer_params, or
        be None.
      joint_fc_layer_params: Optional list of fully connected parameters after
        merging observations and actions, where each item is the number of units
        in the layer.
      joint_dropout_layer_params: Optional list of dropout layer parameters,
        each item is the fraction of input units to drop or a dictionary of
        parameters according to the keras.Dropout documentation. The additional
        parameter `permanent', if set to True, allows to apply dropout at
        inference for approximated Bayesian inference. The dropout layers are
        interleaved with the fully connected layers; there is a dropout layer
        after each fully connected layer, except if the entry in the list is
        None. This list must have the same length of joint_fc_layer_params, or
        be None.
      activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
      output_activation_fn: Activation function for the last layer. This can be
        used to restrict the range of the output. For example, one can pass
        tf.keras.activations.sigmoid here to restrict the output to be bounded
        between 0 and 1.
      name: A string representing name of the network.

    Raises:
      ValueError: If `observation_spec` or `action_spec` contains more than one
        observation.
    """
    super(DistributionalCriticNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        name=name)

    self._num_atoms = num_atoms

    observation_spec, action_spec = input_tensor_spec

    if action_fc_layer_params == None:
        action_fc_layer_params = ()
    if observation_fc_layer_params == None:
        observation_fc_layer_params = ()
    if (action_fc_layer_params != ()) and (observation_fc_layer_params != ()):
        if (observation_fc_layer_params[-1] != action_fc_layer_params[-1]):
            raise ValueError('Observation layer and action layer connot be merged')

    if len(tf.nest.flatten(observation_spec)) > 1:
      raise ValueError('Only a single observation is supported by this network')

    if not kernel_initializer:
      kernel_initializer = tf.keras.initializers.GlorotUniform()

    self._use_obs_encoder = False
    if len(observation_fc_layer_params) > 1:
        self._use_obs_encoder = True
        self._observation_encoder = encoding_network.EncodingNetwork(
            observation_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=observation_fc_layer_params[:-1],
            dropout_layer_params=observation_dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer,
            batch_squash=True,
            name='ObservationEncodingNetwork'
            )
    self._use_obs_postprocessing_layer = False
    if not observation_fc_layer_params == ():
        self._use_obs_postprocessing_layer = True
        self._observation_postprocessing_layer = tf.keras.layers.Dense(
            observation_fc_layer_params[-1],
            activation=None,
            kernel_initializer=kernel_initializer)

    self._use_act_encoder = False
    if len(action_fc_layer_params) > 1:
        self._use_act_encoder = True
        self._action_encoder = encoding_network.EncodingNetwork(
            action_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=action_fc_layer_params[:-1],
            dropout_layer_params=action_dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer,
            batch_squash=True,
            name='ActionEncodingNetwork'
            )
    self._use_act_postprocessing_layer = False
    if not action_fc_layer_params == ():
        self._use_act_postprocessing_layer = True
        self._action_postprocessing_layer = tf.keras.layers.Dense(
            action_fc_layer_params[-1],
            activation=None,
            kernel_initializer=kernel_initializer)

    self._use_joint_encoder = False
    if len(joint_fc_layer_params) > 0:
        self._use_joint_encoder = True
        self._joint_encoder = encoding_network.EncodingNetwork(
            tf.TensorSpec(shape = (action_fc_layer_params[-1]), dtype=tf.float32),
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=joint_fc_layer_params,
            dropout_layer_params=action_dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer,
            batch_squash=True,
            name='JointEncodingNetwork'
            )
    self._postprocessing_layer = tf.keras.layers.Dense(
        self._num_atoms,
        activation=tf.keras.activations.softmax,
        kernel_initializer=tf.keras.initializers.RandomUniform(
        minval=-0.003, maxval=0.003))

  def call(self, inputs, step_type=None, network_state=(), training=False):
    observations, actions = inputs

    if self._use_obs_encoder:
        observations, _ = self._observation_encoder(observations,
            step_type=step_type, network_state=network_state)
    if self._use_obs_postprocessing_layer:
        observations = self._observation_postprocessing_layer(observations)

    if self._use_act_encoder:
        actions, _ = self._action_encoder(actions,
            step_type=step_type, network_state=network_state)
    if self._use_act_postprocessing_layer:
        actions = self._action_postprocessing_layer(actions)

    joint = tf.keras.activations.relu(observations + actions)
    if self._use_joint_encoder:
        joint, _ = self._joint_encoder(joint,
            step_type=step_type, network_state=network_state)
    value = self._postprocessing_layer(joint)

    return value, network_state

  @property
  def num_atoms(self):
      return self._num_atoms

  def get_action_grad(self, inputs, support):
    observations, actions = inputs
    _, original_actions = inputs

    if self._use_obs_encoder:
        observations, _ = self._observation_encoder(observations,
            step_type=1)
    if self._use_obs_postprocessing_layer:
        observations = self._observation_postprocessing_layer(observations)

    if self._use_act_encoder:
        actions, _ = self._action_encoder(actions,
            step_type=1)
    if self._use_act_postprocessing_layer:
        actions = self._action_postprocessing_layer(actions)

    joint = tf.keras.activations.relu(observations + actions)
    if self._use_joint_encoder:
        joint, _ = self._joint_encoder(joint,
            step_type=1)
    value = self._postprocessing_layer(joint)

    return tf.gradients(value, original_actions, support)
