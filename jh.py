from params import train_params
from utils.network import Actor, Critic
import tensorflow as tf
import numpy as np

# critic_net = Critic(train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, train_params.NUM_ATOMS, train_params.V_MIN, train_params.V_MAX, name='critic')
# inputs=(tf.constant(np.zeros(shape=(1,3), dtype=np.float32)), tf.constant(np.zeros(shape=(1,1)), dtype=np.float32))
# critic_net.save_weights('/Users/junhyeokahn/Repository/cs394-rl_project/data/')
# critic_net2 = Critic(train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, train_params.NUM_ATOMS, train_params.V_MIN, train_params.V_MAX, name='critic2')
# critic_net2.load_weights('/Users/junhyeokahn/Repository/cs394-rl_project/data/Pendulum-v0/D4PG_1/eval/critic_10')
# critic_net.save('/Users/Repository/cs394-rl_project/data/')




writer = tf.summary.create_file_writer("/tmp/mylogs/tf_function")

@tf.function
def my_func(step):
  with writer.as_default():
    # other model code would go here
    tf.summary.scalar("my_metric", 0.5, step=step)

for step in range(100):
  my_func(step)
  writer.flush()
