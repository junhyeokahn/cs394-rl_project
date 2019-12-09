# cs394-rl_project
Final project for CS394R: Implementation for Distributed [DDPG](https://arxiv.org/pdf/1509.02971.pdf) and [D4PG](https://arxiv.org/pdf/1804.08617.pdf) Algorithm.

## Code Explanation
- [depricated](https://github.com/junhyeokahn/cs394-rl_project/tree/master/depricated) includes our implementation of Distributed DDPG and D4PG using [tf_agent](https://github.com/tensorflow/agents).
    - [depricated/train_eval.py](https://github.com/junhyeokahn/cs394-rl_project/tree/master/depricated/train_eval.py) is for training and evaluating Distributed DDPG and D4PG based on [depricated/prams.py](https://github.com/junhyeokahn/cs394-rl_project/tree/master/depricated/params.py).
    - [depricated/train_ddpg.py](https://github.com/junhyeokahn/cs394-rl_project/tree/master/depricated/train_ddpg.py) is for training and evaluating DDPG.
    - [depricated/networks/distributional_critic_network.py](https://github.com/junhyeokahn/cs394-rl_project/tree/master/depricated/networks/distributional_critic_network.py) is an implementation of critic network returning the probability over the distribution, and [depricated/d4pg/d4pg_agent.py](https://github.com/junhyeokahn/cs394-rl_project/tree/master/depricated/d4pg/d4pg_agent.py) is an implementation of training.
    - [depricated/utils/misc_utils.py](https://github.com/junhyeokahn/cs394-rl_project/tree/master/depricated/utils/misc_utils.py) is an implementation of running an agent for a replay buffer.
    - Our Distributed DDPG implementation works well on several domains but the D4PG implementation does not show the convergence for some reason. Therefore the experiment results below is *not* from this D4PG implementation.
- In the [root](https://github.com/junhyeokahn/cs394-rl_project/tree/master/), there is another implementation of D4PG starting from existing [repository](https://github.com/msinto93/D4PG).
    - We have directly copied some usefule utility functions, such as [utils/prioritised_experience_replay.py](https://github.com/junhyeokahn/cs394-rl_project/blob/master/utils/prioritised_experience_replay.py), and [utils/env_wrapper.py](https://github.com/junhyeokahn/cs394-rl_project/blob/master/utils/env_wrapper.py).
    - However, we mostly reimplement [actor network](https://github.com/junhyeokahn/cs394-rl_project/blob/107291abc621649591c5db24f3146c013a96b54b/utils/network.py#L86) and [critic network](https://github.com/junhyeokahn/cs394-rl_project/blob/107291abc621649591c5db24f3146c013a96b54b/utils/network.py#L16), as well as [training part](https://github.com/junhyeokahn/cs394-rl_project/blob/107291abc621649591c5db24f3146c013a96b54b/learner.py#L75).
    - [train_eval.py](https://github.com/junhyeokahn/cs394-rl_project/blob/master/train_eval.py) runs training and evaluation of D4PG algorithm based on [prams.py](https://github.com/junhyeokahn/cs394-rl_project/tree/master/params.py).
    - [play.py](https://github.com/junhyeokahn/cs394-rl_project/blob/master/train_eval.py) reads trained networks and records a video.

## Experiment Results

### Pendulum-v0
<img src="https://github.com/junhyeokahn/cs394-rl_project/blob/master/figures/Pendulum-v0.gif" width="250" height="250">

### LunarLanderContinuous-v2
<img src="https://github.com/junhyeokahn/cs394-rl_project/blob/master/figures/LunarLanderContinuous-v2.gif" width="250" height="250">

### BipedalWalker-v2
<img src="https://github.com/junhyeokahn/cs394-rl_project/blob/master/figures/BipedalWalker-v2.gif" width="250" height="250">

## Run the Code Yourself

### Install requirements
- run ```pip install tensorflow``` or ```pip install tensorflow-gpu```
- run ```pip install -r requirements.txt```

### Training the agent with D4PG
- run ```python train_eval.py```
