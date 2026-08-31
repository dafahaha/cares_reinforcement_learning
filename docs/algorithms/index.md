# Algorithms

CARES Reinforcement Learning provides implementations of both value-based and policy-based reinforcement learning algorithms for single-agent and multi-agent settings.

## Value-Based Algorithms

| Algorithm | Action Space | Key Features | Reference |
|-----------|-------------|--------------|-----------|
| [DQN](dqn.md) | Discrete | Experience replay, target network | [Mnih et al., 2015](https://www.nature.com/articles/nature14236) |
| DoubleDQN | Discrete | Decoupled action selection and evaluation | [van Hasselt et al., 2016](https://arxiv.org/abs/1509.06461) |
| DuelingDQN | Discrete | State-value and advantage decomposition | [Wang et al., 2016](https://arxiv.org/abs/1511.06581) |
| NoisyNet | Discrete | Noisy linear layers for exploration | [Fortunato et al., 2018](https://arxiv.org/abs/1706.10295) |
| PERDQN | Discrete | Prioritized experience replay | [Schaul et al., 2016](https://arxiv.org/abs/1511.05952) |
| C51 | Discrete | Distributional RL (Categorical) | [Bellemare et al., 2017](https://arxiv.org/abs/1707.06887) |
| QRDQN | Discrete | Distributional RL (Quantile Regression) | [Dabney et al., 2018](https://arxiv.org/abs/1710.10044) |
| Rainbow | Discrete | Combination of six DQN improvements | [Hessel et al., 2018](https://arxiv.org/abs/1710.02298) |

## Policy-Based Algorithms

| Algorithm | Action Space | Key Features | Reference |
|-----------|-------------|--------------|-----------|
| [PPO](ppo.md) | Discrete / Continuous | Clipped surrogate objective | [Schulman et al., 2017](https://arxiv.org/abs/1707.06347) |
| DDPG | Continuous | Deterministic policy gradient | [Lillicrap et al., 2016](https://arxiv.org/abs/1509.02971) |
| TD3 | Continuous | Twin delayed DDPG | [Fujimoto et al., 2018](https://arxiv.org/abs/1802.09477) |
| SAC | Continuous | Soft actor-critic, maximum entropy | [Haarnoja et al., 2018](https://arxiv.org/abs/1801.01290) |
| DroQ | Continuous | Dropout regularized Q-learning | [Hiraoka et al., 2022](https://arxiv.org/abs/2110.02034) |
| CrossQ | Continuous | Batch normalization in critic | [Bhatt et al., 2023](https://arxiv.org/abs/2302.02259) |
| CTD4 | Continuous | Continuous tempered distributional derivative | [CMS, 2024] |

## Prioritized Experience Replay Variants

The following algorithms combine the base algorithm with Prioritized Experience Replay (PER):

- PERTD3, PERSAC, PERDQN
- LAPTD3, LAPSAC (Loss-Adjusted Prioritization)
- LA3PTD3, LA3PSAC (Look-Ahead Loss-Adjusted Prioritization)
- MAPERTD3, MAPERSAC (Multi-step PER)
- PALTD3 (Prioritized Attention Learning)
- NaSATD3 (Noise-Suppressing Attention)

## Multi-Agent Algorithms

See the [MARL Cross Play](../user_guide/marl_cross_play.md) guide for multi-agent algorithm documentation.

## Selecting an Algorithm

### For discrete action spaces:
- **Start with DQN** as a baseline
- Use **Rainbow** for best performance (combines all improvements)
- Use **C51/QRDQN** if you need distributional value estimates
- Use **NoisyNet** for parameter-space exploration

### For continuous action spaces:
- **Start with TD3** as a stable baseline
- Use **SAC** for sample efficiency and maximum entropy exploration
- Use **PPO** for on-policy learning and discrete/continuous hybrid
- Use **DroQ/CrossQ** for large-batch training

## Stability Metrics

All algorithms log the following metrics for monitoring training stability:

- **Loss**: Policy loss and value/critic loss
- **Returns**: Episode return (mean, max, min over evaluation episodes)
- **Q-values**: Estimated Q-values (mean, max, min)
- **Exploration**: Epsilon (for value-based) or entropy (for SAC)
- **Learning rate**: Current learning rate

See individual algorithm pages for algorithm-specific metrics and stability guidance.
