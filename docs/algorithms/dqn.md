# Deep Q-Network (DQN)

## Overview

Deep Q-Network (DQN) is a value-based reinforcement learning algorithm that uses a deep neural network to approximate the Q-value function. It was the first algorithm to demonstrate successful learning of control policies directly from high-dimensional sensory input, achieving human-level performance on Atari 2600 games.

**Paper**: [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) (Mnih et al., Nature 2015)

**Action Space**: Discrete

**Policy Type**: Off-policy, value-based

## How It Works

DQN learns an action-value function $Q(s, a)$ that estimates the expected cumulative reward of taking action $a$ in state $s$ and following the optimal policy thereafter.

### Key Components

1. **Q-Network**: A neural network that takes state $s$ as input and outputs Q-values for all possible actions.

2. **Target Network**: A slowly-updated copy of the Q-network used to compute target Q-values, stabilizing training.

3. **Experience Replay Buffer**: Stores transitions $(s, a, r, s')$ and samples random mini-batches for training, breaking temporal correlations.

### Update Rule

The Q-network is updated by minimizing the temporal difference (TD) loss:

$$
L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D} \left[ \left( r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a) \right)^2 \right]
$$

Where:
- $\theta$: Q-network parameters
- $\theta^-$: Target network parameters (updated periodically)
- $\gamma$: Discount factor
- $D$: Experience replay buffer

### Target Network Update

The target network is updated either:
- **Hard update**: Copy Q-network weights every $N$ steps
- **Soft update**: $\theta^- \leftarrow \tau \theta + (1-\tau) \theta^-$ every step

## Configuration

### Algorithm Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `learning_rate` | float | 1e-4 | Learning rate for Q-network optimizer |
| `gamma` | float | 0.99 | Discount factor for future rewards |
| `tau` | float | 0.005 | Soft update coefficient for target network |
| `target_update_frequency` | int | 1000 | Steps between hard target network updates |
| `batch_size` | int | 64 | Mini-batch size sampled from replay buffer |
| `learning_starts` | int | 1000 | Steps before learning begins (buffer warmup) |
| `train_frequency` | int | 1 | Steps between training updates |
| `gradient_steps` | int | 1 | Number of gradient steps per update |

### Exploration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `exploration_fraction` | float | 0.1 | Fraction of total steps for epsilon decay |
| `start_epsilon` | float | 1.0 | Initial exploration rate |
| `end_epsilon` | float | 0.05 | Final exploration rate |

### Network Configuration

See [MLP Configuration](../user_guide/mlp_configuration.md) for network architecture settings.

## Usage Example

```python
from cares_reinforcement_learning.algorithm.value import DQN
from cares_reinforcement_learning.algorithm.configurations import AlgorithmConfig
from cares_reinforcement_learning.networks import MLPNetwork

# Configure the algorithm
config = AlgorithmConfig(
    learning_rate=1e-4,
    gamma=0.99,
    tau=0.005,
    batch_size=64,
)

# Create network
network = MLPNetwork(
    observation_size=env.observation_space.shape[0],
    action_size=env.action_space.n,
)

# Initialize DQN
algorithm = DQN(
    network=network,
    config=config,
    device="cuda",
)

# Training loop
for step in range(total_steps):
    action = algorithm.act(observation)
    next_observation, reward, done, info = env.step(action)
    
    memory_buffer.add(observation, action, reward, next_observation, done)
    
    if step > config.learning_starts:
        metrics = algorithm.train(memory_buffer, episode_context)
    
    observation = next_observation
```

## Stability Metrics

Monitor the following metrics to assess DQN training stability:

### Loss Metrics

| Metric | Expected Behavior | Warning Signs |
|--------|------------------|---------------|
| `q_loss` | Gradually decreases then stabilizes | Sudden spikes, continuous growth, NaN |
| `q_values_mean` | Increases during early learning, then plateaus | Monotonic decrease, divergence |
| `q_values_max` | Should remain bounded | Exponential growth (overestimation) |

### Performance Metrics

| Metric | Expected Behavior | Warning Signs |
|--------|------------------|---------------|
| `episode_return` | Improves over time, with noise | No improvement after exploration decay |
| `evaluation_return` | Smoother version of episode return | Consistently below random baseline |

### Exploration Metrics

| Metric | Expected Behavior | Warning Signs |
|--------|------------------|---------------|
| `epsilon` | Linearly decays from 1.0 to end_epsilon | Not decaying (stuck at 1.0) |

## Common Issues and Solutions

### 1. Q-value Divergence

**Symptom**: `q_values_max` grows exponentially, loss becomes NaN.

**Causes**:
- Learning rate too high
- Target network updating too frequently
- Missing gradient clipping

**Solutions**:
- Reduce learning rate to 1e-4 or lower
- Increase `target_update_frequency` to 2000+
- Ensure gradient clipping is enabled (max_norm=10)

### 2. No Learning Progress

**Symptom**: Episode return stays at baseline after exploration decay.

**Causes**:
- `learning_starts` too high
- Replay buffer size too small
- Network architecture too simple for the task

**Solutions**:
- Reduce `learning_starts` to 500-1000
- Increase replay buffer size to 100k+
- Try larger network (256 or 512 hidden units)

### 3. Training Instability

**Symptom**: Performance oscillates wildly, frequent catastrophic forgetting.

**Causes**:
- Batch size too small
- High variance in reward signal
- No reward normalization

**Solutions**:
- Increase batch size to 128 or 256
- Apply reward clipping or normalization
- Use DoubleDQN or DuelingDQN variants for more stable learning

## Variants

For improved performance, consider these DQN variants available in the library:

- **DoubleDQN**: Reduces Q-value overestimation by decoupling action selection from evaluation
- **DuelingDQN**: Separates state-value and advantage estimation for better value learning
- **Rainbow**: Combines six improvements (Double, Dueling, PER, NoisyNet, Distributional, Multi-step)
- **PERDQN**: Prioritized experience replay for more efficient learning
- **NoisyNet**: Parameter-space exploration for better exploration in sparse reward environments

## References

1. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.
2. Mnih, V., et al. (2013). Playing Atari with Deep Reinforcement Learning. *NeurIPS Deep Learning Workshop*.
3. van Hasselt, H., et al. (2016). Deep Reinforcement Learning with Double Q-learning. *AAAI*.
