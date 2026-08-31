# Proximal Policy Optimization (PPO)

## Overview

Proximal Policy Optimization (PPO) is an on-policy policy gradient algorithm that alternates between sampling data through interaction with the environment and optimizing a clipped surrogate objective. PPO strikes a balance between ease of implementation, sample complexity, and ease of tuning, making it one of the most widely used RL algorithms in practice.

**Paper**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017)

**Action Space**: Discrete and Continuous

**Policy Type**: On-policy, policy gradient

## How It Works

PPO learns a stochastic policy $\pi_\theta(a|s)$ that directly maps states to action probabilities. The key innovation is the clipped surrogate objective, which prevents excessively large policy updates that can destabilize training.

### Key Components

1. **Actor Network**: Outputs action probabilities (discrete) or mean and std of a Gaussian distribution (continuous).

2. **Critic Network**: Estimates the state-value function $V(s)$ for advantage estimation.

3. **Generalized Advantage Estimation (GAE)**: Computes advantage estimates by balancing bias and variance through a parameter $\lambda$.

### Clipped Surrogate Objective

PPO optimizes the following objective:

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

Where:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$: Probability ratio
- $A_t$: Advantage estimate
- $\epsilon$: Clip parameter (typically 0.1 or 0.2)

The clipping ensures the policy update does not deviate too far from the old policy, improving training stability.

### Value Function Loss

The critic is updated to minimize the value function loss:

$$
L^{VF}(\theta) = \mathbb{E}_t \left[ (V_\theta(s_t) - V_t^{target})^2 \right]
$$

### Entropy Bonus

An entropy bonus is added to encourage exploration:

$$
L^{S}(\theta) = \mathbb{E}_t [H(\pi_\theta(\cdot|s_t))]
$$

### Total Objective

$$
L(\theta) = L^{CLIP} - c_1 L^{VF} + c_2 L^{S}
$$

Where $c_1$ and $c_2$ are coefficients for value loss and entropy bonus.

## Configuration

### Algorithm Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `learning_rate` | float | 3e-4 | Learning rate for actor and critic optimizers |
| `gamma` | float | 0.99 | Discount factor for future rewards |
| `gae_lambda` | float | 0.95 | GAE lambda parameter for advantage estimation |
| `clip_coef` | float | 0.2 | PPO clipping parameter ($\epsilon$) |
| `ent_coef` | float | 0.01 | Entropy bonus coefficient |
| `vf_coef` | float | 0.5 | Value function loss coefficient |
| `max_grad_norm` | float | 0.5 | Maximum gradient norm for clipping |
| `n_epochs` | int | 10 | Number of optimization epochs per rollout |
| `num_minibatches` | int | 4 | Number of minibatches per epoch |
| `clip_vloss` | bool | True | Whether to clip value function loss |
| `norm_adv` | bool | True | Whether to normalize advantages |

### Rollout Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rollout_length` | int | 2048 | Number of steps per rollout collection |
| `total_timesteps` | int | 1e6 | Total training timesteps |

### Network Configuration

See [MLP Configuration](../user_guide/mlp_configuration.md) for network architecture settings.

## Usage Example

```python
from cares_reinforcement_learning.algorithm.policy import PPO
from cares_reinforcement_learning.algorithm.configurations import AlgorithmConfig
from cares_reinforcement_learning.networks import MLPNetwork

# Configure the algorithm
config = AlgorithmConfig(
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_coef=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    n_epochs=10,
    num_minibatches=4,
)

# Create actor and critic networks
actor_network = MLPNetwork(
    observation_size=env.observation_space.shape[0],
    action_size=env.action_space.shape[0],
    output_activation="tanh",  # for continuous actions
)

critic_network = MLPNetwork(
    observation_size=env.observation_space.shape[0],
    action_size=1,  # value output
)

# Initialize PPO
algorithm = PPO(
    actor_network=actor_network,
    critic_network=critic_network,
    config=config,
    device="cuda",
)

# Training loop
for update in range(num_updates):
    # Collect rollout
    for step in range(config.rollout_length):
        action, log_prob, value = algorithm.act(observation)
        next_observation, reward, done, info = env.step(action)
        
        rollout_buffer.add(observation, action, reward, done, log_prob, value)
        observation = next_observation
    
    # Compute returns and advantages
    rollout_buffer.compute_returns_and_advantages(
        last_value=algorithm.get_value(observation),
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    
    # Update policy
    metrics = algorithm.train(rollout_buffer, episode_context)
```

## Stability Metrics

Monitor the following metrics to assess PPO training stability:

### Policy Metrics

| Metric | Expected Behavior | Warning Signs |
|--------|------------------|---------------|
| `policy_loss` | Negative, gradually approaches 0 | Becomes positive (policy worse than old), large magnitude |
| `approx_kl` | Small (< 0.01-0.02), stable | Exceeds 0.05 consistently (too large updates) |
| `clip_fraction` | 0.05-0.3, stable | Near 0 (no clipping, lr too low) or > 0.5 (too aggressive) |
| `entropy` | Gradually decreases, stabilizes | Drops to near 0 too quickly (premature convergence) |

### Value Metrics

| Metric | Expected Behavior | Warning Signs |
|--------|------------------|---------------|
| `value_loss` | Decreases then stabilizes | Continuous growth, NaN |
| `explained_variance` | Increases toward 1.0 | Negative (critic worse than mean prediction) |
| `values_mean` | Tracks cumulative reward | Diverges from actual returns |

### Performance Metrics

| Metric | Expected Behavior | Warning Signs |
|--------|------------------|---------------|
| `episode_return` | Improves over time | No improvement after 100+ updates |
| `evaluation_return` | Smoother improvement | Consistently below baseline |

## Common Issues and Solutions

### 1. Policy Collapse / Premature Convergence

**Symptom**: Entropy drops to near 0, policy becomes deterministic, performance plateaus at suboptimal level.

**Causes**:
- Entropy coefficient too low
- Learning rate too high
- Too many optimization epochs per rollout

**Solutions**:
- Increase `ent_coef` to 0.02-0.05
- Reduce learning rate to 1e-4 or 5e-5
- Reduce `n_epochs` to 4-8
- Increase `clip_coef` to 0.3 for more permissive updates

### 2. Large KL Divergence

**Symptom**: `approx_kl` consistently exceeds 0.05, training unstable.

**Causes**:
- Learning rate too high
- Too many epochs per rollout
- Rollout length too short

**Solutions**:
- Reduce learning rate by 50%
- Reduce `n_epochs` to 5-8
- Increase `rollout_length` to 4096
- Enable early stopping based on KL threshold

### 3. Value Function Divergence

**Symptom**: `value_loss` grows, `explained_variance` becomes negative.

**Causes**:
- Value learning rate too high
- Reward scale too large
- Missing value clipping

**Solutions**:
- Use separate, lower learning rate for critic
- Normalize or clip rewards
- Enable `clip_vloss`
- Reduce `vf_coef` to 0.25

### 4. No Learning Progress

**Symptom**: Episode return stays at baseline after many updates.

**Causes**:
- Rollout length too short for the task
- Learning rate too low
- Network architecture too simple
- Advantage normalization issues

**Solutions**:
- Increase `rollout_length` to 4096 or 8192
- Increase learning rate to 5e-4 or 1e-3
- Try larger network (256 or 512 hidden units, 2-3 layers)
- Verify `norm_adv` is enabled

## Hyperparameter Tuning Guide

### Start Here (Default Configuration)

```python
config = AlgorithmConfig(
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_coef=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    n_epochs=10,
    num_minibatches=4,
    rollout_length=2048,
)
```

### If Training is Unstable:
- Reduce `learning_rate` to 1e-4
- Reduce `n_epochs` to 5
- Reduce `clip_coef` to 0.1

### If Learning is Too Slow:
- Increase `learning_rate` to 5e-4
- Increase `n_epochs` to 15-20
- Increase `rollout_length` to 4096

### For Sparse Reward Environments:
- Increase `ent_coef` to 0.05-0.1
- Use larger `rollout_length` (4096+)
- Consider reward shaping or curriculum learning

## Comparison with Other Algorithms

| Aspect | PPO | DQN | SAC | TD3 |
|--------|-----|-----|-----|-----|
| Policy Type | On-policy | Off-policy | Off-policy | Off-policy |
| Action Space | Any | Discrete | Continuous | Continuous |
| Sample Efficiency | Low | High | High | High |
| Stability | High | Medium | High | High |
| Implementation Complexity | Medium | Low | High | Medium |
| Hyperparameter Sensitivity | Low | Medium | High | Medium |

**When to choose PPO**:
- Discrete or continuous action spaces
- On-policy learning is acceptable
- Simplicity and robustness are prioritized over sample efficiency
- Reproducibility and ease of tuning are important

## References

1. Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. *arXiv preprint arXiv:1707.06347*.
2. Schulman, J., et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation. *ICLR*.
3. Schulman, J., et al. (2015). Trust Region Policy Optimization. *ICML*.
