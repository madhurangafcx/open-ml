# Reinforcement Learning

In **Reinforcement Learning (RL)**, an **agent** learns to make sequential decisions by interacting with an dynamic **environment**. The agent performs actions, transitions between states, and receives numerical feedback in the form of rewards or penalties, aiming to maximize the cumulative reward over time through trial and error.

```
                  +-------------+
                  | Environment |
                  +-------------+
                     ^       |
       Action (A_t)  |       | State (S_t), Reward (R_t)
                     |       v
                  +-------------+
                  |    Agent    |
                  +-------------+
```

---

## Core Problem Paradigms & Key Algorithms

### 1. Value-Based Methods
Methods where the agent learns a value function that predicts expected future returns, deriving an optimal policy indirectly by selecting actions that maximize expected value.

* **Q-Learning**: An off-policy Temporal Difference (TD) algorithm where the agent learns the quality $Q(s, a)$ of taking a specific action $a$ in a state $s$, updating via the Bellman optimality equation regardless of the action actually chosen by the current exploration policy.
* **SARSA (State-Action-Reward-State-Action)**: An on-policy TD algorithm where the policy is updated based on the actual transitions $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$ executed by the current policy, making it generally safer in hazardous environments.
* **Deep Q-Networks (DQN)**: Combines classical Q-learning with deep convolutional neural networks, using **experience replay buffers** and **target networks** to stabilize training. Famously proved superhuman capability across Atari 2600 games.

### 2. Policy Gradient & Actor-Critic Methods
Methods where the policy $\pi_\theta(a|s)$ is directly parameterized and optimized using gradient ascent on expected return.

* **REINFORCE (Monte Carlo Policy Gradient)**: Direct policy parameter optimization utilizing complete sampled episode trajectories.
* **Proximal Policy Optimization (PPO)**: A robust on-policy actor-critic algorithm using a clipped surrogate objective function that constrains policy updates to avoid destructively large steps. It strikes an optimal balance between sample complexity, ease of tuning, and stability, making it the de-facto standard in robotics and Reinforcement Learning from Human Feedback (**RLHF**) for modern LLMs (e.g., ChatGPT).
* **Actor-Critic (A2C / A3C)**: Dual-architecture paradigm where the **Actor** updates the policy based on directional suggestions provided by the **Critic** (which estimates the value function $V(s)$).
* **Deep Deterministic Policy Gradient (DDPG) & Soft Actor-Critic (SAC)**: Off-policy actor-critic methods tailored for continuous action spaces, with SAC maximizing both expected reward and policy entropy for enhanced exploration.

---

## Roadmap & Planned Notebooks

| Topic | Planned Directory | Description | Status |
| :--- | :--- | :--- | :--- |
| **Tabular Q-Learning** | `Q_Learning/` | GridWorld / FrozenLake discrete environments | :black_square_button: Planned |
| **SARSA** | `SARSA/` | On-policy navigation & cliff walking | :black_square_button: Planned |
| **Deep Q-Networks (DQN)** | `DQN/` | Deep Q-learning on Gymnasium environments | :black_square_button: Planned |
| **Policy Gradients & PPO** | `PPO/` | Continuous control & clipped surrogate objective | :black_square_button: Planned |
| **Actor-Critic (A2C / SAC)** | `Actor_Critic/` | Actor-critic implementations in continuous spaces | :black_square_button: Planned |
