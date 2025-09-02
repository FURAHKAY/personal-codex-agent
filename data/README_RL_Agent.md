# RL GridWorld Agent

## Overview

This project explores reinforcement learning (RL) through two approaches:

1. **Q-Learning Agent** – a tabular RL algorithm that learns action–value functions directly.
2. **Transformer-based RL Agent** – an experimental approach that uses sequence modelling (Transformer encoder) to approximate Q-values.

Both agents are trained in a simple **GridWorld environment**, where the task is to move from a start position to a goal while maximizing reward.

---

## Environment

* **GridWorld**: a 5×5 grid with a fixed start state and a goal state.
* **Actions**: up, down, left, right.
* **Rewards**:

  * `+1` for reaching the goal.
  * `-0.1` penalty for each step (to encourage efficiency).

---

## Agents

### Q-Learning Agent

* Maintains a **Q-table** with state–action values.
* Uses an **ε-greedy policy** for exploration.
* Updates Q-values using the TD target:

  $$
  Q(s,a) \leftarrow Q(s,a) + \alpha \Big(r + \gamma \max_{a'} Q(s',a') - Q(s,a)\Big)
  $$
* Provides a baseline to compare against more complex agents.

### Transformer RL Agent

* Uses an **embedding layer** to represent states.
* Adds **positional encoding** to capture order in state sequences.
* A **Transformer encoder** processes the sequence.
* A **linear head** predicts Q-values for available actions.
* Shows how sequence models can generalize beyond tabular Q-learning.

---

## Results

* **Q-Learning** converges steadily, showing increasing total rewards across episodes.
* **Transformer RL Agent** learns slower but demonstrates the potential of sequence-based models in RL.
* Training curves are saved in the `plots/` folder:

  * `q_learning_reward_curve.png`
  * `transformer_rl_reward_curve.png`

---

## Tech Stack

* **Python**
* **NumPy** (Q-Learning implementation)
* **PyTorch** (Transformer RL agent)
* **Matplotlib** (plots)

---

## How to Run

1. **Q-Learning Agent**

   ```bash
   python train_q_learning.py
   ```

2. **Transformer RL Agent**

   ```bash
   python train_transformer_rl.py
   ```

Plots will be saved in the `plots/` directory.

---

## Future Work

* Extend GridWorld to larger or more complex environments.
* Compare with policy-gradient methods.
* Experiment with attention-based replay buffers.

