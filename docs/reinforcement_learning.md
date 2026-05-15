# Reinforcement Learning in hmls

This guide explains how to train a neural network to play the hmls tank game.

The game's fog-of-war mechanic means a tank can only see a small patch of the map around it.
It must make sequential decisions — move, turn, fire, or wait — based on partial information, and it must *remember* what it saw on previous turns.
This makes the game a natural fit for reinforcement learning (RL), where an agent learns a policy by playing many games and receiving reward signals that tell it how well it's doing.

Several pre-built neural network architectures (called "tanks") are provided.
You can train them against each other or against a rule-based opponent, and you can implement your own tank architectures.

## What is reinforcement learning

In reinforcement learning an agent interacts with an environment over a sequence of time steps (an *episode*).
At each step the agent observes the state of the world, takes an action, and receives a numeric *reward*.
The goal is to learn a *policy* — a mapping from observations to actions — that maximises the total discounted reward over an episode:

```
G = r₁ + γ·r₂ + γ²·r₃ + … + γᵀ⁻¹·rₜ
```

Here `γ` (gamma) is a discount factor between 0 and 1.
Values close to 1 make the agent care about long-term outcomes; smaller values make it more myopic.

The trainer in this repository uses **REINFORCE**, a *policy gradient* algorithm.
The neural network directly outputs a probability distribution over actions (the policy).
After each episode, the algorithm adjusts the network weights to make actions that led to high returns more likely and actions that led to low returns less likely.
A baseline (running average of returns) reduces variance.

In practice, training involves playing thousands of games on randomly generated maps.
Each game is one episode.
The agent starts knowing nothing and gradually learns to explore, avoid walls, and (eventually) fight.

## Running the trainer

A ready-to-use sample configuration is included in the repository.
From the workspace root:

```bash
uv run hmls-reinforcetrainer sample_configs/single_tanks/config.json
```

This trains a Mk-I neural network (`model_a`) against a frozen random opponent (`model_b`).
The trainer generates random maps, plays games, and updates the trainee's weights after each game.
Progress is printed to stdout.

The trainer can be run on a previously trained model.
This will further refine the model's behaviour (although be careful of overfitting).

### Key configuration sections

The configuration file is JSON with several sections. The full reference is in
[`packages/hmls-reinforcetrainer/README.md`](../packages/hmls-reinforcetrainer/README.md);
below is a high-level overview of the most important knobs.

#### Model selection

```json
{
  "model_a": { "dir": "weights/model_a", "train": true },
  "model_b": { "dir": "weights/random_opponent", "train": false }
}
```

Each model entry points to a directory containing a `model_config.json` (which selects the architecture) and, optionally, a pre-trained `model.pt`.
Set `"train": true` to update that model's weights during training, or `false` to freeze it as a static opponent.

#### Reward shaping

Rewards guide learning.
They are configured per-model in a `"reward"` block with categories:

| Category | Examples | Purpose |
|----------|----------|---------|
| **actions** | `move_forward`, `turn_left`, `pass_action` | Encourage/discourage specific movement patterns |
| **firing** | `hit`, `miss`, `neglect` | Reward hitting enemies, penalise wasted shots or failing to fire when an enemy is visible |
| **game_state** | `win`, `loss`, `death`, `step` | Terminal rewards and per-step cost to encourage faster play |
| **exploration** | `see_cell`, `occupy_cell` | Reward discovering new terrain |
| **situational** | `enemy_in_cone` | Reward positioning relative to enemies |

Getting rewards right is the hardest part of training.
Start with the sample config's values and adjust based on observed behaviour in the replay viewer.

#### Hyperparameters

| Parameter | What it controls |
|-----------|-----------------|
| `learning_rate` | How fast the network updates (default 0.001) |
| `gamma` | Discount factor, as introduced above — how much to value future rewards (default 0.995) |
| `entropy_coeff` | Bonus for action diversity; prevents premature convergence |
| `max_grad_norm` | Gradient clipping to stabilise training |
| `seed` | Set for reproducibility; omit or `null` for random |

The entropy coefficient adds a bonus that rewards the policy for keeping its action probabilities spread out rather than collapsing onto a single action too early.
Without it, a single large reward in an early episode (perhaps from a particularly favourable starting position) can snatch the model's attention, and prevent the model exploring finding better strategies.
The value must be ≥ 0 (set it to 0 to disable the bonus entirely).
The default is 0.01; the sample config uses 0.1.
Larger values encourage more exploration but can prevent the policy from ever committing to a clear strategy.
If you do repeated training runs on the same set of weights, you may want to reduce the value in later runs.

#### Output

The trainer periodically saves:

- **Model weights** — `model.pt` in each model directory (every
  `save_weights_interval` games and at the end of training).
- **Sample games** — replay JSON files in `sample_game_dir` (every
  `sample_game_interval` games). View them with:

```bash
uv run hmls-replayviewer output/sample_games/game_000100.json
```

### Training tips

- One training run is unlikely to produce a good model. Expect to run multiple
  sessions, adjusting rewards and hyperparameters between runs.
- Start with small runs (`total_maps: 10`, `games_per_map: 5`) to verify
  everything works before committing to long runs.
- Check sample games regularly — they tell you more than loss numbers.
- Training against the random tank first provides a stable learning signal.
  Switch to self-play (both models training) once the agent can beat the
  random opponent reliably.

## Available tanks

Four tank implementations are included:

### Mk-I — CNN + GRU

```
Encoded Patch → [Conv→ReLU→Pool] × N → GRU → Linear → action logits
```

The general-purpose default.
Convolutional layers extract spatial features from the visibility patch; a GRU cell provides temporal memory across turns.

**Package:** `hmls-singlemki` · **Config field:** `"model_id": "singlemki"`

### Mk-II — CNN + stacked GRU

```
Encoded Patch → [Conv→ReLU→Pool] × N → GRU₁ → GRU₂ → Linear → action logits
```

Like Mk-I but with two stacked GRU cells.
The first GRU compresses spatial features over time; the second learns higher-level temporal patterns from that compressed representation.

**Package:** `hmls-singlemkii` · **Config field:** `"model_id": "singlemkii"`

### Mk-III — GRU only (no CNN)

```
Encoded Patch → Flatten → GRU → Linear → action logits
```

Removes all convolutional layers.
The raw patch is flattened directly into the GRU.

**Package:** `hmls-singlemkiii` · **Config field:** `"model_id": "singlemkiii"`

### Random tank (rule-based)

Not a neural network.
Fires deterministically when an enemy is directly ahead; otherwise moves forward or turns with configurable probabilities.
Useful as an initial training opponent because it provides consistent, non-trivial behaviour — it explores the map and will kill your tank if you stand in front of it, but it doesn't adapt, making the learning signal stable.

**Package:** `hmls-randomtank` · **Config field:** `"model_id": "randomtank"`

## Implementing new tanks

You can create your own tank architecture by adding a new package under `packages/`.
The easiest approach is to copy an existing tank package (e.g. `hmls-singlemkiii` for the simplest structure) and modify it.

### Required components

1. **Model config** — a frozen Pydantic model defining the architecture
   parameters. Serialised as `model_config.json`.

2. **Model class** — wraps the PyTorch `nn.Module`. Must implement the forward
   pass and expose a `config` property.

3. **Player class** — inherits from `hmls.nncore.player.NNPlayerBase`. Handles
   action selection in both `"play"` mode (inference only) and `"learn"` mode
   (records log-probabilities for training).

4. **Persistence object** — a module-level constant that the trainer discovers
   via entry points. It tells the system how to load/save your model.

### Entry point registration

Register your package in its `pyproject.toml`:

```toml
[project.entry-points."hmls.models"]
mytank = "hmls.mytank.persistence:PERSISTENCE"
```

The key (`mytank`) becomes the `model_id` you reference in `model_config.json`.

### Template structure

```
packages/hmls-mytank/
├── pyproject.toml
└── src/hmls/mytank/
    ├── __init__.py
    ├── config.py          # Pydantic model config
    ├── model.py           # PyTorch model
    ├── player.py          # NNPlayerBase subclass
    └── persistence.py     # PERSISTENCE constant
```

See the existing tank packages for working examples of each component.