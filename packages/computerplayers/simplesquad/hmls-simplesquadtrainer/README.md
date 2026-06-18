# hmls-simplesquadtrainer

REINFORCE policy gradient trainer for the simple squad (planner + executor) architecture.

## Getting Started

### 1. Create squad model directories

```
my_squad/
├── planner/
│   └── model_config.json
└── executor/
    └── model_config.json
```

**planner/model_config.json:**
```json
{
    "model_id": "hmls.simplesquadplanner",
    "patch_size": 9,
    "num_orders": 8,
    "max_tanks": 5,
    "cnn_channels": [32, 64],
    "tank_feature_dim": 64,
    "mlp_hidden_dim": 64
}
```

**executor/model_config.json:**
```json
{
    "model_id": "hmls.simplesquadexecutor",
    "patch_size": 9,
    "num_orders": 8,
    "cnn_channels": [32, 64],
    "gru_hidden_size": 128,
    "order_embedding_dim": 16
}
```

### 2. Create a training config

```json
{
    "team_a": {
        "type": "squad",
        "dir": "models/squad_a",
        "train": true
    },
    "team_b": {
        "type": "squad",
        "dir": "models/squad_b",
        "train": true
    },
    "game": {
        "tanks_per_team": 3,
        "max_turns": 200,
        "patch_size": 9,
        "total_maps": 100,
        "games_per_map": 10
    },
    "hyperparameters": {
        "executor_learning_rate": 0.001,
        "planner_learning_rate": 0.0003,
        "gamma": 0.99,
        "entropy_coeff": 0.01,
        "planner_entropy_coeff": 0.01
    }
}
```

### 3. Run training

```bash
uv run hmls-simplesquadtrainer config.json
```

## Configuration Reference

### Team Configuration

The trainer supports asymmetric matches via a discriminated union on the `type` field:

#### Squad team (`"type": "squad"`)

```json
{
    "type": "squad",
    "dir": "path/to/squad_dir",
    "train": true,
    "reward": { ... }
}
```

| Field | Description |
|-------|-------------|
| `dir` | Path to squad directory (contains `planner/` and `executor/` subdirs) |
| `train` | Whether to update this squad's weights during training |
| `reward` | Reward configuration (same as single-tank `RewardConfig`) |

#### Independent team (`"type": "independent"`)

Uses N copies of a single-tank model acting independently (no coordination):

```json
{
    "type": "independent",
    "dir": "path/to/singletank_model_dir",
    "train": false,
    "reward": { ... }
}
```

This is useful for bootstrapping: train your squad against known-good single-tank opponents before attempting squad-vs-squad.

### Game Configuration

```json
{
    "games_per_map": 10,
    "total_maps": 100,
    "max_turns": 200,
    "patch_size": 9,
    "tanks_per_team": 3
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `games_per_map` | 10 | Games per generated map |
| `total_maps` | 100 | Total maps to generate |
| `max_turns` | 200 | Max turns per game before draw |
| `patch_size` | 9 | Visibility patch side length (odd) |
| `tanks_per_team` | 3 | Number of tanks per team |

### Hyperparameters

```json
{
    "executor_learning_rate": 0.001,
    "planner_learning_rate": 0.0003,
    "gamma": 0.99,
    "seed": null,
    "baseline_alpha": 0.99,
    "entropy_coeff": 0.01,
    "planner_entropy_coeff": 0.01,
    "loss_reduction": "sum",
    "max_grad_norm": null
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `executor_learning_rate` | 0.001 | Adam LR for executor |
| `planner_learning_rate` | 0.0003 | Adam LR for planner (typically lower) |
| `gamma` | 0.99 | Discount factor |
| `seed` | null | Random seed (null = non-deterministic) |
| `baseline_alpha` | 0.99 | EMA decay for return baselines |
| `entropy_coeff` | 0.01 | Entropy bonus for executor |
| `planner_entropy_coeff` | 0.01 | Entropy bonus for planner |
| `loss_reduction` | "sum" | `"sum"` or `"mean"` |
| `max_grad_norm` | null | Gradient clipping (null = disabled) |

### Map Configuration

Same as `hmls-reinforcetrainer`:

```json
{
    "min_size": 15,
    "max_size": 25,
    "impassable_fraction": 0.3,
    "strategies": [{"type": "blob_and_line"}]
}
```

### Output Configuration

```json
{
    "sample_game_dir": "sample_games",
    "sample_game_interval": 50,
    "save_weights_interval": 100
}
```

## How Training Works

### Executor Training

The executor is a single model shared across all tanks on a team. Each tank generates an independent trajectory through the model (same weights, independent hidden states). After each game:

1. Per-tank returns are computed with discount factor γ
2. Advantages are computed using a shared cross-episode EMA baseline
3. All per-tank losses are accumulated into a single loss
4. One `optimizer.step()` updates the shared executor weights

This is equivalent to treating each tank as a separate rollout from the same policy.

### Planner Training

The planner has a team-level trajectory (one step per planning round). Its reward at each step is the mean of per-step executor rewards across all alive tanks. This encourages the planner to issue orders that maximise team-wide performance.

### Reward Strategy

- **Executor**: Same shaped per-step rewards as single-tank training (exploration, firing, game state, etc.)
- **Planner**: Aggregated executor rewards (mean across tanks and time steps, distributed across planner steps)

## Tips

- Start with a lower planner LR (3e-4) than executor LR (1e-3) — the planner's reward signal is noisier
- Bootstrap against independent single-tank opponents before squad-vs-squad
- Use 2-3 tanks initially; 5 tanks makes training slower and reward attribution harder
- The entropy bonus is important for the planner to explore different order combinations
