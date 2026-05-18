# Single Tanks Sample Configuration

A simple training configuration pitting two single-tank models against each
other on randomly generated maps.

## Usage

From the repository root:

```shell
uv run hmls-reinforcetrainer sample_configs/single_tanks/config.json
```

## Layout

```
single_tanks/
├── config.json                      # Training configuration
├── weights/
│   ├── model_a/
│   │   └── model_config.json        # MkI architecture config
│   └── model_b/
│       └── model_config.json        # MkI architecture config
└── output/
    └── sample_games/                # Replay files saved periodically
```

The `weights/` directories contain `model_config.json` which tells the
trainer which model architecture to use (singlemki) and its hyperparameters.
Trained weights (`model.pt`) are saved here during training. The `output/`
directory is created automatically.

## Configuration Summary

| Section          | Key choices                              |
|------------------|------------------------------------------|
| Models           | Both models train; basic reward shaping  |
| Maps             | 10–15 cells, 10% impassable, blob_and_line & perlin_noise |
| Games            | 10 games/map × 100 maps, 400 turn limit |
| Hyperparameters  | lr=0.001, γ=0.995, entropy bonus, grad clip |
| Lethargy         | Consecutive-turn limit (10)              |
