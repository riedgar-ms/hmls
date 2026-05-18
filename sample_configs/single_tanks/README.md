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
│   │   └── model_config.json        # MkIII architecture config (unused by default config)
│   ├── model_b/
│   │   └── model_config.json        # MkI architecture config (used as trainee)
│   └── random_opponent/
│       └── model_config.json        # Random tank config (used as frozen opponent)
└── output/
    └── sample_games/                # Replay files saved periodically
```

The `weights/` directories contain `model_config.json` which tells the
trainer which model architecture to use and its hyperparameters.
Trained weights (`model.pt`) are saved here during training. The `output/`
directory is created automatically.

The default `config.json` trains a Mk-I model (`weights/model_b`) against a
frozen random opponent (`weights/random_opponent`).

## Configuration Summary

| Section          | Key choices                              |
|------------------|------------------------------------------|
| Models           | Model A trains (MkI); Model B frozen (random tank) |
| Maps             | 10–15 cells, 10% impassable, blob_and_line & perlin_noise |
| Games            | 10 games/map × 200 maps, 500 turn limit |
| Hyperparameters  | lr=0.001, γ=0.995, entropy bonus, grad clip |
| Lethargy         | Consecutive-turn limit (10)              |
