# New Tank Architecture

Create a new neural network tank architecture for the hmls game.

## Steps

1. **Choose a name** — e.g. `hmls-singlemkiv`. The package name pattern is `hmls-<tankname>`.

2. **Copy an existing tank package** as a starting point:
   - `packages/hmls-singlemkiii` is simplest (no CNN).
   - `packages/hmls-singlemki` is the standard CNN+GRU template.

3. **Rename the namespace** — the Python package is `hmls.<tankname>` under `src/hmls/<tankname>/`.

4. **Define the model config** (`model.py`):
   - Create a frozen Pydantic `BaseModel` subclassing `TankModelConfig` with architecture hyperparameters.
   - Include a `model_id` class variable matching your entry point key.

5. **Implement the model** (`model.py`):
   - In the same file as the config, subclass `torch.nn.Module` via `TankModelBase`.
   - Input: encoded patch tensor of shape `[5, patch_size, patch_size]`.
   - Output: 5 action logits (MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, FIRE, PASS).
   - If using temporal memory (GRU/LSTM), expose hidden state management.

6. **Implement the player** (`player.py`, optional):
   - Only needed if the generic `NNPlayer` from `hmls-nncore` is insufficient
     (e.g. rule-based action logic as in `hmls-randomtank`).
   - If needed, subclass `hmls.nncore.player.NNPlayerBase`.
   - Handle both `"play"` mode (argmax/sample from logits) and `"learn"` mode (record log-probs).

7. **Create persistence** (`persistence.py`):
   - Export a module-level `PERSISTENCE` constant that the trainer uses to load/save the model.

8. **Register the entry point** in `pyproject.toml`:
   ```toml
   [project.entry-points."hmls.models"]
   <tankname> = "hmls.<tankname>.persistence:PERSISTENCE"
   ```

9. **Register in the workspace** — add to root `pyproject.toml`:
   - `[dependency-groups] dev` list
   - `[tool.uv.sources]` with `{ workspace = true }`
   - `[tool.mypy] mypy_path` — append the src path

10. **Add tests** in `packages/hmls-<tankname>/tests/`.

11. **Run checks**: `uv run ruff format --check .`, `uv run ruff check .`, `uv run mypy .`, `uv run pytest`.
