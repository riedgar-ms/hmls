"""Tests for the REINFORCE updater."""

from __future__ import annotations

import torch
from torch.distributions import Categorical

from hmls.nncore.trajectory import Episode
from hmls.reinforcetrainer.updater import reinforce_update
from hmls.singlemki.model import ModelConfig, TankPolicyNetwork


def _make_episode_with_tensors(
    model: TankPolicyNetwork, num_steps: int
) -> tuple[Episode, list[torch.Tensor]]:
    """Helper: run a model to get a real episode with tensor log_probs."""
    from hmls.singlemki.encoding import NUM_CHANNELS

    episode = Episode()
    log_prob_tensors: list[torch.Tensor] = []
    hidden = model.initial_hidden(batch_size=1).squeeze(0)

    for i in range(num_steps):
        # Random input patch
        patch = torch.randn(NUM_CHANNELS, 9, 9)
        logits, hidden = model(patch, hidden)
        hidden = hidden.detach()
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()  # type: ignore[no-untyped-call]
        log_prob = dist.log_prob(action)  # type: ignore[no-untyped-call]
        log_prob_tensors.append(log_prob)
        episode.add_step(action_index=int(action.item()), log_prob=float(log_prob.item()))
        episode.set_reward(i, 0.1 * (i % 3) - 0.05)

    return episode, log_prob_tensors


class TestReinforceUpdate:
    """Tests for reinforce_update function."""

    def test_empty_episode_returns_zero(self) -> None:
        """An empty episode should produce zero loss."""
        episode = Episode()
        model = TankPolicyNetwork(ModelConfig())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss = reinforce_update(episode, optimizer, gamma=0.99)
        assert loss == 0.0

    def test_single_step_with_tensor_log_probs(self) -> None:
        """A single-step episode with tensor log_probs updates the model."""
        model = TankPolicyNetwork(ModelConfig())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        episode, tensors = _make_episode_with_tensors(model, 1)
        episode.set_reward(0, 1.0)

        loss = reinforce_update(episode, optimizer, gamma=0.99, log_prob_tensors=tensors)
        assert isinstance(loss, float)
        assert not torch.isnan(torch.tensor(loss))

    def test_multi_step_episode(self) -> None:
        """A multi-step episode with tensor log_probs produces finite loss."""
        model = TankPolicyNetwork(ModelConfig())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        episode, tensors = _make_episode_with_tensors(model, 10)

        loss = reinforce_update(episode, optimizer, gamma=0.99, log_prob_tensors=tensors)
        assert isinstance(loss, float)
        assert not torch.isnan(torch.tensor(loss))

    def test_update_changes_parameters(self) -> None:
        """Verify that an update with tensor log_probs modifies model params."""
        model = TankPolicyNetwork(ModelConfig())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        # Record initial parameter values
        initial_params = {name: param.clone().detach() for name, param in model.named_parameters()}

        episode, tensors = _make_episode_with_tensors(model, 5)
        # Give a strong reward signal
        for i in range(5):
            episode.set_reward(i, 1.0 if i == 4 else -1.0)

        reinforce_update(episode, optimizer, gamma=0.99, log_prob_tensors=tensors)

        # At least some parameters should have changed
        changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(initial_params[name], param):
                changed = True
                break
        assert changed, "No model parameters changed after update"
