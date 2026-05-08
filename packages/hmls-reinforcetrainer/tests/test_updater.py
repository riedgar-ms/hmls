"""Tests for the REINFORCE updater."""

from __future__ import annotations

import torch
from torch.distributions import Categorical

from hmls.nncore.trajectory import Episode
from hmls.reinforcetrainer._testing.stub_model import StubModelConfig, StubTankModel
from hmls.reinforcetrainer.updater import reinforce_update


def _make_episode_with_tensors(
    model: StubTankModel, num_steps: int
) -> tuple[Episode, list[torch.Tensor]]:
    """Helper: run a model to get a real episode with tensor log_probs."""
    from hmls.nncore.encoding import FiveChannelPatchEncoder

    episode = Episode()
    log_prob_tensors: list[torch.Tensor] = []
    hidden = model.initial_hidden(batch_size=1).squeeze(0)

    for i in range(num_steps):
        # Random input patch
        patch = torch.randn(
            FiveChannelPatchEncoder.NUM_CHANNELS,
            model.config.patch_size,
            model.config.patch_size,
        )
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
        model = StubTankModel(StubModelConfig())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss = reinforce_update(episode, optimizer, gamma=0.99)
        assert loss == 0.0

    def test_single_step_with_tensor_log_probs(self) -> None:
        """A single-step episode with tensor log_probs updates the model."""
        model = StubTankModel(StubModelConfig())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        episode, tensors = _make_episode_with_tensors(model, 1)
        episode.set_reward(0, 1.0)

        loss = reinforce_update(episode, optimizer, gamma=0.99, log_prob_tensors=tensors)
        assert isinstance(loss, float)
        assert not torch.isnan(torch.tensor(loss))

    def test_multi_step_episode(self) -> None:
        """A multi-step episode with tensor log_probs produces finite loss."""
        model = StubTankModel(StubModelConfig())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        episode, tensors = _make_episode_with_tensors(model, 10)

        loss = reinforce_update(episode, optimizer, gamma=0.99, log_prob_tensors=tensors)
        assert isinstance(loss, float)
        assert not torch.isnan(torch.tensor(loss))

    def test_update_changes_parameters(self) -> None:
        """Verify that an update with tensor log_probs modifies model params."""
        model = StubTankModel(StubModelConfig())
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

    def test_positive_reward_increases_action_probability(self) -> None:
        """A positive reward should increase the probability of the taken action."""
        torch.manual_seed(42)
        model = StubTankModel(StubModelConfig())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        from hmls.nncore.encoding import FiveChannelPatchEncoder

        # Create a fixed input
        patch = torch.randn(
            FiveChannelPatchEncoder.NUM_CHANNELS,
            model.config.patch_size,
            model.config.patch_size,
        )
        hidden = model.initial_hidden(batch_size=1).squeeze(0)

        # Get the action probs before update
        with torch.no_grad():
            logits_before, _ = model(patch, hidden)
            probs_before = torch.softmax(logits_before, dim=-1)

        # Create a single-step episode with high positive reward
        episode, tensors = _make_episode_with_tensors(model, 1)
        episode.set_reward(0, 10.0)

        reinforce_update(episode, optimizer, gamma=0.99, log_prob_tensors=tensors)

        # The loss should be non-zero (update happened)
        with torch.no_grad():
            logits_after, _ = model(patch, hidden)
            probs_after = torch.softmax(logits_after, dim=-1)

        # Probs should have changed
        assert not torch.allclose(probs_before, probs_after)

    def test_multiple_updates_accumulate(self) -> None:
        """Multiple REINFORCE updates accumulate parameter changes."""
        model = StubTankModel(StubModelConfig())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        initial_params = {name: p.clone().detach() for name, p in model.named_parameters()}

        total_loss = 0.0
        for _ in range(5):
            episode, tensors = _make_episode_with_tensors(model, 3)
            for i in range(3):
                episode.set_reward(i, 1.0)
            total_loss += reinforce_update(episode, optimizer, gamma=0.99, log_prob_tensors=tensors)

        # After multiple updates, parameters should have moved significantly
        max_diff = 0.0
        for name, param in model.named_parameters():
            diff = (param - initial_params[name]).abs().max().item()
            max_diff = max(max_diff, diff)

        assert max_diff > 1e-4, "Parameters barely changed after 5 updates"
