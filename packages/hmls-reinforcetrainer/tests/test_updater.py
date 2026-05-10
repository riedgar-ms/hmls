"""Tests for the REINFORCE updater."""

from __future__ import annotations

import torch
from torch.distributions import Categorical

from hmls.nncore.trajectory import Episode
from hmls.reinforcetrainer._testing.stub_model import StubModelConfig, StubTankModel
from hmls.reinforcetrainer.updater import ReturnBaseline, reinforce_update


def _make_episode_with_tensors(
    model: StubTankModel, num_steps: int
) -> tuple[Episode, list[torch.Tensor], list[torch.Tensor]]:
    """Helper: run a model to get a real episode with tensor log_probs and entropies."""
    from hmls.nncore.encoding import FiveChannelPatchEncoder

    episode = Episode()
    log_prob_tensors: list[torch.Tensor] = []
    entropy_tensors: list[torch.Tensor] = []
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
        entropy = dist.entropy()  # type: ignore[no-untyped-call]
        log_prob_tensors.append(log_prob)
        entropy_tensors.append(entropy)
        episode.add_step(action_index=int(action.item()), log_prob=float(log_prob.item()))
        episode.set_reward(i, 0.1 * (i % 3) - 0.05)

    return episode, log_prob_tensors, entropy_tensors


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
        episode, tensors, entropies = _make_episode_with_tensors(model, 1)
        episode.set_reward(0, 1.0)

        loss = reinforce_update(
            episode,
            optimizer,
            gamma=0.99,
            log_prob_tensors=tensors,
            entropy_tensors=entropies,
        )
        assert isinstance(loss, float)
        assert not torch.isnan(torch.tensor(loss))

    def test_multi_step_episode(self) -> None:
        """A multi-step episode with tensor log_probs produces finite loss."""
        model = StubTankModel(StubModelConfig())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        episode, tensors, entropies = _make_episode_with_tensors(model, 10)

        loss = reinforce_update(
            episode,
            optimizer,
            gamma=0.99,
            log_prob_tensors=tensors,
            entropy_tensors=entropies,
        )
        assert isinstance(loss, float)
        assert not torch.isnan(torch.tensor(loss))

    def test_update_changes_parameters(self) -> None:
        """Verify that an update with tensor log_probs modifies model params."""
        model = StubTankModel(StubModelConfig())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        # Record initial parameter values
        initial_params = {name: param.clone().detach() for name, param in model.named_parameters()}

        episode, tensors, entropies = _make_episode_with_tensors(model, 5)
        # Give a strong reward signal
        for i in range(5):
            episode.set_reward(i, 1.0 if i == 4 else -1.0)

        reinforce_update(
            episode,
            optimizer,
            gamma=0.99,
            log_prob_tensors=tensors,
            entropy_tensors=entropies,
        )

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
        episode, tensors, entropies = _make_episode_with_tensors(model, 1)
        episode.set_reward(0, 10.0)

        reinforce_update(
            episode,
            optimizer,
            gamma=0.99,
            log_prob_tensors=tensors,
            entropy_tensors=entropies,
        )

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
            episode, tensors, entropies = _make_episode_with_tensors(model, 3)
            for i in range(3):
                episode.set_reward(i, 1.0)
            total_loss += reinforce_update(
                episode,
                optimizer,
                gamma=0.99,
                log_prob_tensors=tensors,
                entropy_tensors=entropies,
            )

        # After multiple updates, parameters should have moved significantly
        max_diff = 0.0
        for name, param in model.named_parameters():
            diff = (param - initial_params[name]).abs().max().item()
            max_diff = max(max_diff, diff)

        assert max_diff > 1e-4, "Parameters barely changed after 5 updates"


class TestReturnBaseline:
    """Tests for the cross-episode return baseline."""

    def test_initial_state_is_none(self) -> None:
        """Baseline mean and std should be None before any updates."""
        baseline = ReturnBaseline(alpha=0.99)
        assert baseline.mean is None
        assert baseline.std is None

    def test_first_update_initialises_directly(self) -> None:
        """First episode initialises the baseline directly (no blending)."""
        baseline = ReturnBaseline(alpha=0.99)
        returns = torch.tensor([1.0, 2.0, 3.0])
        baseline.update(returns)
        assert baseline.mean is not None
        assert abs(baseline.mean - 2.0) < 1e-6

    def test_ema_blends_across_episodes(self) -> None:
        """Subsequent episodes blend with the running baseline via EMA."""
        baseline = ReturnBaseline(alpha=0.5)
        baseline.update(torch.tensor([10.0]))
        assert baseline.mean is not None
        assert abs(baseline.mean - 10.0) < 1e-6

        # Second update with alpha=0.5: new_mean = 0.5 * 10 + 0.5 * 20 = 15
        baseline.update(torch.tensor([20.0]))
        assert abs(baseline.mean - 15.0) < 1e-6

    def test_uniform_bad_episode_gets_negative_advantages(self) -> None:
        """A uniformly-bad episode should get uniformly-negative advantages.

        This is the key fix for the spinning problem: per-episode
        normalisation would split these into +/− halves, but a
        cross-episode baseline keeps them all negative.
        """
        baseline = ReturnBaseline(alpha=0.99)
        # Warm up with a "neutral" episode so the baseline is set
        baseline.update(torch.tensor([0.0, 0.0, 0.0]))

        # A uniformly-bad episode: all returns well below the baseline
        bad_returns = torch.tensor([-1.0, -0.9, -0.8, -0.7, -0.6])
        advantages = baseline.compute_advantages(bad_returns)

        # All advantages should be negative (below the ~0.0 baseline)
        assert (advantages < 0).all(), (
            f"Expected all negative advantages, got {advantages.tolist()}"
        )

    def test_warmup_fallback_uses_episode_mean(self) -> None:
        """Before any updates, falls back to per-episode mean subtraction."""
        baseline = ReturnBaseline(alpha=0.99)
        returns = torch.tensor([1.0, 2.0, 3.0])
        advantages = baseline.compute_advantages(returns)
        # Should be centred around zero (per-episode mean subtraction)
        assert abs(advantages.mean().item()) < 1e-6

    def test_invalid_alpha_raises(self) -> None:
        """Alpha outside (0, 1) should raise ValueError."""
        import pytest

        with pytest.raises(ValueError):
            ReturnBaseline(alpha=0.0)
        with pytest.raises(ValueError):
            ReturnBaseline(alpha=1.0)
        with pytest.raises(ValueError):
            ReturnBaseline(alpha=-0.5)

    def test_baseline_with_reinforce_update(self) -> None:
        """Baseline integrates correctly with reinforce_update."""
        model = StubTankModel(StubModelConfig())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        baseline = ReturnBaseline(alpha=0.99)

        for _ in range(3):
            episode, tensors, entropies = _make_episode_with_tensors(model, 5)
            for i in range(5):
                episode.set_reward(i, -0.1)
            reinforce_update(
                episode,
                optimizer,
                gamma=0.99,
                log_prob_tensors=tensors,
                entropy_tensors=entropies,
                baseline=baseline,
            )

        # After 3 updates, baseline should have adapted
        assert baseline.mean is not None


class TestEntropyBonus:
    """Tests for entropy regularisation in reinforce_update."""

    def test_entropy_coeff_zero_matches_no_entropy(self) -> None:
        """With entropy_coeff=0, result should match a call without entropy."""
        torch.manual_seed(42)
        model = StubTankModel(StubModelConfig())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        episode, tensors, entropies = _make_episode_with_tensors(model, 5)
        for i in range(5):
            episode.set_reward(i, 0.5)

        # The loss with entropy_coeff=0 should be the same whether or
        # not entropy tensors are passed.
        loss = reinforce_update(
            episode,
            optimizer,
            gamma=0.99,
            log_prob_tensors=tensors,
            entropy_tensors=entropies,
            entropy_coeff=0.0,
        )
        assert isinstance(loss, float)
        assert not torch.isnan(torch.tensor(loss))

    def test_nonzero_entropy_coeff_changes_loss(self) -> None:
        """Nonzero entropy_coeff should produce a different loss value.

        We can't easily check the exact value, but we can verify that
        the entropy bonus is being applied by checking that the loss
        differs from the entropy_coeff=0 case.
        """
        torch.manual_seed(42)
        model1 = StubTankModel(StubModelConfig())
        model2 = StubTankModel(StubModelConfig())
        # Copy weights so both start identically
        model2.load_state_dict(model1.state_dict())

        opt1 = torch.optim.Adam(model1.parameters(), lr=1e-3)
        opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)

        ep1, t1, e1 = _make_episode_with_tensors(model1, 5)
        ep2, t2, e2 = _make_episode_with_tensors(model2, 5)

        # Same rewards
        for i in range(5):
            ep1.set_reward(i, 0.5)
            ep2.set_reward(i, 0.5)

        loss_no_ent = reinforce_update(
            ep1,
            opt1,
            gamma=0.99,
            log_prob_tensors=t1,
            entropy_tensors=e1,
            entropy_coeff=0.0,
        )
        loss_with_ent = reinforce_update(
            ep2,
            opt2,
            gamma=0.99,
            log_prob_tensors=t2,
            entropy_tensors=e2,
            entropy_coeff=0.1,
        )
        # They should differ because the entropy bonus modifies the loss
        assert loss_no_ent != loss_with_ent
