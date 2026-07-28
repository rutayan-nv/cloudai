# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the reward-pipeline composition layer.

Covers:
* Pure constructional / structural properties of the rewards subpackage.
* Pipeline composition semantics (chained update + apply).
* :class:`PerContextZScore` running-stats math, including a regression
  fixture sized to the real ilyk 2-drop run (169 / 151 trials per regime,
  μ ≈ 0.7619 / 0.7183, σ ≈ 0.0447 / 0.0251).
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from cloudai.configurator.env_params import EnvParamSpec
from cloudai.configurator.rewards import (
    ContextAutoDetector,
    GlobalMeanStdFilter,
    IdentityTransform,
    PerContextZScore,
    PipelineRewardWrapper,
    RewardPipeline,
    RewardTransform,
    build_default_pipeline,
)


def _make_env(env_params: dict[str, EnvParamSpec] | None) -> SimpleNamespace:
    """Build a duck-typed env exposing only ``test_run.test.env_params``.

    The detector reads exactly that path; constructing a real ``TestRun``
    would couple this test to far too much unrelated machinery.
    """
    test = SimpleNamespace(env_params=env_params or {})
    test_run = SimpleNamespace(test=test)
    return SimpleNamespace(test_run=test_run)


def _categorical(values: list, weights: list[float] | None = None) -> EnvParamSpec:
    payload: dict = {"sampling": {"type": "categorical", "values": values}}
    if weights is not None:
        payload["sampling"]["weights"] = weights
    return EnvParamSpec.model_validate(payload)


def _log_uniform(low: float, high: float) -> EnvParamSpec:
    return EnvParamSpec.model_validate(
        {"sampling": {"type": "loguniform", "low": low, "high": high}}
    )


def _uniform(low: float, high: float) -> EnvParamSpec:
    return EnvParamSpec.model_validate(
        {"sampling": {"type": "uniform", "low": low, "high": high}}
    )


def _fixed(value) -> EnvParamSpec:  # noqa: ANN001 — `value` is intentionally untyped
    """A bare scalar parses as a FixedSampling EnvParamSpec."""
    return EnvParamSpec.model_validate(value)


class TestBuildDefaultPipelineNoContext:
    """No context keys → a single GlobalMeanStdFilter (regime-blind fallback).

    Design X: the default pipeline commits to *exactly one* estimator chosen
    from the context structure. Without a low-cardinality categorical context
    there is nothing to bin on, so a global mean/std filter is the right (and
    only) estimator. No chaining, no trailing IdentityTransform.
    """

    def test_returns_reward_pipeline(self) -> None:
        pipe = build_default_pipeline(())
        assert isinstance(pipe, RewardPipeline)

    def test_has_single_transform(self) -> None:
        pipe = build_default_pipeline(())
        assert len(pipe.transforms) == 1

    def test_only_transform_is_global_mean_std_filter(self) -> None:
        pipe = build_default_pipeline(())
        assert isinstance(pipe.transforms[0], GlobalMeanStdFilter)


class TestBuildDefaultPipelineWithContext:
    """Context keys present → a single PerContextZScore applied to RAW reward.

    Design X: when the detector found a low-cardinality categorical context,
    per-context z-score is the estimator. It runs on the raw reward directly —
    Global is NOT chained in front of it (chaining under EMA amplified the
    regime gap and poisoned the per-context stats with warm-up spikes; see
    ``viz_zscore_chain.py``).
    """

    def test_single_per_context_zscore_transform(self) -> None:
        pipe = build_default_pipeline(("drop_rate",))
        assert len(pipe.transforms) == 1
        assert isinstance(pipe.transforms[0], PerContextZScore)

    def test_no_global_filter_when_context_present(self) -> None:
        pipe = build_default_pipeline(("drop_rate",))
        assert not any(isinstance(t, GlobalMeanStdFilter) for t in pipe.transforms)

    def test_context_keys_propagate_to_transform(self) -> None:
        pipe = build_default_pipeline(("drop_rate", "msg_size"))
        zscore = pipe.transforms[0]
        assert isinstance(zscore, PerContextZScore)
        # Internal field; testing here pins down that the keys reach the transform.
        assert zscore._keys == ("drop_rate", "msg_size")  # noqa: SLF001


class TestPipelineConstruction:
    """RewardPipeline composes any sequence satisfying the RewardTransform protocol."""

    def test_empty_pipeline(self) -> None:
        pipe = RewardPipeline()
        assert pipe.transforms == ()

    def test_none_treated_as_empty(self) -> None:
        pipe = RewardPipeline(transforms=None)
        assert pipe.transforms == ()

    def test_transforms_immutable_view(self) -> None:
        """``pipeline.transforms`` returns a tuple; mutating it must not affect state."""
        pipe = RewardPipeline([IdentityTransform()])
        view = pipe.transforms
        assert isinstance(view, tuple)
        # touching the view obviously cannot rebind the underlying list:
        assert pipe.transforms == view


class TestRewardTransformProtocol:
    """Concrete transforms satisfy RewardTransform without inheriting from it."""

    def test_identity_satisfies_protocol(self) -> None:
        assert isinstance(IdentityTransform(), RewardTransform)

    def test_global_filter_satisfies_protocol(self) -> None:
        assert isinstance(GlobalMeanStdFilter(), RewardTransform)

    def test_per_context_satisfies_protocol(self) -> None:
        assert isinstance(PerContextZScore(("drop_rate",)), RewardTransform)

    def test_user_defined_class_satisfies_protocol_structurally(self) -> None:
        """Composition-by-structure: anything with update/apply qualifies, no
        inheritance from a CloudAI-defined base class is required."""

        class _UserTransform:
            def update(self, reward, ctx):  # noqa: ARG002
                return None

            def apply(self, reward, ctx):  # noqa: ARG002
                return reward

        assert isinstance(_UserTransform(), RewardTransform)


class _RecordingTransform:
    """Minimal RewardTransform that records the (call_kind, reward, ctx) it sees.

    Lets pipeline.step tests pin down ordering: which method is called first,
    what value flows in, and that ``ctx`` is forwarded unchanged.
    """

    def __init__(self, name: str, log: list, increment: float = 0.0) -> None:
        self._name = name
        self._log = log
        self._increment = float(increment)

    def update(self, reward, ctx) -> None:
        self._log.append((self._name, "update", float(reward), dict(ctx)))

    def apply(self, reward, ctx) -> float:
        self._log.append((self._name, "apply", float(reward), dict(ctx)))
        return float(reward) + self._increment


class TestPipelineStepEmpty:
    """An empty pipeline is the identity over (reward, ctx)."""

    def test_returns_reward_unchanged(self) -> None:
        pipe = RewardPipeline()
        out = pipe.step(0.5, {"drop_rate": 0.001})
        assert out == 0.5

    def test_does_not_mutate_ctx(self) -> None:
        ctx = {"drop_rate": 0.001}
        pipe = RewardPipeline()
        pipe.step(0.5, ctx)
        assert ctx == {"drop_rate": 0.001}


class TestPipelineStepIdentitySingle:
    """A single IdentityTransform must not change the reward."""

    def test_returns_reward_unchanged(self) -> None:
        pipe = RewardPipeline([IdentityTransform()])
        assert pipe.step(0.7619, {"drop_rate": 0.0}) == 0.7619


class TestPipelineStepIdentityChain:
    """Chained Identities must not change the reward (composition is identity)."""

    def test_two_identities(self) -> None:
        pipe = RewardPipeline([IdentityTransform(), IdentityTransform()])
        assert pipe.step(0.42, {"drop_rate": 0.001}) == 0.42

    def test_three_identities(self) -> None:
        pipe = RewardPipeline([IdentityTransform()] * 3)
        assert pipe.step(-1.5, {"k": "v"}) == -1.5


class TestPipelineStepOrdering:
    """Verify the chained semantics: per transform, update first, then apply;
    each transform receives the PREVIOUS transform's output (not the raw reward).
    """

    def test_update_precedes_apply_within_a_transform(self) -> None:
        log: list = []
        t = _RecordingTransform("t1", log, increment=0.0)
        pipe = RewardPipeline([t])
        pipe.step(0.5, {"drop_rate": 0.0})
        assert [(name, kind) for name, kind, *_ in log] == [
            ("t1", "update"),
            ("t1", "apply"),
        ]

    def test_two_transforms_interleave_update_apply(self) -> None:
        """Expected order is t1.update, t1.apply, t2.update, t2.apply."""
        log: list = []
        t1 = _RecordingTransform("t1", log, increment=0.0)
        t2 = _RecordingTransform("t2", log, increment=0.0)
        pipe = RewardPipeline([t1, t2])
        pipe.step(0.5, {"drop_rate": 0.0})
        assert [(name, kind) for name, kind, *_ in log] == [
            ("t1", "update"),
            ("t1", "apply"),
            ("t2", "update"),
            ("t2", "apply"),
        ]

    def test_value_threads_through_chain(self) -> None:
        """Each transform's apply increments the reward; downstream sees the
        incremented value in its update + apply, proving the chain semantics.
        """
        log: list = []
        t1 = _RecordingTransform("t1", log, increment=10.0)
        t2 = _RecordingTransform("t2", log, increment=100.0)
        pipe = RewardPipeline([t1, t2])
        out = pipe.step(1.0, {})
        assert out == 1.0 + 10.0 + 100.0
        # t1 receives the raw 1.0; t2 receives 1.0 + 10.0 = 11.0
        seen = {(name, kind): val for name, kind, val, _ in log}
        assert seen[("t1", "update")] == 1.0
        assert seen[("t1", "apply")] == 1.0
        assert seen[("t2", "update")] == 11.0
        assert seen[("t2", "apply")] == 11.0

    def test_ctx_is_forwarded_unchanged(self) -> None:
        """Each transform should see the same ctx dict the caller passed."""
        log: list = []
        t1 = _RecordingTransform("t1", log)
        t2 = _RecordingTransform("t2", log)
        pipe = RewardPipeline([t1, t2])
        pipe.step(0.5, {"drop_rate": 0.001, "msg_size": 64})
        for _, _, _, ctx in log:
            assert ctx == {"drop_rate": 0.001, "msg_size": 64}


class TestPipelineBatchStepEmpty:
    """Empty batch returns empty list and is a no-op."""

    def test_returns_empty_list(self) -> None:
        pipe = RewardPipeline([IdentityTransform()])
        assert pipe.batch_step([], []) == []

    def test_does_not_modify_state(self) -> None:
        z = PerContextZScore(("drop_rate",))
        pipe = RewardPipeline([z])
        pipe.batch_step([], [])
        assert z._stats == {}


class TestPipelineBatchStepIdentity:
    """Identity pipeline returns the same list of rewards unchanged."""

    def test_returns_inputs_unchanged(self) -> None:
        pipe = RewardPipeline([IdentityTransform()])
        rewards = [0.5, 0.7, 0.9]
        ctxs = [{"drop_rate": 0.0}] * 3
        assert pipe.batch_step(rewards, ctxs) == rewards


class TestPipelineBatchStepEquivalence:
    """``batch_step`` is observably equivalent to a loop over ``step``.

    This pins down that the API surface is a drop-in for the sequential
    path; future parallel implementations must preserve this equivalence
    on whatever ordering they choose to define.
    """

    @staticmethod
    def _make_dataset(seed: int = 0, n: int = 50) -> list[tuple[float, dict]]:
        rng = np.random.default_rng(seed)
        out = []
        for r in rng.normal(0.7619, 0.0447, n):
            out.append((float(r), {"drop_rate": 0.0}))
        for r in rng.normal(0.7183, 0.0251, n):
            out.append((float(r), {"drop_rate": 0.001}))
        rng.shuffle(out)
        return out

    def test_matches_loop_of_step(self) -> None:
        rows = self._make_dataset()
        rewards = [r for r, _ in rows]
        ctxs = [c for _, c in rows]

        pipe_loop = RewardPipeline(
            [PerContextZScore(("drop_rate",), momentum=0.99, min_samples=5)]
        )
        loop_outputs = [pipe_loop.step(r, c) for r, c in zip(rewards, ctxs)]

        pipe_batch = RewardPipeline(
            [PerContextZScore(("drop_rate",), momentum=0.99, min_samples=5)]
        )
        batch_outputs = pipe_batch.batch_step(rewards, ctxs)

        assert batch_outputs == loop_outputs

    def test_internal_stats_match_loop(self) -> None:
        rows = self._make_dataset()
        rewards = [r for r, _ in rows]
        ctxs = [c for _, c in rows]

        z_loop = PerContextZScore(("drop_rate",), momentum=0.99, min_samples=5)
        pipe_loop = RewardPipeline([z_loop])
        for r, c in zip(rewards, ctxs):
            pipe_loop.step(r, c)

        z_batch = PerContextZScore(("drop_rate",), momentum=0.99, min_samples=5)
        pipe_batch = RewardPipeline([z_batch])
        pipe_batch.batch_step(rewards, ctxs)

        # Same sequence of (update, apply) calls → identical state
        assert z_batch._stats == z_loop._stats


class TestPipelineBatchStepValidation:
    """Length mismatch is a programming error and must be surfaced loudly."""

    def test_mismatched_lengths_raise(self) -> None:
        pipe = RewardPipeline([IdentityTransform()])
        with pytest.raises(ValueError, match="length"):
            pipe.batch_step([0.5, 0.6], [{"drop_rate": 0.0}])


class TestGlobalMeanStdFilterFirstUpdate:
    """First sample initialises (mean=r, var=0, count=1) — same unbiased
    init pattern as PerContextZScore, but global instead of per-bin."""

    def test_creates_state_on_first_update(self) -> None:
        f = GlobalMeanStdFilter()
        f.update(0.7619, {"drop_rate": 0.0})
        assert f._mean == pytest.approx(0.7619)
        assert f._var == 0.0
        assert f._count == 1

    def test_ctx_is_ignored(self) -> None:
        """Global filter should not branch on context — same state regardless."""
        f1 = GlobalMeanStdFilter()
        f2 = GlobalMeanStdFilter()
        f1.update(0.5, {"drop_rate": 0.0})
        f2.update(0.5, {"drop_rate": 0.001})
        assert f1._mean == f2._mean
        assert f1._var == f2._var
        assert f1._count == f2._count


class TestGlobalMeanStdFilterEMAUpdate:
    """Subsequent samples apply EMA over (mean, var, count)."""

    def test_second_update_mean(self) -> None:
        f = GlobalMeanStdFilter(momentum=0.9)
        f.update(0.5, {})
        f.update(0.6, {})
        # 0.9 * 0.5 + 0.1 * 0.6 = 0.51
        assert f._mean == pytest.approx(0.51)

    def test_second_update_var_uses_old_mean(self) -> None:
        f = GlobalMeanStdFilter(momentum=0.9)
        f.update(0.5, {})  # mean_old = 0.5, var = 0
        f.update(0.6, {})
        # 0.9 * 0 + 0.1 * (0.6 - 0.5)^2 = 0.001
        assert f._var == pytest.approx(0.001)

    def test_count_increments(self) -> None:
        f = GlobalMeanStdFilter(momentum=0.99)
        for r in [0.5, 0.6, 0.7, 0.8]:
            f.update(r, {})
        assert f._count == 4


class TestGlobalMeanStdFilterApply:
    """``apply`` is pass-through during warm-up, normalized after."""

    def test_passthrough_when_no_samples(self) -> None:
        f = GlobalMeanStdFilter()
        # count=0; nothing to normalize against
        assert f.apply(0.7, {}) == 0.7

    def test_passthrough_after_one_sample(self) -> None:
        """count=1 → var=0; dividing by √ε would explode. Must pass through."""
        f = GlobalMeanStdFilter()
        f.update(0.7619, {})
        assert f.apply(0.7619, {}) == 0.7619

    def test_z_score_at_mean_is_zero(self) -> None:
        f = GlobalMeanStdFilter(momentum=0.5)
        f.update(1.0, {})
        f.update(3.0, {})
        # mean = 0.5*1.0 + 0.5*3.0 = 2.0
        out = f.apply(2.0, {})
        assert abs(out) < 1e-2

    def test_z_score_one_sigma_offset(self) -> None:
        """Feed N(μ, σ²); verify (μ + σ) maps to ≈+1."""
        f = GlobalMeanStdFilter(momentum=0.99)
        rng = np.random.default_rng(0)
        for r in rng.normal(0.5, 0.1, size=2000):
            f.update(float(r), {})
        sigma = math.sqrt(f._var)
        out = f.apply(f._mean + sigma, {})
        assert out == pytest.approx(1.0, abs=0.05)


class TestTransformClip:
    """``clip`` bounds the transformed reward — a numerical safety net.

    A near-zero early std can otherwise produce a |z| ≈ 10 spike (see
    ``viz_zscore_chain.py``). ``clip`` caps the output magnitude so a
    pathological warm-up sample cannot dominate the policy gradient. It is
    inert in steady state (real z rarely exceeds ~4), so it does not bias the
    normal-regime gradient.
    """

    def test_global_clip_caps_spike(self) -> None:
        # Two near-identical samples → tiny std → huge raw z; clip caps it.
        f = GlobalMeanStdFilter(momentum=0.5, clip=3.0)
        f.update(0.5000, {})
        f.update(0.5001, {})
        out = f.apply(1.0, {})  # far from mean, tiny std → would be enormous
        assert out == pytest.approx(3.0)

    def test_global_clip_symmetric_negative(self) -> None:
        f = GlobalMeanStdFilter(momentum=0.5, clip=3.0)
        f.update(0.5000, {})
        f.update(0.5001, {})
        out = f.apply(-1.0, {})
        assert out == pytest.approx(-3.0)

    def test_global_clip_none_means_unbounded(self) -> None:
        f = GlobalMeanStdFilter(momentum=0.5, clip=None)
        f.update(0.5000, {})
        f.update(0.5001, {})
        assert abs(f.apply(1.0, {})) > 100.0

    def test_global_clip_inert_in_normal_range(self) -> None:
        f = GlobalMeanStdFilter(momentum=0.99, clip=10.0)
        rng = np.random.default_rng(0)
        for r in rng.normal(0.5, 0.1, size=2000):
            f.update(float(r), {})
        sigma = math.sqrt(f._var)
        out = f.apply(f._mean + sigma, {})  # ≈ +1, well within clip
        assert out == pytest.approx(1.0, abs=0.05)

    def test_per_context_clip_caps_spike(self) -> None:
        z = PerContextZScore(context_keys=("d",), momentum=0.5, min_samples=2, clip=3.0)
        ctx = {"d": 0.0}
        z.update(0.5000, ctx)
        z.update(0.5001, ctx)
        out = z.apply(1.0, ctx)
        assert out == pytest.approx(3.0)

    def test_per_context_clip_none_means_unbounded(self) -> None:
        z = PerContextZScore(context_keys=("d",), momentum=0.5, min_samples=2, clip=None)
        ctx = {"d": 0.0}
        z.update(0.5000, ctx)
        z.update(0.5001, ctx)
        assert abs(z.apply(1.0, ctx)) > 100.0


class TestPerContextZScoreBinKey:
    """The bin key is the tuple of values from ctx, in declared key order."""

    def test_single_key(self) -> None:
        z = PerContextZScore(("drop_rate",))
        assert z._bin_key({"drop_rate": 0.001}) == (0.001,)

    def test_multiple_keys_preserves_order(self) -> None:
        z = PerContextZScore(("drop_rate", "msg_size"))
        assert z._bin_key({"drop_rate": 0.001, "msg_size": 64}) == (0.001, 64)
        # verify order is by ``context_keys`` not by ctx insertion order
        assert z._bin_key({"msg_size": 64, "drop_rate": 0.001}) == (0.001, 64)

    def test_missing_key_raises(self) -> None:
        z = PerContextZScore(("drop_rate",))
        with pytest.raises(KeyError):
            z._bin_key({"msg_size": 64})


class TestPerContextZScoreFirstUpdate:
    """First sample for a bin initialises (mean=r, var=0, count=1)."""

    def test_creates_bin(self) -> None:
        z = PerContextZScore(("drop_rate",))
        z.update(0.5, {"drop_rate": 0.001})
        assert (0.001,) in z._stats

    def test_initial_mean_is_first_sample(self) -> None:
        z = PerContextZScore(("drop_rate",))
        z.update(0.7619, {"drop_rate": 0.0})
        mean, _, _ = z._stats[(0.0,)]
        assert mean == pytest.approx(0.7619)

    def test_initial_var_is_zero(self) -> None:
        z = PerContextZScore(("drop_rate",))
        z.update(0.7619, {"drop_rate": 0.0})
        _, var, _ = z._stats[(0.0,)]
        assert var == 0.0

    def test_initial_count_is_one(self) -> None:
        z = PerContextZScore(("drop_rate",))
        z.update(0.7619, {"drop_rate": 0.0})
        _, _, count = z._stats[(0.0,)]
        assert count == 1


class TestPerContextZScoreEMAUpdate:
    """Subsequent samples apply EMA: mean ← α·mean + (1−α)·r,
    var ← α·var + (1−α)·(r − mean_old)²."""

    def test_second_update_mean(self) -> None:
        z = PerContextZScore(("drop_rate",), momentum=0.9)
        z.update(0.5, {"drop_rate": 0.0})  # init
        z.update(0.6, {"drop_rate": 0.0})
        # 0.9 * 0.5 + 0.1 * 0.6 = 0.51
        mean, _, _ = z._stats[(0.0,)]
        assert mean == pytest.approx(0.51)

    def test_second_update_var_uses_old_mean(self) -> None:
        z = PerContextZScore(("drop_rate",), momentum=0.9)
        z.update(0.5, {"drop_rate": 0.0})  # mean_old = 0.5, var = 0
        z.update(0.6, {"drop_rate": 0.0})
        # 0.9 * 0 + 0.1 * (0.6 - 0.5)^2 = 0.001
        _, var, _ = z._stats[(0.0,)]
        assert var == pytest.approx(0.001)

    def test_count_increments(self) -> None:
        z = PerContextZScore(("drop_rate",), momentum=0.99)
        for r in [0.5, 0.6, 0.7]:
            z.update(r, {"drop_rate": 0.0})
        _, _, count = z._stats[(0.0,)]
        assert count == 3


class TestPerContextZScoreApply:
    """``apply`` is pass-through during warm-up, normalized after."""

    def test_passthrough_when_bin_unknown(self) -> None:
        z = PerContextZScore(("drop_rate",), min_samples=1)
        z.update(0.5, {"drop_rate": 0.0})
        assert z.apply(0.7, {"drop_rate": 0.001}) == 0.7

    def test_passthrough_under_min_samples(self) -> None:
        z = PerContextZScore(("drop_rate",), min_samples=5)
        for r in [0.6, 0.7, 0.8]:
            z.update(r, {"drop_rate": 0.0})
        assert z.apply(0.5, {"drop_rate": 0.0}) == 0.5

    def test_z_score_at_mean_is_zero(self) -> None:
        z = PerContextZScore(("drop_rate",), momentum=0.5, min_samples=2)
        z.update(1.0, {"drop_rate": 0.0})
        z.update(3.0, {"drop_rate": 0.0})
        # mean = 0.5*1.0 + 0.5*3.0 = 2.0; apply at r=2.0 ⇒ 0 (within ε)
        out = z.apply(2.0, {"drop_rate": 0.0})
        assert abs(out) < 1e-2

    def test_z_score_one_sigma_offset(self) -> None:
        """Feed a regime with known μ and σ²; verify (μ+σ) maps to ≈+1."""
        z = PerContextZScore(("drop_rate",), momentum=0.99, min_samples=10)
        rng = np.random.default_rng(0)
        for r in rng.normal(0.5, 0.1, size=2000):
            z.update(float(r), {"drop_rate": 0.0})
        mean, var, _ = z._stats[(0.0,)]
        sigma = math.sqrt(var)
        # apply at mean + sigma should be approximately +1
        out = z.apply(mean + sigma, {"drop_rate": 0.0})
        assert out == pytest.approx(1.0, abs=0.05)


class TestPerContextZScoreBinIndependence:
    """Updates to one bin must not affect another bin's stats."""

    def test_two_bins_isolated(self) -> None:
        z = PerContextZScore(("drop_rate",))
        z.update(0.5, {"drop_rate": 0.0})
        z.update(0.7, {"drop_rate": 0.001})
        assert z._stats[(0.0,)] == (0.5, 0.0, 1)
        assert z._stats[(0.001,)] == (0.7, 0.0, 1)


class TestPerContextZScoreILykRegressionFixture:
    """Synthetic dataset with the same per-regime structure as the real
    ilyk 2-drop run (169 / 151 trials, μ ≈ 0.7619 / 0.7183, σ ≈ 0.0447 / 0.0251).

    Numbers are reproducible (seeded RNG); the test pins down the EMA
    converges close to the underlying distribution stats.
    """

    @staticmethod
    def _ilyk_like_dataset(seed: int = 0) -> list[tuple[float, dict]]:
        rng = np.random.default_rng(seed)
        rows: list[tuple[float, dict]] = []
        for r in rng.normal(loc=0.7619, scale=0.0447, size=169):
            rows.append((float(r), {"drop_rate": 0.0}))
        for r in rng.normal(loc=0.7183, scale=0.0251, size=151):
            rows.append((float(r), {"drop_rate": 0.001}))
        rng.shuffle(rows)
        return rows

    def test_per_regime_means_close_to_truth(self) -> None:
        z = PerContextZScore(("drop_rate",), momentum=0.99, min_samples=5)
        for r, ctx in self._ilyk_like_dataset():
            z.update(r, ctx)
        mean_d0, _, _ = z._stats[(0.0,)]
        mean_d1, _, _ = z._stats[(0.001,)]
        assert mean_d0 == pytest.approx(0.7619, abs=0.02)
        assert mean_d1 == pytest.approx(0.7183, abs=0.02)

    def test_per_regime_stds_close_to_truth(self) -> None:
        z = PerContextZScore(("drop_rate",), momentum=0.99, min_samples=5)
        for r, ctx in self._ilyk_like_dataset():
            z.update(r, ctx)
        _, var_d0, _ = z._stats[(0.0,)]
        _, var_d1, _ = z._stats[(0.001,)]
        assert math.sqrt(var_d0) == pytest.approx(0.0447, abs=0.015)
        assert math.sqrt(var_d1) == pytest.approx(0.0251, abs=0.015)

    def test_normalized_outputs_have_zero_mean_unit_var(self) -> None:
        """After warm-up, applying to all rewards should produce per-regime
        outputs that look ~N(0, 1) — that's the whole point of the transform.
        """
        z = PerContextZScore(("drop_rate",), momentum=0.99, min_samples=20)
        rows = self._ilyk_like_dataset()
        for r, ctx in rows:
            z.update(r, ctx)
        normalized_d0 = [z.apply(r, ctx) for r, ctx in rows if ctx["drop_rate"] == 0.0]
        normalized_d1 = [z.apply(r, ctx) for r, ctx in rows if ctx["drop_rate"] == 0.001]
        assert abs(float(np.mean(normalized_d0))) < 0.5
        assert abs(float(np.std(normalized_d0)) - 1.0) < 0.3
        assert abs(float(np.mean(normalized_d1))) < 0.5
        assert abs(float(np.std(normalized_d1)) - 1.0) < 0.3


# ---------------------------------------------------------------------------
# ContextAutoDetector — schema gate
# ---------------------------------------------------------------------------


class TestContextAutoDetectorSchemaGateMissingFields:
    """The detector must cope with envs that lack any link in the chain."""

    def test_no_test_run_returns_empty(self) -> None:
        env = SimpleNamespace()  # no .test_run at all
        det = ContextAutoDetector(env)
        assert det.initial_context_keys() == ()

    def test_no_test_returns_empty(self) -> None:
        env = SimpleNamespace(test_run=SimpleNamespace())
        det = ContextAutoDetector(env)
        assert det.initial_context_keys() == ()

    def test_no_env_params_returns_empty(self) -> None:
        env = SimpleNamespace(test_run=SimpleNamespace(test=SimpleNamespace()))
        det = ContextAutoDetector(env)
        assert det.initial_context_keys() == ()

    def test_empty_env_params_dict_returns_empty(self) -> None:
        det = ContextAutoDetector(_make_env({}))
        assert det.initial_context_keys() == ()


class TestContextAutoDetectorSchemaGateSamplingTypes:
    """Each sampling family is admitted or rejected per the documented rule."""

    def test_low_cardinality_categorical_admitted(self) -> None:
        env = _make_env({"drop_rate": _categorical([0.0, 0.001, 0.003, 0.01])})
        det = ContextAutoDetector(env)
        assert det.initial_context_keys() == ("drop_rate",)

    def test_log_uniform_excluded(self) -> None:
        env = _make_env({"drop_rate": _log_uniform(1e-4, 1e-1)})
        det = ContextAutoDetector(env)
        assert det.initial_context_keys() == ()

    def test_uniform_excluded(self) -> None:
        env = _make_env({"drop_rate": _uniform(0.0, 0.1)})
        det = ContextAutoDetector(env)
        assert det.initial_context_keys() == ()

    def test_fixed_sampling_excluded(self) -> None:
        """A fixed env_param has zero variance; binning it would yield a
        single bucket identical to a global filter."""
        env = _make_env({"drop_rate": _fixed(0.001)})
        det = ContextAutoDetector(env)
        assert det.initial_context_keys() == ()


class TestContextAutoDetectorSchemaGateCardinality:
    """``max_cardinality`` is the inclusive upper bound on categorical size."""

    def test_at_boundary_admitted(self) -> None:
        values = list(range(8))  # cardinality == 8
        env = _make_env({"k": _categorical(values)})
        det = ContextAutoDetector(env)
        assert det.initial_context_keys(max_cardinality=8) == ("k",)

    def test_above_boundary_excluded(self) -> None:
        values = list(range(9))  # cardinality 9 > 8
        env = _make_env({"k": _categorical(values)})
        det = ContextAutoDetector(env)
        assert det.initial_context_keys(max_cardinality=8) == ()

    def test_default_max_cardinality_is_16(self) -> None:
        """Sixteen-way categoricals should still pass under the default."""
        values = list(range(16))
        env = _make_env({"k": _categorical(values)})
        det = ContextAutoDetector(env)
        assert det.initial_context_keys() == ("k",)


class TestContextAutoDetectorSchemaGateMixed:
    """Mixed env_params: only categoricals below threshold survive."""

    def test_only_categorical_returned_from_mix(self) -> None:
        env = _make_env(
            {
                "drop_rate": _categorical([0.0, 0.001]),
                "noise": _log_uniform(1e-4, 1e-1),
                "fan_in": _fixed(64),
            }
        )
        det = ContextAutoDetector(env)
        assert det.initial_context_keys() == ("drop_rate",)

    def test_two_categoricals_preserve_insertion_order(self) -> None:
        env = _make_env(
            {
                "topology": _categorical(["fat-tree", "dragonfly"]),
                "drop_rate": _categorical([0.0, 0.001, 0.003]),
            }
        )
        det = ContextAutoDetector(env)
        assert det.initial_context_keys() == ("topology", "drop_rate")


# ---------------------------------------------------------------------------
# RLAgentBase.setup_rewards — wiring detector + builder + pipeline
# ---------------------------------------------------------------------------


class _StubRLAgent:
    """Minimal subclass to exercise ``RLAgentBase.setup_rewards`` without
    pulling in concrete library bindings. Manually drives the parts of
    ``__init__`` we need so we don't depend on ``BaseGym`` / pydantic config.
    """

    def __init__(self, env) -> None:  # noqa: ANN001 — duck-typed stub env
        self.env = env
        # Mirror RLAgentBase.__init__ side-effects without invoking ABC machinery.
        self.pipeline = RewardPipeline()
        self._context_keys: tuple[str, ...] = ()

    setup_rewards = __import__(
        "cloudai.configurator.base_rl_agent", fromlist=["RLAgentBase"]
    ).RLAgentBase.setup_rewards
    _build_pipeline = __import__(
        "cloudai.configurator.base_rl_agent", fromlist=["RLAgentBase"]
    ).RLAgentBase._build_pipeline


class TestRLAgentBaseSetupRewardsNoContext:
    """Env without categorical env_params → single GlobalMeanStdFilter."""

    def test_pipeline_starts_empty(self) -> None:
        agent = _StubRLAgent(_make_env({}))
        assert agent.pipeline.transforms == ()

    def test_setup_rewards_with_no_env_params_yields_single_global(self) -> None:
        agent = _StubRLAgent(_make_env({}))
        agent.setup_rewards()
        ts = agent.pipeline.transforms
        assert len(ts) == 1
        assert isinstance(ts[0], GlobalMeanStdFilter)

    def test_setup_rewards_with_only_continuous_env_params(self) -> None:
        agent = _StubRLAgent(_make_env({"noise": _log_uniform(1e-4, 1e-1)}))
        agent.setup_rewards()
        ts = agent.pipeline.transforms
        assert len(ts) == 1
        assert isinstance(ts[0], GlobalMeanStdFilter)


class TestRLAgentBaseSetupRewardsWithContext:
    """Env with categorical env_params → single PerContextZScore on those keys."""

    def test_categorical_env_param_promotes_to_per_context(self) -> None:
        agent = _StubRLAgent(
            _make_env({"drop_rate": _categorical([0.0, 0.001, 0.003, 0.01])})
        )
        agent.setup_rewards()
        ts = agent.pipeline.transforms
        assert len(ts) == 1
        assert isinstance(ts[0], PerContextZScore)
        assert not any(isinstance(t, GlobalMeanStdFilter) for t in ts)

    def test_per_context_uses_detected_keys(self) -> None:
        agent = _StubRLAgent(
            _make_env(
                {
                    "drop_rate": _categorical([0.0, 0.001]),
                    "topology": _categorical(["fat-tree", "dragonfly"]),
                }
            )
        )
        agent.setup_rewards()
        per_ctx = agent.pipeline.transforms[0]
        assert isinstance(per_ctx, PerContextZScore)
        assert per_ctx._keys == ("drop_rate", "topology")

    def test_setup_rewards_replaces_initial_empty_pipeline(self) -> None:
        """The cold-start empty pipeline must be replaced, not appended to."""
        agent = _StubRLAgent(_make_env({"drop_rate": _categorical([0.0, 0.001])}))
        original = agent.pipeline
        agent.setup_rewards()
        assert agent.pipeline is not original
        assert len(agent.pipeline.transforms) == 1

    def test_setup_rewards_records_context_keys(self) -> None:
        """``_context_keys`` is populated from the detector so ``wrap_env`` can
        project the context dict before forwarding to transforms.
        """
        agent = _StubRLAgent(_make_env({"drop_rate": _categorical([0.0, 0.001])}))
        agent.setup_rewards()
        assert agent._context_keys == ("drop_rate",)


# ---------------------------------------------------------------------------
# PipelineRewardWrapper — gym/gymnasium-style reward injection
# ---------------------------------------------------------------------------


class _FakeEnv:
    """Minimal duck-typed env for PipelineRewardWrapper tests.

    Returns a deterministic reward sequence; ``observation_space``,
    ``action_space``, and ``unwrapped`` are sentinels so we can verify
    the wrapper forwards them without inspecting their types.
    """

    def __init__(self, rewards: list[float], info_seed: dict | None = None) -> None:
        self._rewards = list(rewards)
        self._idx = 0
        self._info_seed = dict(info_seed or {})
        self.observation_space = object()
        self.action_space = object()
        self.metadata = {"render_modes": ["human"]}
        self._reset_called = 0
        self._closed = False

    @property
    def unwrapped(self):
        return self

    def reset(self, *args, **kwargs):
        self._reset_called += 1
        self._idx = 0
        return ("obs0", dict(self._info_seed))

    def step(self, action):
        r = float(self._rewards[self._idx])
        self._idx += 1
        info = dict(self._info_seed)
        return ("obs", r, False, False, info)

    def render(self, *args, **kwargs):
        return "rendered"

    def close(self):
        self._closed = True


class TestPipelineRewardWrapperReset:
    """``reset()`` is fully transparent."""

    def test_reset_forwards_to_env(self) -> None:
        env = _FakeEnv(rewards=[1.0])
        pipe = build_default_pipeline(())
        wrapper = PipelineRewardWrapper(env, pipe, context_provider=lambda: {})
        obs, info = wrapper.reset()
        assert obs == "obs0"
        assert isinstance(info, dict)
        assert env._reset_called == 1


class TestPipelineRewardWrapperStepEmptyPipeline:
    """Empty pipeline is identity on reward; raw_reward still injected."""

    def test_reward_unchanged_when_pipeline_empty(self) -> None:
        env = _FakeEnv(rewards=[0.5])
        wrapper = PipelineRewardWrapper(env, RewardPipeline(), context_provider=lambda: {})
        obs, reward, term, trunc, info = wrapper.step({"a": 0})
        assert reward == 0.5
        assert info["raw_reward"] == 0.5


class TestPipelineRewardWrapperStepGlobalFilter:
    """Pipeline with only global filter z-scores against running stats."""

    def test_first_step_passes_through_then_warmup(self) -> None:
        env = _FakeEnv(rewards=[1.0, 2.0, 3.0])
        pipe = build_default_pipeline(())
        wrapper = PipelineRewardWrapper(env, pipe, context_provider=lambda: {})

        # n=1 sample → GlobalMeanStdFilter still in warmup, returns raw
        _, r1, *_ = wrapper.step({})
        # n=2 samples → z-score kicks in
        _, r2, *_, info2 = wrapper.step({})
        _, r3, *_, info3 = wrapper.step({})

        assert r1 == 1.0
        assert r2 != 2.0  # transformed
        assert info2["raw_reward"] == 2.0
        assert info3["raw_reward"] == 3.0


class TestPipelineRewardWrapperStepPerContext:
    """Pipeline with PerContextZScore reads ctx via the provider."""

    def test_context_provider_called_per_step(self) -> None:
        ctx_log: list[dict] = []

        def provider() -> dict:
            d = {"drop_rate": 0.001 if len(ctx_log) % 2 == 0 else 0.0}
            ctx_log.append(d)
            return d

        env = _FakeEnv(rewards=[0.7, 0.6, 0.75])
        pipe = build_default_pipeline(("drop_rate",))
        wrapper = PipelineRewardWrapper(
            env, pipe, context_provider=provider, context_keys=("drop_rate",)
        )

        wrapper.step({})
        wrapper.step({})
        wrapper.step({})

        assert len(ctx_log) == 3
        assert ctx_log[0] == {"drop_rate": 0.001}
        assert ctx_log[1] == {"drop_rate": 0.0}
        assert ctx_log[2] == {"drop_rate": 0.001}


class TestPipelineRewardWrapperContextProjection:
    """``context_keys`` projects the provider's dict to relevant keys only."""

    def test_keys_outside_projection_dropped(self) -> None:
        seen: list[dict] = []

        class _Spy:
            def update(self, reward, ctx) -> None:
                seen.append(dict(ctx))

            def apply(self, reward, ctx):
                return reward

        spy = _Spy()
        pipe = RewardPipeline([spy])
        env = _FakeEnv(rewards=[1.0])
        wrapper = PipelineRewardWrapper(
            env,
            pipe,
            context_provider=lambda: {"drop_rate": 0.0, "topology": "fat-tree", "seed": 42},
            context_keys=("drop_rate",),
        )
        wrapper.step({})
        assert seen == [{"drop_rate": 0.0}]

    def test_empty_keys_forwards_full_ctx(self) -> None:
        seen: list[dict] = []

        class _Spy:
            def update(self, reward, ctx) -> None:
                seen.append(dict(ctx))

            def apply(self, reward, ctx):
                return reward

        pipe = RewardPipeline([_Spy()])
        env = _FakeEnv(rewards=[1.0])
        wrapper = PipelineRewardWrapper(
            env,
            pipe,
            context_provider=lambda: {"drop_rate": 0.0, "topology": "fat-tree"},
            context_keys=(),
        )
        wrapper.step({})
        assert seen == [{"drop_rate": 0.0, "topology": "fat-tree"}]


class TestPipelineRewardWrapperFourTuple:
    """Old-gym 4-tuple step is supported; ``info`` still carries raw_reward."""

    def test_four_tuple_supported(self) -> None:
        class _OldGym(_FakeEnv):
            def step(self, action):
                r = float(self._rewards[self._idx])
                self._idx += 1
                return ("obs", r, False, dict(self._info_seed))

        env = _OldGym(rewards=[2.5])
        wrapper = PipelineRewardWrapper(
            env, RewardPipeline(), context_provider=lambda: {}
        )
        result = wrapper.step({})
        assert len(result) == 4
        obs, reward, done, info = result
        assert reward == 2.5
        assert info["raw_reward"] == 2.5


class TestPipelineRewardWrapperPassthrough:
    """Properties + attribute forwarding match a transparent wrapper."""

    def test_observation_and_action_space_forward(self) -> None:
        env = _FakeEnv(rewards=[])
        wrapper = PipelineRewardWrapper(env, RewardPipeline(), context_provider=lambda: {})
        assert wrapper.observation_space is env.observation_space
        assert wrapper.action_space is env.action_space

    def test_metadata_forwarded_via_getattr(self) -> None:
        env = _FakeEnv(rewards=[])
        wrapper = PipelineRewardWrapper(env, RewardPipeline(), context_provider=lambda: {})
        assert wrapper.metadata == {"render_modes": ["human"]}

    def test_unwrapped_returns_inner_env(self) -> None:
        env = _FakeEnv(rewards=[])
        wrapper = PipelineRewardWrapper(env, RewardPipeline(), context_provider=lambda: {})
        assert wrapper.unwrapped is env

    def test_close_propagates(self) -> None:
        env = _FakeEnv(rewards=[])
        wrapper = PipelineRewardWrapper(env, RewardPipeline(), context_provider=lambda: {})
        wrapper.close()
        assert env._closed is True

    def test_render_propagates(self) -> None:
        env = _FakeEnv(rewards=[])
        wrapper = PipelineRewardWrapper(env, RewardPipeline(), context_provider=lambda: {})
        assert wrapper.render() == "rendered"


class TestPipelineRewardWrapperInfoMerge:
    """``info["raw_reward"]`` is added without clobbering env-supplied info."""

    def test_existing_keys_preserved(self) -> None:
        env = _FakeEnv(rewards=[1.0], info_seed={"trial_index": 7})
        wrapper = PipelineRewardWrapper(env, RewardPipeline(), context_provider=lambda: {})
        _, _, _, _, info = wrapper.step({})
        assert info["trial_index"] == 7
        assert info["raw_reward"] == 1.0

    def test_none_info_treated_as_empty(self) -> None:
        class _NoneInfoEnv(_FakeEnv):
            def step(self, action):
                r = float(self._rewards[self._idx])
                self._idx += 1
                return ("obs", r, False, False, None)

        env = _NoneInfoEnv(rewards=[0.5])
        wrapper = PipelineRewardWrapper(env, RewardPipeline(), context_provider=lambda: {})
        _, _, _, _, info = wrapper.step({})
        assert info == {"raw_reward": 0.5}


class TestPipelineRewardWrapperBadTuple:
    """Defensive: malformed step return raises a clear ValueError."""

    def test_three_tuple_rejected(self) -> None:
        class _BadEnv(_FakeEnv):
            def step(self, action):
                return ("obs", 1.0, False)

        env = _BadEnv(rewards=[])
        wrapper = PipelineRewardWrapper(env, RewardPipeline(), context_provider=lambda: {})
        with pytest.raises(ValueError, match="4- or 5-tuple"):
            wrapper.step({})


# ---------------------------------------------------------------------------
# snapshot() observability hooks
# ---------------------------------------------------------------------------


class TestGlobalMeanStdFilterSnapshot:
    def test_inactive_before_two_samples(self) -> None:
        f = GlobalMeanStdFilter()
        assert f.snapshot()["active"] is False
        f.update(1.0, {})
        assert f.snapshot()["count"] == 1
        assert f.snapshot()["active"] is False

    def test_active_and_stats_after_two_samples(self) -> None:
        f = GlobalMeanStdFilter()
        f.update(1.0, {})
        f.update(3.0, {})
        snap = f.snapshot()
        assert snap["active"] is True
        assert snap["count"] == 2
        # mean after EMA: 0.99*1 + 0.01*3 = 1.02
        assert math.isclose(snap["mean"], 1.02, rel_tol=1e-9)
        assert snap["std"] > 0.0


class TestPerContextZScoreSnapshot:
    def test_bin_inactive_during_warmup(self) -> None:
        z = PerContextZScore(context_keys=("d",), min_samples=5)
        for _ in range(3):
            z.update(1.0, {"d": 0})
        s = z.snapshot({"d": 0})
        assert s["bin"] == (0,)
        assert s["count"] == 3
        assert s["active"] is False

    def test_bin_active_after_min_samples(self) -> None:
        z = PerContextZScore(context_keys=("d",), min_samples=5)
        for _ in range(5):
            z.update(1.0, {"d": 0})
        s = z.snapshot({"d": 0})
        assert s["count"] == 5
        assert s["active"] is True

    def test_unknown_bin_is_inactive_zero(self) -> None:
        z = PerContextZScore(context_keys=("d",), min_samples=5)
        s = z.snapshot({"d": 99})
        assert s["count"] == 0
        assert s["active"] is False

    def test_all_bins_dump(self) -> None:
        z = PerContextZScore(context_keys=("d",), min_samples=2)
        z.update(1.0, {"d": 0})
        z.update(2.0, {"d": 1})
        dump = z.snapshot()
        assert set(dump["bins"].keys()) == {(0,), (1,)}


class TestPipelineRewardWrapperPerStepLogging:
    """The wrapper emits one INFO line per step with verifiable stats."""

    @staticmethod
    def _capture_wrapper_logs():
        """Attach a handler directly to the wrapper logger; return
        ``(messages, cleanup)``.

        Attaching a handler to the logger under test (rather than relying on
        pytest's ``caplog``, which listens on the root logger) is the
        idiomatic way to assert on a logger whose records do not propagate to
        root. cloudai's ``setup_logging`` sets ``propagate=False`` on the root
        config, so caplog can miss these records; a local handler is exact.

        Note: this intentionally does NOT touch ``logging.disable`` or
        ``logger.disabled``. With ``cloudai`` named in
        ``cli.setup_logging``'s config, ``disable_existing_loggers=True`` no
        longer disables ``cloudai.*`` children, so no re-enabling is needed.
        """
        import logging

        wlog = logging.getLogger("cloudai.configurator.rewards.wrapper")
        old_level = wlog.level
        wlog.setLevel(logging.INFO)
        messages: list[str] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                messages.append(record.getMessage())

        handler = _Capture()
        wlog.addHandler(handler)

        def cleanup() -> None:
            wlog.removeHandler(handler)
            wlog.setLevel(old_level)

        return messages, cleanup

    def test_log_line_emitted_with_ctx_and_stats(self) -> None:
        env = _FakeEnv(rewards=[0.7, 0.8, 0.75])
        pipe = build_default_pipeline(("drop_rate",))
        provider = lambda: {"drop_rate": 0.0}  # noqa: E731
        wrapper = PipelineRewardWrapper(
            env, pipe, context_provider=provider, context_keys=("drop_rate",)
        )
        messages, cleanup = self._capture_wrapper_logs()
        try:
            wrapper.step({})
            wrapper.step({})
        finally:
            cleanup()

        lines = [m for m in messages if "[reward-pipeline]" in m]
        assert len(lines) == 2
        assert "ctx={'drop_rate': 0.0}" in lines[0]
        # Design X: context present → single PerContextZScore, no Global chained.
        assert "PerContextZScore" in lines[0]
        assert "GlobalMeanStdFilter" not in lines[0]
        assert "raw=" in lines[0] and "reward=" in lines[0]

    def test_no_snapshot_transforms_logs_placeholder(self) -> None:
        env = _FakeEnv(rewards=[1.0])
        # empty pipeline => no stateful transforms
        wrapper = PipelineRewardWrapper(env, RewardPipeline(), context_provider=lambda: {})
        messages, cleanup = self._capture_wrapper_logs()
        try:
            wrapper.step({})
        finally:
            cleanup()
        lines = [m for m in messages if "[reward-pipeline]" in m]
        assert len(lines) == 1
        assert "no stateful transforms" in lines[0]
