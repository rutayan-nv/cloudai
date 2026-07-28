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

import copy
import csv
import dataclasses
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cloudai.core import METRIC_ERROR, BaseRunner, Registry, TestRun
from cloudai.util.lazy_imports import lazy

from .base_agent import RewardOverrides
from .base_gym import BaseGym
from .env_params import CsvSink, EnvParamsObserver, ObsLeafDescriptor, StepObserver


@dataclasses.dataclass(frozen=True)
class TrajectoryEntry:
    """Represents a trajectory entry."""

    step: int
    action: dict[str, Any]
    reward: float
    observation: list
    env_params: dict[str, Any] = dataclasses.field(default_factory=dict)


class CloudAIGymEnv(BaseGym):
    """
    Custom Gym environment for CloudAI integration.

    Uses the TestRun object and actual runner methods to execute jobs.
    """

    def __init__(self, test_run: TestRun, runner: BaseRunner, rewards: RewardOverrides):
        """
        Initialize the Gym environment using the TestRun object.

        Args:
            test_run (TestRun): A test run object that encapsulates cmd_args, extra_cmd_args, etc.
            runner (BaseRunner): The runner object to execute jobs.
            rewards: Reward / observation overrides from agent config.
        """
        self.test_run = test_run
        self.original_test_run = copy.deepcopy(test_run)  # Preserve clean state for DSE
        self.runner = runner
        self.rewards = rewards
        self.max_steps = test_run.test.agent_steps
        self.reward_function = Registry().get_reward_function(test_run.test.agent_reward_function)
        self.trajectory: dict[int, list[TrajectoryEntry]] = {}
        self.observers: List[StepObserver] = self._build_observers()
        self._prime_cache_if_configured()
        super().__init__()

    def _prime_cache_if_configured(self) -> None:
        """
        Pre-populate ``self.trajectory[0]`` from a prior run when configured.

        Reads ``test.cache_warm_start_path`` (set in the workload TOML) and
        loads its rows as ``TrajectoryEntry`` objects so trials repeating a
        prior ``(action, env_params)`` pair short-circuit cluster execution.
        Missing/unreadable paths log a warning and leave the cache empty
        rather than aborting the run.
        """
        path = getattr(self.test_run.test, "cache_warm_start_path", None)
        if path is None:
            return
        from .trajectory_loader import load_trajectory_with_env

        try:
            entries = load_trajectory_with_env(path)
        except FileNotFoundError as exc:
            logging.warning("cache_warm_start_path: %s; trajectory cache will start empty.", exc)
            return
        if entries:
            self.trajectory[0] = list(entries)
            logging.info("Primed trajectory cache with %d entries from %s", len(entries), path)

    def _build_observers(self) -> List[StepObserver]:
        """
        Construct the per-step observers implied by the TestDefinition.

        Workloads opt in to env_params via a TOML ``[env_params.<name>]`` block;
        an empty mapping yields no observers and zero overhead.
        """
        observers: List[StepObserver] = []
        if self.test_run.test.env_params:
            seed = int((self.test_run.test.agent_config or {}).get("random_seed", 0))
            sink = CsvSink(self._env_csv_path())
            observers.append(EnvParamsObserver(self.test_run.test.env_params, sink, seed))
        return observers

    def _env_csv_path(self) -> Path:
        """``env.csv`` lives alongside ``trajectory.csv`` so a plain ``merge`` joins them."""
        return self.trajectory_file_path.parent / "env.csv"

    def define_action_space(self) -> Dict[str, list[Any]]:
        return self.test_run.param_space

    @property
    def first_sweep(self) -> dict[str, Any]:
        """Builds a sweep using first elements of each explorable parameter."""
        return {k: v[0] for k, v in self.define_action_space().items()}

    def define_observation_space(self) -> list:
        """
        Define the observation space for the environment.

        Returns:
            list: One float slot per declared observation name. Uses
            ``agent_observation`` when set, otherwise falls back to
            ``agent_metrics`` for backward compatibility. Always at least one slot
            so adapters that derive ``gymnasium.spaces.Box`` from this output get
            a valid shape.
        """
        names = self.test_run.test.agent_observation or self.test_run.test.agent_metrics
        return [0.0] * max(len(names), 1)

    def observation_names(self) -> list[str]:
        """Ordered observation names: ``agent_observation`` or ``agent_metrics`` fallback."""
        return list(self.test_run.test.agent_observation or self.test_run.test.agent_metrics)

    def structured_observation_descriptors(self) -> Optional[Dict[str, ObsLeafDescriptor]]:
        """
        Descriptors for a structured (Dict) obs space, or ``None`` for the flat path.

        Opt-in gate: returns descriptors only when at least one observed name
        is a declared ``[env_params.<name>]`` (so it carries an encoding worth
        a named leaf, e.g. a log-encoded ``drop_rate``). Pure-metric
        observations — and any env without a ``test_run`` — keep the legacy
        flat ``define_observation_space`` (Box) path unchanged, containing the
        blast radius to domain-randomized RL workloads.
        """
        test_run = getattr(self, "test_run", None)
        if test_run is None:
            return None
        env_params = test_run.test.env_params
        if not any(name in env_params for name in self.observation_names()):
            return None
        return self.observation_descriptors()

    def observation_descriptors(self) -> Dict[str, ObsLeafDescriptor]:
        """
        Per-name observation-leaf descriptors for building a structured (Dict) obs space.

        A declared ``[env_params.<name>]`` contributes its encoding-derived
        leaf (e.g. ``log`` -> ``Box(2)``); any other observed name (a measured
        metric) is a raw ``Box(1)`` scalar. RL adapters use this to construct a
        ``gymnasium.spaces.Dict``; non-RL agents (BO/GA/MAB) ignore it and keep
        using the flat ``define_observation_space`` path.
        """
        env_params = self.test_run.test.env_params
        descriptors: Dict[str, ObsLeafDescriptor] = {}
        for name in self.observation_names():
            spec = env_params.get(name)
            descriptors[name] = (
                spec.observation_descriptor() if spec is not None else ObsLeafDescriptor(kind="box", dim=1)
            )
        return descriptors

    def encode_observation(self, raw_values: list) -> Dict[str, Any]:
        """
        Encode a flat raw observation into named, per-leaf encoded values.

        ``raw_values`` is the flat list returned by ``reset()``/``get_observation``
        (the same vector that feeds the reward and ``trajectory.csv``). Each
        env-randomized name is encoded by its declared ``EnvParamSpec`` (e.g.
        ``log`` -> ``[is_zero, log10]``); every other name passes through as a
        raw ``[float]`` leaf. The keys/leaf widths align 1:1 with
        :meth:`observation_descriptors`.
        """
        env_params = self.test_run.test.env_params
        names = self.observation_names()
        encoded: Dict[str, Any] = {}
        for name, raw in zip(names, raw_values, strict=True):
            spec = env_params.get(name)
            encoded[name] = spec.encode_observation(raw) if spec is not None else [float(raw)]
        return encoded

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,  # noqa: Vulture
    ) -> Tuple[list, dict[str, Any]]:
        """
        Reset the environment and start a new trial.

        Trial-boundary semantics: this is the canonical "trial begins" hook.
        The trial counter advances here (``increment_step``), env_param
        observers fire here (so ``current_env_params`` is populated *before*
        the agent reads obs), and the returned observation already includes
        any env-randomized context declared via ``agent_observation``. This
        replaces the prior design where ``step()`` advanced the counter and
        sampled env_params; the timing made the contextual signal arrive
        *after* the policy had already chosen its action.

        Args:
            seed: Seed for the environment's random number generator.
            options: Additional options for reset.

        Returns:
            Tuple of (observation, info).
        """
        if seed is not None:
            lazy.np.random.seed(seed)
        self.test_run.current_iteration = 0

        self.test_run.increment_step()
        for observer in self.observers:
            observer.before_step(self.test_run)

        observation = self._observation_at_reset()
        info: dict[str, Any] = {}
        return observation, info

    def _observation_at_reset(self) -> list:
        """
        Build the obs vector returned by ``reset()``.

        Measured-metric slots are set to ``rewards.metric_failure`` (the same
        sentinel ``get_observation`` uses on the post-step path) because no
        sim has run yet. Slots that name a declared env_param resolve to the
        sample written by the env_params observer in this same ``reset()``
        call, exposing the trial's context to the policy *before* it picks
        an action.
        """
        names = self.test_run.test.agent_observation or self.test_run.test.agent_metrics
        if not names:
            return [0.0]
        observation: list = []
        for name in names:
            if name in self.test_run.test.env_params:
                observation.append(self.test_run.current_env_params.get(name, self.rewards.metric_failure))
            else:
                observation.append(self.rewards.metric_failure)
        return observation

    def step(self, action: Any) -> Tuple[list, float, bool, dict]:
        """
        Execute one step in the environment.

        Trial counter advancement and env_param sampling happen in
        ``reset()`` (the trial-boundary hook). ``step()`` is pure
        action-handling: apply the action to the test_run, run the workload
        (or hit the cache), compute the post-step observation and reward.

        Args:
            action: Action chosen by the agent.

        Returns:
            Tuple of (observation, reward, done, info).
        """
        self.test_run = self.test_run.apply_params_set(action)

        cached_result = self.get_cached_trajectory_result(action)
        if cached_result is not None:
            logging.info(
                "Retrieved cached result from trajectory with reward %s (from step %s). Skipping execution.",
                cached_result.reward,
                cached_result.step,
            )
            self.write_trajectory(
                TrajectoryEntry(
                    step=self.test_run.step,
                    action=action,
                    reward=cached_result.reward,
                    observation=cached_result.observation,
                    env_params=dict(self.test_run.current_env_params),
                )
            )
            for observer in self.observers:
                observer.after_step(self.test_run, cached_result.observation, cached_result.reward)
            return cached_result.observation, cached_result.reward, False, {}

        if not self.test_run.test.constraint_check(self.test_run, self.runner.system):
            logging.info("Constraint check failed. Skipping step.")
            return [-1.0], self.rewards.constraint_failure, True, {}

        new_tr = copy.deepcopy(self.test_run)
        new_tr.output_path = self.runner.get_job_output_path(new_tr)
        self.runner.test_scenario.test_runs = [new_tr]

        self.runner.shutting_down = False
        self.runner.jobs.clear()
        self.runner.testrun_to_job_map.clear()

        try:
            self.runner.run()
        except Exception as e:
            logging.error(f"Error running step {self.test_run.step}: {e}")

        if self.runner.test_scenario.test_runs and self.runner.test_scenario.test_runs[0].output_path.exists():
            self.test_run = self.runner.test_scenario.test_runs[0]
        else:
            self.test_run = copy.deepcopy(self.original_test_run)
            self.test_run.step = new_tr.step
            self.test_run.output_path = new_tr.output_path

        observation = self.get_observation(action)
        reward = self.compute_reward(observation)

        self.write_trajectory(
            TrajectoryEntry(
                step=self.test_run.step,
                action=action,
                reward=reward,
                observation=observation,
                env_params=dict(self.test_run.current_env_params),
            )
        )

        for observer in self.observers:
            observer.after_step(self.test_run, observation, reward)

        return observation, reward, False, {}

    def render(self, mode: str = "human"):
        """
        Render the current state of the TestRun.

        Args:
            mode (str): The mode to render with. Default is "human".
        """
        print(f"Step {self.test_run.current_iteration}: Parameters {self.test_run.test.cmd_args}")

    def seed(self, seed: Optional[int] = None):
        """
        Set the seed for the environment's random number generator.

        Args:
            seed (Optional[int]): Seed for the environment's random number generator.
        """
        if seed is not None:
            lazy.np.random.seed(seed)

    def current_context(self) -> dict[str, Any]:
        """Expose ``test_run.current_env_params`` as the env's context.

        The :class:`EnvParamsObserver` populates ``current_env_params`` on
        every ``reset()`` (before ``compute_reward`` runs). Returning a copy
        from here lets reward transforms key on the trial's sampled
        env_params without binding to ``test_run`` directly. Empty dict when
        the test defines no env_params or hasn't been ``reset()`` yet.

        Defensive: tests sometimes construct ``CloudAIGymEnv`` subclasses
        without setting ``test_run`` (synthetic envs used to drive RLlib's
        plumbing tests). Returning an empty dict in that case keeps the
        contract total -- this method must never raise.
        """
        test_run = getattr(self, "test_run", None)
        if test_run is None:
            return {}
        return dict(getattr(test_run, "current_env_params", {}) or {})

    def compute_reward(self, observation: list) -> float:
        """Compute the reward by delegating to the registered reward function.

        Reward inputs come from two independent sources, concatenated in this order:

        1. **Metric values** — resolved from ``agent_metrics`` via
           :meth:`TestRun.get_metric_value` (env_params short-circuit, otherwise
           the workload's report). ``METRIC_ERROR`` is substituted with
           :attr:`rewards.metric_failure`.
        2. **Observation** — the caller-provided list (built by
           :meth:`get_observation` from ``agent_observation``).

        This decouples the reward source (``agent_metrics``) from the agent's
        feature view (``agent_observation``): legacy TOMLs that listed metrics
        in ``agent_observation`` continue to work (the metric value is just
        duplicated in the list, harmlessly), while clean TOMLs can keep the
        two specs disjoint without breaking the reward function's "slot 0 is
        the first metric" contract.

        Args:
            observation: Observation list (or ``[]`` for reward functions that
                only need metric values).

        Returns:
            The reward function's output.
        """
        metric_values = self._resolve_metric_values()
        return self.reward_function(metric_values + list(observation))

    def _resolve_metric_values(self) -> list:
        """Resolve ``agent_metrics`` names to their current values.

        Mirrors the same lookup path as the observation layer
        (``TestRun.get_metric_value`` → env_params short-circuit → report
        fallback) and applies the same ``METRIC_ERROR`` → ``metric_failure``
        substitution, so the reward function sees uniform numeric input
        regardless of whether a name resolves through env_params or the
        workload report.
        """
        out: list = []
        for name in self.test_run.test.agent_metrics:
            v = self.test_run.get_metric_value(self.runner.system, name)
            if v is METRIC_ERROR:
                v = self.rewards.metric_failure
            out.append(v)
        return out

    def get_observation(self, action: Any) -> list:
        """
        Get the observation from the TestRun object.

        Resolves each name in ``agent_observation`` (or ``agent_metrics`` when
        ``agent_observation`` is empty, for backward compatibility) via
        ``TestRun.get_metric_value``, which short-circuits to
        ``current_env_params`` when the name matches a declared env-randomized
        parameter. METRIC_ERROR sentinels become ``rewards.metric_failure``.

        Args:
            action (Any): Action taken by the agent.

        Returns:
            list: The observation.
        """
        names = self.test_run.test.agent_observation or self.test_run.test.agent_metrics
        if not names:
            raise ValueError("No agent observation or metrics defined for the test run")

        observation = []
        for name in names:
            v = self.test_run.get_metric_value(self.runner.system, name)
            if v is METRIC_ERROR:
                v = self.rewards.metric_failure
            observation.append(v)
        return observation

    def write_trajectory(self, entry: TrajectoryEntry):
        """Append the trajectory to the CSV file and to the local attribute."""
        self.current_trajectory.append(entry)

        file_exists = self.trajectory_file_path.exists()
        logging.debug(f"Writing trajectory into {self.trajectory_file_path} (exists: {file_exists})")
        self.trajectory_file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.trajectory_file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["step", "action", "reward", "observation"])
            writer.writerow([entry.step, entry.action, entry.reward, entry.observation])

    @property
    def trajectory_file_path(self) -> Path:
        return self.runner.scenario_root / self.test_run.name / f"{self.test_run.current_iteration}" / "trajectory.csv"

    @property
    def current_trajectory(self) -> list[TrajectoryEntry]:
        return self.trajectory.setdefault(self.test_run.current_iteration, [])

    def get_cached_trajectory_result(self, action: Any) -> TrajectoryEntry | None:
        """
        Return a cached entry only when the full trial identity matches.

        Trial identity is ``(action, env_params)``: env-randomized parameters
        change the workload's behaviour, so a trial repeating the same action
        under a different ``env_params`` sample must miss and re-run. Empty
        env_params on both sides is the back-compat path for workloads that
        do not declare any ``[env_params.*]`` block.
        """
        current_env_params = getattr(self.test_run, "current_env_params", {}) or {}
        for entry in self.current_trajectory:
            if not self._values_match_exact(entry.action, action):
                continue
            entry_env = getattr(entry, "env_params", {}) or {}
            if self._values_match_exact(entry_env, current_env_params):
                return entry

        return None

    @classmethod
    def _values_match_exact(cls, left: Any, right: Any) -> bool:
        if type(left) is not type(right):
            return False

        elif isinstance(left, dict):
            left_keys = set(left.keys())
            right_keys = set(right.keys())
            if left_keys != right_keys:
                return False

            return all(cls._values_match_exact(left[key], right[key]) for key in left_keys)

        elif isinstance(left, (list, tuple)):
            if len(left) != len(right):
                return False

            for left_item, right_item in zip(left, right, strict=True):
                if not cls._values_match_exact(left_item, right_item):
                    return False

            return True

        else:
            return left == right
