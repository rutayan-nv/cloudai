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

from pathlib import Path
from typing import Union, cast
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from pydantic import BaseModel, Field

from cloudai._core.action_space import ContinuousSpace
from cloudai.configurator import CloudAIGymEnv, GridSearchAgent, TrajectoryEntry
from cloudai.configurator.env_params import EnvParamSpec
from cloudai.core import BaseRunner, RewardOverrides, Runner, TestRun, TestScenario
from cloudai.models.workload import CmdArgs, TestDefinition
from cloudai.systems.slurm import SlurmSystem
from cloudai.util import flatten_dict
from cloudai.workloads.nemo_run import (
    Data,
    NeMoRunCmdArgs,
    NeMoRunTestDefinition,
    Trainer,
    TrainerStrategy,
)
from cloudai.workloads.nemo_run.report_generation_strategy import NeMoRunReportGenerationStrategy
from cloudai.workloads.nixl_bench import NIXLBenchCmdArgs, NIXLBenchTestDefinition


@pytest.fixture
def nemorun() -> NeMoRunTestDefinition:
    return NeMoRunTestDefinition(
        name="NemoModel",
        description="Nemo Model",
        test_template_name="nemo_template",
        cmd_args=NeMoRunCmdArgs(docker_image_url="https://docker/url", task="some_task", recipe_name="some_recipe"),
    )


@pytest.fixture
def setup_env(slurm_system: SlurmSystem, nemorun: NeMoRunTestDefinition) -> tuple[TestRun, BaseRunner]:
    tdef = nemorun.model_copy(deep=True)
    tdef.cmd_args.trainer = Trainer(
        max_steps=[1000, 2000],
        val_check_interval=[100, 200],
        strategy=TrainerStrategy(
            tensor_model_parallel_size=[1, 2],
            pipeline_model_parallel_size=[1, 2],
            context_parallel_size=[2, 4],
        ),
    )
    tdef.cmd_args.data = Data(micro_batch_size=[1, 2])
    tdef.agent_metrics = ["default"]

    mock_command_gen = MagicMock()
    mock_command_gen.gen_srun_command.return_value = "srun mock command"
    mock_command_gen.generate_test_command.return_value = ["python", "run.py", "--arg", "value"]

    test_template_mock = MagicMock()
    test_template_mock.command_gen_strategy = mock_command_gen

    test_run = TestRun(
        name="mock_test_run", test=tdef, num_nodes=1, nodes=[], reports={NeMoRunReportGenerationStrategy}
    )

    test_scenario = TestScenario(name="mock_test_scenario", test_runs=[test_run])
    test_run.output_path = (
        slurm_system.output_path / test_scenario.name / test_run.name / f"{test_run.current_iteration}"
    )

    runner = Runner(mode="dry-run", system=slurm_system, test_scenario=test_scenario)

    return test_run, runner.runner


def test_observation_space(setup_env: tuple[TestRun, BaseRunner]):
    test_run, runner = setup_env
    env = CloudAIGymEnv(test_run=test_run, runner=runner, rewards=RewardOverrides())
    observation_space = env.define_observation_space()

    expected_observation_space = [0.0]

    assert observation_space == expected_observation_space


def test_structured_observation_descriptors_and_encoding(setup_env: tuple[TestRun, BaseRunner]):
    """Structured obs: a log-encoded env_param contributes a Box(2) [is_zero, log10] leaf.

    The flat raw obs (which feeds reward + trajectory.csv) is unchanged; the
    encoded Dict is derived from it via the env's declared per-param encoding.
    """
    import math

    test_run, runner = setup_env
    test_run.test.env_params = {
        "drop_rate": EnvParamSpec(
            sampling={"type": "loguniform", "low": 1e-4, "high": 1e-1, "zero_prob": 0.2},
            encoding={"type": "log"},
        )
    }
    test_run.test.agent_observation = ["bus_bw", "drop_rate"]
    env = CloudAIGymEnv(test_run=test_run, runner=runner, rewards=RewardOverrides())

    descriptors = env.observation_descriptors()
    assert list(descriptors.keys()) == ["bus_bw", "drop_rate"]
    assert descriptors["bus_bw"].kind == "box" and descriptors["bus_bw"].dim == 1
    assert descriptors["drop_rate"].kind == "box" and descriptors["drop_rate"].dim == 2

    encoded = env.encode_observation([11.5, 0.01])
    assert encoded["bus_bw"] == [11.5]
    assert encoded["drop_rate"][0] == 0.0  # not the zero baseline
    assert encoded["drop_rate"][1] == pytest.approx(math.log10(0.01))

    encoded_zero = env.encode_observation([11.5, 0.0])
    assert encoded_zero["drop_rate"][0] == 1.0  # is_zero indicator
    assert encoded_zero["drop_rate"][1] == pytest.approx(math.log10(1e-4))  # floor=low anchor


@pytest.mark.parametrize(
    "reward_function,test_cases",
    [
        (
            "inverse",
            [
                ([0.34827126874999986], pytest.approx(2.871, 0.001)),
                ([0.0], 0.0),
                ([], 0.0),
                ([2.0, 2.0], 0.5),
            ],
        ),
        (
            "negative",
            [
                ([2.0], -2.0),
                ([-1.5], 1.5),
                ([0.0], 0.0),
                ([], 0.0),
            ],
        ),
        (
            "identity",
            [
                ([2.0], 2.0),
                ([-1.5], -1.5),
                ([0.0], 0.0),
                ([], 0.0),
            ],
        ),
    ],
)
def test_compute_reward(reward_function, test_cases, base_tr: TestRun):
    base_tr.test.agent_reward_function = reward_function
    base_tr.test.agent_metrics = []  # focus this unit test on the reward-fn dispatch path;
    # metric resolution + concat is exercised separately by TestComputeRewardMetricsConcat below.
    env = CloudAIGymEnv(test_run=base_tr, runner=MagicMock(), rewards=RewardOverrides())

    for input_value, expected_reward in test_cases:
        reward = env.compute_reward(input_value)
        assert reward == expected_reward


# ---------------------------------------------------------------------------
# compute_reward — agent_metrics resolution + concat with observation
# ---------------------------------------------------------------------------


class TestComputeRewardMetricsConcat:
    """Decoupling contract:

    * ``agent_metrics`` is the canonical reward source. ``compute_reward``
      resolves each metric name via ``TestRun.get_metric_value`` and prepends
      the resulting values to the ``observation`` argument before delegating
      to the registered reward function.
    * ``agent_observation`` is the agent's *feature* view; it does not need
      to (and should not, for RL) repeat the metric names.
    * METRIC_ERROR sentinels are substituted with ``rewards.metric_failure``,
      same as the observation path, so the reward function never sees the
      sentinel.
    """

    def _make_env(
        self,
        base_tr: TestRun,
        *,
        agent_metrics: list[str],
        reward_fn_name: str = "identity",
        metric_failure: float = -1.0,
    ) -> CloudAIGymEnv:
        base_tr.test.agent_metrics = agent_metrics
        base_tr.test.agent_reward_function = reward_fn_name
        return CloudAIGymEnv(
            test_run=base_tr,
            runner=MagicMock(),
            rewards=RewardOverrides(metric_failure=metric_failure),
        )

    def test_no_metrics_passes_observation_through(self, base_tr: TestRun) -> None:
        """Empty ``agent_metrics`` ⇒ no concat ⇒ legacy delegation."""
        env = self._make_env(base_tr, agent_metrics=[])
        assert env.compute_reward([42.0]) == 42.0

    def test_metric_failure_substitutes_for_unresolved_metric(self, base_tr: TestRun) -> None:
        """No metric_reporter for ``"default"`` (mocked runner) ⇒ METRIC_ERROR
        ⇒ replaced by ``metric_failure`` before reaching the reward fn."""
        env = self._make_env(base_tr, agent_metrics=["default"], metric_failure=-7.5)
        assert env.compute_reward([0.42]) == -7.5  # identity reads slot 0

    def test_metric_value_resolved_from_env_params(self, base_tr: TestRun) -> None:
        """When a metric name matches a declared env_param, ``get_metric_value``
        returns the current trial's sampled value — same path the observation
        layer uses, but the reward gets it directly without going through obs."""
        base_tr.test.env_params = {"drop_rate": EnvParamSpec(values=[0.0, 0.001])}
        base_tr.current_env_params = {"drop_rate": 0.001}
        env = self._make_env(base_tr, agent_metrics=["drop_rate"])
        assert env.compute_reward([]) == pytest.approx(0.001)

    def test_metric_prepends_then_observation(self, base_tr: TestRun) -> None:
        """Reward fn receives ``[*metric_values, *observation]`` so the
        documented "slot 0 is the first agent_metric" contract holds."""
        base_tr.test.env_params = {"drop_rate": EnvParamSpec(values=[0.0, 0.001])}
        base_tr.current_env_params = {"drop_rate": 0.001}
        env = self._make_env(base_tr, agent_metrics=["drop_rate"], reward_fn_name="negative")
        # negative reads slot 0 → -drop_rate, ignoring observation entirely.
        assert env.compute_reward([99.0, 99.0]) == pytest.approx(-0.001)

    def test_observation_only_used_when_metrics_exhausted(self, base_tr: TestRun) -> None:
        """A reward fn that reads slot N can pick up observation entries when
        N exceeds the metric count. Locks in the concat ordering."""

        # Custom reward fn that reads slot 1 (second element of concat list).
        from cloudai._core.registry import Registry

        registry = Registry()
        registry.update_reward_function("__test_slot1", lambda obs: float(obs[1]) if len(obs) > 1 else 0.0)
        try:
            base_tr.test.env_params = {"drop_rate": EnvParamSpec(values=[0.0, 0.001])}
            base_tr.current_env_params = {"drop_rate": 0.001}
            env = self._make_env(base_tr, agent_metrics=["drop_rate"], reward_fn_name="__test_slot1")
            # Concat: [drop_rate=0.001] + [73.2] → slot 1 = 73.2 (the obs value).
            assert env.compute_reward([73.2]) == pytest.approx(73.2)
        finally:
            del registry.reward_functions_map["__test_slot1"]

    def test_backward_compat_when_observation_includes_metric(self, base_tr: TestRun) -> None:
        """Legacy TOMLs duplicate the metric in ``agent_observation`` (e.g.
        ``["bus_bw", "drop_rate"]`` with metrics=``["bus_bw"]``). Concat means
        the metric value is duplicated in the list, but the reward fn still
        reads slot 0 → correct value. No silent corruption."""
        base_tr.test.env_params = {"drop_rate": EnvParamSpec(values=[0.0, 0.001])}
        base_tr.current_env_params = {"drop_rate": 0.001}
        env = self._make_env(base_tr, agent_metrics=["drop_rate"])  # identity reads slot 0
        # Legacy obs would include drop_rate at some slot; identity ignores it.
        assert env.compute_reward([0.001, 99.0]) == pytest.approx(0.001)


def test_compute_reward_invalid(base_tr: TestRun):
    base_tr.test.agent_reward_function = "nonexistent"

    with pytest.raises(KeyError) as exc_info:
        CloudAIGymEnv(test_run=base_tr, runner=MagicMock(), rewards=RewardOverrides())

    assert "Reward function 'nonexistent' not found" in str(exc_info.value)
    assert (
        "Available functions: ['inverse', 'negative', 'identity', "
        "'ai_dynamo_weighted_normalized', 'ai_dynamo_ratio_normalized', 'ai_dynamo_log_scale']" in str(exc_info.value)
    )


def test_tr_output_path(setup_env: tuple[TestRun, BaseRunner]):
    test_run, runner = setup_env
    test_run.test.cmd_args.data.global_batch_size = 8  # avoid constraint check failure
    env = CloudAIGymEnv(test_run=test_run, runner=runner, rewards=RewardOverrides())
    agent = GridSearchAgent(env, GridSearchAgent.get_config_class()())

    env.test_run.step = 41
    env.reset()
    _, action = agent.select_action()
    env.step(action)

    assert env.test_run.output_path.name == "42", (
        "CloudAIGymEnv.reset() must advance test_run.step (the trial-boundary mutator) "
        "before step() computes output_path; starting at 41, step #42's artifacts must "
        "land in dir '42'."
    )


@pytest.mark.parametrize(
    "rewards, expected_reward",
    [
        pytest.param(RewardOverrides(), -1.0, id="default_penalty"),
        pytest.param(RewardOverrides(constraint_failure=-2.5), -2.5, id="custom_penalty"),
    ],
)
def test_constraint_failure(nemorun: NeMoRunTestDefinition, rewards: RewardOverrides, expected_reward: float):
    tdef = nemorun.model_copy(deep=True)
    tdef.cmd_args.data.global_batch_size = 8
    tdef.agent_metrics = ["default"]
    test_run = TestRun(
        name="constraint_fail_tr",
        test=tdef,
        num_nodes=1,
        nodes=[],
        reports={NeMoRunReportGenerationStrategy},
    )
    runner = MagicMock(spec=BaseRunner)
    runner.system = MagicMock()
    env = CloudAIGymEnv(test_run=test_run, runner=runner, rewards=rewards)

    bad = {"trainer.strategy.context_parallel_size": 3}  # induce constraint failure
    obs, reward, done, info = env.step(bad)

    assert obs == [-1.0]
    assert reward == expected_reward
    assert done is True
    assert info == {}


def test_action_space(nemorun: NeMoRunTestDefinition, setup_env: tuple[TestRun, BaseRunner]):
    tr, _ = setup_env
    nemorun.cmd_args.trainer = Trainer(
        max_steps=[1000, 2000], strategy=TrainerStrategy(tensor_model_parallel_size=[1, 2])
    )
    nemorun.cmd_args.data.micro_batch_size = [1, 2]
    nemorun.extra_env_vars["DSE_VAR"] = ["1", "2"]

    tr.test = nemorun
    tr.num_nodes = [1, 2]

    action_space = tr.param_space

    assert len(action_space) == 5
    assert action_space["data.micro_batch_size"] == nemorun.cmd_args.data.micro_batch_size
    assert action_space["trainer.max_steps"] == nemorun.cmd_args.trainer.max_steps
    assert (
        action_space["trainer.strategy.tensor_model_parallel_size"]
        == nemorun.cmd_args.trainer.strategy.tensor_model_parallel_size
    )
    assert action_space["extra_env_vars.DSE_VAR"] == nemorun.extra_env_vars["DSE_VAR"]
    assert action_space["NUM_NODES"] == tr.num_nodes


def test_action_space_excludes_configured_cmd_arg_prefix(
    nemorun: NeMoRunTestDefinition, setup_env: tuple[TestRun, BaseRunner]
):
    tr, _ = setup_env
    nemorun.cmd_args.trainer = Trainer(
        max_steps=[1000, 2000], strategy=TrainerStrategy(tensor_model_parallel_size=[1, 2])
    )
    nemorun.dse_excluded_args = ["cmd_args.trainer.strategy"]
    tr.test = nemorun

    action_space = tr.param_space

    assert action_space["trainer.max_steps"] == [1000, 2000]
    assert "trainer.strategy.tensor_model_parallel_size" not in action_space


@pytest.mark.parametrize("num_nodes", (1, [1, 2], [3]))
def test_all_combinations(nemorun: NeMoRunTestDefinition, setup_env: tuple[TestRun, BaseRunner], num_nodes: int):
    tr, _ = setup_env
    nemorun.cmd_args.trainer = Trainer(max_steps=[1000], strategy=TrainerStrategy(tensor_model_parallel_size=[1, 2]))
    nemorun.extra_env_vars["DSE_VAR"] = ["1", "2", "3"]
    tr.test = nemorun
    tr.num_nodes = num_nodes

    expected_num_combinations = 6
    if isinstance(num_nodes, list):
        expected_num_combinations *= len(num_nodes)

    real_combinations = tr.all_combinations
    assert len(real_combinations) == expected_num_combinations
    _combinations = [
        {"trainer.max_steps": 1000, "trainer.strategy.tensor_model_parallel_size": 1, "extra_env_vars.DSE_VAR": "1"},
        {"trainer.max_steps": 1000, "trainer.strategy.tensor_model_parallel_size": 1, "extra_env_vars.DSE_VAR": "1"},
        {"trainer.max_steps": 1000, "trainer.strategy.tensor_model_parallel_size": 1, "extra_env_vars.DSE_VAR": "2"},
        {"trainer.max_steps": 1000, "trainer.strategy.tensor_model_parallel_size": 1, "extra_env_vars.DSE_VAR": "2"},
        {"trainer.max_steps": 1000, "trainer.strategy.tensor_model_parallel_size": 1, "extra_env_vars.DSE_VAR": "3"},
        {"trainer.max_steps": 1000, "trainer.strategy.tensor_model_parallel_size": 1, "extra_env_vars.DSE_VAR": "3"},
        {"trainer.max_steps": 1000, "trainer.strategy.tensor_model_parallel_size": 2, "extra_env_vars.DSE_VAR": "1"},
        {"trainer.max_steps": 1000, "trainer.strategy.tensor_model_parallel_size": 2, "extra_env_vars.DSE_VAR": "1"},
        {"trainer.max_steps": 1000, "trainer.strategy.tensor_model_parallel_size": 2, "extra_env_vars.DSE_VAR": "2"},
        {"trainer.max_steps": 1000, "trainer.strategy.tensor_model_parallel_size": 2, "extra_env_vars.DSE_VAR": "2"},
        {"trainer.max_steps": 1000, "trainer.strategy.tensor_model_parallel_size": 2, "extra_env_vars.DSE_VAR": "3"},
        {"trainer.max_steps": 1000, "trainer.strategy.tensor_model_parallel_size": 2, "extra_env_vars.DSE_VAR": "3"},
    ]
    expected_combinations = []
    for param_set in _combinations:
        if isinstance(num_nodes, list):
            for nnodes in num_nodes:
                expected_combinations.append(param_set | {"NUM_NODES": nnodes})
        else:
            expected_combinations.append(param_set)

    for expected in expected_combinations:
        assert expected in real_combinations, f"Expected {expected} in all_combinations"


def test_all_combinations_non_dse(nemorun: NeMoRunTestDefinition, setup_env: tuple[TestRun, BaseRunner]):
    tr, _ = setup_env
    tr.test = nemorun
    assert len(tr.all_combinations) == 0


def test_all_combinations_non_dse_but_with_space(nemorun: NeMoRunTestDefinition, setup_env: tuple[TestRun, Runner]):
    tr, _ = setup_env
    tr.test = nemorun
    with patch.object(type(tr.test), "is_dse_job", new_callable=PropertyMock(return_value=True)):
        assert len(tr.all_combinations) == 0


def test_all_combinations_dse_on_num_nodes(nemorun: NeMoRunTestDefinition, setup_env: tuple[TestRun, Runner]):
    tr, _ = setup_env
    tr.test = NeMoRunTestDefinition(
        name="NemoModel",
        description="Nemo Model",
        test_template_name="nemo_template",
        cmd_args=NeMoRunCmdArgs(docker_image_url="https://docker/url", task="some_task", recipe_name="some_recipe"),
    )
    tr.num_nodes = [1, 2]
    assert len(tr.all_combinations) == 2


class _StubInner(BaseModel):
    """Nested cmd_args group exercising the recursive walk in ``_collect_action_spaces``."""

    knob: Union[int, list[int], ContinuousSpace] = 0


class _StubCmdArgs(CmdArgs):
    """Minimal CmdArgs accepting both list-typed and ContinuousSpace tunables."""

    threshold: Union[int, list[int], ContinuousSpace] = 0
    mode: Union[str, list[str]] = "default"
    inner: _StubInner = Field(default_factory=_StubInner)


class _StubTestDefinition(TestDefinition):
    """TestDefinition with stub cmd_args and a no-op metric reporter."""

    cmd_args: _StubCmdArgs = Field(default_factory=_StubCmdArgs)


def _stub_tr(cmd_args: _StubCmdArgs) -> TestRun:
    tdef = _StubTestDefinition(name="stub", description="stub", test_template_name="stub_template", cmd_args=cmd_args)
    return TestRun(name="stub_run", test=tdef, num_nodes=1, nodes=[])


def test_param_space_surfaces_continuous_space_as_single_tunable() -> None:
    """A ContinuousSpace value lives in cmd_args as a single tunable, not as exploded sub-keys.

    Without the ``_collect_action_spaces`` pre-pass, ``flatten_dict(model_dump())`` would
    surface ``threshold.low`` / ``.high`` / ``.dtype`` as three independent scalar
    entries — none of which are ``list``-typed, so they'd be filtered out and the
    tunable would silently disappear from ``param_space``.
    """
    space = ContinuousSpace(low=0.0, high=200.0, dtype="int")
    tr = _stub_tr(_StubCmdArgs(threshold=space))

    action_space = tr.param_space

    assert "threshold" in action_space, (
        f"ContinuousSpace must appear as a single tunable; got keys={sorted(action_space)}"
    )
    assert action_space["threshold"] is space
    for exploded in ("threshold.low", "threshold.high", "threshold.dtype"):
        assert exploded not in action_space, f"ContinuousSpace must not be flattened into {exploded}"


def test_param_space_mixes_list_and_continuous_space_tunables() -> None:
    """Discrete-list and ContinuousSpace tunables co-exist in the same param_space."""
    space = ContinuousSpace(low=0.0, high=200.0, dtype="int")
    tr = _stub_tr(_StubCmdArgs(threshold=space, mode=["a", "b"]))

    action_space = tr.param_space

    assert action_space["threshold"] is space
    assert action_space["mode"] == ["a", "b"]


def test_param_space_finds_nested_action_space() -> None:
    """``_collect_action_spaces`` recurses into nested cmd_args groups."""
    space = ContinuousSpace(low=0.0, high=10.0, dtype="int")
    tr = _stub_tr(_StubCmdArgs(inner=_StubInner(knob=space)))

    action_space = tr.param_space

    assert action_space["inner.knob"] is space
    for exploded in ("inner.knob.low", "inner.knob.high", "inner.knob.dtype"):
        assert exploded not in action_space


def test_all_combinations_rejects_continuous_space() -> None:
    """Continuous action spaces cannot be enumerated; grid-search must fail loudly."""
    space = ContinuousSpace(low=0.0, high=200.0, dtype="int")
    tr = _stub_tr(_StubCmdArgs(threshold=space, mode=["a", "b"]))

    with pytest.raises(TypeError, match="continuous action spaces"):
        _ = tr.all_combinations


def test_continuous_space_rejects_invalid_bounds() -> None:
    """``low >= high`` is a config bug; fail at construction with a clear message."""
    with pytest.raises(ValueError, match="low < high"):
        ContinuousSpace(low=10.0, high=10.0)
    with pytest.raises(ValueError, match="low < high"):
        ContinuousSpace(low=10.0, high=5.0)


def _stub_tdef(cmd_args: _StubCmdArgs) -> _StubTestDefinition:
    return _StubTestDefinition(name="stub", description="stub", test_template_name="stub_template", cmd_args=cmd_args)


def test_is_dse_job_detects_continuous_space() -> None:
    """Workloads with only a ContinuousSpace tunable (no list-typed args) are still DSE jobs.

    ``model_dump()`` lowers ContinuousSpace to a plain dict, so the legacy
    ``check_dict`` walk over ``cmd_args_dict`` no longer surfaces it. Without
    the typed-cmd_args fallback, the runner mis-classifies this as a non-DSE
    job, hits the standalone path, and the workload errors at command-emit
    time because no agent has resolved the action.
    """
    space = ContinuousSpace(low=0.0, high=200.0, dtype="int")
    tdef = _stub_tdef(_StubCmdArgs(threshold=space))
    assert tdef.is_dse_job, "ContinuousSpace must mark the test as a DSE job"


def test_is_dse_job_false_when_only_fixed_scalars() -> None:
    """Sanity: a cmd_args with only fixed scalars (no list, no ActionSpace) is not DSE."""
    tdef = _stub_tdef(_StubCmdArgs(threshold=42, mode="fixed"))
    assert not tdef.is_dse_job


@pytest.mark.parametrize("num_nodes", (1, [1, 2], [3]))
def test_params_set(setup_env: tuple[TestRun, Runner], num_nodes: int):
    tr, _ = setup_env
    tr.num_nodes = num_nodes
    assert len(tr.all_combinations) > 1
    for action in tr.all_combinations:
        new_tr = tr.apply_params_set(action)
        cmd_args = flatten_dict(new_tr.test.cmd_args.model_dump())
        for key, value in action.items():
            if key.startswith("extra_env_vars."):
                assert new_tr.test.extra_env_vars[key[len("extra_env_vars.") :]] == value
            elif key == "NUM_NODES":
                assert new_tr.num_nodes == value
            else:
                assert cmd_args[key] == value


def test_params_set_validated(setup_env: tuple[TestRun, Runner], nemorun: NeMoRunTestDefinition):
    tr, _ = setup_env
    nemorun.cmd_args.trainer = Trainer(max_steps=[1000])
    tr.test = nemorun
    action_space = tr.param_space
    action_space["trainer.max_steps"] = "invalid"

    with pytest.raises(UserWarning) as excinfo:
        tr.apply_params_set(action_space)

    assert excinfo.type is UserWarning
    assert "Pydantic serializer warnings:" in str(excinfo.value)
    assert "serialized value may not be as expected" in str(excinfo.value)
    assert "input_value='invalid'" in str(excinfo.value)


def test_apply_params_set__preserves_installables_state(setup_env: tuple[TestRun, Runner], tmp_path: Path):
    tr, _ = setup_env
    tr.test = NIXLBenchTestDefinition(
        name="NIXLBench",
        description="NIXL Bench",
        test_template_name="NIXLBench",
        cmd_args=NIXLBenchCmdArgs(
            docker_image_url="https://docker/url",
            path_to_benchmark="https://benchmark/path",
        ),
    )
    tr.test.docker_image.installed_path = tmp_path

    new_tr = tr.apply_params_set({"backend": "VRAM"})

    upd_tdef = cast(NIXLBenchTestDefinition, new_tr.test)

    assert upd_tdef.docker_image.installed_path == tmp_path


@pytest.mark.parametrize(
    ("trajectory", "current_iteration", "action", "expected_step"),
    [
        ({}, 0, {"x": 1}, None),
        ({0: [TrajectoryEntry(1, {"x": 1}, 1, [1])]}, 0, {"x": 1}, 1),
        ({0: [TrajectoryEntry(1, {"x": 1.0}, 1, [1])]}, 0, {"x": 1}, None),
        (
            {
                0: [
                    TrajectoryEntry(1, {"x": 1.0}, 1, [1]),
                    TrajectoryEntry(2, {"x": 1}, 1, [1]),
                ]
            },
            0,
            {"x": 1},
            2,
        ),
        ({0: [TrajectoryEntry(1, {"x": 1}, 1, [1])]}, 1, {"x": 1}, None),
        ({1: [TrajectoryEntry(3, {"x": 1}, 1, [1])]}, 1, {"x": 1}, 3),
    ],
)
def test_get_cached_trajectory_result(
    base_tr: TestRun,
    tmp_path: Path,
    trajectory: dict[int, list[TrajectoryEntry]],
    current_iteration: int,
    action: dict[str, object],
    expected_step: int | None,
) -> None:
    runner = MagicMock()
    runner.scenario_root = tmp_path / "scenario"
    runner.system = MagicMock()
    runner.test_scenario = MagicMock(test_runs=[])
    runner.jobs = {}
    runner.testrun_to_job_map = {}
    runner.get_job_output_path.return_value = tmp_path / "scenario" / base_tr.name / "0" / "7"

    env = CloudAIGymEnv(test_run=base_tr, runner=runner, rewards=RewardOverrides())
    env.test_run.current_iteration = current_iteration
    env.trajectory = trajectory

    actual = env.get_cached_trajectory_result(action)
    if actual is None:
        assert expected_step is None
    else:
        assert actual.step == expected_step


def test_cached_step_appends_trajectory_row(nemorun: NeMoRunTestDefinition, tmp_path: Path) -> None:
    """Cache hits must still append a row to trajectory.csv so the visible step list matches agent_steps."""
    tdef = nemorun.model_copy(deep=True)
    tdef.cmd_args.data.global_batch_size = 8
    tdef.agent_metrics = ["default"]
    test_run = TestRun(
        name="cache_tr",
        test=tdef,
        num_nodes=1,
        nodes=[],
        reports={NeMoRunReportGenerationStrategy},
    )

    runner = MagicMock(spec=BaseRunner)
    runner.scenario_root = tmp_path / "scenario"
    runner.system = MagicMock()

    env = CloudAIGymEnv(test_run=test_run, runner=runner, rewards=RewardOverrides())
    cached_action = {"trainer.max_steps": 1000}
    env.test_run.current_iteration = 0
    env.trajectory = {0: [TrajectoryEntry(step=1, action=cached_action, reward=0.42, observation=[0.84])]}

    env.test_run.step = 4
    env.reset()
    obs, reward, done, _info = env.step(cached_action)

    runner.run.assert_not_called()
    assert reward == 0.42
    assert obs == [0.84]
    assert done is False
    rows = env.trajectory[0]
    assert len(rows) == 2
    assert rows[-1].step == 5, (
        "CloudAIGymEnv.reset() advances test_run.step at the trial boundary; the cached row "
        "appended in step() must carry the advanced trial index, not the pre-reset value."
    )
    assert rows[-1].reward == 0.42
    assert rows[-1].action == cached_action

    csv_path = env.trajectory_file_path
    assert csv_path.exists()
    contents = csv_path.read_text().strip().splitlines()
    assert contents[0] == "step,action,reward,observation"
    assert contents[-1].startswith("5,")


def test_trajectory_csv_records_int_action_verbatim(nemorun: NeMoRunTestDefinition, tmp_path: Path) -> None:
    """trajectory.csv records the exact int action handed to ``step()`` — no float upcast.

    Continuous action spaces are rounded at the cloudaix adapter boundary
    (``GymnasiumAdapter._decode_continuous``). After quantization, ``CloudAIGymEnv.step()``
    receives a Python int (e.g. ``{"prt_ooo_threshold": 47}``); the CSV row
    must mirror that — no ``47.0``, no policy-side float like ``47.34``. This
    pins the actuator-quantization invariant from the cloudai side.
    """
    tdef = nemorun.model_copy(deep=True)
    tdef.cmd_args.data.global_batch_size = 8
    tdef.agent_metrics = ["default"]
    test_run = TestRun(
        name="int_action_tr",
        test=tdef,
        num_nodes=1,
        nodes=[],
        reports={NeMoRunReportGenerationStrategy},
    )

    runner = MagicMock(spec=BaseRunner)
    runner.scenario_root = tmp_path / "scenario"
    runner.system = MagicMock()

    env = CloudAIGymEnv(test_run=test_run, runner=runner, rewards=RewardOverrides())
    int_action = {"trainer.max_steps": 47}
    env.test_run.current_iteration = 0
    env.trajectory = {0: [TrajectoryEntry(step=1, action=int_action, reward=0.42, observation=[0.84])]}

    env.test_run.step = 0
    env.reset()
    env.step(int_action)

    csv_path = env.trajectory_file_path
    assert csv_path.exists()
    last_row = csv_path.read_text().strip().splitlines()[-1]
    assert "{'trainer.max_steps': 47}" in last_row, f"row should contain int action; got: {last_row}"
    assert "47.0" not in last_row, f"int action must not be upcast to float; got: {last_row}"
    assert "47.3" not in last_row, f"trajectory must not leak the policy's pre-quantized float; got: {last_row}"


def _seed_cached_entry_with_env_params(
    env: CloudAIGymEnv, action: dict[str, object], env_params: dict[str, object]
) -> None:
    """Seed env.trajectory with one entry carrying the given env_params."""
    entry = TrajectoryEntry(step=1, action=action, reward=0.5, observation=[100.0], env_params=env_params)
    env.test_run.current_iteration = 0
    env.trajectory = {0: [entry]}


def test_cache_miss_when_env_params_differ(base_tr: TestRun, tmp_path: Path) -> None:
    """Cache MUST miss when env_params differ, even if action is identical.

    Without this property the agent receives stale rewards on every cache hit
    under domain randomization. PPO/DQN/BO all silently train on labels that
    do not correspond to the env they were nominally generated under.
    """
    runner = MagicMock()
    runner.scenario_root = tmp_path / "scenario"
    runner.system = MagicMock()
    runner.test_scenario = MagicMock(test_runs=[])
    runner.jobs = {}
    runner.testrun_to_job_map = {}

    env = CloudAIGymEnv(test_run=base_tr, runner=runner, rewards=RewardOverrides())
    _seed_cached_entry_with_env_params(env, {"x": 10}, env_params={"drop_rate": 0.001})

    env.test_run.current_env_params = {"drop_rate": 0.01}

    assert env.get_cached_trajectory_result({"x": 10}) is None, (
        "Cache must include env_params in its key. The current implementation "
        "keys on action alone, so trials repeating the same action under a "
        "different env_params sample receive a stale cached reward. See "
        "env-params-cloudai-corpus-plan.md."
    )


def test_cache_hit_when_action_and_env_params_match(base_tr: TestRun, tmp_path: Path) -> None:
    """Same action AND same env_params must still HIT the cache."""
    runner = MagicMock()
    runner.scenario_root = tmp_path / "scenario"
    runner.system = MagicMock()
    runner.test_scenario = MagicMock(test_runs=[])
    runner.jobs = {}
    runner.testrun_to_job_map = {}

    env = CloudAIGymEnv(test_run=base_tr, runner=runner, rewards=RewardOverrides())
    _seed_cached_entry_with_env_params(env, {"x": 10}, env_params={"drop_rate": 0.001})

    env.test_run.current_env_params = {"drop_rate": 0.001}

    result = env.get_cached_trajectory_result({"x": 10})
    assert result is not None and result.step == 1


def test_cache_hit_when_neither_has_env_params(base_tr: TestRun, tmp_path: Path) -> None:
    """Workloads without env_params behave exactly as today (back-compat)."""
    runner = MagicMock()
    runner.scenario_root = tmp_path / "scenario"
    runner.system = MagicMock()
    runner.test_scenario = MagicMock(test_runs=[])
    runner.jobs = {}
    runner.testrun_to_job_map = {}

    env = CloudAIGymEnv(test_run=base_tr, runner=runner, rewards=RewardOverrides())
    env.test_run.current_iteration = 0
    env.trajectory = {0: [TrajectoryEntry(step=1, action={"x": 10}, reward=0.5, observation=[100.0])]}
    # Note: neither the cached entry nor test_run carries env_params -> existing behavior.

    result = env.get_cached_trajectory_result({"x": 10})
    assert result is not None and result.step == 1


def test_step_reruns_workload_when_env_params_change(nemorun: NeMoRunTestDefinition, tmp_path: Path) -> None:
    """Integration: env.step() with same action but different env_params re-runs the workload.

    Counterpart to test_cache_miss_when_env_params_differ but exercising the
    full step() flow: increment_step -> apply_params_set -> cache lookup ->
    runner.run() -> write_trajectory.
    """
    tdef = nemorun.model_copy(deep=True)
    tdef.cmd_args.data.global_batch_size = 8
    tdef.agent_metrics = ["default"]
    test_run = TestRun(
        name="dr_tr",
        test=tdef,
        num_nodes=1,
        nodes=[],
        output_path=tmp_path / "out" / "dr_tr" / "0",
        reports={NeMoRunReportGenerationStrategy},
    )
    test_scenario = TestScenario(name="dr_scenario", test_runs=[test_run])

    runner = MagicMock(spec=BaseRunner)
    runner.scenario_root = tmp_path / "scenario"
    runner.system = MagicMock()
    runner.test_scenario = test_scenario
    runner.jobs = {}
    runner.testrun_to_job_map = {}
    runner.shutting_down = False
    runner.get_job_output_path.return_value = test_run.output_path

    env = CloudAIGymEnv(test_run=test_run, runner=runner, rewards=RewardOverrides())
    action = {"trainer.max_steps": 1000}
    fake_obs = iter([[100.0], [50.0]])

    with patch.object(env, "get_observation", side_effect=lambda _action: next(fake_obs)):
        env.test_run.step = 0
        env.test_run.current_env_params = {"drop_rate": 0.001}
        obs1, _r1, *_ = env.step(action)

        env.test_run.current_env_params = {"drop_rate": 0.01}
        obs2, _r2, *_ = env.step(action)

    assert runner.run.call_count == 2, (
        "Different env_params between two env.step() calls with the same action "
        "must trigger a workload re-run; the cache lookup must miss."
    )
    assert obs1 != obs2, "fresh workload run should produce a fresh observation"


def test_env_csv_is_step_aligned_with_trajectory(nemorun: NeMoRunTestDefinition, tmp_path: Path) -> None:
    """env.csv must have exactly one row per env.step() call, with steps aligned 1:1 to trajectory.csv.

    This pins the corpus-friendly contract: a downstream consumer can
    ``pd.merge(traj, env, on="step")`` without losing rows on either side,
    independent of whether the trial hit the trajectory cache.
    """
    tdef = nemorun.model_copy(deep=True)
    tdef.cmd_args.data.global_batch_size = 8
    tdef.agent_metrics = ["default"]
    tdef.env_params = {"drop_rate": EnvParamSpec(values=[0.0, 0.001, 0.01])}
    tdef.agent_config = {"random_seed": 42}

    test_run = TestRun(
        name="dr_tr",
        test=tdef,
        num_nodes=1,
        nodes=[],
        output_path=tmp_path / "out" / "dr_tr" / "0",
        reports={NeMoRunReportGenerationStrategy},
    )
    test_scenario = TestScenario(name="dr_scenario", test_runs=[test_run])

    runner = MagicMock(spec=BaseRunner)
    runner.scenario_root = tmp_path / "scenario"
    runner.system = MagicMock()
    runner.test_scenario = test_scenario
    runner.jobs, runner.testrun_to_job_map, runner.shutting_down = {}, {}, False
    runner.get_job_output_path.return_value = test_run.output_path

    env = CloudAIGymEnv(test_run=test_run, runner=runner, rewards=RewardOverrides())
    action_a, action_b = {"trainer.max_steps": 1000}, {"trainer.max_steps": 2000}
    fake_obs = iter([[100.0], [50.0], [25.0]])

    with patch.object(env, "get_observation", side_effect=lambda _action: next(fake_obs)):
        env.test_run.step = 0
        for action in (action_a, action_b, action_a):
            env.reset()
            env.step(action)

    env_csv = env._env_csv_path()
    traj_csv = env.trajectory_file_path
    assert env_csv.exists(), "env.csv must be written when env_params is declared"

    env_steps = [int(line.split(",", 1)[0]) for line in env_csv.read_text().strip().splitlines()[1:]]
    traj_steps = [int(line.split(",", 1)[0]) for line in traj_csv.read_text().strip().splitlines()[1:]]
    assert env_steps == traj_steps == [1, 2, 3], (
        f"step columns must align 1:1 across env.csv ({env_steps}) and trajectory.csv ({traj_steps})"
    )


def test_step_cache_hit_with_declared_env_params_still_writes_env_csv(
    nemorun: NeMoRunTestDefinition, tmp_path: Path
) -> None:
    """End-to-end: cache HIT under observer-driven env_params still records env.csv.

    This is the contract that was broken before the fix. With env_params
    declared on the TestDefinition, ``CloudAIGymEnv`` must fire its
    EnvParamsObserver *before* the cache lookup so every trial - hit or
    miss - appends to env.csv and stays step-aligned with trajectory.csv.
    Asserts: (a) the workload is NOT re-run (cache short-circuit), (b)
    env.csv gains a row, (c) trajectory.csv gains a row carrying the
    sampled env_params.
    """
    import random as _random

    tdef = nemorun.model_copy(deep=True)
    tdef.cmd_args.data.global_batch_size = 8
    tdef.agent_metrics = ["default"]
    tdef.env_params = {"drop_rate": EnvParamSpec(values=[0.0, 0.001, 0.01])}
    tdef.agent_config = {"random_seed": 42}

    test_run = TestRun(
        name="dr_tr",
        test=tdef,
        num_nodes=1,
        nodes=[],
        output_path=tmp_path / "out" / "dr_tr" / "0",
        reports={NeMoRunReportGenerationStrategy},
    )
    runner = MagicMock(spec=BaseRunner)
    runner.scenario_root = tmp_path / "scenario"
    runner.system = MagicMock()
    runner.test_scenario = TestScenario(name="dr_scenario", test_runs=[test_run])
    runner.jobs, runner.testrun_to_job_map, runner.shutting_down = {}, {}, False
    runner.get_job_output_path.return_value = test_run.output_path

    env = CloudAIGymEnv(test_run=test_run, runner=runner, rewards=RewardOverrides())
    assert env.observers, "TestDefinition.env_params declared -> observer must be built"

    expected_sample = {"drop_rate": _random.Random("42:drop_rate:1").choice([0.0, 0.001, 0.01])}
    action = {"trainer.max_steps": 1000}
    env.test_run.current_iteration = 0
    env.trajectory = {
        0: [TrajectoryEntry(step=0, action=action, reward=0.42, observation=[0.84], env_params=expected_sample)]
    }
    env.test_run.step = 0
    env.reset()

    with patch.object(env, "get_observation", side_effect=AssertionError("cache miss path must not run")):
        obs, reward, _done, _info = env.step(action)

    runner.run.assert_not_called()
    assert reward == 0.42 and obs == [0.84]

    env_csv = env._env_csv_path()
    assert env_csv.exists(), "cache HIT must NOT skip the observer; env.csv must record the trial"
    env_rows = env_csv.read_text().strip().splitlines()
    assert env_rows[0] == "step,env"
    assert env_rows[1].startswith("1,"), f"expected step 1 row in env.csv, got {env_rows[1]!r}"

    traj_rows = env.trajectory[0]
    assert len(traj_rows) == 2 and traj_rows[-1].env_params == expected_sample, (
        "cache-hit trajectory entry must record the per-trial env_params sample"
    )


def test_no_env_csv_when_env_params_not_declared(nemorun: NeMoRunTestDefinition, tmp_path: Path) -> None:
    """Workloads without [env_params.*] pay zero overhead: no observer, no env.csv."""
    tdef = nemorun.model_copy(deep=True)
    tdef.cmd_args.data.global_batch_size = 8
    test_run = TestRun(
        name="plain_tr",
        test=tdef,
        num_nodes=1,
        nodes=[],
        output_path=tmp_path / "out" / "plain_tr" / "0",
        reports={NeMoRunReportGenerationStrategy},
    )
    runner = MagicMock(spec=BaseRunner)
    runner.scenario_root = tmp_path / "scenario"
    runner.system = MagicMock()

    env = CloudAIGymEnv(test_run=test_run, runner=runner, rewards=RewardOverrides())

    assert env.observers == [], "no env_params declared -> no per-step observers"
    assert not env._env_csv_path().exists()


def test_reset_exposes_env_param_in_observation(nemorun: NeMoRunTestDefinition, tmp_path: Path) -> None:
    """Contextual-bandit pipeline: reset() must surface the trial's sampled env_param value.

    The policy in an RL contextual bandit reads obs from ``reset()`` and
    chooses an action *before* ``step()`` runs. For domain-randomized
    parameters declared in ``agent_observation``, the value must therefore
    be sampled and visible in the obs vector returned by reset(), not later.
    Asserts: (a) the obs slot for ``drop_rate`` equals the deterministic
    sample for trial 1 under seed 42, and (b) the measured-metric slot is
    the metric_failure sentinel (no sim has run yet).
    """
    import random as _random

    tdef = nemorun.model_copy(deep=True)
    tdef.cmd_args.data.global_batch_size = 8
    tdef.agent_metrics = ["default"]
    tdef.agent_observation = ["drop_rate", "default"]
    tdef.env_params = {"drop_rate": EnvParamSpec(values=[0.0, 0.001, 0.01])}
    tdef.agent_config = {"random_seed": 42}

    test_run = TestRun(
        name="ctx_tr",
        test=tdef,
        num_nodes=1,
        nodes=[],
        output_path=tmp_path / "out" / "ctx_tr" / "0",
        reports={NeMoRunReportGenerationStrategy},
    )
    runner = MagicMock(spec=BaseRunner)
    runner.scenario_root = tmp_path / "scenario"
    runner.system = MagicMock()
    runner.test_scenario = TestScenario(name="ctx_scenario", test_runs=[test_run])

    env = CloudAIGymEnv(test_run=test_run, runner=runner, rewards=RewardOverrides(metric_failure=-7.5))
    expected_drop_rate = _random.Random("42:drop_rate:1").choice([0.0, 0.001, 0.01])

    obs, info = env.reset()

    assert env.test_run.step == 1, "reset() advances test_run.step at the trial boundary"
    assert env.test_run.current_env_params == {"drop_rate": expected_drop_rate}, (
        "reset() must fire env_params observers so current_env_params is populated before obs is built"
    )
    assert obs == [expected_drop_rate, -7.5], (
        f"obs at reset must expose the sampled drop_rate in slot 0 and use metric_failure (-7.5) "
        f"for the unmeasured metric in slot 1; got {obs}"
    )
    assert info == {}


def test_observation_at_step_combines_metric_and_env_param(nemorun: NeMoRunTestDefinition, tmp_path: Path) -> None:
    """End-to-end: step() returns obs combining a measured metric with a sampled env_param.

    Pins the contract that ``agent_observation`` can mix names that resolve
    via the post-run report (measured metric ``default``) and names that
    resolve via ``current_env_params`` (declared env_param ``drop_rate``).
    The metric-resolution path is patched so we can drive the test
    deterministically without standing up a runner.
    """
    import random as _random

    tdef = nemorun.model_copy(deep=True)
    tdef.cmd_args.data.global_batch_size = 8
    tdef.agent_metrics = ["default"]
    tdef.agent_observation = ["default", "drop_rate"]
    tdef.env_params = {"drop_rate": EnvParamSpec(values=[0.0, 0.001, 0.01])}
    tdef.agent_config = {"random_seed": 42}

    test_run = TestRun(
        name="ctx_tr",
        test=tdef,
        num_nodes=1,
        nodes=[],
        output_path=tmp_path / "out" / "ctx_tr" / "0",
        reports={NeMoRunReportGenerationStrategy},
    )
    runner = MagicMock(spec=BaseRunner)
    runner.scenario_root = tmp_path / "scenario"
    runner.system = MagicMock()
    runner.test_scenario = TestScenario(name="ctx_scenario", test_runs=[test_run])

    env = CloudAIGymEnv(test_run=test_run, runner=runner, rewards=RewardOverrides())
    expected_drop_rate = _random.Random("42:drop_rate:1").choice([0.0, 0.001, 0.01])
    measured_bus_bw = 84.0

    env.reset()

    action = {"trainer.max_steps": 1000}
    env.trajectory = {
        0: [
            TrajectoryEntry(
                step=1,
                action=action,
                reward=0.42,
                observation=[measured_bus_bw, expected_drop_rate],
                env_params={"drop_rate": expected_drop_rate},
            )
        ]
    }

    obs, _reward, _done, _info = env.step(action)

    assert obs == [measured_bus_bw, expected_drop_rate], (
        f"obs at step must carry the measured metric in slot 0 and the trial's drop_rate in slot 1; got {obs}"
    )


def test_raw_zero_drop_rate_preserved_in_trajectory_while_encoded_is_finite(
    nemorun: NeMoRunTestDefinition, tmp_path: Path
) -> None:
    """C1: the raw obs (and trajectory.csv) carries ``drop_rate == 0.0`` unmodified.

    The encoded Dict leaf ``[1.0, log10(floor)]`` is *derived* from the raw
    value and used only by the policy. The reward and trajectory must see the
    raw ``0.0`` — never the log-encoded form — so logs stay lossless and the
    reward is not computed on transformed values. Pins the raw-vs-encoded
    separation that contract C1 declares.
    """
    import csv as _csv
    import math

    tdef = nemorun.model_copy(deep=True)
    tdef.cmd_args.data.global_batch_size = 8
    tdef.agent_metrics = ["default"]
    tdef.agent_observation = ["drop_rate"]
    # zero_prob=1.0 forces a deterministic exact-0.0 draw (the dangerous atom).
    tdef.env_params = {
        "drop_rate": EnvParamSpec(
            sampling={"type": "loguniform", "low": 1e-4, "high": 1e-1, "zero_prob": 1.0},
            encoding={"type": "log"},
        )
    }
    tdef.agent_config = {"random_seed": 42}

    test_run = TestRun(
        name="c1_tr",
        test=tdef,
        num_nodes=1,
        nodes=[],
        output_path=tmp_path / "out" / "c1_tr" / "0",
        reports={NeMoRunReportGenerationStrategy},
    )
    runner = MagicMock(spec=BaseRunner)
    runner.scenario_root = tmp_path / "scenario"
    runner.system = MagicMock()
    runner.test_scenario = TestScenario(name="c1_scenario", test_runs=[test_run])

    env = CloudAIGymEnv(test_run=test_run, runner=runner, rewards=RewardOverrides())
    obs, _info = env.reset()

    # Raw side: the sampled value and the obs slot are exactly 0.0 (not encoded).
    assert env.test_run.current_env_params["drop_rate"] == 0.0
    assert obs == [0.0], f"raw obs must carry drop_rate=0.0 unmodified; got {obs}"

    # Encoded side: derived from the same raw 0.0, finite, distinct from the raw.
    encoded = env.encode_observation(obs)
    assert encoded["drop_rate"][0] == 1.0
    assert encoded["drop_rate"][1] == pytest.approx(math.log10(1e-4))
    assert all(math.isfinite(v) for v in encoded["drop_rate"])

    # Trajectory side: the persisted observation cell is the raw [0.0], not the encoded leaf.
    env.write_trajectory(
        TrajectoryEntry(
            step=env.test_run.step,
            action={"trainer.max_steps": 1000},
            reward=0.0,
            observation=obs,
            env_params=dict(env.test_run.current_env_params),
        )
    )
    with open(env.trajectory_file_path, newline="") as f:
        rows = list(_csv.reader(f))
    persisted_observation = rows[1][3]
    assert persisted_observation == "[0.0]", (
        f"trajectory.csv must persist the raw observation [0.0], not the encoded leaf; got {persisted_observation}"
    )


def _write_prior_trajectory(tmp_path: Path, rows: list[tuple[int, dict, float, list]]) -> Path:
    """Write a fake prior-run trajectory.csv and return its path."""
    import csv as _csv

    traj = tmp_path / "prior" / "trajectory.csv"
    traj.parent.mkdir(parents=True, exist_ok=True)
    with traj.open("w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(["step", "action", "reward", "observation"])
        for step, action, reward, observation in rows:
            writer.writerow([step, action, reward, observation])
    return traj


def _write_prior_env(traj_csv: Path, rows: list[tuple[int, dict]]) -> None:
    """Write a fake prior-run env.csv next to ``traj_csv`` (auto-discovery target)."""
    import csv as _csv

    env_path = traj_csv.parent / "env.csv"
    with env_path.open("w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(["step", "env"])
        for step, env in rows:
            writer.writerow([step, env])


def _make_runner(tmp_path: Path) -> MagicMock:
    runner = MagicMock(spec=BaseRunner)
    runner.scenario_root = tmp_path / "scenario"
    runner.system = MagicMock()
    return runner


def test_prime_cache_loads_prior_trajectory(base_tr: TestRun, tmp_path: Path) -> None:
    """A configured prior trajectory.csv pre-populates env.trajectory[0]."""
    traj = _write_prior_trajectory(
        tmp_path,
        [
            (1, {"x": 1}, 0.5, [0.5]),
            (2, {"x": 2}, 0.6, [0.6]),
        ],
    )
    base_tr.test.cache_warm_start_path = traj

    env = CloudAIGymEnv(test_run=base_tr, runner=_make_runner(tmp_path), rewards=RewardOverrides())

    primed = env.trajectory.get(0, [])
    assert [e.step for e in primed] == [1, 2]
    assert primed[0].action == {"x": 1}
    assert primed[0].reward == pytest.approx(0.5)


def test_prime_cache_serves_hit_on_action_and_env_match(base_tr: TestRun, tmp_path: Path) -> None:
    """Primed entries participate in the normal cache lookup with full key semantics."""
    traj = _write_prior_trajectory(tmp_path, [(1, {"x": 1}, 0.5, [0.5])])
    _write_prior_env(traj, [(1, {"drop_rate": 0.0})])
    base_tr.test.cache_warm_start_path = traj

    env = CloudAIGymEnv(test_run=base_tr, runner=_make_runner(tmp_path), rewards=RewardOverrides())
    env.test_run.current_iteration = 0
    env.test_run.current_env_params = {"drop_rate": 0.0}

    hit = env.get_cached_trajectory_result({"x": 1})
    assert hit is not None
    assert hit.reward == pytest.approx(0.5)


def test_prime_cache_misses_when_env_params_differ(base_tr: TestRun, tmp_path: Path) -> None:
    """Primed entries respect the env_params component of the cache key."""
    traj = _write_prior_trajectory(tmp_path, [(1, {"x": 1}, 0.5, [0.5])])
    _write_prior_env(traj, [(1, {"drop_rate": 0.0})])
    base_tr.test.cache_warm_start_path = traj

    env = CloudAIGymEnv(test_run=base_tr, runner=_make_runner(tmp_path), rewards=RewardOverrides())
    env.test_run.current_iteration = 0
    env.test_run.current_env_params = {"drop_rate": 0.01}

    assert env.get_cached_trajectory_result({"x": 1}) is None


def test_prime_cache_skips_failure_rows(base_tr: TestRun, tmp_path: Path) -> None:
    """Constraint-failure rows (-1.0) from the prior run must NOT pre-fill."""
    traj = _write_prior_trajectory(
        tmp_path,
        [
            (1, {"x": 1}, 0.5, [0.5]),
            (2, {"x": 2}, -1.0, [-1.0]),
        ],
    )
    base_tr.test.cache_warm_start_path = traj

    env = CloudAIGymEnv(test_run=base_tr, runner=_make_runner(tmp_path), rewards=RewardOverrides())

    primed = env.trajectory.get(0, [])
    assert [e.step for e in primed] == [1]


def test_prime_cache_no_path_is_noop(base_tr: TestRun, tmp_path: Path) -> None:
    """``cache_warm_start_path=None`` (default) leaves the cache empty without erroring."""
    assert base_tr.test.cache_warm_start_path is None

    env = CloudAIGymEnv(test_run=base_tr, runner=_make_runner(tmp_path), rewards=RewardOverrides())

    assert env.trajectory == {}


def test_prime_cache_missing_path_logs_warning(
    base_tr: TestRun, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A misconfigured path warns and degrades to an empty cache (the run still works)."""
    base_tr.test.cache_warm_start_path = tmp_path / "does_not_exist.csv"

    with caplog.at_level("WARNING"):
        env = CloudAIGymEnv(test_run=base_tr, runner=_make_runner(tmp_path), rewards=RewardOverrides())

    assert env.trajectory == {}
    assert any("cache_warm_start_path" in rec.message for rec in caplog.records)
