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

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Literal

from pydantic import BaseModel, ConfigDict, Field

from .base_gym import BaseGym


class RewardOverrides(BaseModel):
    """Optional reward and observation overrides for the agent."""

    model_config = ConfigDict(extra="forbid")

    constraint_failure: float = Field(
        default=-1.0,
        description="Reward when a constraint check fails.",
    )
    metric_failure: float = Field(
        default=-1.0,
        description="Observation value when a metric is missing or failed.",
    )


class BaseAgentConfig(BaseModel):
    """Base config class for all agents in the CloudAI framework."""

    model_config = ConfigDict(extra="forbid")

    random_seed: int = 42
    start_action: Literal["random", "first"] = "random"
    rewards: RewardOverrides = Field(
        default_factory=RewardOverrides,
        description="Reward and observation overrides for the agent.",
    )


class BaseAgent(ABC):
    """
    Base class for all agents in the CloudAI framework.

    Provides a unified interface and parameter management for action spaces.
    """

    def __init__(self, env: BaseGym, config: BaseAgentConfig):
        """
        Initialize the agent with the environment.

        Args:
            env (BaseGym): The environment instance for the agent.
            config (BaseAgentConfig): The agent configuration. Class is defined by `get_config_class` static method.
        """
        self.env = env
        self.config = config

        self.action_space = {}
        self.max_steps = 0

    @staticmethod
    @abstractmethod
    def get_config_class() -> type[BaseAgentConfig]:
        pass

    @abstractmethod
    def configure(self, config: dict[str, Any]) -> None:
        """
        Configure the agent with additional settings.

        Args:
            config (Dict[str, Any]): Configuration settings for the agent.
        """
        pass

    @abstractmethod
    def select_action(self, observation: list[float] | None = None) -> tuple[int, dict[str, Any]]:
        """
        Select an action from the action space.

        Args:
            observation: Latest observation produced by the environment (``env.reset()`` on the
                first call, then the result of the prior ``env.step()``). Stateless agents such
                as grid search or Bayesian optimization may ignore this; observation-conditioned
                agents (RL, contextual bandits) should use it.

        Returns:
            Tuple[int, Dict[str, Any]]: The current step index and a dictionary mapping action keys to selected values.
        """
        pass

    @abstractmethod
    def update_policy(self, _feedback: Dict[str, Any]) -> None:
        """
        Update the agent state based on feedback from the environment.

        Args:
            feedback (Dict[str, Any]): Feedback information from the environment.
        """
        pass

    def run(self) -> int:
        """
        Orchestrate this agent's exploration over ``self.env``.

        Default: a (reset → select_action → step → update_policy) loop, one
        cycle per trial. Calling ``env.reset()`` per trial is the gym
        contract and is also what triggers the env's trial-boundary work
        (counter advancement, env_param sampling) so the observation passed
        to ``select_action`` carries any contextual signal declared via
        ``agent_observation``. Agents that drive their own training loop
        (e.g. RLlib-based agents calling ``algo.train()``) override this.

        Returns:
            int: Process-style return code (``0`` success, non-zero failure).
            ``handle_dse_job`` accumulates this via ``err |= agent.run()``.
        """
        for _ in range(self.max_steps):
            observation, _ = self.env.reset()
            result = self.select_action(observation=observation)
            if result is None:
                break
            step, action = result
            logging.info(f"Running step {step} (of {self.max_steps}) with action {action}")
            observation, reward, *_ = self.env.step(action)
            self.update_policy({"trial_index": step, "value": reward, "observation": observation})
            logging.info(
                f"Step {step}: Observation: {[round(obs, 4) for obs in observation]}, Reward: {reward:.4f}"
            )
        return 0
