# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Reward transform pipeline for RL agents.

Lib-agnostic abstractions used by :class:`RLAgentBase` to build a reward
transformation chain. The chain is wired into the env via
:class:`PipelineRewardWrapper` -- a single, library-agnostic
``gym/gymnasium``-style wrapper. RLlib, CleanRL, SB3 and custom torch
loops all consume gymnasium envs, so one wrapper covers every backend.

We deliberately did NOT pick an RLlib-specific surface (e.g.
``ConnectorV2``) because reward shaping is agent-agnostic and
workload-agnostic; binding it to one library's plugin system would force
re-implementation for every future RL backend and double the test
surface.

The pipeline state is **agent-owned** and **in-memory only**. It is not
serialized into the env's reporting plane (``trajectory.csv`` /
``env.csv``) and does not depend on those report formats. The wrapper
preserves the original reward in ``info["raw_reward"]`` so reports and
auditing tools recover the un-normalized signal without re-running the
workload.
"""

from .context_detector import ContextAutoDetector
from .pipeline import RewardPipeline, build_default_pipeline
from .transforms import (
    GlobalMeanStdFilter,
    IdentityTransform,
    PerContextZScore,
    RewardTransform,
)
from .wrapper import PipelineRewardWrapper

__all__ = [
    "ContextAutoDetector",
    "GlobalMeanStdFilter",
    "IdentityTransform",
    "PerContextZScore",
    "PipelineRewardWrapper",
    "RewardPipeline",
    "RewardTransform",
    "build_default_pipeline",
]
