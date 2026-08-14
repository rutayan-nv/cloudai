# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cloudai.systems.standalone.standalone_job import StandaloneJob
from cloudai.systems.standalone.standalone_system import StandaloneSystem


@pytest.fixture
def standalone_system():
    """
    Fixture to create a StandaloneSystem instance for testing.

    Returns:
        StandaloneSystem: A new instance of StandaloneSystem for testing.
    """
    return StandaloneSystem(
        name="StandaloneTestSystem",
        install_path=Path("/fake/install/path"),
        output_path=Path("/fake/output/path"),
    )


@pytest.fixture
def mock_test():
    """
    Fixture to create a mock Test instance for testing.

    Returns:
        MagicMock: A mocked Test instance.
    """
    return MagicMock(name="MockTest")


@pytest.fixture
def standalone_job(standalone_system, mock_test):
    """
    Fixture to create a StandaloneJob instance for testing.

    Args:
        standalone_system (StandaloneSystem): The system where the job will be executed.
        mock_test (Test): The mock test instance associated with the job.

    Returns:
        StandaloneJob: A new instance of StandaloneJob for testing.
    """
    return StandaloneJob(mock_test, id=12345)


@pytest.mark.parametrize(
    "ps_output, expected_result",
    [
        ("12345\n", True),  # Job is running, PID is in ps output
        ("", False),  # Job is not running, ps output is empty
    ],
)
@patch("cloudai.util.CommandShell.execute")
def test_is_job_running(mock_execute, standalone_system, standalone_job, ps_output, expected_result):
    """
    Test if a job is running using a mocked CommandShell.

    Args:
        mock_execute (MagicMock): Mocked CommandShell execute method.
        standalone_system (StandaloneSystem): Instance of the system under test.
        standalone_job (StandaloneJob): Job instance to check.
        ps_output (str): Mocked output of the ps command.
        expected_result (bool): Expected result for the job running status.
    """
    mock_process = MagicMock()
    mock_process.communicate.return_value = (ps_output, "")
    mock_execute.return_value = mock_process

    assert standalone_system.is_job_running(standalone_job) == expected_result


@pytest.mark.parametrize(
    "returncode, expected_result",
    [
        (None, True),  # poll() returns None while the process is alive
        (0, False),  # a returncode means it has exited
        (137, False),  # including a non-zero one
    ],
)
@patch("cloudai.util.CommandShell.execute")
def test_is_job_running_uses_process_handle(
    mock_execute, standalone_system, mock_test, returncode, expected_result
):
    """With a handle available, status comes from poll() rather than ps."""
    process = MagicMock()
    process.poll.return_value = returncode
    job = StandaloneJob(mock_test, id=12345, process=process)

    assert standalone_system.is_job_running(job) is expected_result
    process.poll.assert_called_once()
    # The point of the handle: no subprocess is created to answer the question.
    mock_execute.assert_not_called()


def test_is_job_running_on_a_real_process(standalone_system, mock_test):
    """End to end against a real process, no mocks.

    Guards the whole path: a live process reads as running, and once it exits the
    handle reports completion. The ps-based version could get this wrong if the
    pid were recycled.
    """
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    job = StandaloneJob(mock_test, id=process.pid, process=process)
    try:
        assert standalone_system.is_job_running(job) is True
    finally:
        process.kill()
        process.wait()
    assert standalone_system.is_job_running(job) is False


def test_is_job_running_falls_back_to_ps_without_a_handle(standalone_system, mock_test):
    """Dry-run jobs and any caller that kept only a pid must still work."""
    job = StandaloneJob(mock_test, id=12345)
    assert job.process is None
    with patch("cloudai.util.CommandShell.execute") as mock_execute:
        mock_execute.return_value.communicate.return_value = ("12345\n", "")
        assert standalone_system.is_job_running(job) is True
    mock_execute.assert_called_once_with("ps -p 12345")


@patch("cloudai.util.CommandShell.execute")
def test_kill_uses_handle_and_reaps(mock_execute, standalone_system, mock_test):
    process = MagicMock()
    process.poll.return_value = None  # still running
    job = StandaloneJob(mock_test, id=12345, process=process)

    standalone_system.kill(job)

    process.kill.assert_called_once()
    process.wait.assert_called_once()  # reaped, so no zombie is left
    mock_execute.assert_not_called()


@patch("cloudai.util.CommandShell.execute")
def test_kill_already_exited_process_only_reaps(mock_execute, standalone_system, mock_test):
    process = MagicMock()
    process.poll.return_value = 0  # already finished
    job = StandaloneJob(mock_test, id=12345, process=process)

    standalone_system.kill(job)

    process.kill.assert_not_called()
    process.wait.assert_called_once()
    mock_execute.assert_not_called()


def test_monitor_interval_accepts_sub_second_values():
    """Short trials need a poll interval between 0 and 1 second."""
    system = StandaloneSystem(
        name="s",
        install_path=Path("/fake"),
        output_path=Path("/fake"),
        monitor_interval=0.005,
    )
    assert system.monitor_interval == pytest.approx(0.005)


@patch("cloudai.util.CommandShell.execute")
def test_kill_job(mock_execute, standalone_system, standalone_job):
    """
    Test if a job can be killed using a mocked CommandShell.

    Args:
        mock_execute (MagicMock): Mocked CommandShell execute method.
        standalone_system (StandaloneSystem): Instance of the system under test.
        standalone_job (StandaloneJob): Job instance to kill.
    """
    mock_process = MagicMock()
    mock_execute.return_value = mock_process

    standalone_system.kill(standalone_job)
    kill_command = f"kill -9 {standalone_job.id}"

    mock_execute.assert_called_once_with(kill_command)
