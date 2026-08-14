# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for CommandShell's optional shell bypass.

``shell=True`` puts a ``/bin/bash -c`` process in front of every command, which is
invisible when a job takes 80 s and dominant when it takes 40 ms -- measured at
14.2 ms. ``use_shell=False`` skips it when the command does not need a shell.

The bypass stays inside ``CommandShell`` rather than in the caller so that
``CommandShell.execute`` remains the single process-launch seam. Tests elsewhere
(test_cache_prefill_e2e) patch it to count simulator invocations, and moving the
launch out from under it silently broke those assertions while leaking real
processes.
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cloudai.systems.standalone.standalone_runner import StandaloneRunner
from cloudai.systems.standalone.standalone_system import StandaloneSystem
from cloudai.util import CommandShell


@pytest.fixture
def shell() -> CommandShell:
    return CommandShell()


class TestShellIsSkippedWhenPossible:
    @pytest.mark.parametrize(
        "command, expected_argv",
        [
            ('bash "/tmp/entrypoint.sh"', ["bash", "/tmp/entrypoint.sh"]),
            ("bash /tmp/entrypoint.sh", ["bash", "/tmp/entrypoint.sh"]),
            ('bash "/tmp/a dir/entrypoint.sh"', ["bash", "/tmp/a dir/entrypoint.sh"]),
            ("python /tmp/x.py --flag value", ["python", "/tmp/x.py", "--flag", "value"]),
        ],
    )
    def test_plain_command_is_spawned_as_argv(self, shell: CommandShell, command: str, expected_argv: list) -> None:
        with patch("subprocess.Popen") as popen:
            shell.execute(command, use_shell=False)

        assert popen.call_args[0][0] == expected_argv
        assert "shell" not in popen.call_args[1], "must not ask for a shell"

    def test_stdout_stays_a_pipe_so_callers_can_still_communicate(self, shell: CommandShell) -> None:
        """is_job_running's ps fallback reads stdout; changing this would break it."""
        with patch("subprocess.Popen") as popen:
            shell.execute("ps -p 12345", use_shell=False)
        kwargs = popen.call_args[1]
        assert kwargs["stdout"] is subprocess.PIPE
        assert kwargs["stderr"] is subprocess.PIPE


class TestShellIsKeptWhenNeeded:
    @pytest.mark.parametrize(
        "command",
        [
            'bash "/tmp/x.sh" > /tmp/out.txt 2> /tmp/err.txt',  # redirection
            "cat /tmp/a | grep b",  # pipe
            "/tmp/a.sh && /tmp/b.sh",  # sequencing
            "echo $HOME",  # substitution
            "ls /tmp/*.log",  # globbing
            "sleep 1 &",  # backgrounding
        ],
    )
    def test_shell_syntax_still_goes_through_the_shell(self, shell: CommandShell, command: str) -> None:
        """Asking for no shell must not turn metacharacters into literal args."""
        with patch("subprocess.Popen") as popen:
            shell.execute(command, use_shell=False)

        assert popen.call_args[0][0] == command
        assert popen.call_args[1]["shell"] is True

    def test_empty_command_falls_back_rather_than_spawning_nothing(self, shell: CommandShell) -> None:
        with patch("subprocess.Popen") as popen:
            shell.execute("   ", use_shell=False)
        assert popen.call_args[1]["shell"] is True

    def test_default_is_unchanged_for_every_existing_caller(self, shell: CommandShell) -> None:
        with patch("subprocess.Popen") as popen:
            shell.execute('bash "/tmp/entrypoint.sh"')
        assert popen.call_args[1]["shell"] is True


class TestAgainstRealProcesses:
    @pytest.mark.parametrize("command, expected", [("/usr/bin/true", 0), ("/usr/bin/false", 1)])
    def test_argv_path_runs_and_reports_exit_code(self, shell: CommandShell, command: str, expected: int) -> None:
        process = shell.execute(command, use_shell=False)
        # communicate() rather than wait(): stdout/stderr are pipes, and leaving
        # them open raises a ResourceWarning that pytest surfaces as a failure.
        process.communicate()
        assert process.returncode == expected

    def test_both_paths_agree_on_output(self, shell: CommandShell) -> None:
        with_shell = shell.execute("echo hello", use_shell=True).communicate()[0]
        without = shell.execute("echo hello", use_shell=False).communicate()[0]
        assert with_shell == without == "hello\n"


class TestRunnerAsksForNoShell:
    def test_submit_test_requests_the_bypass(self, tmp_path: Path) -> None:
        system = StandaloneSystem(name="s", install_path=tmp_path, output_path=tmp_path)
        scenario = MagicMock()
        scenario.test_runs = []
        runner = StandaloneRunner(mode="run", system=system, test_scenario=scenario, output_path=tmp_path)

        tr = MagicMock()
        tr.name = "t"
        with (
            patch.object(runner, "get_job_output_path", return_value=tmp_path),
            patch.object(runner, "get_cmd_gen_strategy") as strategy,
            patch.object(runner.cmd_shell, "execute") as execute,
        ):
            strategy.return_value.gen_exec_command.return_value = 'bash "/tmp/entrypoint.sh"'
            execute.return_value = MagicMock(pid=4242)
            job = runner._submit_test(tr)

        assert execute.call_args[1]["use_shell"] is False
        assert job.id == 4242
        assert job.process is execute.return_value, "the handle must be retained for polling"
