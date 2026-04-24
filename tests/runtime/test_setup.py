"""Tests for the setup script."""

from conftest import (
    _load_runtime,
)

from openhands.events.action import FileReadAction, FileWriteAction
from openhands.events.observation import FileReadObservation, FileWriteObservation


def test_maybe_run_setup_script(temp_dir, runtime_cls, run_as_openhands):
    """Test that setup script is executed when it exists."""
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)

    setup_script = '.openhands/setup.sh'
    write_obs = runtime.write(
        FileWriteAction(
            path=setup_script, content="#!/bin/bash\necho 'Hello World' >> README.md\n"
        )
    )
    assert isinstance(write_obs, FileWriteObservation)

    # Run setup script
    runtime.maybe_run_setup_script()

    # Verify script was executed by checking output
    read_obs = runtime.read(FileReadAction(path='README.md'))
    assert isinstance(read_obs, FileReadObservation)
    assert read_obs.content == 'Hello World\n'


def test_maybe_run_setup_script_with_long_timeout(
    temp_dir, runtime_cls, run_as_openhands
):
    """Test that setup script is executed when it exists."""
    runtime, config = _load_runtime(
        temp_dir,
        runtime_cls,
        run_as_openhands,
        runtime_startup_env_vars={'NO_CHANGE_TIMEOUT_SECONDS': '1'},
    )

    setup_script = '.openhands/setup.sh'
    write_obs = runtime.write(
        FileWriteAction(
            path=setup_script,
            content="#!/bin/bash\nsleep 3 && echo 'Hello World' >> README.md\n",
        )
    )
    assert isinstance(write_obs, FileWriteObservation)

    # Run setup script
    runtime.maybe_run_setup_script()

    # Verify script was executed by checking output
    read_obs = runtime.read(FileReadAction(path='README.md'))
    assert isinstance(read_obs, FileReadObservation)
    assert read_obs.content == 'Hello World\n'
