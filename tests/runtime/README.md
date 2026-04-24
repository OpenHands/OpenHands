## Runtime Tests

This folder contains integration tests that verify the functionality of OpenHands' runtime environments and their interactions with various tools and features.

### What are Runtime Tests?

Runtime tests focus on testing:
- Tool interactions within a runtime environment (bash commands, browsing, file operations)
- Environment setup and configuration
- Resource management and cleanup
- Browser-based operations and file viewing capabilities
- IPython/Jupyter integration
- Environment variables and configuration handling

The tests can be run against different runtime environments (Docker, Local, Remote, Runloop, or Daytona) by setting the TEST_RUNTIME environment variable. By default, tests run using the Docker runtime.

### How are they different from Unit Tests?

While unit tests in `tests/unit/` focus on testing individual components in isolation, runtime tests verify:
1. Integration between components
2. Actual execution of commands in different runtime environments
3. System-level interactions (file system, network, browser)
4. Environment setup and teardown
5. Tool functionality in real runtime contexts

### Running the Tests

Run all runtime tests:

```bash
poetry run pytest ./tests/runtime
```

Run specific test file:

```bash
poetry run pytest ./tests/runtime/test_bash.py
```

Run specific test:

```bash
poetry run pytest ./tests/runtime/test_bash.py::test_bash_command_env
```

For verbose output, add the `-v` flag (more verbose: `-vv` and `-vvv`):

```bash
poetry run pytest -v ./tests/runtime/test_bash.py
```

### Environment Variables

The runtime tests can be configured using environment variables:
- `TEST_IN_CI`: Set to 'True' when running in CI environment
- `TEST_RUNTIME`: Specify the runtime to test ('docker', 'local', 'remote', 'runloop', 'daytona')
- `RUN_AS_OPENHANDS`: Set to 'True' to run tests as openhands user (default), 'False' for root
- `SANDBOX_BASE_CONTAINER_IMAGE`: Specify a custom base container image for Docker runtime

For more details on pytest usage, see the [pytest documentation](https://docs.pytest.org/en/latest/contents.html).
