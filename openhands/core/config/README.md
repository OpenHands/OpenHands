# Configuration Management in OpenHands

## Overview

OpenHands uses a flexible configuration system that allows settings to be defined through environment variables, TOML files, and command-line arguments. The configuration is managed through a package structure in `openhands/core/config/`.

## Configuration Classes

The main configuration class is:

- `OpenHandsConfig`: The root configuration class

This class is defined as a Pydantic model, with field defaults for all settings.

## Loading Configuration from Environment Variables

The `load_from_env` function in the config package is responsible for loading configuration values from environment variables. It recursively processes the configuration classes, mapping environment variable names to class attributes.

### Naming Convention for Environment Variables

- Field Names: all uppercase
- Full Variable Name: Field Name (e.g., `DEFAULT_AGENT`, `WORKSPACE_BASE`)

## Type Handling

The `load_from_env` function attempts to cast environment variable values to the types specified in the models. It handles:

- Basic types (str, int, bool)
- Optional types (e.g., `str | None`)
- Nested models

If type casting fails, an error is logged, and the default value is retained.

## Default Values

If an environment variable is not set, the default value specified in the model is used.

## Security Considerations

Be cautious when setting sensitive information like API keys in environment variables. Ensure your environment is secure.

## Usage

The `load_openhands_config()` function is the recommended way to initialize your configuration. It performs the following steps:

1. Creates an instance of `OpenHandsConfig`
2. Loads settings from the `config.toml` file (if present)
3. Loads settings from environment variables, overriding TOML settings if applicable
4. Applies final tweaks and validations to the configuration, falling back to the default values specified in the code
5. Optionally sets global logging levels based on the configuration

There are also command line args, which may work to override other sources.

Here's an example of how to use `load_openhands_config()`:

````python
from openhands.core.config import load_openhands_config

# Load all configuration settings
config = load_openhands_config()

# Now you can access your configuration
print(f"Default agent: {config.default_agent}")
print(f"Max iterations: {config.max_iterations}")
````

By using `load_openhands_config()`, you ensure that all configuration sources are properly loaded and processed, providing a consistent and fully initialized configuration for your application.

## Additional Configuration Methods

While this document focuses on environment variable configuration, OpenHands also supports:

- Loading from TOML files
- Parsing command-line arguments

These methods are handled by separate functions in the config package.

## Conclusion

The OpenHands configuration system provides a flexible and type-safe way to manage application settings. By following the naming conventions and utilizing the provided functions, developers can easily customize the behavior of OpenHands components through environment variables and other configuration sources.
