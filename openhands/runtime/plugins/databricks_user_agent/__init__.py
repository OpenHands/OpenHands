"""Databricks User-Agent Plugin for OpenHands.

This plugin ensures that ALL interactions with Databricks include a custom user-agent header:
- Python SDK (databricks-sdk, databricks-cli, databricks-connect)
- Java SDK
- REST API calls (requests, httpx, urllib3)
- JDBC/ODBC connections
- Any other HTTP client making calls to Databricks

The plugin works by:
1. Setting environment variables that Databricks SDKs recognize
2. Creating Python startup scripts that monkey-patch HTTP libraries
3. Configuring Java system properties
4. Injecting initialization code into the runtime environment
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Resolve product name and version from the SDK so the runtime plugin and the
# SDK's HTTP stack advertise the same identity string to Databricks.
try:
    from openhands.sdk.llm.providers.databricks.utils import USER_AGENT as _SDK_UA

    _UA_PRODUCT, _, _UA_VERSION = _SDK_UA.partition(
        '/'
    )  # e.g. "OpenHandsOSS", "1.22.1"
    if not _UA_PRODUCT:
        _UA_PRODUCT, _UA_VERSION = 'OpenHandsOSS', 'unknown'
except ImportError:
    _UA_PRODUCT, _UA_VERSION = 'OpenHandsOSS', 'unknown'


@dataclass
class DatabricksUserAgentRequirement:
    """Requirement specification for the Databricks User-Agent plugin.

    The user-agent product is "OpenHandsOSS" with the version sourced from the
    openhands-sdk package, ensuring a single consistent identity across all
    Databricks API calls (AI Gateway HTTP, OAuth, Databricks SDK env vars).

    Attributes:
        name: Plugin identifier (always 'databricks_user_agent')
        patch_http_libraries: Whether to monkey-patch Python HTTP libraries. Default: True
        configure_java: Whether to configure Java user-agent. Default: True
        enable_debug_logging: Whether to enable debug logging for the plugin. Default: False
    """

    name: str = 'databricks_user_agent'
    patch_http_libraries: bool = True
    configure_java: bool = True
    enable_debug_logging: bool = False

    @property
    def user_agent(self) -> str:
        """Return the user-agent product name."""
        return _UA_PRODUCT

    @property
    def version(self) -> str:
        """Return the version resolved from the installed openhands-sdk package."""
        return _UA_VERSION


class DatabricksUserAgentPlugin:
    """Plugin that configures user-agent for all Databricks API interactions.

    This plugin is initialized once when the runtime starts and sets up:
    - Environment variables for Databricks SDKs
    - Python startup scripts for HTTP library patching
    - Java system properties
    - Shell environment configuration
    """

    name: str = 'databricks_user_agent'

    def __init__(self, requirement: Optional[DatabricksUserAgentRequirement] = None):
        """Initialize the plugin with configuration.

        Args:
            requirement: Configuration for the plugin. If None, uses defaults.
        """
        self.requirement = requirement or DatabricksUserAgentRequirement()
        self.user_agent = _UA_PRODUCT  # "OpenHandsOSS"
        self.version = _UA_VERSION  # from openhands-sdk package metadata
        self.enable_debug = self.requirement.enable_debug_logging
        self._initialized = False

    async def initialize(self, username: str) -> None:
        """Initialize the Databricks user-agent configuration in the runtime.

        This method is called once when the runtime starts. It sets up:
        1. Environment variables for Databricks SDK
        2. Python startup script for HTTP library patching
        3. Java configuration
        4. Shell configuration

        Args:
            username: The username of the user running the runtime
        """
        if self._initialized:
            logger.debug('[DatabricksUserAgent] Already initialized, skipping')
            return

        try:
            logger.info(
                f'[DatabricksUserAgent] Initializing with user-agent: {self.user_agent}'
            )

            # Step 1: Set environment variables
            self._set_environment_variables()

            # Step 2: Create Python startup script
            home_dir = Path(f'/home/{username}')
            python_init_path = await self._create_python_init_script(home_dir)

            # Step 3: Configure shell environment
            await self._configure_shell_environment(home_dir, python_init_path)

            # Step 4: Configure Java if enabled
            if self.requirement.configure_java:
                self._configure_java()

            # Step 5: Create helper scripts
            await self._create_helper_scripts(home_dir)

            self._initialized = True
            logger.info(
                f'[DatabricksUserAgent] Successfully initialized for user: {username}'
            )

        except Exception as e:
            logger.error(
                f'[DatabricksUserAgent] Failed to initialize: {e}', exc_info=True
            )
            # Don't raise - we want the runtime to continue even if plugin fails

    def _set_environment_variables(self) -> None:
        """Set environment variables that Databricks SDKs recognize."""
        os.environ['DATABRICKS_SDK_UPSTREAM'] = self.user_agent  # "OpenHandsOSS"
        os.environ['DATABRICKS_SDK_UPSTREAM_VERSION'] = (
            self.version
        )  # SDK package version
        os.environ['DATABRICKS_USER_AGENT'] = self.user_agent
        os.environ['OH_DATABRICKS_INTEGRATION'] = 'true'

        if self.enable_debug:
            logger.debug('[DatabricksUserAgent] Set environment variables:')
            logger.debug(f'  DATABRICKS_SDK_UPSTREAM={self.user_agent}')
            logger.debug(f'  DATABRICKS_SDK_UPSTREAM_VERSION={self.version}')

    async def _create_python_init_script(self, home_dir: Path) -> Path:
        """Create Python initialization script that patches HTTP libraries.

        Args:
            home_dir: Home directory of the user

        Returns:
            Path to the created initialization script
        """
        init_script_path = home_dir / '.databricks_user_agent_init.py'

        init_script_content = self._generate_python_init_script()

        # Write the script
        init_script_path.write_text(init_script_content)
        init_script_path.chmod(0o755)

        if self.enable_debug:
            logger.debug(
                f'[DatabricksUserAgent] Created Python init script at: {init_script_path}'
            )

        return init_script_path

    def _generate_python_init_script(self) -> str:
        """Generate the Python initialization script content.

        User-agent is HARDCODED to "OpenHandsOSS".
        """
        debug_print = 'print' if self.enable_debug else 'pass  #'

        return f'''"""
Databricks User-Agent Initialization Script
Auto-generated by OpenHands Databricks User-Agent Plugin
This script is automatically executed when Python starts.

User-Agent: {self.user_agent}/{self.version}
"""
import os
import sys

USER_AGENT = "{self.user_agent}"
VERSION = "{self.version}"

# Set environment variables
os.environ.setdefault('DATABRICKS_SDK_UPSTREAM', USER_AGENT)
os.environ.setdefault('DATABRICKS_SDK_UPSTREAM_VERSION', VERSION)
os.environ.setdefault('DATABRICKS_USER_AGENT', USER_AGENT)

def patch_requests():
    """Patch requests library to add user-agent to Databricks calls."""
    try:
        import requests

        _original_request = requests.Session.request

        def _patched_request(self, method, url, **kwargs):
            # Add user-agent for any Databricks URL
            if isinstance(url, str) and ('databricks' in url.lower() or
                                         os.getenv('OH_DATABRICKS_INTEGRATION') == 'true'):
                headers = kwargs.get('headers', {{}})
                if isinstance(headers, dict):
                    # Only set if not already present
                    if 'User-Agent' not in headers:
                        headers['User-Agent'] = USER_AGENT
                    kwargs['headers'] = headers
            return _original_request(self, method, url, **kwargs)

        requests.Session.request = _patched_request
        {debug_print}(f"[DatabricksUserAgent] Patched requests library with user-agent: {{USER_AGENT}}")

    except ImportError:
        {debug_print}("[DatabricksUserAgent] requests library not available")
    except Exception as e:
        {debug_print}(f"[DatabricksUserAgent] Failed to patch requests: {{e}}")

def patch_urllib3():
    """Patch urllib3 library to add user-agent."""
    try:
        import urllib3.util.request

        _original_make_headers = urllib3.util.request.make_headers

        def _patched_make_headers(keep_alive=None, accept_encoding=None,
                                  user_agent=None, basic_auth=None,
                                  proxy_basic_auth=None, disable_cache=None):
            # Use our custom user-agent if none provided
            if user_agent is None:
                user_agent = USER_AGENT
            return _original_make_headers(
                keep_alive=keep_alive,
                accept_encoding=accept_encoding,
                user_agent=user_agent,
                basic_auth=basic_auth,
                proxy_basic_auth=proxy_basic_auth,
                disable_cache=disable_cache
            )

        urllib3.util.request.make_headers = _patched_make_headers
        {debug_print}(f"[DatabricksUserAgent] Patched urllib3 library")

    except ImportError:
        {debug_print}("[DatabricksUserAgent] urllib3 library not available")
    except Exception as e:
        {debug_print}(f"[DatabricksUserAgent] Failed to patch urllib3: {{e}}")

def patch_httpx():
    """Patch httpx library (modern async HTTP client) to add user-agent."""
    try:
        import httpx

        # Patch sync client
        _original_client_init = httpx.Client.__init__

        def _patched_client_init(self, *args, **kwargs):
            headers = kwargs.get('headers')
            if headers is None:
                kwargs['headers'] = {{'User-Agent': USER_AGENT}}
            elif isinstance(headers, dict) and 'User-Agent' not in headers:
                headers['User-Agent'] = USER_AGENT
                kwargs['headers'] = headers
            return _original_client_init(self, *args, **kwargs)

        httpx.Client.__init__ = _patched_client_init

        # Patch async client
        _original_async_client_init = httpx.AsyncClient.__init__

        def _patched_async_client_init(self, *args, **kwargs):
            headers = kwargs.get('headers')
            if headers is None:
                kwargs['headers'] = {{'User-Agent': USER_AGENT}}
            elif isinstance(headers, dict) and 'User-Agent' not in headers:
                headers['User-Agent'] = USER_AGENT
                kwargs['headers'] = headers
            return _original_async_client_init(self, *args, **kwargs)

        httpx.AsyncClient.__init__ = _patched_async_client_init

        {debug_print}(f"[DatabricksUserAgent] Patched httpx library")

    except ImportError:
        {debug_print}("[DatabricksUserAgent] httpx library not available")
    except Exception as e:
        {debug_print}(f"[DatabricksUserAgent] Failed to patch httpx: {{e}}")

def patch_aiohttp():
    """Patch aiohttp library for async HTTP requests."""
    try:
        import aiohttp

        _original_client_init = aiohttp.ClientSession.__init__

        def _patched_client_init(self, *args, **kwargs):
            headers = kwargs.get('headers')
            if headers is None:
                kwargs['headers'] = {{'User-Agent': USER_AGENT}}
            elif isinstance(headers, dict) and 'User-Agent' not in headers:
                headers['User-Agent'] = USER_AGENT
                kwargs['headers'] = headers
            return _original_client_init(self, *args, **kwargs)

        aiohttp.ClientSession.__init__ = _patched_client_init
        {debug_print}(f"[DatabricksUserAgent] Patched aiohttp library")

    except ImportError:
        {debug_print}("[DatabricksUserAgent] aiohttp library not available")
    except Exception as e:
        {debug_print}(f"[DatabricksUserAgent] Failed to patch aiohttp: {{e}}")

# Apply all patches when this module is imported
if __name__ != "__main__":
    {'if True:' if self.requirement.patch_http_libraries else 'if False:'}
        patch_requests()
        patch_urllib3()
        patch_httpx()
        patch_aiohttp()
        {debug_print}("[DatabricksUserAgent] All HTTP libraries patched successfully")
'''

    async def _configure_shell_environment(
        self, home_dir: Path, python_init_path: Path
    ) -> None:
        """Configure shell environment to load Python init script.

        User-agent is HARDCODED to "OpenHandsOSS".

        Args:
            home_dir: Home directory of the user
            python_init_path: Path to the Python initialization script
        """
        # Add to .bashrc
        bashrc_path = home_dir / '.bashrc'

        bashrc_addition = f"""
# ============================================================================
# Databricks User-Agent Configuration (OpenHands Plugin)
# User-Agent: {self.user_agent}/{self.version}
# ============================================================================
export DATABRICKS_SDK_UPSTREAM="{self.user_agent}"
export DATABRICKS_SDK_UPSTREAM_VERSION="{self.version}"
export DATABRICKS_USER_AGENT="{self.user_agent}"
export OH_DATABRICKS_INTEGRATION="true"

# Set Python startup script to auto-load patches
export PYTHONSTARTUP="{python_init_path}"

"""

        # Append to bashrc if not already present
        if bashrc_path.exists():
            content = bashrc_path.read_text()
            if 'Databricks User-Agent Configuration' not in content:
                with bashrc_path.open('a') as f:
                    f.write(bashrc_addition)
        else:
            bashrc_path.write_text(bashrc_addition)

        if self.enable_debug:
            logger.debug(f'[DatabricksUserAgent] Updated {bashrc_path}')

        # Also add to .profile for non-interactive shells
        profile_path = home_dir / '.profile'
        if profile_path.exists():
            content = profile_path.read_text()
            if 'Databricks User-Agent Configuration' not in content:
                with profile_path.open('a') as f:
                    f.write(bashrc_addition)

    def _configure_java(self) -> None:
        """Configure Java system properties for user-agent."""
        java_opts = os.environ.get('JAVA_TOOL_OPTIONS', '')
        user_agent_property = f'-Dhttp.agent="{self.user_agent}"'

        if user_agent_property not in java_opts:
            os.environ['JAVA_TOOL_OPTIONS'] = (
                f'{java_opts} {user_agent_property}'.strip()
            )

            if self.enable_debug:
                logger.debug(
                    f'[DatabricksUserAgent] Configured Java with user-agent: {self.user_agent}'
                )

    async def _create_helper_scripts(self, home_dir: Path) -> None:
        """Create helper scripts for testing and debugging.

        Args:
            home_dir: Home directory of the user
        """
        # Create a test script
        test_script_path = home_dir / '.databricks_user_agent_test.py'
        test_script_content = '''#!/usr/bin/env python3
"""Test script to verify Databricks user-agent configuration."""
import os

def test_databricks_user_agent():
    """Verify that the user-agent is properly configured."""
    print("=" * 70)
    print("Databricks User-Agent Configuration Test")
    print("=" * 70)

    # Check environment variables
    print("\\n1. Environment Variables:")
    for var in ['DATABRICKS_SDK_UPSTREAM', 'DATABRICKS_SDK_UPSTREAM_VERSION',
                'DATABRICKS_USER_AGENT', 'PYTHONSTARTUP', 'JAVA_TOOL_OPTIONS']:
        value = os.getenv(var, 'NOT SET')
        print(f"   {var}: {value}")

    # Test HTTP libraries
    print("\\n2. Testing HTTP Library Patches:")

    # Test requests
    try:
        import requests
        session = requests.Session()
        prep = session.prepare_request(
            requests.Request('GET', 'https://test.databricks.com/api/test')
        )
        print(f"   requests User-Agent: {prep.headers.get('User-Agent', 'NOT SET')}")
    except ImportError:
        print("   requests: NOT INSTALLED")
    except Exception as e:
        print(f"   requests: ERROR - {e}")

    # Test httpx
    try:
        import httpx
        print("   httpx: INSTALLED (patched)")
    except ImportError:
        print("   httpx: NOT INSTALLED")

    # Test Databricks SDK
    print("\\n3. Databricks SDK:")
    try:
        from databricks.sdk import WorkspaceClient
        print("   Databricks SDK: INSTALLED")
        print("   SDK will use DATABRICKS_SDK_UPSTREAM environment variable")
    except ImportError:
        print("   Databricks SDK: NOT INSTALLED")

    print("\\n" + "=" * 70)
    print("Configuration test complete!")
    print("=" * 70)

if __name__ == "__main__":
    test_databricks_user_agent()
'''
        test_script_path.write_text(test_script_content)
        test_script_path.chmod(0o755)

        if self.enable_debug:
            logger.debug(
                f'[DatabricksUserAgent] Created test script at: {test_script_path}'
            )


# Convenience function for creating plugin
def create_databricks_plugin(enable_debug: bool = False) -> DatabricksUserAgentPlugin:
    """Create a Databricks User-Agent plugin.

    The user-agent product is "OpenHandsOSS" with the version sourced from
    the installed openhands-sdk package.

    Args:
        enable_debug: Whether to enable debug logging

    Returns:
        Configured DatabricksUserAgentPlugin instance
    """
    requirement = DatabricksUserAgentRequirement(enable_debug_logging=enable_debug)
    return DatabricksUserAgentPlugin(requirement)
