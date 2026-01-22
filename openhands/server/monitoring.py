# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
# This module belongs to the old V0 web server. The V1 application server lives under openhands/app_server/.
from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.events.event import Event


class MonitoringListener:
    """Abstract base class for monitoring application activity.

    This is an extension point in OpenHands that allows applications to customize how
    application activity is monitored. Applications can substitute their own implementation by:
    1. Creating a class that inherits from MonitoringListener
    2. Implementing desired methods (all methods have default no-op implementations)
    3. Setting server_config.monitoring_listener_class to the fully qualified name of the class

    The class is instantiated via get_impl() in openhands.server.shared.py.

    Implementations should be non-disruptive, do not raise or block to perform I/O.
    """

    def on_session_event(self, event: Event) -> None:
        """Track metrics about events being added to a Session's EventStream."""
        pass

    def on_agent_session_start(self, success: bool, duration: float) -> None:
        """Track an agent session start.
        Success is true if startup completed without error.
        Duration is start time in seconds observed by AgentSession.
        """
        pass

    def on_create_conversation(self) -> None:
        """Track the beginning of conversation creation.
        Does not currently capture whether it succeed.
        """
        pass

    @classmethod
    def get_instance(
        cls,
        config: OpenHandsConfig,
    ) -> 'MonitoringListener':
        return cls()
