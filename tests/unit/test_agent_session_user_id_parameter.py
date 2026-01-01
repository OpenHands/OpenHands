"""
Tests for AgentSession user_id parameter passing to Runtime.

This module tests that user_id is correctly passed to Runtime initialization,
ensuring consistency between DockerRuntime and RemoteRuntime for dynamic token refresh.
"""

import ast
import inspect
import textwrap


class TestAgentSessionUserIdParameter:
    """Test suite for AgentSession user_id parameter."""

    def test_runtime_initialization_includes_user_id_parameter(self):
        """
        Test: Runtime initialization includes user_id parameter in agent_session.py.

        This test verifies that the code change was made to pass user_id to the
        runtime constructor. It checks the source code directly to ensure the
        parameter is present in the runtime initialization call.

        Arrange: Import agent_session module
        Act: Parse the _create_runtime method source code
        Assert: user_id parameter is passed to runtime_cls constructor
        """
        # Arrange
        from openhands.server.session import agent_session

        # Act
        source = inspect.getsource(agent_session.AgentSession._create_runtime)

        # Assert
        # Verify that user_id is passed in the runtime initialization
        # The code should contain: user_id=self.user_id
        assert 'user_id=self.user_id' in source, (
            "Runtime initialization should include 'user_id=self.user_id' parameter"
        )

    def test_user_id_parameter_in_runtime_call_ast_verification(self):
        """
        Test: Verify user_id parameter using AST parsing.

        This test uses AST parsing to verify that the runtime constructor call
        includes the user_id keyword argument.

        Arrange: Get source code of _create_runtime method
        Act: Parse with AST and find the runtime_cls call
        Assert: user_id keyword argument is present
        """
        # Arrange
        from openhands.server.session import agent_session

        source = inspect.getsource(agent_session.AgentSession._create_runtime)
        # Dedent to remove leading whitespace for AST parsing
        source = textwrap.dedent(source)
        tree = ast.parse(source)

        # Act
        # Find all Call nodes in the AST
        found_user_id_kwarg = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check if this call has a 'user_id' keyword argument
                for keyword in node.keywords:
                    if keyword.arg == 'user_id':
                        found_user_id_kwarg = True
                        break

        # Assert
        assert found_user_id_kwarg, (
            "Runtime constructor call should include 'user_id' keyword argument"
        )
