import os
import re
import unittest


class TestCircularImports(unittest.TestCase):
    """Test to detect circular imports in the codebase."""

    def test_no_circular_imports_in_key_modules(self):
        """Test that there are no circular imports in key modules that were previously problematic.

        This test specifically checks the modules that were involved in a previous circular import issue:
        - openhands.utils.prompt
        - openhands.agenthub.codeact_agent.tools.bash
        - openhands.agenthub.codeact_agent.tools.prompt
        - openhands.memory.memory
        - openhands.memory.conversation_memory
        """
        # Get the project root directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

        # Map module names to file paths
        module_paths = {
            'openhands.utils.prompt': os.path.join(
                project_root, 'openhands/utils/prompt.py'
            ),
            'openhands.agenthub.codeact_agent.tools.bash': os.path.join(
                project_root, 'openhands/agenthub/codeact_agent/tools/bash.py'
            ),
            'openhands.agenthub.codeact_agent.tools.prompt': os.path.join(
                project_root, 'openhands/agenthub/codeact_agent/tools/prompt.py'
            ),
            'openhands.memory.memory': os.path.join(
                project_root, 'openhands/memory/memory.py'
            ),
            'openhands.memory.conversation_memory': os.path.join(
                project_root, 'openhands/memory/conversation_memory.py'
            ),
        }

        # Check for the specific circular import pattern that was problematic
        circular_imports = self._find_circular_imports(module_paths)

        # If there are any circular imports, fail the test
        if circular_imports:
            circular_import_str = '\n'.join(
                [
                    f'{module1} -> {module2} -> {module1}'
                    for module1, module2 in circular_imports
                ]
            )
            self.fail(f'Circular imports detected:\n{circular_import_str}')

    def _find_circular_imports(
        self, module_paths: dict[str, str]
    ) -> list[tuple[str, str]]:
        """Find circular imports between modules.

        Args:
            module_paths: Dictionary mapping module names to file paths

        Returns:
            List of tuples (module1, module2) where module1 imports module2 and module2 imports module1
        """
        # Dictionary to store imports for each module
        module_imports = {}

        # Extract imports for each module
        for module_name, file_path in module_paths.items():
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    source_code = f.read()

                # Extract import statements
                import_lines = [
                    line.strip()
                    for line in source_code.split('\n')
                    if line.strip().startswith(('import ', 'from '))
                    and not line.strip().startswith('# ')
                ]

                # Parse import statements to get imported modules
                imported_modules = []
                for line in import_lines:
                    if line.startswith('import '):
                        # Handle "import module" or "import module as alias"
                        parts = line[7:].split(',')
                        for part in parts:
                            module_part = part.strip().split(' as ')[0].strip()
                            if module_part.startswith('openhands.'):
                                imported_modules.append(module_part)
                    elif line.startswith('from '):
                        # Handle "from module import name" or "from module import name as alias"
                        module_part = line[5:].split(' import ')[0].strip()
                        if module_part.startswith('openhands.'):
                            imported_modules.append(module_part)

                module_imports[module_name] = imported_modules

        # Check for circular imports
        circular_imports = []
        for module1, imports1 in module_imports.items():
            for module2 in imports1:
                if module2 in module_imports and module1 in module_imports[module2]:
                    # Found a circular import
                    circular_imports.append((module1, module2))

        return circular_imports

    def test_specific_circular_import_pattern(self):
        """Test for the specific circular import pattern that caused the issue in the stack trace.

        The problematic pattern was:
        openhands.utils.prompt imports from openhands.agenthub.codeact_agent.tools.bash
        openhands.agenthub.codeact_agent.tools.bash imports from openhands.agenthub.codeact_agent.tools.prompt
        openhands.agenthub.codeact_agent.tools.prompt imports from openhands.utils.prompt
        """
        # Get the project root directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

        # Check if the problematic pattern exists
        prompt_path = os.path.join(project_root, 'openhands/utils/prompt.py')
        bash_path = os.path.join(
            project_root, 'openhands/agenthub/codeact_agent/tools/bash.py'
        )
        tools_prompt_path = os.path.join(
            project_root, 'openhands/agenthub/codeact_agent/tools/prompt.py'
        )

        # Check if all files exist
        if not all(
            os.path.exists(path) for path in [prompt_path, bash_path, tools_prompt_path]
        ):
            self.skipTest('One or more required files do not exist')

        # Read the files
        with open(prompt_path, 'r') as f:
            prompt_code = f.read()

        with open(bash_path, 'r') as f:
            bash_code = f.read()

        with open(tools_prompt_path, 'r') as f:
            tools_prompt_code = f.read()

        # Check for the problematic imports
        prompt_imports_bash = (
            re.search(
                r'from openhands\.agenthub\.codeact_agent\.tools\.bash import',
                prompt_code,
            )
            is not None
        )
        bash_imports_tools_prompt = (
            re.search(
                r'from openhands\.agenthub\.codeact_agent\.tools\.prompt import',
                bash_code,
            )
            is not None
        )
        tools_prompt_imports_prompt = (
            re.search(r'from openhands\.utils\.prompt import', tools_prompt_code)
            is not None
        )

        # If all three imports exist, we have a circular import
        if (
            prompt_imports_bash
            and bash_imports_tools_prompt
            and tools_prompt_imports_prompt
        ):
            self.fail(
                'Circular import pattern detected:\n'
                'openhands.utils.prompt imports from openhands.agenthub.codeact_agent.tools.bash\n'
                'openhands.agenthub.codeact_agent.tools.bash imports from openhands.agenthub.codeact_agent.tools.prompt\n'
                'openhands.agenthub.codeact_agent.tools.prompt imports from openhands.utils.prompt'
            )

    def test_detect_circular_imports_in_server_modules(self):
        """Test for circular imports in the server modules that were involved in the stack trace.

        The problematic modules were:
        - openhands.server.shared
        - openhands.server.conversation_manager.conversation_manager
        - openhands.server.session.agent_session
        - openhands.server.session
        - openhands.server.session.session
        """
        # Get the project root directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

        # Map module names to file paths
        module_paths = {
            'openhands.server.shared': os.path.join(
                project_root, 'openhands/server/shared.py'
            ),
            'openhands.server.conversation_manager.conversation_manager': os.path.join(
                project_root,
                'openhands/server/conversation_manager/conversation_manager.py',
            ),
            'openhands.server.session.agent_session': os.path.join(
                project_root, 'openhands/server/session/agent_session.py'
            ),
            'openhands.server.session.__init__': os.path.join(
                project_root, 'openhands/server/session/__init__.py'
            ),
            'openhands.server.session.session': os.path.join(
                project_root, 'openhands/server/session/session.py'
            ),
        }

        # Check for circular imports
        circular_imports = self._find_circular_imports(module_paths)

        # If there are any circular imports, fail the test
        if circular_imports:
            circular_import_str = '\n'.join(
                [
                    f'{module1} -> {module2} -> {module1}'
                    for module1, module2 in circular_imports
                ]
            )
            self.fail(
                f'Circular imports detected in server modules:\n{circular_import_str}'
            )

    def test_detect_circular_imports_in_mcp_modules(self):
        """Test for circular imports in the MCP modules that were involved in the stack trace.

        The problematic modules were:
        - openhands.mcp
        - openhands.mcp.utils
        - openhands.memory.memory
        """
        # Get the project root directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

        # Map module names to file paths
        module_paths = {
            'openhands.mcp.__init__': os.path.join(
                project_root, 'openhands/mcp/__init__.py'
            ),
            'openhands.mcp.utils': os.path.join(project_root, 'openhands/mcp/utils.py'),
            'openhands.memory.memory': os.path.join(
                project_root, 'openhands/memory/memory.py'
            ),
        }

        # Check for circular imports
        circular_imports = self._find_circular_imports(module_paths)

        # If there are any circular imports, fail the test
        if circular_imports:
            circular_import_str = '\n'.join(
                [
                    f'{module1} -> {module2} -> {module1}'
                    for module1, module2 in circular_imports
                ]
            )
            self.fail(
                f'Circular imports detected in MCP modules:\n{circular_import_str}'
            )

    def test_detect_complex_circular_import_chains(self):
        """Test for complex circular import chains involving multiple modules.

        This test checks for circular dependencies that involve more than two modules,
        such as A imports B, B imports C, and C imports A.
        """
        # Get the project root directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

        # Define the modules involved in the stack trace
        modules = [
            'openhands.utils.prompt',
            'openhands.agenthub.codeact_agent.tools.bash',
            'openhands.agenthub.codeact_agent.tools.prompt',
            'openhands.memory.memory',
            'openhands.memory.conversation_memory',
            'openhands.server.shared',
            'openhands.server.conversation_manager.conversation_manager',
            'openhands.server.session.agent_session',
            'openhands.server.session.__init__',
            'openhands.server.session.session',
            'openhands.mcp.__init__',
            'openhands.mcp.utils',
        ]

        # Map module names to file paths
        module_paths = {}
        for module in modules:
            if module.endswith('.__init__'):
                # Handle __init__.py files
                module_path = module[:-9].replace('.', '/')
                file_path = os.path.join(project_root, f'{module_path}/__init__.py')
            else:
                # Handle regular .py files
                module_path = module.replace('.', '/')
                file_path = os.path.join(project_root, f'{module_path}.py')

            if os.path.exists(file_path):
                module_paths[module] = file_path

        # Build the import graph
        import_graph = {}
        for module_name, file_path in module_paths.items():
            with open(file_path, 'r') as f:
                source_code = f.read()

            # Extract import statements
            import_lines = [
                line.strip()
                for line in source_code.split('\n')
                if line.strip().startswith(('import ', 'from '))
                and not line.strip().startswith('# ')
            ]

            # Parse import statements to get imported modules
            imported_modules = []
            for line in import_lines:
                if line.startswith('import '):
                    # Handle "import module" or "import module as alias"
                    parts = line[7:].split(',')
                    for part in parts:
                        module_part = part.strip().split(' as ')[0].strip()
                        if module_part.startswith('openhands.'):
                            imported_modules.append(module_part)
                elif line.startswith('from '):
                    # Handle "from module import name" or "from module import name as alias"
                    module_part = line[5:].split(' import ')[0].strip()
                    if module_part.startswith('openhands.'):
                        imported_modules.append(module_part)

            import_graph[module_name] = [
                m for m in imported_modules if m in module_paths
            ]

        # Check for circular import chains
        circular_chains = self._find_circular_chains(import_graph)

        # If there are any circular chains, fail the test
        if circular_chains:
            circular_chain_str = '\n'.join(
                [' -> '.join(chain) for chain in circular_chains]
            )
            self.fail(f'Complex circular import chains detected:\n{circular_chain_str}')

    def _find_circular_chains(
        self, import_graph: dict[str, list[str]]
    ) -> list[list[str]]:
        """Find circular import chains in the import graph.

        Args:
            import_graph: Dictionary mapping module names to lists of imported modules

        Returns:
            List of circular import chains, where each chain is a list of module names
        """
        circular_chains = []

        def dfs(module: str, path: list[str], visited: set[str]):
            """Depth-first search to find circular import chains.

            Args:
                module: Current module being visited
                path: Current path in the DFS
                visited: Set of modules visited in the current DFS path
            """
            if module in visited:
                # Found a circular import chain
                cycle_start = path.index(module)
                circular_chains.append(path[cycle_start:] + [module])
                return

            visited.add(module)
            path.append(module)

            for imported_module in import_graph.get(module, []):
                dfs(imported_module, path.copy(), visited.copy())

        # Start DFS from each module
        for module in import_graph:
            dfs(module, [], set())

        return circular_chains


if __name__ == '__main__':
    unittest.main()
