# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

_SIMPLIFIED_STRUCTURE_EXPLORER_DESCRIPTION = """
A unified tool that traverses a pre-built code graph to retrieve dependency structure around specified entities,
with options to explore upstream or downstream, and control traversal depth and filters for entity and dependency types.
"""


_SIMPLIFIED_TREE_EXAMPLE = """
Example Usage:
1. Exploring Downstream Dependencies:
    ```
    explore_tree_structure(
        start_entities=['src/module_a.py:ClassA'],
        direction='downstream',
        traversal_depth=2,
        dependency_type_filter=['invokes', 'imports']
    )
    ```
2. Exploring the repository structure from the root directory (/) up to two levels deep:
    ```
    explore_tree_structure(
      start_entities=['/'],
      traversal_depth=2,
      dependency_type_filter=['contains']
    )
    ```
3. Generate Class Diagrams:
    ```
    explore_tree_structure(
        start_entities=selected_entity_ids,
        direction='both',
        traverse_depth=-1,
        dependency_type_filter=['inherits']
    )
    ```
"""


_DETAILED_STRUCTURE_EXPLORER_DESCRIPTION = """
Unified repository exploring tool that traverses a pre-built code graph to retrieve dependency structure around specified entities.
The search can be controlled to traverse upstream (exploring dependencies that entities rely on) or downstream (exploring how entities impact others), with optional limits on traversal depth and filters for entity and dependency types.

Code Graph Definition:
* Entity Types: 'directory', 'file', 'class', 'function'.
* Dependency Types: 'contains', 'imports', 'invokes', 'inherits'.
* Hierarchy:
    - Directories contain files and subdirectories.
    - Files contain classes and functions.
    - Classes contain inner classes and methods.
    - Functions can contain inner functions.
* Interactions:
    - Files/classes/functions can import classes and functions.
    - Classes can inherit from other classes.
    - Classes and functions can invoke others (invocations in a class's `__init__` are attributed to the class).
Entity ID:
* Unique identifier including file path and module path.
* Here's an example of an Entity ID: `"interface/C.py:C.method_a.inner_func"` identifies function `inner_func` within `method_a` of class `C` in `"interface/C.py"`.

Notes:
* Traversal Control: The `traversal_depth` parameter specifies how deep the function should explore the graph starting from the input entities.
* Filtering: Use `entity_type_filter` and `dependency_type_filter` to narrow down the scope of the search, focusing on specific entity types and relationships.

"""


_DETAILED_TREE_EXAMPLE = """
Example Usage:
1. Exploring Outward Dependencies:
    ```
    explore_tree_structure(
        start_entities=['src/module_a.py:ClassA'],
        direction='downstream',
        traversal_depth=2,
        dependency_type_filter=['invokes', 'imports']
    )
    ```
    This retrieves the dependencies of `ClassA` up to 2 levels deep, focusing only on classes and functions with 'invokes' and 'imports' relationships.

2. Exploring Inward Dependencies:
    ```
    explore_tree_structure(
        start_entities=['src/module_b.py:FunctionY'],
        direction='upstream',
        traversal_depth=-1
    )
    ```
    This finds all entities that depend on `FunctionY` without restricting the traversal depth.
3. Exploring Repository Structure:
    ```
    explore_tree_structure(
      start_entities=['/'],
      traversal_depth=2,
      dependency_type_filter=['contains']
    )
    ```
    This retrieves the tree repository structure from the root directory (/), traversing up to two levels deep and focusing only on 'contains' relationship.
4. Generate Class Diagrams:
    ```
    explore_tree_structure(
        start_entities=selected_entity_ids,
        direction='both',
        traverse_depth=-1,
        dependency_type_filter=['inherits']
    )
    ```
"""


_STRUCTURE_EXPLORER_PARAMETERS = {
    'type': 'object',
    'properties': {
        'start_entities': {
            'description': (
                'List of entities (e.g., class, function, file, or directory paths) to begin the search from.\n'
                'Entities representing classes or functions must be formatted as "file_path:QualifiedName" (e.g., `interface/C.py:C.method_a.inner_func`).\n'
                'For files or directories, provide only the file or directory path (e.g., `src/module_a.py` or `src/`).'
            ),
            'type': 'array',
            'items': {'type': 'string'},
        },
        'direction': {
            'description': (
                'Direction of traversal in the code graph; allowed options are: `upstream`, `downstream`, `both`.\n'
                "- 'upstream': Traversal to explore dependencies that the specified entities rely on (how they depend on others).\n"
                "- 'downstream': Traversal to explore the effects or interactions of the specified entities on others (how others depend on them).\n"
                "- 'both': Traversal on both direction."
            ),
            'type': 'string',
            'enum': ['upstream', 'downstream', 'both'],
            'default': 'downstream',
        },
        'traversal_depth': {
            'description': (
                'Maximum depth of traversal. A value of -1 indicates unlimited depth (subject to a maximum limit).'
                'Must be either `-1` or a non-negative integer (≥ 0).'
            ),
            'type': 'integer',
            'default': 2,
        },
        'entity_type_filter': {
            'description': (
                "List of entity types (e.g., 'class', 'function', 'file', 'directory') to include in the traversal. If None, all entity types are included."
            ),
            'type': ['array', 'null'],
            'items': {'type': 'string'},
            'default': None,
        },
        'dependency_type_filter': {
            'description': (
                "List of dependency types (e.g., 'contains', 'imports', 'invokes', 'inherits') to include in the traversal. If None, all dependency types are included."
            ),
            'type': ['array', 'null'],
            'items': {'type': 'string'},
            'default': None,
        },
    },
    'required': ['start_entities'],
}


def create_explore_tree_structure_tool(
    use_simplified_description: bool = False,
) -> ChatCompletionToolParam:
    description = (
        _SIMPLIFIED_STRUCTURE_EXPLORER_DESCRIPTION
        if use_simplified_description
        else _DETAILED_STRUCTURE_EXPLORER_DESCRIPTION
    )
    example = (
        _SIMPLIFIED_TREE_EXAMPLE
        if use_simplified_description
        else _DETAILED_TREE_EXAMPLE
    )
    return ChatCompletionToolParam(
        type='function',
        function=ChatCompletionToolParamFunctionChunk(
            name='explore_tree_structure',
            description=description + example,
            parameters=_STRUCTURE_EXPLORER_PARAMETERS,
        ),
    )
