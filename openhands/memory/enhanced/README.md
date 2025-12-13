# OpenHands Enhanced Memory System

> **Transform OpenHands from a goldfish into a long-lived software engineer**

This enhanced memory system provides persistent learning and context-aware assistance for OpenHands, enabling it to:

- 🧠 **Remember** past tasks and their outcomes
- 🔍 **Understand** codebases through semantic analysis
- 🎯 **Apply** learned skills to new similar tasks
- 📈 **Improve** performance over time through experience

## Quick Start

```python
from openhands.memory.enhanced import MemorySystem

# Initialize memory system
memory = MemorySystem(repo_path="./my_project")

# Create and execute a task with memory
task = memory.create_task_context("Fix authentication bug")
context = memory.retrieve_context(task)  # Gets relevant past experience
```

## Architecture

```
Memory System
├── Working Memory      # Session context tracking
├── Semantic Memory     # Code understanding & retrieval
├── Episodic Memory     # Task experience storage
├── Skill Memory        # Reusable pattern extraction
└── Integration Layer   # OpenHands hooks
```

## Core Components

### 📝 Working Memory (`working_memory.py`)
- Tracks current session state
- Monitors file modifications
- Records action outcomes
- Provides session statistics

### 🔍 Semantic Code Memory (`semantic_memory.py`)
- Vector-based code understanding
- Relevant snippet retrieval
- Dependency graph analysis
- Incremental repository indexing

### 📚 Episodic Task Memory (`episodic_memory.py`)
- Stores completed task episodes
- Finds similar past experiences
- Analyzes success/failure patterns
- Provides statistical insights

### 🎯 Skill Memory (`skill_memory.py`)
- Extracts reusable task patterns
- Matches skills to new tasks
- Tracks usage and confidence
- Evolves through feedback

### 🔗 Integration Layer (`integration.py`)
- Memory-aware agent extension
- Enhanced prompt generation
- Post-task learning hooks
- OpenHands core integration

## Key Features

### 🎯 Context-Aware Assistance
- Automatically retrieves relevant code snippets
- Suggests applicable skills for current tasks
- Warns about potential pitfalls from past failures
- Provides success patterns from similar tasks

### 📈 Continuous Learning
- Learns from every task completion
- Extracts reusable skills from successful patterns
- Updates confidence scores based on outcomes
- Cleans up outdated or unused knowledge

### 🔄 Incremental Knowledge Building
- Repository-specific understanding grows over time
- Skills improve with successful usage
- Failed approaches reduce in confidence
- New patterns emerge from repeated successes

## Usage Examples

### Basic Memory Operations

```python
# Create task context
task = memory.create_task_context(
    intent="Fix login validation bug",
    files_in_scope=["auth.py", "login.py"],
    constraints=["Don't break existing tests"]
)

# Get relevant context
context = memory.retrieve_context(task)
print(context.to_prompt_context())

# Update working memory
memory.update_file_state("auth.py", new_content)

# Complete task and learn
outcome = TaskOutcome(task_id=task.task_id, success=True, ...)
await memory.learn_from_task(task, outcome)
```

### Memory-Enhanced Agent

```python
from openhands.memory.enhanced import MemoryAwareCodeActAgent

agent = MemoryAwareCodeActAgent(
    config=config,
    llm_registry=llm_registry,
    repo_path="./my_project"
)

# Agent automatically uses memory for enhanced context
action = agent.step(state)  # Includes relevant past experience
```

### Full Integration

```python
from openhands.memory.enhanced.integration import create_memory_enhanced_openhands

setup = create_memory_enhanced_openhands(
    config=config,
    llm_registry=llm_registry,
    event_stream=event_stream,
    sid=session_id,
    repo_path="./my_project"
)

controller = setup['controller']  # Memory-enhanced controller
memory_system = setup['memory_system']  # Full memory system
```

## File Structure

```
openhands/memory/enhanced/
├── __init__.py              # Public API exports
├── types.py                 # Core type definitions
├── working_memory.py        # Session context tracking
├── semantic_memory.py       # Code understanding & retrieval
├── episodic_memory.py       # Task experience storage
├── skill_memory.py          # Reusable pattern extraction
├── memory_system.py         # Main coordination layer
├── enhanced_recall.py       # Memory-aware recall actions
├── memory_aware_agent.py    # Enhanced CodeAct agent
├── integration.py           # OpenHands integration hooks
└── README.md               # This file
```

## Testing

Run the basic functionality test:

```bash
python test_memory_basic.py
```

Expected output shows all memory components working correctly:
- ✅ Working Memory: Session tracking, file states, action outcomes
- ✅ Type System: Task classification, scope determination
- ✅ Episodic Memory: Task storage, similarity search, statistics
- ✅ Skill Memory: Skill storage, pattern matching, retrieval

## Memory Storage

The system creates a `.openhands/memory/` directory containing:

```
.openhands/memory/
├── semantic_memory.db      # ChromaDB vector database
├── episodic_memory.db      # SQLite task episodes
├── skill_memory.db         # SQLite skill patterns
└── embeddings/             # Cached embeddings
```

## Performance

- **Memory Usage**: 1-500 MB depending on repository size
- **Scalability**: Supports repositories up to 100k files
- **Efficiency**: Incremental indexing and LRU caching
- **Cleanup**: Automatic removal of old unused data

## Integration Points

The memory system hooks into OpenHands at key points:

1. **Before Planning**: Retrieve relevant context and applicable skills
2. **During Planning**: Enhance prompts with memory-derived insights
3. **During Execution**: Track file changes and action outcomes
4. **After Completion**: Learn from task results and extract patterns

## Task Classification

Automatically classifies tasks into types for better memory organization:

- **DEBUG**: Bug fixes and error resolution
- **IMPLEMENT**: New feature development
- **REFACTOR**: Code restructuring
- **TEST**: Test creation and verification
- **DOCUMENT**: Documentation writing
- **ANALYZE**: Code analysis and understanding
- **OPTIMIZE**: Performance improvements
- **REVIEW**: Code review and auditing

## Dependencies

```bash
pip install chromadb sentence-transformers networkx sqlalchemy
```

## Future Enhancements

- 🌐 Multi-repository memory sharing
- 🤖 Advanced ML-based skill extraction
- ⚡ Real-time code analysis
- 🗜️ Memory compression and optimization
- 👥 Collaborative team learning

## Contributing

The memory system is designed to be:
- **Extensible**: Easy to add new memory types
- **Composable**: Components work independently
- **Hook-based**: Minimal changes to OpenHands core
- **Incremental**: Can be adopted gradually

To contribute:
1. Follow existing patterns in component design
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure backward compatibility

---

**Result**: OpenHands transforms from a stateless agent into a learning software engineer that gets better with every task, remembers what works, and applies past experience to new challenges.
