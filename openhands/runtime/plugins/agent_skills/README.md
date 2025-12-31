# OpenHands Skill Sets

This folder implements a skill/tool set `agentskills` for OpenHands.

It is intended to be used by the agent **inside sandbox**.
The skill set will be exposed as a `pip` package that can be installed as a plugin inside the sandbox.

The skill set can contain a bunch of wrapped tools for agent ([many examples here](https://github.com/OpenHands/OpenHands/pull/1914)), for example:
- Audio/Video to text (these are a temporary solution, and we should switch to multimodal models when they are sufficiently cheap
- PDF to text
- etc.

# Inclusion Criteria

We are walking a fine line here.
We DON't want to *wrap* every possible python packages and re-teach agent their usage (e.g., LLM already knows `pandas` pretty well, so we don't really need create a skill that reads `csv` - it can just use `pandas`).

We ONLY want to add a new skill, when:
- Such skill is not easily achievable for LLM to write code directly (e.g., edit code and replace certain line)
- It involves calling an external model (e.g., you need to call a speech to text model, editor model for speculative editing)

# Intended functionality

- Tool/skill usage (through `IPythonRunAction`)

```python
# In[1]
from agentskills import open_file, edit_file
open_file("/workspace/a.txt")
# Out[1]
[SWE-agent open output]

# In[2]
edit_file(
    "/workspace/a.txt",
    start=1, end=3,
    content=(
        ("REPLACE TEXT")
))
# Out[1]
[SWE-agent edit output]
```

- Tool/skill retrieval (through `IPythonRunAction`)

```python
# In[1]
from agentskills import help_me

help_me("I want to solve a task that involves reading a bunch of PDFs and reason about them")

# Out[1]
"Here are the top skills that may be helpful to you:
- `pdf_to_text`: [documentation about the tools]
...
"
```
