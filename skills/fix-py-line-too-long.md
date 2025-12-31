---
name: fix-py-line-too-long
type: knowledge
version: 1.0.0
agent: CodeActAgent
triggers:
- E501
- line too long
---

# Instructions for fixing "E501 Line too long"

## For code lines
Break into multiple lines using parentheses or brackets:
```python
result = some_very_long_function_name(
    parameter1, parameter2, parameter3
)
```

## For single-line strings
Use string concatenation: `"ABC"` → `("A" "B" "C")`
```python
message = ("This is a very long string "
           "that needs to be broken up")
```

## For long multi-line strings (docstrings)
Add `# noqa: E501` AFTER the ending `"""`. NEVER add it inside the docstring.
```python
def example_function():
    """This is a very long docstring that exceeds the line length limit."""  # noqa: E501
    pass
```

## What NOT to do
- Do not add `# noqa: E501` inside docstrings or multi-line strings
- Do not break strings in the middle of words
- Do not sacrifice code readability for line length compliance
