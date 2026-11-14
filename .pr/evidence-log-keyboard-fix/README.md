# Automation log tab keyboard fix

Running-Canvas evidence for #19522, captured from source commit
`d7d74dd4e21ef99ba77f75cdb006875493280bad` with local Agent Server 1.60.1 and
automation 1.30.1. No model was configured or called.

The real script automation used the same two-stream fixture as the issue:

```python
import sys

print("stdout-visible")
print("stderr-keyboard-proof", file=sys.stderr)
```

The recording opens that run's Logs dialog, focuses Output, and drives
ArrowRight, wrapping ArrowRight, ArrowLeft, Home, and End. The observations
along the way confirmed that:

- ArrowRight selected Error and displayed `stderr-keyboard-proof`.
- ArrowRight from Error wrapped to Output and displayed `stdout-visible`.
- ArrowLeft wrapped from Output to Error.
- Home selected Output and End selected Error.
- The selected tab kept `aria-selected="true"` and focus moved with selection.

Artifacts:

- `after.webm`: complete keyboard navigation recording.
- `after-arrow-error.png`: Error selected after ArrowRight with the real stderr.

The browser reported no application errors, and the isolated stack's final
health check passed before the fixture was deleted.
