# PR reviewer automation — test fixture

Temporary file used to exercise the OpenHands PR review automation
end to end. It is not referenced by any build, test, or runtime path.

Delete this file and close the PR once the automation has posted a review.

## Deliberate review bait

```python
def divide(numerator, denominator):
    # no zero check
    return numerator / denominator


def read_config(path):
    # file handle is never closed
    return open(path).read()
```
