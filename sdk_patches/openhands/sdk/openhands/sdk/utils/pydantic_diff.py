from collections.abc import Mapping, Sequence

from pydantic import BaseModel


def _normalize(x):
    # Convert Pydantic models to dicts
    if isinstance(x, BaseModel):
        return x.model_dump(exclude_none=True)
    # Recurse mappings and sequences (but not strings/bytes)
    if isinstance(x, Mapping):
        return {k: _normalize(v) for k, v in x.items()}
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray)):
        return [_normalize(v) for v in x]
    return x


def _structured_diff(a, b):
    a = _normalize(a)
    b = _normalize(b)

    # Equal after normalization -> no diff
    if a == b:
        return {}

    # Dict vs dict: diff by keys
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        keys = set(a) | set(b)
        out = {}
        for k in sorted(keys, key=lambda x: (str(type(x)), str(x))):
            ak = a.get(k, ...)
            bk = b.get(k, ...)
            if ak is ...:
                out[k] = ("<missing>", bk)
            elif bk is ...:
                out[k] = (ak, "<missing>")
            else:
                sub = _structured_diff(ak, bk)
                out[k] = sub if sub else (ak, bk) if ak != bk else {}
        # Remove entries that ended up equal (empty dicts)
        return {k: v for k, v in out.items() if v != {}}

    # List/tuple vs list/tuple: diff by index
    if (
        isinstance(a, Sequence)
        and isinstance(b, Sequence)
        and not isinstance(a, (str, bytes, bytearray))
        and not isinstance(b, (str, bytes, bytearray))
    ):
        out = {}
        n = max(len(a), len(b))
        for i in range(n):
            ai = a[i] if i < len(a) else ...
            bi = b[i] if i < len(b) else ...
            if ai is ...:
                out[i] = ("<missing>", bi)
            elif bi is ...:
                out[i] = (ai, "<missing>")
            else:
                sub = _structured_diff(ai, bi)
                out[i] = sub if sub else (ai, bi) if ai != bi else {}
        return {k: v for k, v in out.items() if v != {}}

    # Fallback leaf difference
    return (a, b)


def _format_diff(d, indent=0):
    if not isinstance(d, Mapping):
        old, new = d
        return f"{'  ' * indent}{old!r} -> {new!r}"
    lines = []
    pad = "  " * indent
    for key, val in d.items():
        if isinstance(val, Mapping):
            lines.append(f"{pad}{key}:")
            lines.append(_format_diff(val, indent + 1))
        else:
            lines.append(f"{pad}{key}: {_format_diff(val, indent + 1).lstrip()}")
    return "\n".join(lines)


def pretty_pydantic_diff(a: BaseModel, b: BaseModel) -> str:
    diff = _structured_diff(a, b)
    return "No differences" if not diff else _format_diff(diff)
