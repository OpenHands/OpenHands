def looks_like_jwt(token_value: str) -> bool:
    parts = token_value.split('.')
    return len(parts) == 3 and all(parts)
