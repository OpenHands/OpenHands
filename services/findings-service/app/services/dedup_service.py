import hashlib


def compute_dedupe_hash(
    engagement_id: str, title: str, asset: str | None, endpoint: str | None
) -> str:
    key = f"{engagement_id}:{title}:{asset or ''}:{endpoint or ''}"
    return hashlib.sha256(key.encode()).hexdigest()[:64]
