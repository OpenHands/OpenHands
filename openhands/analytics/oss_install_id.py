"""OSS install ID utility.

Provides a stable distinct_id for OSS installations by persisting a UUID
to a file in the persistence directory.
"""

import uuid
from pathlib import Path


def get_or_create_install_id(persistence_dir: Path) -> str:
    """Return a stable install UUID for an OSS installation.

    On first call, generates a new UUID4, writes it to
    ``{persistence_dir}/analytics_id.txt``, and returns it.
    On subsequent calls, reads and returns the stored UUID.

    On any IOError (e.g., read-only filesystem), returns an ephemeral UUID
    without crashing — the caller still gets a usable ID.
    """
    id_file = persistence_dir / 'analytics_id.txt'

    try:
        if id_file.exists():
            stored = id_file.read_text().strip()
            if stored:
                return stored
    except OSError:
        pass

    new_id = str(uuid.uuid4())

    try:
        id_file.write_text(new_id)
    except OSError:
        # File system is not writable — return ephemeral UUID
        pass

    return new_id
