from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker


def fingerprint(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]


def import_models() -> None:
    from seed_live_db import import_models as _import_models

    _import_models()


def probe(db_path: Path) -> dict[str, object]:
    import_models()
    from storage.org_member import OrgMember

    engine = create_engine(
        f'sqlite:///{db_path}', connect_args={'check_same_thread': False}
    )
    Session = sessionmaker(bind=engine)
    with Session() as session:
        rows = session.execute(select(OrgMember)).scalars().all()
        return {
            'members': [
                {
                    'user_id': str(row.user_id),
                    'org_id': str(row.org_id),
                    'has_custom_llm_api_key': row.has_custom_llm_api_key,
                    'llm_key_fingerprint': fingerprint(
                        row.llm_api_key.get_secret_value() if row._llm_api_key else None
                    ),
                    'byor_key_fingerprint': fingerprint(
                        row.llm_api_key_for_byor.get_secret_value()
                        if row._llm_api_key_for_byor
                        else None
                    ),
                }
                for row in rows
            ]
        }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit('usage: probe_live_db.py <sqlite-db-path>')
    print(json.dumps(probe(Path(sys.argv[1])), indent=2))


if __name__ == '__main__':
    main()
