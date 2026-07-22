"""Postgres SSL mode helpers shared by app and migration database setup."""

SUPPORTED_DB_SSL_MODES: frozenset[str] = frozenset({'prefer', 'require', 'disable'})


def normalize_db_ssl_mode(db_ssl_mode: str | None) -> str | None:
    if db_ssl_mode is None:
        return None

    mode = db_ssl_mode.strip().lower()
    if not mode or mode == 'prefer':
        return None
    if mode not in SUPPORTED_DB_SSL_MODES:
        raise ValueError(
            f'Unsupported DB_SSL_MODE "{db_ssl_mode}". '
            f'Supported values are: {", ".join(sorted(SUPPORTED_DB_SSL_MODES))}.'
        )
    return mode


def build_pg8000_connect_args(db_ssl_mode: str | None) -> dict[str, bool]:
    mode = normalize_db_ssl_mode(db_ssl_mode)
    if mode == 'require':
        return {'ssl_context': True}
    if mode == 'disable':
        return {'ssl_context': False}
    return {}


def build_asyncpg_connect_args(db_ssl_mode: str | None) -> dict[str, str]:
    mode = normalize_db_ssl_mode(db_ssl_mode)
    if mode:
        return {'ssl': mode}
    return {}


def build_db_url_query(db_ssl_mode: str | None) -> str:
    mode = normalize_db_ssl_mode(db_ssl_mode)
    if mode:
        return f'?sslmode={mode}'
    return ''
