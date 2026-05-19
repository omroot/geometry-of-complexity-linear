"""WRDS credential loader: env vars > project .env > existing ~/.pgpass.

The :mod:`wrds` package (and OpenAP, which uses it internally) delegate to
libpq, which reads PostgreSQL's standard ``~/.pgpass`` (or whatever path is
in ``PGPASSFILE``). To let new contributors avoid setting up a global
``~/.pgpass``, this module reads ``WRDS_USERNAME`` / ``WRDS_PASSWORD`` from
the environment or from a project-local ``.env`` file, materializes a
runtime ``.pgpass`` (mode 600) inside the cache directory, and points
``PGPASSFILE`` at it.

Precedence:
    1. ``WRDS_USERNAME`` / ``WRDS_PASSWORD`` already in ``os.environ``
    2. Same keys in a ``.env`` file at the project root
    3. Caller falls back to the existing ``~/.pgpass`` (or interactive prompt)
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

# WRDS PostgreSQL endpoint — stable for years, unlikely to change.
WRDS_HOST = "wrds-pgdata.wharton.upenn.edu"
WRDS_PORT = 9737
WRDS_DATABASE = "wrds"


def _parse_dotenv(path: Path) -> dict[str, str]:
    """Tiny ``KEY=VALUE`` parser. Strips matching outer single/double quotes.

    Does not handle escapes, multi-line values, or variable interpolation —
    if you need any of those, switch to ``python-dotenv``. For WRDS creds
    this is sufficient.

    Args:
        path: Path to the ``.env`` file. Missing file → empty dict.

    Returns:
        Mapping of keys to values.
    """
    pairs: dict[str, str] = {}
    if not path.exists():
        return pairs
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        pairs[key.strip()] = value
    return pairs


def materialize_wrds_pgpass(
    project_root: Path,
    runtime_pgpass: Path,
    env_path: Path | None = None,
) -> bool:
    """Resolve WRDS credentials and write a runtime ``.pgpass``.

    Resolution order (first hit wins):
        1. ``WRDS_USERNAME`` and ``WRDS_PASSWORD`` in :data:`os.environ`
        2. Same keys in the project-root ``.env`` file

    If neither source provides both values, returns False without touching
    the filesystem or ``PGPASSFILE`` — the caller is then free to rely on
    the user's existing ``~/.pgpass`` or the interactive WRDS prompt.

    Args:
        project_root: Repository root; used as the default ``.env`` location.
        runtime_pgpass: Where to write the runtime ``.pgpass`` file. Will
            be created with mode 600 (owner read/write only).
        env_path: Override the default ``project_root / ".env"`` path.

    Returns:
        True if credentials were resolved and ``PGPASSFILE`` is now set
        to the runtime pgpass; False if no credentials were found.
    """
    username = os.environ.get("WRDS_USERNAME")
    password = os.environ.get("WRDS_PASSWORD")

    if not (username and password):
        if env_path is None:
            env_path = project_root / ".env"
        pairs = _parse_dotenv(env_path)
        username = username or pairs.get("WRDS_USERNAME")
        password = password or pairs.get("WRDS_PASSWORD")

    if not (username and password):
        return False

    line = f"{WRDS_HOST}:{WRDS_PORT}:{WRDS_DATABASE}:{username}:{password}\n"
    runtime_pgpass.parent.mkdir(parents=True, exist_ok=True)
    runtime_pgpass.write_text(line, encoding="utf-8")
    runtime_pgpass.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0o600
    os.environ["PGPASSFILE"] = str(runtime_pgpass)
    return True


def have_wrds_credentials(project_root: Path) -> str:
    """Report which credential source is available, for diagnostics.

    Args:
        project_root: Repository root; used to locate ``.env``.

    Returns:
        One of ``"env"`` (env vars set), ``"dotenv"`` (.env has both keys),
        ``"pgpass"`` (~/.pgpass exists), or ``"none"``.
    """
    if os.environ.get("WRDS_USERNAME") and os.environ.get("WRDS_PASSWORD"):
        return "env"
    env_pairs = _parse_dotenv(project_root / ".env")
    if env_pairs.get("WRDS_USERNAME") and env_pairs.get("WRDS_PASSWORD"):
        return "dotenv"
    if (Path.home() / ".pgpass").exists():
        return "pgpass"
    return "none"
