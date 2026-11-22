"""Settings read from environment variables (or a .env file in the project root)."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    """Minimal .env loader: KEY=VALUE lines.

    Empty values are ignored, a later line overrides an earlier one, and
    variables already set in the environment always win.
    """
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    values: dict[str, str] = {}
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            values[key.strip()] = value
    for key, value in values.items():
        if not os.environ.get(key):
            os.environ[key] = value


_load_dotenv()

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")
OPENAI_TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "0"))
# gpt-5 models accept a temperature only with reasoning off ("none", also the model default)
OPENAI_REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "none")

WAREHOUSE_DB_PATH = Path(os.environ.get("WAREHOUSE_DB_PATH", PROJECT_ROOT / "data" / "generated" / "warehouse.db"))

# SQL tool limits
SQL_MAX_ROWS = int(os.environ.get("SQL_MAX_ROWS", "200"))
SQL_TIMEOUT_SECONDS = float(os.environ.get("SQL_TIMEOUT_SECONDS", "10"))
