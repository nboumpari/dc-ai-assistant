"""Run read-only SQL against the warehouse database, safely.

Safety is enforced by SQLite itself, not by inspecting the query text:

1. The database file is opened read-only.
2. An authorizer approves every operation SQLite prepares. Only reading
   (SELECT, column reads, functions, recursive CTEs) is allowed. Writes,
   schema changes, PRAGMA and ATTACH are denied, and so is reading any column
   marked sensitive in the semantic layer, wherever it appears in the query.
3. A progress handler stops queries that run longer than the timeout.
4. At most max_rows rows are returned.

Errors are returned as text, not raised, so the agent can read them and fix
its query.
"""

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from dc_ai_assistant import settings
from dc_ai_assistant.data_assistant.semantic_layer import sensitive_columns

_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}


@dataclass
class QueryResult:
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    truncated: bool = False
    error: str | None = None
    elapsed_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_text(self) -> str:
        """Compact text for the LLM: header, rows separated by ' | ', and a summary line."""
        if self.error:
            return f"ERROR: {self.error}"
        if not self.rows:
            return f"{' | '.join(self.columns)}\n(0 rows)"
        lines = [" | ".join(self.columns)]
        lines += [" | ".join("NULL" if v is None else str(v) for v in row) for row in self.rows]
        summary = f"({len(self.rows)} rows"
        summary += (
            f", truncated at {len(self.rows)}: add LIMIT or aggregate to see the full answer)"
            if self.truncated
            else ")"
        )
        return "\n".join(lines + [summary])


class SQLTool:
    def __init__(
        self,
        db_path: Path | None = None,
        max_rows: int | None = None,
        timeout_seconds: float | None = None,
        blocked_columns: set[tuple[str, str]] | None = None,
    ) -> None:
        self.db_path = Path(db_path or settings.WAREHOUSE_DB_PATH)
        self.max_rows = max_rows or settings.SQL_MAX_ROWS
        self.timeout_seconds = timeout_seconds or settings.SQL_TIMEOUT_SECONDS
        self.blocked_columns = sensitive_columns() if blocked_columns is None else blocked_columns
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Database not found at {self.db_path}. Run: uv run python -m scripts.data_generator.generate"
            )

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(f"{self.db_path.resolve().as_uri()}?mode=ro", uri=True, check_same_thread=False)
        con.set_authorizer(self._authorize)
        return con

    def _authorize(self, action: int, arg1: str | None, arg2: str | None, _db: str | None, _trigger: str | None) -> int:
        if action not in _ALLOWED_ACTIONS:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_READ and (arg1, arg2) in self.blocked_columns:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def run(self, query: str) -> QueryResult:
        query = query.strip().rstrip(";").strip()
        if not query:
            return QueryResult(error="Empty query.")

        start = time.perf_counter()
        deadline = start + self.timeout_seconds
        con = self._connect()
        con.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 10_000)
        try:
            cursor = con.execute(query)
            columns = [d[0] for d in cursor.description or []]
            rows = cursor.fetchmany(self.max_rows + 1)
            truncated = len(rows) > self.max_rows
            return QueryResult(
                columns=columns,
                rows=rows[: self.max_rows],
                truncated=truncated,
                elapsed_ms=round((time.perf_counter() - start) * 1000),
            )
        except sqlite3.DatabaseError as exc:
            return QueryResult(error=self._explain(exc), elapsed_ms=round((time.perf_counter() - start) * 1000))
        except (sqlite3.Warning, sqlite3.ProgrammingError) as exc:
            return QueryResult(error=self._explain(exc))
        finally:
            con.close()

    def _explain(self, exc: Exception) -> str:
        message = str(exc)
        if message == "not authorized" or "prohibited" in message:
            blocked = ", ".join(f"{t}.{c}" for t, c in sorted(self.blocked_columns))
            return (
                "Not allowed. Only read-only SELECT queries are permitted, and these columns may not be read: "
                f"{blocked}. If you used SELECT *, list the columns you need instead."
            )
        if message == "interrupted":
            return f"Query stopped after {self.timeout_seconds:g} seconds. Simplify it or filter earlier."
        if "one statement at a time" in message:
            return "Only one statement per call is allowed."
        return message
