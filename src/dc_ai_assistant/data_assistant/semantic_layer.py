"""Load the semantic layer and render it as text for the system prompt."""

from functools import cache
from pathlib import Path
from typing import Any

import yaml

SEMANTIC_LAYER_PATH = Path(__file__).with_name("semantic_layer.yaml")


@cache
def load_semantic_layer(path: Path = SEMANTIC_LAYER_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def sensitive_columns(layer: dict[str, Any] | None = None) -> set[tuple[str, str]]:
    """(table, column) pairs marked sensitive. The SQL tool blocks these."""
    layer = layer or load_semantic_layer()
    return {
        (table, column)
        for table, spec in layer["tables"].items()
        for column, col in spec["columns"].items()
        if col.get("sensitive")
    }


def render_semantic_layer(layer: dict[str, Any] | None = None, include_kpi_sql: bool = True) -> str:
    """Compact text version of the semantic layer for the system prompt."""
    layer = layer or load_semantic_layer()
    out: list[str] = []

    db = layer["database"]
    out.append(f"## Database\n{db['dialect']}. Data period: {db['period']}. Today is {db['today']}.")
    out += [f"- {note}" for note in db["notes"]]

    company = layer["company"]
    out.append(f"\n## Company\n{company['description']}")
    out += [f"- {flow}" for flow in company["flows"]]

    out.append("\n## Tables")
    for table, spec in layer["tables"].items():
        out.append(f"\n### {table}\n{spec['description']} Grain: {spec['grain']}.")
        for column, col in spec["columns"].items():
            parts = [col["description"]] if col.get("description") else []
            if col.get("values"):
                parts.append("Values: " + ", ".join(f"'{v}'" for v in col["values"]) + ".")
            if col.get("sensitive"):
                parts.append("SENSITIVE: never select or show this column.")
            out.append(f"- {column}: {' '.join(parts)}".rstrip())

    out.append("\n## Joins")
    out += [f"- {rel}" for rel in layer["relationships"]]

    out.append("\n## Business terms")
    out += [f"- {term}: {meaning}" for term, meaning in layer["business_terms"].items()]

    out.append("\n## Known data quality issues")
    out += [f"- {issue}" for issue in layer["data_quality"]]

    out.append("\n## KPIs")
    for key, kpi in layer["kpis"].items():
        out.append(f"\n### {kpi['name']} ({key})")
        out.append(f"{kpi['description']} Unit: {kpi['unit']}. Better: {kpi['better']}.")
        out.append(f"Formula: {kpi['formula']}")
        out += [f"- {rule}" for rule in kpi.get("rules", [])]
        splits = ", ".join(f"{name} = {expr}" for name, expr in kpi["split_by"].items())
        out.append(f"Can be split by: {splits}")
        if include_kpi_sql:
            out.append(f"Reference SQL (whole period, no split):\n```sql\n{kpi['sql'].strip()}\n```")

    return "\n".join(out)
