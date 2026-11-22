# DC AI Assistant

Two separate AI assistants for the control tower of a fictional FMCG manufacturer with several distribution centres.

| | Part 1: Data assistant | Part 2: Rules assistant |
|---|---|---|
| Answers | How many, which, when, where | What's the rule, what's allowed, what's required |
| Source | Synthetic 17-table warehouse database (orders, transfers, returns, shipments, pallets/SSCC, inventory, dock appointments) | Real public documents: GS1 standards, CMR, EU driving times, ADR (aerosols), EU GPSR, EU-OSHA guides, EPAL |
| Technique | Text-to-SQL | Retrieval-augmented generation with citations |

## Setup

```powershell
uv sync
```

## Generate the warehouse database

```powershell
uv run python -m scripts.data_generator.generate
```

This simulates one year (2025) of a fictional FMCG company: one plant, three DCs (Athens, Thessaloniki, Patras), 150 products and 80 customers. It writes:

- `data/generated/warehouse.db`: SQLite database, 18 tables, about 1.4 million rows
- `data/generated/injected_errors.csv`: every deliberate data error (duplicate scans, missing scans, wrong SSCC check digits, and so on)

The same seed always produces the same data, and the run stops if any consistency check fails.

## Run the tests

```powershell
uv run pytest
```

## Project layout

```
src/dc_ai_assistant/
  data_assistant/    Part 1: text-to-SQL
  rules_assistant/   Part 2: RAG over public documents
scripts/
  data_generator/    builds the synthetic warehouse database
data/                generated database and downloaded documents (not committed)
tests/
```
