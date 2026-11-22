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

## Project layout

```
src/dc_ai_assistant/
  data_assistant/    Part 1: text-to-SQL
  rules_assistant/   Part 2: RAG over public documents
data/                generated database and downloaded documents (not committed)
tests/
```
