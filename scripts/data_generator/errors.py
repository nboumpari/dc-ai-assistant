"""Deliberate data errors, so the data assistant can show it notices bad data.

The simulation produces clean data. This module damages a small, controlled
part of it and logs every change, so we know exactly what the assistant
should find. Rates are set in config.ERROR_RATES.
"""

import csv
import random
from pathlib import Path

from . import config
from .master_data import gs1_check_digit

Rows = dict[str, list[tuple]]
LogEntry = tuple[str, str, str, str]  # error_type, table, record_id, detail

# column positions in the row tuples (see schema.sql)
PE_ID, PE_SSCC, PE_TYPE, PE_AT = 0, 1, 2, 3
PAL_SSCC, PAL_SHIPMENT = 0, 1
PC_SSCC = 0
RL_ID, RL_REQUESTED, RL_RECEIVED = 0, 5, 6
SH_ID, SH_TYPE = 0, 1


def inject_errors(rows: Rows, rng: random.Random) -> list[LogEntry]:
    log: list[LogEntry] = []
    _bad_sscc_check_digits(rows, rng, log)  # first, so later log entries use the final SSCCs
    _duplicate_pallet_scans(rows, rng, log)
    _missing_loaded_scans(rows, rng, log)
    _transfer_pallets_never_received(rows, rng, log)
    _return_qty_mismatches(rows, rng, log)
    return log


def _duplicate_pallet_scans(rows: Rows, rng: random.Random, log: list[LogEntry]) -> None:
    events = rows["pallet_events"]
    n = round(len(events) * config.ERROR_RATES["duplicate_pallet_scan"])
    for i, original in enumerate(rng.sample(events, n), start=1):
        dup = list(original)
        dup[PE_ID] = f"PE-DUP{i:06d}"
        events.append(tuple(dup))
        log.append(
            (
                "duplicate_pallet_scan",
                "pallet_events",
                dup[PE_ID],
                f"duplicate of {original[PE_ID]} ({original[PE_TYPE]})",
            )
        )


def _missing_loaded_scans(rows: Rows, rng: random.Random, log: list[LogEntry]) -> None:
    events = rows["pallet_events"]
    loaded = {e[PE_SSCC]: e for e in events if e[PE_TYPE] == "loaded"}
    departed = {e[PE_SSCC] for e in events if e[PE_TYPE] == "departed"}
    candidates = sorted(s for s in loaded if s in departed)
    n = round(len(rows["pallets"]) * config.ERROR_RATES["missing_loaded_scan"])
    chosen = sorted(rng.sample(candidates, min(n, len(candidates))))
    remove = {loaded[s][PE_ID] for s in chosen}
    rows["pallet_events"] = [e for e in events if e[PE_ID] not in remove]
    for sscc in chosen:
        log.append(
            ("missing_loaded_scan", "pallets", sscc, f"loaded scan {loaded[sscc][PE_ID]} removed; pallet departed")
        )


def _transfer_pallets_never_received(rows: Rows, rng: random.Random, log: list[LogEntry]) -> None:
    transfer_shipments = {s[SH_ID] for s in rows["shipments"] if s[SH_TYPE] == "stock transfer"}
    transfer_pallets = {p[PAL_SSCC] for p in rows["pallets"] if p[PAL_SHIPMENT] in transfer_shipments}
    received = {
        e[PE_ID]: e[PE_SSCC]
        for e in rows["pallet_events"]
        if e[PE_TYPE] == "received" and e[PE_SSCC] in transfer_pallets
    }
    n = round(len(transfer_pallets) * config.ERROR_RATES["transfer_pallet_never_received"])
    remove = set(rng.sample(sorted(received), min(n, len(received))))
    rows["pallet_events"] = [e for e in rows["pallet_events"] if e[PE_ID] not in remove]
    for event_id in sorted(remove):
        log.append(
            (
                "transfer_pallet_never_received",
                "pallets",
                received[event_id],
                f"received scan {event_id} removed at the destination DC",
            )
        )


def _bad_sscc_check_digits(rows: Rows, rng: random.Random, log: list[LogEntry]) -> None:
    pallets = rows["pallets"]
    n = round(len(pallets) * config.ERROR_RATES["bad_sscc_check_digit"])
    chosen = rng.sample([p[PAL_SSCC] for p in pallets], n)
    renamed: dict[str, str] = {}
    for sscc in chosen:
        body, correct = sscc[:17], sscc[17]
        assert gs1_check_digit(body) == correct
        wrong = rng.choice([d for d in "0123456789" if d != correct])
        renamed[sscc] = body + wrong
    # the wrong code was printed on the label, so it is wrong everywhere it was scanned
    rows["pallets"] = [(renamed.get(p[0], p[0]), *p[1:]) for p in pallets]
    rows["pallet_contents"] = [(renamed.get(c[0], c[0]), *c[1:]) for c in rows["pallet_contents"]]
    rows["pallet_events"] = [(e[0], renamed.get(e[PE_SSCC], e[PE_SSCC]), *e[2:]) for e in rows["pallet_events"]]
    for old, new in sorted(renamed.items()):
        log.append(("bad_sscc_check_digit", "pallets", new, f"check digit should be {old[17]}, label has {new[17]}"))


def _return_qty_mismatches(rows: Rows, rng: random.Random, log: list[LogEntry]) -> None:
    lines = rows["customer_return_lines"]
    eligible = [i for i, ln in enumerate(lines) if ln[RL_RECEIVED] > 0]
    n = round(len(eligible) * config.ERROR_RATES["return_qty_mismatch"])
    for i in sorted(rng.sample(eligible, n)):
        ln = list(lines[i])
        requested = ln[RL_REQUESTED]
        delta = rng.choice([-1, 1]) * rng.randint(1, max(1, min(5, requested // 2)))
        received = max(0, requested + delta)
        if received == requested:
            received = requested + 1
        ln[RL_RECEIVED] = received
        lines[i] = tuple(ln)
        log.append(
            ("return_qty_mismatch", "customer_return_lines", ln[RL_ID], f"requested {requested}, received {received}")
        )


def write_log(log: list[LogEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["error_type", "table", "record_id", "detail"])
        writer.writerows(log)
