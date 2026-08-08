#!/usr/bin/env python3
"""
Registry Metrics Snapshot

Fetches the current public metrics for this node from the ComfyUI Registry
API and appends one JSON-Lines record to metrics/registry-history.jsonl.

- Append-only: existing lines are never rewritten.
- Stdlib only (urllib), no extra dependencies, no secrets required.
- Safe to run locally: `python scripts/registry_metrics_snapshot.py`

Exit codes:
  0  success (record appended, or nothing to do)
  1  the registry request failed / returned unexpected data
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

NODE_ID = "imagemetahub-comfyui-save"
REGISTRY_URL = f"https://api.comfy.org/nodes/{NODE_ID}"
HISTORY_FILE = Path(__file__).resolve().parent.parent / "metrics" / "registry-history.jsonl"
REQUEST_TIMEOUT = 30


def fetch_registry_data(url: str) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            body = response.read()
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"Failed to reach {url}: {error}") from error

    try:
        data = json.loads(body)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Registry response was not valid JSON: {error}") from error

    if not isinstance(data, dict) or "id" not in data:
        raise RuntimeError(f"Unexpected registry payload shape: {data!r}")

    return data


def read_last_record(path: Path) -> dict | None:
    if not path.exists():
        return None
    last_line = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                last_line = line
    if last_line is None:
        return None
    try:
        return json.loads(last_line)
    except json.JSONDecodeError:
        return None


def compute_delta(current: dict, previous: dict | None) -> dict | None:
    if previous is None:
        return None

    def numeric_delta(key: str):
        cur_val = current.get(key)
        prev_val = previous.get(key)
        if not isinstance(cur_val, (int, float)) or not isinstance(prev_val, (int, float)):
            return None
        delta = cur_val - prev_val
        return round(delta, 4) if isinstance(delta, float) else delta

    return {
        "downloads": numeric_delta("downloads"),
        "github_stars": numeric_delta("github_stars"),
        "rating": numeric_delta("rating"),
        "search_ranking": numeric_delta("search_ranking"),
        "version_changed": current.get("latest_version") != previous.get("latest_version"),
        "status_changed": current.get("status") != previous.get("status"),
    }


def build_record(payload: dict, previous: dict | None) -> dict:
    latest_version = payload.get("latest_version") or {}
    record = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "node_id": payload.get("id", NODE_ID),
        "downloads": payload.get("downloads"),
        "github_stars": payload.get("github_stars"),
        "rating": payload.get("rating"),
        "search_ranking": payload.get("search_ranking"),
        "latest_version": latest_version.get("version"),
        "status": payload.get("status"),
    }
    record["delta"] = compute_delta(record, previous)
    return record


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def write_step_summary(record: dict) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    delta = record.get("delta") or {}
    lines = [
        "### Registry metrics snapshot",
        "",
        f"- timestamp: `{record['timestamp']}`",
        f"- downloads: **{record['downloads']}** (Δ {delta.get('downloads', 'n/a')})",
        f"- github_stars: **{record['github_stars']}** (Δ {delta.get('github_stars', 'n/a')})",
        f"- rating: **{record['rating']}** (Δ {delta.get('rating', 'n/a')})",
        f"- search_ranking: **{record['search_ranking']}** (Δ {delta.get('search_ranking', 'n/a')})",
        f"- latest_version: **{record['latest_version']}**"
        + (" (changed)" if delta.get("version_changed") else ""),
        f"- status: **{record['status']}**"
        + (" (changed)" if delta.get("status_changed") else ""),
    ]
    with open(summary_path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> int:
    try:
        payload = fetch_registry_data(REGISTRY_URL)
    except RuntimeError as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1

    previous = read_last_record(HISTORY_FILE)
    record = build_record(payload, previous)
    append_record(HISTORY_FILE, record)
    write_step_summary(record)

    print(f"Appended snapshot to {HISTORY_FILE}: {json.dumps(record, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
