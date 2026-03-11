"""dream_engine/recorder.py — Dream journal: record, tag, search, and review dreams.

Storage: ~/.local/share/blackmagic-gen/dreams/journal.json
Each entry: timestamp, title, narrative, tags, mood, vividness (1-10),
            lucidity (0-5), recurring (bool), people, locations, notes.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_DATA_DIR = Path.home() / ".local/share/blackmagic-gen/dreams"
_JOURNAL = _DATA_DIR / "journal.json"


def _ensure_dir():
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_journal() -> list[dict]:
    _ensure_dir()
    if _JOURNAL.exists():
        try:
            return json.loads(_JOURNAL.read_text())
        except Exception:
            pass
    return []


def _save_journal(entries: list[dict]):
    _ensure_dir()
    _JOURNAL.write_text(json.dumps(entries, indent=2))


def record(
    title: str,
    narrative: str,
    tags: list[str] | None = None,
    mood: str = "",
    vividness: int = 5,
    lucidity: int = 0,
    recurring: bool = False,
    people: list[str] | None = None,
    locations: list[str] | None = None,
    notes: str = "",
    sleep_quality: int = 5,
) -> dict:
    """Record a new dream entry. Returns the created entry."""
    entry = {
        "id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "title": title,
        "narrative": narrative,
        "tags": tags or [],
        "mood": mood,
        "vividness": max(1, min(10, vividness)),
        "lucidity": max(0, min(5, lucidity)),
        "recurring": recurring,
        "people": people or [],
        "locations": locations or [],
        "notes": notes,
        "sleep_quality": max(1, min(10, sleep_quality)),
    }
    journal = _load_journal()
    journal.append(entry)
    _save_journal(journal)
    return entry


def get_all() -> list[dict]:
    """Return all dream entries, newest first."""
    return list(reversed(_load_journal()))


def get_by_id(dream_id: str) -> dict | None:
    for entry in _load_journal():
        if entry["id"] == dream_id:
            return entry
    return None


def search(query: str) -> list[dict]:
    """Search dreams by keyword in title, narrative, tags, people, locations."""
    q = query.lower()
    results = []
    for entry in _load_journal():
        searchable = " ".join([
            entry.get("title", ""),
            entry.get("narrative", ""),
            " ".join(entry.get("tags", [])),
            " ".join(entry.get("people", [])),
            " ".join(entry.get("locations", [])),
            entry.get("mood", ""),
            entry.get("notes", ""),
        ]).lower()
        if q in searchable:
            results.append(entry)
    return list(reversed(results))


def search_by_tag(tag: str) -> list[dict]:
    tag_l = tag.lower()
    return list(reversed([
        e for e in _load_journal()
        if tag_l in [t.lower() for t in e.get("tags", [])]
    ]))


def get_recurring() -> list[dict]:
    """Return all dreams marked as recurring."""
    return list(reversed([e for e in _load_journal() if e.get("recurring")]))


def get_lucid() -> list[dict]:
    """Return all dreams with lucidity > 0."""
    return list(reversed([e for e in _load_journal() if e.get("lucidity", 0) > 0]))


def delete(dream_id: str) -> bool:
    journal = _load_journal()
    filtered = [e for e in journal if e["id"] != dream_id]
    if len(filtered) < len(journal):
        _save_journal(filtered)
        return True
    return False


def stats() -> dict:
    """Return summary stats about the dream journal."""
    journal = _load_journal()
    if not journal:
        return {"total": 0}

    vividness_vals = [e.get("vividness", 5) for e in journal]
    lucidity_vals = [e.get("lucidity", 0) for e in journal]
    sleep_vals = [e.get("sleep_quality", 5) for e in journal]

    all_tags = {}
    all_people = {}
    all_moods = {}
    for e in journal:
        for t in e.get("tags", []):
            all_tags[t] = all_tags.get(t, 0) + 1
        for p in e.get("people", []):
            all_people[p] = all_people.get(p, 0) + 1
        mood = e.get("mood", "")
        if mood:
            all_moods[mood] = all_moods.get(mood, 0) + 1

    return {
        "total": len(journal),
        "recurring": sum(1 for e in journal if e.get("recurring")),
        "lucid": sum(1 for e in journal if e.get("lucidity", 0) > 0),
        "avg_vividness": round(sum(vividness_vals) / len(vividness_vals), 1),
        "avg_lucidity": round(sum(lucidity_vals) / len(lucidity_vals), 1),
        "avg_sleep_quality": round(sum(sleep_vals) / len(sleep_vals), 1),
        "top_tags": sorted(all_tags.items(), key=lambda x: -x[1])[:10],
        "top_people": sorted(all_people.items(), key=lambda x: -x[1])[:10],
        "moods": sorted(all_moods.items(), key=lambda x: -x[1]),
    }
