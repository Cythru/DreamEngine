"""dream_engine/dream_atlas.py — Maps the geography of your dream world.

Every place you've dreamed becomes a node in a growing atlas — connected by
the dreams that link them.  Over time, the map reveals the hidden architecture
of your subconscious landscape.
"""
from __future__ import annotations

import json
import os
import sys
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from . import recorder

# ── paths ────────────────────────────────────────────────────────────────────

_DATA_DIR = Path.home() / ".local/share/blackmagic-gen/dreams"
_ATLAS_FILE = _DATA_DIR / "atlas.json"

# ── fuzzy matching threshold ─────────────────────────────────────────────────

_FUZZY_THRESHOLD = 0.70  # ratio above which two location names are "the same"


# ── dataclass ────────────────────────────────────────────────────────────────

@dataclass
class DreamLocation:
    name: str
    description: str = ""
    connected_to: list[str] = field(default_factory=list)
    dream_ids: list[str] = field(default_factory=list)
    moods: list[str] = field(default_factory=list)
    visit_count: int = 0
    first_seen: str = ""
    last_seen: str = ""
    characters_seen: list[str] = field(default_factory=list)
    notable_events: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> DreamLocation:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# ── helpers ──────────────────────────────────────────────────────────────────

def _ensure_dir():
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


def _normalize(name: str) -> str:
    """Lowercase, strip articles and whitespace for comparison."""
    n = name.lower().strip()
    for prefix in ("the ", "a ", "an "):
        if n.startswith(prefix):
            n = n[len(prefix):]
    return n.strip()


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _find_canonical(name: str, locations: dict[str, DreamLocation]) -> str | None:
    """Return the canonical name for *name* if a fuzzy match exists."""
    norm = _normalize(name)
    best_name: str | None = None
    best_score = 0.0
    for existing in locations:
        score = _similarity(name, existing)
        # exact normalised match is always best
        if _normalize(existing) == norm:
            return existing
        if score > best_score:
            best_score = score
            best_name = existing
    if best_score >= _FUZZY_THRESHOLD:
        return best_name
    return None


def _add_unique(lst: list, item) -> None:
    """Append *item* if not already present."""
    if item and item not in lst:
        lst.append(item)


# ── Atlas ────────────────────────────────────────────────────────────────────

class DreamAtlas:
    """Graph of dream locations, built from the dream journal."""

    def __init__(self):
        self._atlas_path: Path = _ATLAS_FILE
        self._locations: dict[str, DreamLocation] = {}
        self.load()

    # ── persistence ──────────────────────────────────────────────────────

    def save(self) -> None:
        _ensure_dir()
        payload = {name: loc.to_dict() for name, loc in self._locations.items()}
        self._atlas_path.write_text(json.dumps(payload, indent=2))

    def load(self) -> None:
        if self._atlas_path.exists():
            try:
                raw = json.loads(self._atlas_path.read_text())
                self._locations = {
                    name: DreamLocation.from_dict(d) for name, d in raw.items()
                }
            except Exception:
                self._locations = {}
        else:
            self._locations = {}

    # ── LLM extraction ───────────────────────────────────────────────────

    def extract_locations_llm(self, dream_entry: dict) -> list[dict]:
        """Use the LLM to pull detailed location info from a dream narrative.

        Returns a list of dicts with keys:
            name, description, characters, notable_events
        Falls back to the plain `locations` field if the LLM is unavailable.
        """
        narrative = dream_entry.get("narrative", "")
        raw_locations = dream_entry.get("locations", [])

        # attempt LLM extraction
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from grimoire.ai_backend import bm_call
        except ImportError:
            bm_call = None

        if bm_call and narrative.strip():
            system = (
                "You extract dream locations from dream narratives. "
                "For each distinct location, return a JSON array of objects with keys: "
                "name (short, evocative), description (1-2 sentences), "
                "characters (list of people/entities seen there), "
                "notable_events (list of brief event descriptions). "
                "Return ONLY valid JSON, no markdown fences."
            )
            prompt = f"Dream narrative:\n{narrative[:2000]}"
            messages = [{"role": "user", "content": prompt}]
            try:
                result = bm_call(messages, max_tokens=400, temperature=0.3, system=system)
                if result and "[BlackMagic Offline]" not in result:
                    # strip markdown fences if the model added them
                    text = result.strip()
                    if text.startswith("```"):
                        text = text.split("\n", 1)[-1]
                    if text.endswith("```"):
                        text = text.rsplit("```", 1)[0]
                    parsed = json.loads(text.strip())
                    if isinstance(parsed, list) and parsed:
                        return [
                            {
                                "name": loc.get("name", "unknown"),
                                "description": loc.get("description", ""),
                                "characters": loc.get("characters", []),
                                "notable_events": loc.get("notable_events", []),
                            }
                            for loc in parsed
                        ]
            except (json.JSONDecodeError, Exception):
                pass  # fall through to manual

        # manual fallback — build from the locations field
        return [
            {
                "name": loc,
                "description": "",
                "characters": dream_entry.get("people", []),
                "notable_events": [],
            }
            for loc in raw_locations
        ]

    # ── build from journal ───────────────────────────────────────────────

    def build_from_journal(self) -> int:
        """Scan every dream and (re)build the atlas. Returns location count."""
        self._locations.clear()
        dreams = recorder.get_all()

        for entry in dreams:
            dream_id = entry.get("id", "")
            timestamp = entry.get("timestamp", "")
            mood = entry.get("mood", "")
            people = entry.get("people", [])

            extracted = self.extract_locations_llm(entry)
            loc_names_this_dream: list[str] = []

            for loc_info in extracted:
                raw_name = loc_info["name"]
                canonical = _find_canonical(raw_name, self._locations)
                name = canonical if canonical else raw_name

                if name not in self._locations:
                    self._locations[name] = DreamLocation(
                        name=name,
                        first_seen=timestamp,
                        last_seen=timestamp,
                    )

                dl = self._locations[name]
                _add_unique(dl.dream_ids, dream_id)
                dl.visit_count = len(dl.dream_ids)
                if mood:
                    _add_unique(dl.moods, mood)
                for person in people:
                    _add_unique(dl.characters_seen, person)
                for person in loc_info.get("characters", []):
                    _add_unique(dl.characters_seen, person)
                for ev in loc_info.get("notable_events", []):
                    _add_unique(dl.notable_events, ev)
                if loc_info.get("description") and not dl.description:
                    dl.description = loc_info["description"]

                # update timestamps
                if timestamp:
                    if not dl.first_seen or timestamp < dl.first_seen:
                        dl.first_seen = timestamp
                    if not dl.last_seen or timestamp > dl.last_seen:
                        dl.last_seen = timestamp

                loc_names_this_dream.append(name)

            # connect locations that appeared in the same dream
            for i, a in enumerate(loc_names_this_dream):
                for b in loc_names_this_dream[i + 1:]:
                    if a != b:
                        _add_unique(self._locations[a].connected_to, b)
                        _add_unique(self._locations[b].connected_to, a)

        self.save()
        return len(self._locations)

    # ── manual curation ──────────────────────────────────────────────────

    def add_location(self, location: DreamLocation) -> None:
        """Add or overwrite a location manually."""
        self._locations[location.name] = location
        self.save()

    def merge_locations(self, name1: str, name2: str) -> DreamLocation | None:
        """Merge *name2* into *name1*. Returns the merged location or None."""
        loc1 = self._locations.get(name1)
        loc2 = self._locations.get(name2)
        if not loc1 or not loc2:
            return None

        # absorb everything from loc2 into loc1
        for did in loc2.dream_ids:
            _add_unique(loc1.dream_ids, did)
        loc1.visit_count = len(loc1.dream_ids)
        for m in loc2.moods:
            _add_unique(loc1.moods, m)
        for c in loc2.characters_seen:
            _add_unique(loc1.characters_seen, c)
        for ev in loc2.notable_events:
            _add_unique(loc1.notable_events, ev)
        for conn in loc2.connected_to:
            if conn != name1:
                _add_unique(loc1.connected_to, conn)
        if not loc1.description and loc2.description:
            loc1.description = loc2.description
        if loc2.first_seen and (not loc1.first_seen or loc2.first_seen < loc1.first_seen):
            loc1.first_seen = loc2.first_seen
        if loc2.last_seen and (not loc1.last_seen or loc2.last_seen > loc1.last_seen):
            loc1.last_seen = loc2.last_seen

        # rewrite connections from other locations that pointed at name2
        for other_name, other_loc in self._locations.items():
            if name2 in other_loc.connected_to:
                other_loc.connected_to.remove(name2)
                if other_name != name1:
                    _add_unique(other_loc.connected_to, name1)

        del self._locations[name2]
        self.save()
        return loc1

    # ── queries ──────────────────────────────────────────────────────────

    def get_location(self, name: str) -> DreamLocation | None:
        loc = self._locations.get(name)
        if loc:
            return loc
        # try fuzzy
        canonical = _find_canonical(name, self._locations)
        if canonical:
            return self._locations[canonical]
        return None

    def get_all_locations(self) -> list[DreamLocation]:
        return list(self._locations.values())

    def get_connections(self, name: str) -> list[str]:
        loc = self.get_location(name)
        return list(loc.connected_to) if loc else []

    def get_map(self) -> dict:
        """Return the full atlas graph: nodes + edges, ready for visualization."""
        nodes = []
        edges_set: set[tuple[str, str]] = set()

        for name, loc in self._locations.items():
            nodes.append({
                "id": name,
                "label": name,
                "visit_count": loc.visit_count,
                "mood": loc.moods[0] if loc.moods else "",
                "description": loc.description,
            })
            for conn in loc.connected_to:
                edge = tuple(sorted((name, conn)))
                edges_set.add(edge)

        edges = [{"source": a, "target": b} for a, b in edges_set]
        return {"nodes": nodes, "edges": edges}

    def get_most_visited(self) -> list[DreamLocation]:
        return sorted(self._locations.values(), key=lambda l: -l.visit_count)

    def find_path(self, from_loc: str, to_loc: str) -> list[str] | None:
        """BFS shortest path between two locations. Returns name list or None."""
        start = self.get_location(from_loc)
        end = self.get_location(to_loc)
        if not start or not end:
            return None
        if start.name == end.name:
            return [start.name]

        visited: set[str] = {start.name}
        queue: deque[list[str]] = deque([[start.name]])

        while queue:
            path = queue.popleft()
            current = path[-1]
            loc = self._locations.get(current)
            if not loc:
                continue
            for neighbor in loc.connected_to:
                if neighbor == end.name:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])

        return None  # no path

    def get_neighborhood(self, name: str, depth: int = 1) -> list[DreamLocation]:
        """All locations within *depth* connections of *name*."""
        loc = self.get_location(name)
        if not loc:
            return []

        visited: set[str] = {loc.name}
        frontier: set[str] = {loc.name}

        for _ in range(depth):
            next_frontier: set[str] = set()
            for n in frontier:
                node = self._locations.get(n)
                if not node:
                    continue
                for conn in node.connected_to:
                    if conn not in visited:
                        visited.add(conn)
                        next_frontier.add(conn)
            frontier = next_frontier

        return [
            self._locations[n]
            for n in visited
            if n in self._locations and n != loc.name
        ]


# ── CLI ──────────────────────────────────────────────────────────────────────

def cmd_atlas(args: list[str] | None = None) -> str:
    """CLI entry-point for the atlas.

    Usage:
        atlas                    — list all locations with visit counts
        atlas connections        — show all connections
        atlas detail <name>      — show details for a location
        atlas build              — rebuild the atlas from the journal
        atlas path <A> -> <B>    — find a path between two locations
        atlas map                — dump the graph as JSON
    """
    args = args or []
    atlas = DreamAtlas()

    # ── no args: overview ────────────────────────────────────────────────
    if not args:
        locations = atlas.get_most_visited()
        if not locations:
            return "Atlas is empty. Run 'atlas build' to populate it from your journal."
        lines = ["Dream Atlas — All Locations", "=" * 40]
        for loc in locations:
            conns = len(loc.connected_to)
            lines.append(
                f"  {loc.name}  —  {loc.visit_count} visit(s), "
                f"{conns} connection(s)"
                + (f"  [{loc.moods[0]}]" if loc.moods else "")
            )
        lines.append(f"\nTotal: {len(locations)} location(s)")
        return "\n".join(lines)

    cmd = args[0].lower()

    # ── build ────────────────────────────────────────────────────────────
    if cmd == "build":
        count = atlas.build_from_journal()
        return f"Atlas rebuilt. {count} location(s) mapped."

    # ── connections ──────────────────────────────────────────────────────
    if cmd == "connections":
        graph = atlas.get_map()
        if not graph["edges"]:
            return "No connections yet."
        lines = ["Dream Atlas — Connections", "=" * 40]
        for edge in graph["edges"]:
            lines.append(f"  {edge['source']}  <-->  {edge['target']}")
        return "\n".join(lines)

    # ── detail ───────────────────────────────────────────────────────────
    if cmd == "detail":
        name = " ".join(args[1:])
        if not name:
            return "Usage: atlas detail <location name>"
        loc = atlas.get_location(name)
        if not loc:
            return f"Location '{name}' not found."
        lines = [
            f"Location: {loc.name}",
            "=" * 40,
            f"  Description:    {loc.description or '(none)'}",
            f"  Visits:         {loc.visit_count}",
            f"  First seen:     {loc.first_seen[:10] if loc.first_seen else 'unknown'}",
            f"  Last seen:      {loc.last_seen[:10] if loc.last_seen else 'unknown'}",
            f"  Moods:          {', '.join(loc.moods) or '(none)'}",
            f"  Characters:     {', '.join(loc.characters_seen) or '(none)'}",
            f"  Connected to:   {', '.join(loc.connected_to) or '(none)'}",
            f"  Dream IDs:      {', '.join(loc.dream_ids)}",
        ]
        if loc.notable_events:
            lines.append("  Notable events:")
            for ev in loc.notable_events:
                lines.append(f"    - {ev}")
        return "\n".join(lines)

    # ── path ─────────────────────────────────────────────────────────────
    if cmd == "path":
        rest = " ".join(args[1:])
        if "->" not in rest:
            return "Usage: atlas path <location A> -> <location B>"
        parts = rest.split("->", 1)
        a, b = parts[0].strip(), parts[1].strip()
        path = atlas.find_path(a, b)
        if path is None:
            return f"No path found between '{a}' and '{b}'."
        return " -> ".join(path)

    # ── map ──────────────────────────────────────────────────────────────
    if cmd == "map":
        graph = atlas.get_map()
        return json.dumps(graph, indent=2)

    return f"Unknown atlas command: {cmd}. Try: build, connections, detail, path, map"
