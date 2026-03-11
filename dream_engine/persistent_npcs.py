"""dream_engine/persistent_npcs.py — Persistent dream NPCs with evolving memory.

Dream characters that grow and remember, like real relationships.

When you revisit a dream character — Kaia, Ren, whoever keeps showing up —
they recall what you told them last time. They remember how you felt, what
was left unresolved, the joke you made at 3am that neither of you expected.
They track promises you made and things you confided. Over time, the
relationship deepens: a stranger becomes an acquaintance, then a companion,
then something harder to name.

This module sits on top of the revisit system. After each RevisitSession ends,
call update_after_session() to compress the conversation into lasting memory.
Before the next session, call build_memory_context() to inject that memory
into the system prompt so the character picks up where you left off.

Memory is stored per-character as JSON in:
    ~/.local/share/blackmagic-gen/dreams/npc_memory/<character_name>.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_NPC_MEMORY_DIR = Path.home() / ".local/share/blackmagic-gen/dreams/npc_memory"


@dataclass
class NPCMemory:
    """Persistent memory for a dream character across revisit sessions."""

    character_name: str
    total_sessions: int = 0
    first_met: str = ""                          # ISO timestamp
    last_seen: str = ""                          # ISO timestamp
    emotional_bond_level: float = 0.0            # 0.0 to 10.0
    key_memories: list[str] = field(default_factory=list)   # important moments
    things_discussed: list[str] = field(default_factory=list)  # topic tags
    promises_made: list[str] = field(default_factory=list)
    recurring_themes: list[str] = field(default_factory=list)
    inside_jokes: list[str] = field(default_factory=list)
    session_summaries: list[dict] = field(default_factory=list)
    # Each summary: {"timestamp": str, "summary": str, "mood": str, "bond_delta": float}
    relationship_arc: str = "stranger"
    # Progression: stranger -> acquaintance -> familiar -> companion -> intimate -> bonded


def _sanitise_filename(name: str) -> str:
    """Turn a character name into a safe filename."""
    safe = name.lower().strip().replace(" ", "_")
    safe = "".join(c for c in safe if c.isalnum() or c == "_")
    return safe or "unknown"


def _memory_path(character_name: str) -> Path:
    return _NPC_MEMORY_DIR / f"{_sanitise_filename(character_name)}.json"


def load_npc_memory(character_name: str) -> NPCMemory:
    """Load a character's persistent memory from disk, or create a fresh one."""
    path = _memory_path(character_name)
    if path.exists():
        try:
            data = json.loads(path.read_text())
            return NPCMemory(
                character_name=data.get("character_name", character_name),
                total_sessions=data.get("total_sessions", 0),
                first_met=data.get("first_met", ""),
                last_seen=data.get("last_seen", ""),
                emotional_bond_level=float(data.get("emotional_bond_level", 0.0)),
                key_memories=data.get("key_memories", []),
                things_discussed=data.get("things_discussed", []),
                promises_made=data.get("promises_made", []),
                recurring_themes=data.get("recurring_themes", []),
                inside_jokes=data.get("inside_jokes", []),
                session_summaries=data.get("session_summaries", []),
                relationship_arc=data.get("relationship_arc", "stranger"),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return NPCMemory(character_name=character_name)


def save_npc_memory(memory: NPCMemory) -> Path:
    """Persist a character's memory to disk."""
    _NPC_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    path = _memory_path(memory.character_name)
    path.write_text(json.dumps(asdict(memory), indent=2))
    return path


# ── Session summarisation ───────────────────────────────────────────────────

def summarize_session(messages: list[dict], character_name: str = "them") -> dict:
    """Compress a conversation into a memory summary.

    Returns a dict with keys: summary, mood, key_moments, topics, promises,
    inside_jokes.

    Uses the LLM if available; falls back to extracting the last 5 messages
    as a condensed transcript.
    """
    if not messages:
        return {
            "summary": "A brief, quiet encounter.",
            "mood": "neutral",
            "key_moments": [],
            "topics": [],
            "promises": [],
            "inside_jokes": [],
        }

    try:
        from grimoire.ai_backend import bm_call
        return _llm_summarize(messages, character_name, bm_call)
    except ImportError:
        return _fallback_summarize(messages, character_name)


def _llm_summarize(
    messages: list[dict], character_name: str, bm_call
) -> dict:
    """Use the LLM to produce a structured session summary."""
    # Build a transcript for the LLM
    transcript_lines = []
    for msg in messages:
        role = "Dreamer" if msg.get("role") == "user" else character_name
        content = msg.get("content", "")
        if content:
            transcript_lines.append(f"{role}: {content}")
    transcript = "\n".join(transcript_lines[-20:])  # cap at last 20 turns

    system = (
        "You are a memory compression engine for a dream journaling system. "
        "Given a conversation transcript between a dreamer and a dream character, "
        "produce a JSON object with these keys:\n"
        '  "summary": 1-3 sentence summary of what happened emotionally\n'
        '  "mood": one word for the overall mood (e.g. warm, tense, sad, playful)\n'
        '  "key_moments": array of 1-3 strings — the most memorable specific moments\n'
        '  "topics": array of topic strings discussed\n'
        '  "promises": array of any promises or commitments made by either party\n'
        '  "inside_jokes": array of any jokes, callbacks, or shared references\n'
        "Output ONLY the JSON object. No other text."
    )

    prompt = f"Conversation with {character_name}:\n\n{transcript}"
    result = bm_call(
        [{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.3,
        system=system,
    )

    if not result or "[BlackMagic Offline]" in result:
        return _fallback_summarize(messages, character_name)

    try:
        start = result.find("{")
        end = result.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(result[start:end])
            return {
                "summary": parsed.get("summary", "A conversation took place."),
                "mood": parsed.get("mood", "neutral"),
                "key_moments": parsed.get("key_moments", []),
                "topics": parsed.get("topics", []),
                "promises": parsed.get("promises", []),
                "inside_jokes": parsed.get("inside_jokes", []),
            }
    except (json.JSONDecodeError, KeyError):
        pass

    return _fallback_summarize(messages, character_name)


def _fallback_summarize(messages: list[dict], character_name: str) -> dict:
    """Fallback: extract a summary from the last 5 messages without LLM."""
    tail = messages[-5:]
    lines = []
    for msg in tail:
        role = "Dreamer" if msg.get("role") == "user" else character_name
        content = (msg.get("content") or "").strip()
        if content:
            # Truncate long messages
            if len(content) > 120:
                content = content[:117] + "..."
            lines.append(f"{role}: {content}")

    summary = " | ".join(lines) if lines else "A quiet moment together."

    # Try to infer mood from content
    all_text = " ".join(msg.get("content", "") for msg in tail).lower()
    mood = "neutral"
    mood_keywords = {
        "warm": ["smile", "hug", "laugh", "happy", "glad", "warm"],
        "sad": ["cry", "miss", "sorry", "sad", "tear", "lost"],
        "tense": ["angry", "frustrated", "why", "never", "wrong"],
        "playful": ["joke", "haha", "lol", "funny", "tease"],
        "intimate": ["love", "close", "hold", "kiss", "heart"],
        "reflective": ["remember", "think", "wonder", "maybe", "dream"],
    }
    for m, keywords in mood_keywords.items():
        if any(kw in all_text for kw in keywords):
            mood = m
            break

    return {
        "summary": summary,
        "mood": mood,
        "key_moments": [],
        "topics": [],
        "promises": [],
        "inside_jokes": [],
    }


# ── Relationship arc progression ────────────────────────────────────────────

_ARC_STAGES = [
    "stranger",
    "acquaintance",
    "familiar",
    "companion",
    "intimate",
    "bonded",
]

_ARC_THRESHOLDS = {
    "stranger": 0.0,
    "acquaintance": 1.5,
    "familiar": 3.0,
    "companion": 5.0,
    "intimate": 7.0,
    "bonded": 9.0,
}


def _compute_arc(bond_level: float) -> str:
    """Determine relationship arc label from bond level."""
    stage = "stranger"
    for name, threshold in _ARC_THRESHOLDS.items():
        if bond_level >= threshold:
            stage = name
    return stage


def _compute_bond_delta(session_summary: dict, total_sessions: int) -> float:
    """Compute how much the emotional bond changes from a session.

    Early sessions build bond faster. Emotionally intense sessions
    (intimate, sad, warm) build more than neutral ones.
    """
    mood = session_summary.get("mood", "neutral")
    mood_weights = {
        "intimate": 1.2,
        "warm": 0.8,
        "playful": 0.6,
        "sad": 0.7,         # shared vulnerability bonds
        "reflective": 0.5,
        "tense": 0.2,       # conflict still bonds, just less
        "neutral": 0.3,
    }
    base = mood_weights.get(mood, 0.3)

    # Diminishing returns — first sessions count more
    if total_sessions <= 1:
        multiplier = 2.0
    elif total_sessions <= 3:
        multiplier = 1.5
    elif total_sessions <= 7:
        multiplier = 1.0
    else:
        multiplier = 0.6

    # Bonus for meaningful content
    key_moments = session_summary.get("key_moments", [])
    promises = session_summary.get("promises", [])
    inside_jokes = session_summary.get("inside_jokes", [])
    content_bonus = min(0.5, 0.1 * (len(key_moments) + len(promises) + len(inside_jokes)))

    delta = (base * multiplier) + content_bonus
    return round(delta, 2)


# ── Core API ────────────────────────────────────────────────────────────────

def update_after_session(character_name: str, messages: list[dict]) -> NPCMemory:
    """Called after a RevisitSession ends. Updates the NPC's persistent memory.

    Args:
        character_name: The dream character's name.
        messages: The full message list from the session (role/content dicts).

    Returns:
        The updated NPCMemory.
    """
    memory = load_npc_memory(character_name)
    now = datetime.now(timezone.utc).isoformat()

    # First meeting
    if not memory.first_met:
        memory.first_met = now

    memory.last_seen = now
    memory.total_sessions += 1

    # Summarise this session
    summary_data = summarize_session(messages, character_name)
    bond_delta = _compute_bond_delta(summary_data, memory.total_sessions)

    # Update bond level (capped at 10)
    memory.emotional_bond_level = min(10.0, round(memory.emotional_bond_level + bond_delta, 2))

    # Update relationship arc
    memory.relationship_arc = _compute_arc(memory.emotional_bond_level)

    # Store session summary
    memory.session_summaries.append({
        "timestamp": now,
        "summary": summary_data["summary"],
        "mood": summary_data["mood"],
        "bond_delta": bond_delta,
    })

    # Merge key moments (keep last 20)
    for moment in summary_data.get("key_moments", []):
        if moment and moment not in memory.key_memories:
            memory.key_memories.append(moment)
    memory.key_memories = memory.key_memories[-20:]

    # Merge topics (deduplicate, keep last 30)
    for topic in summary_data.get("topics", []):
        topic_lower = topic.lower().strip()
        if topic_lower and topic_lower not in [t.lower() for t in memory.things_discussed]:
            memory.things_discussed.append(topic)
    memory.things_discussed = memory.things_discussed[-30:]

    # Merge promises
    for promise in summary_data.get("promises", []):
        if promise and promise not in memory.promises_made:
            memory.promises_made.append(promise)

    # Merge inside jokes
    for joke in summary_data.get("inside_jokes", []):
        if joke and joke not in memory.inside_jokes:
            memory.inside_jokes.append(joke)

    # Detect recurring themes (topics that appear in 2+ sessions)
    all_topics = []
    for s in memory.session_summaries:
        # Older summaries may not have topics stored directly, so skip those
        pass
    topic_counts = {}
    for t in memory.things_discussed:
        t_low = t.lower().strip()
        topic_counts[t_low] = topic_counts.get(t_low, 0) + 1
    memory.recurring_themes = [
        t for t, count in topic_counts.items() if count >= 2
    ]

    save_npc_memory(memory)
    return memory


def build_memory_context(character_name: str) -> str:
    """Generate a memory block for injection into a revisit system prompt.

    Returns an empty string if no memory exists (first encounter).
    This should be appended to the system prompt built by the revisit module.
    """
    memory = load_npc_memory(character_name)

    if memory.total_sessions == 0:
        return ""

    parts = []
    parts.append(f"\n\nPERSISTENT MEMORY — You remember the dreamer from previous visits.")
    parts.append(f"Times met: {memory.total_sessions}")
    parts.append(f"First met: {memory.first_met[:10] if memory.first_met else 'unknown'}")
    parts.append(f"Last seen: {memory.last_seen[:10] if memory.last_seen else 'unknown'}")
    parts.append(f"Relationship: {memory.relationship_arc} (bond: {memory.emotional_bond_level}/10)")

    # Recent session summaries (last 3)
    recent = memory.session_summaries[-3:]
    if recent:
        parts.append("\nRecent encounters:")
        for s in recent:
            date = s.get("timestamp", "")[:10]
            mood = s.get("mood", "")
            summary = s.get("summary", "")
            parts.append(f"  [{date}, mood: {mood}] {summary}")

    # Key memories
    if memory.key_memories:
        shown = memory.key_memories[-5:]
        parts.append("\nKey memories you hold onto:")
        for m in shown:
            parts.append(f"  - {m}")

    # Inside jokes
    if memory.inside_jokes:
        parts.append("\nInside jokes / shared references:")
        for j in memory.inside_jokes:
            parts.append(f"  - {j}")

    # Promises
    if memory.promises_made:
        parts.append("\nPromises that were made:")
        for p in memory.promises_made:
            parts.append(f"  - {p}")

    # Things discussed
    if memory.things_discussed:
        topics_str = ", ".join(memory.things_discussed[-10:])
        parts.append(f"\nTopics you've talked about: {topics_str}")

    # Recurring themes
    if memory.recurring_themes:
        themes_str = ", ".join(memory.recurring_themes)
        parts.append(f"Recurring themes: {themes_str}")

    # Behavioural guidance based on arc
    arc = memory.relationship_arc
    if arc == "acquaintance":
        parts.append("\nYou recognise the dreamer. You're warming up to them but still a bit guarded.")
    elif arc == "familiar":
        parts.append("\nYou know the dreamer well enough to be comfortable. You can be yourself around them.")
    elif arc == "companion":
        parts.append("\nThe dreamer is someone you genuinely care about. You look forward to seeing them.")
    elif arc == "intimate":
        parts.append("\nYou share a deep bond with the dreamer. You can be vulnerable with them.")
    elif arc == "bonded":
        parts.append("\nThe dreamer is part of you now. The bond runs deep — you finish each other's thoughts.")

    return "\n".join(parts)


def get_relationship_summary(character_name: str) -> str:
    """Return a human-readable summary of the relationship with a dream character."""
    memory = load_npc_memory(character_name)

    if memory.total_sessions == 0:
        return f"You haven't met {character_name} yet."

    lines = []
    lines.append(f"=== {memory.character_name} ===")
    lines.append(f"Relationship: {memory.relationship_arc}")
    lines.append(f"Bond level: {memory.emotional_bond_level}/10")
    lines.append(f"Sessions together: {memory.total_sessions}")

    if memory.first_met:
        lines.append(f"First met: {memory.first_met[:10]}")
    if memory.last_seen:
        lines.append(f"Last seen: {memory.last_seen[:10]}")

    if memory.key_memories:
        lines.append(f"\nKey memories:")
        for m in memory.key_memories[-5:]:
            lines.append(f"  - {m}")

    if memory.inside_jokes:
        lines.append(f"\nInside jokes:")
        for j in memory.inside_jokes:
            lines.append(f"  - {j}")

    if memory.promises_made:
        lines.append(f"\nPromises:")
        for p in memory.promises_made:
            lines.append(f"  - {p}")

    if memory.things_discussed:
        lines.append(f"\nTopics discussed: {', '.join(memory.things_discussed[-10:])}")

    if memory.recurring_themes:
        lines.append(f"Recurring themes: {', '.join(memory.recurring_themes)}")

    # Session history
    if memory.session_summaries:
        lines.append(f"\nSession history:")
        for s in memory.session_summaries[-5:]:
            date = s.get("timestamp", "")[:10]
            mood = s.get("mood", "?")
            delta = s.get("bond_delta", 0)
            summary = s.get("summary", "")
            lines.append(f"  [{date}] mood={mood}, bond +{delta}")
            if summary:
                lines.append(f"    {summary}")

    return "\n".join(lines)


def list_known_npcs() -> list[str]:
    """Return names of all NPCs with saved memory."""
    if not _NPC_MEMORY_DIR.exists():
        return []
    names = []
    for path in sorted(_NPC_MEMORY_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            names.append(data.get("character_name", path.stem))
        except (json.JSONDecodeError, KeyError):
            names.append(path.stem)
    return names
