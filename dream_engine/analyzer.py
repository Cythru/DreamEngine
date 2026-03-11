"""dream_engine/analyzer.py — Pattern analysis and dream interpretation.

Analyzes the dream journal for:
- Recurring symbols, themes, and people
- Sleep quality correlations (vividness vs sleep quality, lucidity trends)
- Dream frequency and streaks
- LLM-powered dream interpretation
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from . import recorder


def theme_frequency() -> dict[str, int]:
    """Count how often each tag appears across all dreams."""
    tags = Counter()
    for entry in recorder.get_all():
        for tag in entry.get("tags", []):
            tags[tag.lower()] += 1
    return dict(tags.most_common())


def people_frequency() -> dict[str, int]:
    """Count how often each person appears."""
    people = Counter()
    for entry in recorder.get_all():
        for person in entry.get("people", []):
            people[person.lower()] += 1
    return dict(people.most_common())


def location_frequency() -> dict[str, int]:
    locations = Counter()
    for entry in recorder.get_all():
        for loc in entry.get("locations", []):
            locations[loc.lower()] += 1
    return dict(locations.most_common())


def mood_distribution() -> dict[str, int]:
    moods = Counter()
    for entry in recorder.get_all():
        mood = entry.get("mood", "").lower()
        if mood:
            moods[mood] += 1
    return dict(moods.most_common())


def lucidity_trend(last_n: int = 30) -> list[dict]:
    """Return lucidity scores for the last N dreams (oldest first)."""
    entries = recorder.get_all()[:last_n]
    return [
        {"date": e["timestamp"][:10], "lucidity": e.get("lucidity", 0), "title": e.get("title", "")}
        for e in reversed(entries)
    ]


def vividness_vs_sleep() -> list[dict]:
    """Return vividness and sleep quality pairs for correlation analysis."""
    return [
        {
            "vividness": e.get("vividness", 5),
            "sleep_quality": e.get("sleep_quality", 5),
            "lucidity": e.get("lucidity", 0),
        }
        for e in recorder.get_all()
    ]


def streak() -> dict:
    """Calculate current recording streak (consecutive days with entries)."""
    entries = recorder.get_all()
    if not entries:
        return {"current_streak": 0, "longest_streak": 0}

    dates = sorted({e["timestamp"][:10] for e in entries})
    if not dates:
        return {"current_streak": 0, "longest_streak": 0}

    streaks = []
    current = 1
    for i in range(1, len(dates)):
        d1 = datetime.fromisoformat(dates[i - 1])
        d2 = datetime.fromisoformat(dates[i])
        if (d2 - d1).days == 1:
            current += 1
        else:
            streaks.append(current)
            current = 1
    streaks.append(current)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    yesterday = (datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)
                 - __import__("datetime").timedelta(days=1)).strftime("%Y-%m-%d")
    is_active = dates[-1] in (today, yesterday)

    return {
        "current_streak": current if is_active else 0,
        "longest_streak": max(streaks),
        "total_days": len(dates),
    }


def interpret_with_llm(dream_id: str) -> str:
    """Use the local LLM to interpret a specific dream."""
    try:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from grimoire.ai_backend import bm_call
    except ImportError:
        return "LLM backend not available. Install grimoire or configure an OpenAI-compatible endpoint."

    entry = recorder.get_by_id(dream_id)
    if not entry:
        return "Dream not found."

    patterns = recorder.stats()
    recurring_tags = [t[0] for t in patterns.get("top_tags", [])[:5]]

    system = (
        "You are a dream analyst. Interpret the following dream concisely. "
        "Consider symbolism, emotional tone, and recurring patterns. "
        "Be insightful but not prescriptive. Suggest what the dreamer's "
        "subconscious might be processing. Keep it to 2-3 paragraphs."
    )

    context = f"""Dream: "{entry['title']}"
Narrative: {entry['narrative'][:1500]}
Mood: {entry.get('mood', 'unknown')}
Vividness: {entry.get('vividness', 5)}/10
Lucidity: {entry.get('lucidity', 0)}/5
People: {', '.join(entry.get('people', [])) or 'none'}
Locations: {', '.join(entry.get('locations', [])) or 'unknown'}
Tags: {', '.join(entry.get('tags', [])) or 'none'}
Recurring journal themes: {', '.join(recurring_tags) or 'none yet'}"""

    messages = [{"role": "user", "content": context}]
    result = bm_call(messages, max_tokens=300, temperature=0.7, system=system)
    if "[BlackMagic Offline]" in result:
        return "LLM offline — interpretation unavailable."
    return result
