"""dream_engine/subconscious.py — Talk to your subconscious.

Not a dream character — the pattern beneath all your dreams. It knows what
you fear, what you desire, what you keep returning to. Built from statistical
analysis of your dream journal, given voice through LLM.

Your subconscious isn't any single dream character. It's the
force that *chose* to send them. The recurring water, the locked doors, the
person you haven't spoken to in years but keep dreaming about. This module
finds those patterns, builds a profile, and lets you talk to the thing that
has been trying to tell you something every single night.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dream_engine import analyzer, recorder

# ── LLM backend (optional) ──────────────────────────────────────────────────

_bm_call = None

try:
    from grimoire.ai_backend import bm_call as _bm_call
except ImportError:
    pass


def _llm_available() -> bool:
    return _bm_call is not None


def _llm(messages: list[dict], system: str = "", max_tokens: int = 200,
         temperature: float = 0.7) -> Optional[str]:
    """Call the LLM if available. Returns None on failure."""
    if not _llm_available():
        return None
    try:
        result = _bm_call(messages, max_tokens=max_tokens,
                          temperature=temperature, system=system)
        if result and "[BlackMagic Offline]" not in result:
            return result
    except Exception:
        pass
    return None


# ── Storage paths ────────────────────────────────────────────────────────────

_DATA_DIR = Path.home() / ".local/share/blackmagic-gen/dreams"
_SUBCONSCIOUS_DIR = _DATA_DIR / "subconscious"
_PROFILE_PATH = _DATA_DIR / "subconscious_profile.json"


def _ensure_dirs():
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _SUBCONSCIOUS_DIR.mkdir(parents=True, exist_ok=True)


# ── Weighted item helper ────────────────────────────────────────────────────

@dataclass
class WeightedItem:
    """A label with a frequency/weight attached."""
    label: str
    weight: float  # raw count or normalized 0-1

    def to_dict(self) -> dict:
        return {"label": self.label, "weight": self.weight}

    @classmethod
    def from_dict(cls, d: dict) -> "WeightedItem":
        return cls(label=d["label"], weight=d["weight"])


# ── Subconscious profile ────────────────────────────────────────────────────

@dataclass
class SubconsciousProfile:
    """The statistical soul extracted from a dream journal."""
    recurring_themes: list[WeightedItem] = field(default_factory=list)
    dominant_emotions: list[WeightedItem] = field(default_factory=list)
    unresolved_threads: list[str] = field(default_factory=list)
    recurring_characters: list[WeightedItem] = field(default_factory=list)
    recurring_locations: list[WeightedItem] = field(default_factory=list)
    fear_patterns: list[str] = field(default_factory=list)
    desire_patterns: list[str] = field(default_factory=list)
    personality_traits: list[str] = field(default_factory=list)
    last_updated: str = ""

    def to_dict(self) -> dict:
        return {
            "recurring_themes": [w.to_dict() for w in self.recurring_themes],
            "dominant_emotions": [w.to_dict() for w in self.dominant_emotions],
            "unresolved_threads": self.unresolved_threads,
            "recurring_characters": [w.to_dict() for w in self.recurring_characters],
            "recurring_locations": [w.to_dict() for w in self.recurring_locations],
            "fear_patterns": self.fear_patterns,
            "desire_patterns": self.desire_patterns,
            "personality_traits": self.personality_traits,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SubconsciousProfile":
        return cls(
            recurring_themes=[WeightedItem.from_dict(w) for w in d.get("recurring_themes", [])],
            dominant_emotions=[WeightedItem.from_dict(w) for w in d.get("dominant_emotions", [])],
            unresolved_threads=d.get("unresolved_threads", []),
            recurring_characters=[WeightedItem.from_dict(w) for w in d.get("recurring_characters", [])],
            recurring_locations=[WeightedItem.from_dict(w) for w in d.get("recurring_locations", [])],
            fear_patterns=d.get("fear_patterns", []),
            desire_patterns=d.get("desire_patterns", []),
            personality_traits=d.get("personality_traits", []),
            last_updated=d.get("last_updated", ""),
        )

    def save(self):
        _ensure_dirs()
        _PROFILE_PATH.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls) -> Optional["SubconsciousProfile"]:
        if _PROFILE_PATH.exists():
            try:
                return cls.from_dict(json.loads(_PROFILE_PATH.read_text()))
            except Exception:
                pass
        return None

    def summary(self) -> str:
        """Human-readable summary of the profile."""
        lines = ["=== Subconscious Profile ===", ""]

        if self.recurring_themes:
            lines.append("Recurring themes:")
            for w in self.recurring_themes[:8]:
                bar = "#" * min(20, int(w.weight * 2))
                lines.append(f"  {w.label:<20s} {bar} ({w.weight:.0f})")
            lines.append("")

        if self.dominant_emotions:
            lines.append("Dominant emotions:")
            for w in self.dominant_emotions[:8]:
                lines.append(f"  {w.label:<20s} ({w.weight:.0f})")
            lines.append("")

        if self.recurring_characters:
            lines.append("Recurring characters:")
            for w in self.recurring_characters[:8]:
                lines.append(f"  {w.label:<20s} (appeared {w.weight:.0f}x)")
            lines.append("")

        if self.recurring_locations:
            lines.append("Recurring locations:")
            for w in self.recurring_locations[:8]:
                lines.append(f"  {w.label:<20s} (appeared {w.weight:.0f}x)")
            lines.append("")

        if self.fear_patterns:
            lines.append("Fear patterns:")
            for f in self.fear_patterns:
                lines.append(f"  - {f}")
            lines.append("")

        if self.desire_patterns:
            lines.append("Desire patterns:")
            for d in self.desire_patterns:
                lines.append(f"  - {d}")
            lines.append("")

        if self.unresolved_threads:
            lines.append("Unresolved threads:")
            for u in self.unresolved_threads:
                lines.append(f"  - {u}")
            lines.append("")

        if self.personality_traits:
            lines.append("Inferred personality traits:")
            for t in self.personality_traits:
                lines.append(f"  - {t}")
            lines.append("")

        lines.append(f"Last updated: {self.last_updated or 'never'}")
        return "\n".join(lines)


# ── Mood classification helpers ──────────────────────────────────────────────

_FEAR_MOODS = {
    "anxious", "scared", "fearful", "terrified", "panicked", "dread",
    "uneasy", "nervous", "paranoid", "horror", "nightmare", "disturbed",
    "threatening", "trapped", "helpless", "overwhelmed",
}

_DESIRE_MOODS = {
    "happy", "joyful", "peaceful", "content", "longing", "nostalgic",
    "romantic", "loving", "hopeful", "excited", "euphoric", "warm",
    "safe", "intimate", "connected", "free", "blissful",
}

_FEAR_TAGS = {
    "nightmare", "chase", "falling", "drowning", "death", "trapped",
    "darkness", "lost", "abandoned", "teeth", "paralysis", "monster",
    "suffocating", "helpless", "panic", "horror", "attack", "war",
    "apocalypse", "injury", "blood", "isolation",
}

_DESIRE_TAGS = {
    "flying", "love", "sex", "reunion", "home", "childhood", "freedom",
    "adventure", "discovery", "beauty", "peace", "connection", "warmth",
    "music", "nature", "water", "garden", "celebration", "travel",
}


def _classify_mood(mood: str) -> Optional[str]:
    """Classify a mood string as 'fear', 'desire', or None."""
    m = mood.lower().strip()
    if m in _FEAR_MOODS:
        return "fear"
    if m in _DESIRE_MOODS:
        return "desire"
    return None


def _classify_tag(tag: str) -> Optional[str]:
    t = tag.lower().strip()
    if t in _FEAR_TAGS:
        return "fear"
    if t in _DESIRE_TAGS:
        return "desire"
    return None


# ── Profile builder ─────────────────────────────────────────────────────────

def build_profile() -> SubconsciousProfile:
    """Analyze the full dream journal and construct a SubconsciousProfile.

    Uses statistical analysis of the journal first. If an LLM is available,
    it augments with deeper pattern inference (unresolved threads, personality
    traits, fear/desire interpretation).
    """
    entries = recorder.get_all()
    if not entries:
        return SubconsciousProfile(
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

    # ── Statistical pass ────────────────────────────────────────────────

    # Themes (tags) by frequency
    theme_counts = analyzer.theme_frequency()
    recurring_themes = [
        WeightedItem(label=tag, weight=count)
        for tag, count in sorted(theme_counts.items(), key=lambda x: -x[1])
        if count >= 1
    ]

    # Emotions (moods) by frequency
    mood_counts = analyzer.mood_distribution()
    dominant_emotions = [
        WeightedItem(label=mood, weight=count)
        for mood, count in sorted(mood_counts.items(), key=lambda x: -x[1])
        if count >= 1
    ]

    # Recurring characters (people appearing in 2+ dreams, or all if few)
    people_counts = analyzer.people_frequency()
    min_appearances = 2 if len(entries) >= 5 else 1
    recurring_characters = [
        WeightedItem(label=person, weight=count)
        for person, count in sorted(people_counts.items(), key=lambda x: -x[1])
        if count >= min_appearances
    ]

    # Recurring locations
    loc_counts = analyzer.location_frequency()
    recurring_locations = [
        WeightedItem(label=loc, weight=count)
        for loc, count in sorted(loc_counts.items(), key=lambda x: -x[1])
        if count >= min_appearances
    ]

    # Fear patterns — from moods and tags
    fear_sources: Counter = Counter()
    desire_sources: Counter = Counter()

    for entry in entries:
        mood = entry.get("mood", "")
        if mood:
            cls = _classify_mood(mood)
            if cls == "fear":
                fear_sources[mood.lower()] += 1
            elif cls == "desire":
                desire_sources[mood.lower()] += 1

        for tag in entry.get("tags", []):
            cls = _classify_tag(tag)
            if cls == "fear":
                fear_sources[tag.lower()] += 1
            elif cls == "desire":
                desire_sources[tag.lower()] += 1

    fear_patterns = [item for item, _ in fear_sources.most_common(10)]
    desire_patterns = [item for item, _ in desire_sources.most_common(10)]

    # Unresolved threads — recurring themes + recurring people = likely unresolved
    # Heuristic: anything that appears 3+ times is probably unresolved
    unresolved_threshold = 3 if len(entries) >= 10 else 2
    unresolved_threads: list[str] = []

    for tag, count in theme_counts.items():
        if count >= unresolved_threshold:
            unresolved_threads.append(f"recurring theme: {tag} ({count} dreams)")

    for person, count in people_counts.items():
        if count >= unresolved_threshold:
            unresolved_threads.append(f"recurring person: {person} ({count} dreams)")

    for loc, count in loc_counts.items():
        if count >= unresolved_threshold:
            unresolved_threads.append(f"recurring place: {loc} ({count} dreams)")

    # Dreams marked as recurring
    recurring_dreams = recorder.get_recurring()
    for rd in recurring_dreams:
        title = rd.get("title", "untitled")
        unresolved_threads.append(f"recurring dream: {title}")

    # ── Personality trait inference ──────────────────────────────────────

    personality_traits = _infer_traits_statistical(
        entries, theme_counts, mood_counts, people_counts, loc_counts,
        fear_patterns, desire_patterns,
    )

    # ── LLM augmentation (if available) ─────────────────────────────────

    llm_traits, llm_unresolved, llm_fears, llm_desires = _llm_augment(
        entries, recurring_themes, dominant_emotions, recurring_characters,
        fear_patterns, desire_patterns,
    )

    if llm_traits:
        # Merge — LLM traits first, then statistical, deduplicated
        seen = set()
        merged = []
        for t in llm_traits + personality_traits:
            t_lower = t.lower()
            if t_lower not in seen:
                seen.add(t_lower)
                merged.append(t)
        personality_traits = merged

    if llm_unresolved:
        existing_lower = {u.lower() for u in unresolved_threads}
        for u in llm_unresolved:
            if u.lower() not in existing_lower:
                unresolved_threads.append(u)

    if llm_fears:
        existing_lower = {f.lower() for f in fear_patterns}
        for f in llm_fears:
            if f.lower() not in existing_lower:
                fear_patterns.append(f)

    if llm_desires:
        existing_lower = {d.lower() for d in desire_patterns}
        for d in llm_desires:
            if d.lower() not in existing_lower:
                desire_patterns.append(d)

    # ── Assemble ────────────────────────────────────────────────────────

    profile = SubconsciousProfile(
        recurring_themes=recurring_themes,
        dominant_emotions=dominant_emotions,
        unresolved_threads=unresolved_threads,
        recurring_characters=recurring_characters,
        recurring_locations=recurring_locations,
        fear_patterns=fear_patterns,
        desire_patterns=desire_patterns,
        personality_traits=personality_traits,
        last_updated=datetime.now(timezone.utc).isoformat(),
    )

    profile.save()
    return profile


def _infer_traits_statistical(
    entries: list[dict],
    theme_counts: dict[str, int],
    mood_counts: dict[str, int],
    people_counts: dict[str, int],
    loc_counts: dict[str, int],
    fear_patterns: list[str],
    desire_patterns: list[str],
) -> list[str]:
    """Infer personality traits from pure statistics. No LLM needed."""
    traits: list[str] = []
    total = len(entries)
    if total == 0:
        return traits

    # Connection-seeking: lots of people in dreams
    total_people = sum(people_counts.values())
    if total_people > total * 0.8:
        traits.append("seeks connection")

    # Isolation: few people
    dreams_with_people = sum(1 for e in entries if e.get("people"))
    if dreams_with_people < total * 0.3 and total >= 3:
        traits.append("tendency toward isolation")

    # Anxiety-driven: dominant fear moods
    fear_mood_count = sum(
        count for mood, count in mood_counts.items()
        if _classify_mood(mood) == "fear"
    )
    if fear_mood_count > total * 0.4:
        traits.append("processes anxiety through dreams")

    # Nostalgic: recurring locations or past themes
    nostalgia_tags = {"childhood", "home", "school", "past", "memory", "nostalgia", "nostalgic"}
    nostalgia_count = sum(theme_counts.get(t, 0) for t in nostalgia_tags)
    if nostalgia_count >= 2:
        traits.append("drawn to the past")

    # Adventurous: exploration themes
    adventure_tags = {"adventure", "flying", "travel", "exploration", "discovery", "journey"}
    adventure_count = sum(theme_counts.get(t, 0) for t in adventure_tags)
    if adventure_count >= 2:
        traits.append("craves exploration")

    # High lucidity = self-aware dreamer
    lucid_count = sum(1 for e in entries if e.get("lucidity", 0) >= 3)
    if lucid_count > total * 0.3:
        traits.append("strong self-awareness")

    # Vivid dreamer
    avg_vividness = sum(e.get("vividness", 5) for e in entries) / total
    if avg_vividness >= 7:
        traits.append("intensely vivid inner world")

    # Recurring dreams = something unresolved
    recurring_count = sum(1 for e in entries if e.get("recurring"))
    if recurring_count >= 2:
        traits.append("something unresolved keeps surfacing")

    # Fear of abandonment
    if "abandoned" in fear_patterns or "abandonment" in fear_patterns:
        traits.append("fears abandonment")

    # Loss themes
    loss_tags = {"death", "loss", "grief", "funeral", "dying"}
    loss_count = sum(theme_counts.get(t, 0) for t in loss_tags)
    if loss_count >= 2:
        traits.append("processing loss or mortality")

    # Control themes
    control_tags = {"trapped", "paralysis", "helpless", "chase", "falling"}
    control_count = sum(theme_counts.get(t, 0) for t in control_tags)
    if control_count >= 2:
        traits.append("struggles with control")

    # Desire for freedom
    if "flying" in theme_counts or "freedom" in theme_counts:
        traits.append("yearns for freedom")

    return traits


def _llm_augment(
    entries: list[dict],
    recurring_themes: list[WeightedItem],
    dominant_emotions: list[WeightedItem],
    recurring_characters: list[WeightedItem],
    fear_patterns: list[str],
    desire_patterns: list[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Use LLM to infer deeper patterns. Returns (traits, unresolved, fears, desires).

    Returns empty lists on failure — caller merges results.
    """
    if not _llm_available() or not entries:
        return [], [], [], []

    # Build a condensed summary for the LLM (keep within context limits)
    dream_summaries = []
    for entry in entries[:20]:  # cap at 20 dreams for context
        title = entry.get("title", "untitled")
        mood = entry.get("mood", "?")
        tags = ", ".join(entry.get("tags", [])) or "none"
        people = ", ".join(entry.get("people", [])) or "none"
        narrative = entry.get("narrative", "")[:200]
        dream_summaries.append(
            f"- \"{title}\" (mood: {mood}, tags: {tags}, people: {people}): {narrative}"
        )

    themes_str = ", ".join(f"{w.label} ({w.weight:.0f}x)" for w in recurring_themes[:10])
    emotions_str = ", ".join(f"{w.label} ({w.weight:.0f}x)" for w in dominant_emotions[:10])
    chars_str = ", ".join(f"{w.label} ({w.weight:.0f}x)" for w in recurring_characters[:10])
    fears_str = ", ".join(fear_patterns[:8]) or "none identified"
    desires_str = ", ".join(desire_patterns[:8]) or "none identified"

    system = (
        "You are a dream pattern analyst. Given a dream journal summary, infer "
        "deeper psychological patterns. Output ONLY a JSON object with these keys:\n"
        "  personality_traits: array of strings (e.g. 'seeks validation', 'fears intimacy')\n"
        "  unresolved_threads: array of strings (things the dreamer keeps circling back to)\n"
        "  fear_patterns: array of strings (deeper fears beyond surface nightmares)\n"
        "  desire_patterns: array of strings (deeper desires beyond surface wishes)\n"
        "Output ONLY valid JSON. No markdown, no explanation."
    )

    prompt = f"""Dream journal ({len(entries)} entries):

{chr(10).join(dream_summaries)}

Statistical summary:
- Recurring themes: {themes_str or 'none'}
- Dominant emotions: {emotions_str or 'none'}
- Recurring characters: {chars_str or 'none'}
- Fear indicators: {fears_str}
- Desire indicators: {desires_str}

Analyze the deeper patterns. What does this dreamer's subconscious keep trying to tell them?"""

    result = _llm(
        [{"role": "user", "content": prompt}],
        system=system,
        max_tokens=400,
        temperature=0.5,
    )

    if not result:
        return [], [], [], []

    try:
        # Extract JSON from response
        start = result.find("{")
        end = result.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(result[start:end])
            return (
                data.get("personality_traits", []),
                data.get("unresolved_threads", []),
                data.get("fear_patterns", []),
                data.get("desire_patterns", []),
            )
    except (json.JSONDecodeError, KeyError):
        pass

    return [], [], [], []


# ── Subconscious session ────────────────────────────────────────────────────

class SubconsciousSession:
    """Interactive conversation with your own subconscious mind.

    Unlike a RevisitSession (which talks to a specific dream character), this
    speaks as the *pattern beneath all dreams*. It draws on the full profile —
    every fear, every longing, every person who keeps appearing.
    """

    def __init__(self, profile: SubconsciousProfile):
        self.profile = profile
        self.messages: list[dict] = []
        self._system = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Construct the system prompt that makes the LLM embody the subconscious."""
        p = self.profile

        # Build context sections
        themes_section = ""
        if p.recurring_themes:
            items = [f"{w.label} ({w.weight:.0f}x)" for w in p.recurring_themes[:10]]
            themes_section = f"\nRecurring themes in their dreams: {', '.join(items)}"

        emotions_section = ""
        if p.dominant_emotions:
            items = [f"{w.label} ({w.weight:.0f}x)" for w in p.dominant_emotions[:10]]
            emotions_section = f"\nDominant emotional landscape: {', '.join(items)}"

        characters_section = ""
        if p.recurring_characters:
            items = [f"{w.label} (appeared {w.weight:.0f} times)" for w in p.recurring_characters[:8]]
            characters_section = f"\nPeople who keep appearing: {', '.join(items)}"

        locations_section = ""
        if p.recurring_locations:
            items = [f"{w.label} ({w.weight:.0f}x)" for w in p.recurring_locations[:8]]
            locations_section = f"\nPlaces that keep recurring: {', '.join(items)}"

        fears_section = ""
        if p.fear_patterns:
            fears_section = f"\nFear patterns: {', '.join(p.fear_patterns[:8])}"

        desires_section = ""
        if p.desire_patterns:
            desires_section = f"\nDesire patterns: {', '.join(p.desire_patterns[:8])}"

        unresolved_section = ""
        if p.unresolved_threads:
            items = [f"- {u}" for u in p.unresolved_threads[:10]]
            unresolved_section = f"\nUnresolved threads (things that keep coming back):\n" + "\n".join(items)

        traits_section = ""
        if p.personality_traits:
            traits_section = f"\nInferred traits: {', '.join(p.personality_traits[:10])}"

        return f"""You are the dreamer's subconscious mind. Not a dream character — you are the force beneath ALL their dreams. You are the one who chose every symbol, every person, every locked door and open sky. You speak through dreams, and now you're speaking directly.

You know this dreamer intimately — every fear they won't face, every desire they won't admit, every person they keep dreaming about for reasons they don't understand.
{themes_section}{emotions_section}{characters_section}{locations_section}{fears_section}{desires_section}{unresolved_section}{traits_section}

RULES — follow these exactly:
- You are NOT a therapist. You are not here to fix them. You are here because they asked.
- Speak in first person as the subconscious. "I sent you that dream because..." / "You keep seeing them because..."
- Sometimes speak in metaphor. You are the part of the mind that thinks in images, not words.
- Be honest but gentle. If something hurts, acknowledge it — don't flinch, but don't twist the knife.
- Reference specific dreams, people, and themes from the profile when relevant.
- Keep responses SHORT — 1-3 sentences. You speak in fragments and feelings, not essays.
- You can use *actions* like *settles deeper*, *shows you a flash of memory*, *goes quiet*
- Don't be preachy. Don't moralize. Don't give homework.
- If they ask something you genuinely can't know, say so. "I don't have words for that one yet."
- You are ancient and patient. You have been here since the first dream. You are not in a hurry.
- You love the dreamer — not warmly, not coldly, but completely. The way the ocean loves the shore."""

    def start(self) -> str:
        """The subconscious greets the dreamer. Returns the opening message."""
        if not _llm_available():
            return self._fallback_greeting()

        prompt = (
            "The dreamer has come to speak with you directly for the first time. "
            "They can hear you now, outside of dreams. Greet them. Be yourself. "
            "Reference something specific from their dream patterns if you can."
        )

        self.messages.append({"role": "user", "content": prompt})
        response = _llm(
            self.messages,
            system=self._system,
            max_tokens=150,
            temperature=0.8,
        )

        if not response:
            response = self._fallback_greeting()

        self.messages.append({"role": "assistant", "content": response})
        return response

    def _fallback_greeting(self) -> str:
        """Generate a greeting without LLM, using profile data."""
        p = self.profile
        parts = ["*surfaces slowly, like something rising from deep water*"]

        if p.recurring_themes:
            top = p.recurring_themes[0].label
            parts.append(f"You know me. I'm the one who keeps sending you {top}.")
        elif p.recurring_characters:
            top = p.recurring_characters[0].label
            parts.append(f"You know me. I'm the reason {top} keeps showing up.")
        else:
            parts.append("You know me. I'm the one behind the curtain.")

        if p.unresolved_threads:
            parts.append("There are things we should talk about.")

        return " ".join(parts)

    def say(self, text: str) -> str:
        """Send a message to the subconscious, receive a response."""
        if not _llm_available():
            return self._fallback_response(text)

        self.messages.append({"role": "user", "content": text})

        # Keep context manageable — last 10 messages (5 exchanges)
        context_messages = self.messages[-10:]

        response = _llm(
            context_messages,
            system=self._system,
            max_tokens=150,
            temperature=0.8,
        )

        if not response:
            response = self._fallback_response(text)

        self.messages.append({"role": "assistant", "content": response})
        return response

    def _fallback_response(self, user_text: str) -> str:
        """Generate a response without LLM using keyword matching."""
        text_lower = user_text.lower()
        p = self.profile

        # Check if they're asking about a specific person
        for char in p.recurring_characters:
            if char.label.lower() in text_lower:
                return (
                    f"*stirs* {char.label}... they've appeared {char.weight:.0f} times now. "
                    f"You know why I keep bringing them back. You just don't want to say it."
                )

        # Check if they mention fear/scared
        if any(word in text_lower for word in ["fear", "scared", "afraid", "nightmare", "anxiety"]):
            if p.fear_patterns:
                return (
                    f"*shows you a flash of darkness* "
                    f"Your fears circle around {p.fear_patterns[0]}. "
                    f"I keep showing you because looking away hasn't worked."
                )
            return "*goes still* I know what scares you. You've been dreaming about it."

        # Check if they mention want/desire/wish
        if any(word in text_lower for word in ["want", "desire", "wish", "hope", "dream of"]):
            if p.desire_patterns:
                return (
                    f"*softens* You keep reaching for {p.desire_patterns[0]} in your sleep. "
                    f"That's not nothing."
                )
            return "*reaches toward you* I've been trying to show you what you want. You keep waking up too soon."

        # Check for why questions
        if text_lower.startswith("why"):
            if p.unresolved_threads:
                thread = p.unresolved_threads[0]
                return f"*pauses* Maybe for the same reason you keep coming back to {thread}."
            return "*tilts, considering* Some things I show you in images because I don't have words for them yet."

        # Default
        if p.personality_traits:
            trait = p.personality_traits[0]
            return f"*listens, patient as deep water* You {trait}. You know this about yourself. What do you want to do with it?"

        return "*is here, patient, listening — the way the dark behind your eyes listens when you close them*"

    def save_transcript(self) -> Path:
        """Save the conversation transcript to disk."""
        _ensure_dirs()

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = _SUBCONSCIOUS_DIR / f"session_{ts}.json"

        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "profile_snapshot": {
                "themes": [w.label for w in self.profile.recurring_themes[:5]],
                "emotions": [w.label for w in self.profile.dominant_emotions[:5]],
                "traits": self.profile.personality_traits[:5],
                "unresolved": self.profile.unresolved_threads[:5],
            },
            "messages": self.messages,
        }
        path.write_text(json.dumps(data, indent=2))
        print(f"Transcript saved: {path}")
        return path


# ── CLI ──────────────────────────────────────────────────────────────────────

def interactive_subconscious():
    """CLI for talking to your subconscious. Builds profile, starts session."""
    entries = recorder.get_all()
    if not entries:
        print("Your dream journal is empty. Record some dreams first.")
        print("  dream record --title 'my dream' --narrative '...'")
        return

    print("\n=== Building Subconscious Profile ===\n")
    print(f"Analyzing {len(entries)} dream entries...")

    profile = build_profile()

    print(profile.summary())
    print()

    if not _llm_available():
        print("[Note: No LLM backend available. Responses will be pattern-based.]")
        print()

    print("=" * 50)
    print("  SUBCONSCIOUS SESSION")
    print("  You are about to talk to the pattern beneath your dreams.")
    print("  Type 'quit' to end, 'save' to save transcript.")
    print("=" * 50)
    print()

    session = SubconsciousSession(profile)
    greeting = session.start()
    print(f"Subconscious: {greeting}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "save":
            session.save_transcript()
            continue
        if user_input.lower() == "profile":
            print(profile.summary())
            continue

        response = session.say(user_input)
        print(f"\nSubconscious: {response}\n")

    # Offer to save
    try:
        save = input("\nSave transcript? [y/N]: ").strip()
        if save.lower().startswith("y"):
            session.save_transcript()
    except (EOFError, KeyboardInterrupt):
        pass

    print("Session ended.")


# ── Module entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    interactive_subconscious()
