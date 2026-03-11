"""dream_engine/suggester.py — Dream suggestion and lucid dreaming induction.

Generates pre-sleep suggestions to influence dream content, using:
1. Dream journal patterns (recurring themes, people, locations)
2. User-requested dream scenarios
3. Lucid dreaming techniques (MILD, WILD, reality checks)
4. LLM-powered dream scene generation

The suggester builds a "dream seed" — a short, vivid scene description
designed to be read/listened to before sleep to incubate the dream.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

from . import recorder

# ── Suggestion storage ───────────────────────────────────────────────────────

_DATA_DIR = Path.home() / ".local/share/blackmagic-gen/dreams"
_SUGGESTIONS_FILE = _DATA_DIR / "suggestions.json"
_TECHNIQUES_FILE = _DATA_DIR / "techniques.json"


def _load_suggestions() -> list[dict]:
    if _SUGGESTIONS_FILE.exists():
        try:
            return json.loads(_SUGGESTIONS_FILE.read_text())
        except Exception:
            pass
    return []


def _save_suggestions(entries: list[dict]):
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _SUGGESTIONS_FILE.write_text(json.dumps(entries, indent=2))


# ── Built-in lucid dreaming techniques ───────────────────────────────────────

TECHNIQUES = {
    "mild": {
        "name": "MILD (Mnemonic Induction of Lucid Dreams)",
        "steps": [
            "As you fall asleep, repeat: 'Next time I'm dreaming, I will realize I'm dreaming.'",
            "Visualize yourself in a recent dream, but this time recognizing it as a dream.",
            "Hold that intention as you drift off. The last thought before sleep matters most.",
        ],
        "best_for": "beginners, after waking from a dream mid-night",
    },
    "wild": {
        "name": "WILD (Wake-Initiated Lucid Dream)",
        "steps": [
            "Lie still with eyes closed. Focus on hypnagogic imagery (the patterns behind your eyelids).",
            "Let your body fall asleep while keeping your mind aware.",
            "You'll feel sleep paralysis — don't panic. Let it pass into the dream.",
            "You'll transition directly into a lucid dream with full awareness.",
        ],
        "best_for": "experienced, during WBTB (wake-back-to-bed)",
    },
    "wbtb": {
        "name": "WBTB (Wake Back To Bed)",
        "steps": [
            "Set an alarm for 5-6 hours after falling asleep.",
            "Wake up, stay awake for 20-30 minutes. Read about lucid dreaming.",
            "Go back to sleep using MILD or WILD technique.",
            "REM rebound makes lucid dreams much more likely.",
        ],
        "best_for": "highest success rate when combined with MILD",
    },
    "reality_checks": {
        "name": "Reality Checks",
        "steps": [
            "Throughout the day, genuinely ask: 'Am I dreaming right now?'",
            "Check: try pushing your finger through your palm.",
            "Check: read text, look away, read again (text changes in dreams).",
            "Check: count your fingers (often wrong number in dreams).",
            "Check: try breathing with your nose pinched shut.",
            "Do 10+ checks daily. The habit carries into dreams.",
        ],
        "best_for": "building the foundation, works with all other techniques",
    },
    "ssild": {
        "name": "SSILD (Senses Initiated Lucid Dream)",
        "steps": [
            "Wake after 5 hours of sleep.",
            "Cycle through senses with eyes closed:",
            "  - Sight: notice colors/patterns behind eyelids (15-30s)",
            "  - Sound: listen to ambient sounds or inner ear (15-30s)",
            "  - Touch: feel your body, blanket, temperature (15-30s)",
            "Repeat 4-5 cycles, then fall asleep normally.",
            "Lucid dreams often occur in the following REM period.",
        ],
        "best_for": "easy to learn, high success rate",
    },
}


# ── Dream seed generation ────────────────────────────────────────────────────

def _build_pattern_context() -> dict:
    """Analyze dream journal for recurring patterns to feed into suggestions."""
    s = recorder.stats()
    recent = recorder.get_all()[:10]

    return {
        "top_tags": [t[0] for t in s.get("top_tags", [])[:5]],
        "top_people": [p[0] for p in s.get("top_people", [])[:5]],
        "moods": [m[0] for m in s.get("moods", [])[:3]],
        "recurring_themes": [
            e.get("title", "") for e in recorder.get_recurring()[:5]
        ],
        "recent_locations": list({
            loc for e in recent for loc in e.get("locations", [])
        }),
        "avg_lucidity": s.get("avg_lucidity", 0),
        "total_dreams": s.get("total", 0),
    }


def suggest_dream_seed(
    scenario: str = "",
    include_lucidity: bool = True,
    use_journal_patterns: bool = True,
) -> dict:
    """Generate a dream seed — a pre-sleep visualization prompt.

    Args:
        scenario: Optional user-requested dream scenario (e.g. "flying over mountains")
        include_lucidity: Whether to weave in lucid dreaming cues
        use_journal_patterns: Whether to incorporate patterns from dream journal

    Returns:
        Dict with 'seed' (the visualization text), 'technique', and 'tips'.
    """
    patterns = _build_pattern_context() if use_journal_patterns else {}

    # Build the seed request for LLM
    seed_parts = []

    if scenario:
        seed_parts.append(f"Requested scenario: {scenario}")
    if patterns.get("top_tags"):
        seed_parts.append(f"Recurring dream themes: {', '.join(patterns['top_tags'])}")
    if patterns.get("recent_locations"):
        seed_parts.append(f"Dream locations: {', '.join(patterns['recent_locations'][:3])}")

    # Pick a technique recommendation
    if patterns.get("avg_lucidity", 0) < 1:
        technique = TECHNIQUES["reality_checks"]
        technique_key = "reality_checks"
    elif patterns.get("total_dreams", 0) > 20:
        technique = TECHNIQUES["ssild"]
        technique_key = "ssild"
    else:
        technique = TECHNIQUES["mild"]
        technique_key = "mild"

    result = {
        "seed_prompt": seed_parts,
        "scenario": scenario or "(journal-based)",
        "technique": technique_key,
        "technique_info": technique,
        "tips": [
            "Read this seed slowly 2-3 times before closing your eyes.",
            "Visualize each detail — colors, sounds, textures, smells.",
            "As you drift off, hold the scene gently. Don't force it.",
            "If you notice something odd in the dream, do a reality check.",
        ],
        "patterns": patterns,
        "llm_prompt": None,  # filled by generate_seed_with_llm()
    }

    return result


def generate_seed_with_llm(scenario: str = "", use_journal: bool = True) -> str:
    """Use the local LLM to generate a vivid dream seed scene.

    Returns the generated dream seed text, or a fallback if LLM is unavailable.
    """
    try:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from grimoire.ai_backend import bm_call
    except ImportError:
        # Standalone mode — use OpenAI-compatible endpoint if available
        bm_call = None

    patterns = _build_pattern_context() if use_journal else {}

    context_parts = []
    if scenario:
        context_parts.append(f"The dreamer wants to dream about: {scenario}")
    if patterns.get("top_tags"):
        context_parts.append(f"Their recurring dream themes include: {', '.join(patterns['top_tags'])}")
    if patterns.get("top_people"):
        context_parts.append(f"People who appear in their dreams: {', '.join(patterns['top_people'])}")
    if patterns.get("recent_locations"):
        context_parts.append(f"Familiar dream locations: {', '.join(patterns['recent_locations'][:3])}")

    context = "\n".join(context_parts) if context_parts else "Generate a peaceful, immersive dream scene."

    system = (
        "You are a dream architect. Write a short, vivid dream scene (3-5 paragraphs) "
        "designed to be read before sleep. Use second person ('you'). "
        "Include rich sensory details — what you see, hear, feel, smell. "
        "Make it immersive and slightly surreal. "
        "Weave in a subtle cue the dreamer can use to realize they're dreaming "
        "(like noticing their hands glow, or text that shifts). "
        "End with the scene gently opening into possibility, inviting the dreamer to explore."
    )

    messages = [{"role": "user", "content": context}]

    if bm_call is None:
        return _fallback_seed(scenario)
    result = bm_call(messages, max_tokens=400, temperature=0.85, system=system)
    if result and "[BlackMagic Offline]" not in result:
        return result
    return _fallback_seed(scenario)


def _fallback_seed(scenario: str = "") -> str:
    """Static fallback if LLM is unavailable."""
    seeds = [
        (
            "You're standing at the edge of a vast, still lake. The water is impossibly clear — "
            "you can see the bottom, hundreds of feet down, where soft light pulses like a heartbeat. "
            "The air smells of rain and cedar. You look at your hands — they shimmer faintly, "
            "translucent at the edges. You realize: this is a dream. The lake invites you in."
        ),
        (
            "You're walking through a library that stretches endlessly in every direction. "
            "The books hum softly, each one a different frequency. You pull one from the shelf — "
            "the text on the cover rearranges itself as you read it. You smile. You know this place. "
            "You're dreaming. Every book here contains a world you can step into."
        ),
        (
            "You're on a rooftop at twilight. The city below is familiar but wrong — the buildings "
            "are too tall, the sky has two moons. A warm wind carries the sound of distant music. "
            "You look at your hands and count six fingers. This is a dream. "
            "You step to the edge, knowing you can fly."
        ),
    ]
    if scenario:
        return f"You find yourself in a dream. {scenario}. Look at your hands — they shimmer. You're lucid now. Explore."
    return random.choice(seeds)


def get_technique(name: str) -> dict | None:
    """Get details for a specific lucid dreaming technique."""
    return TECHNIQUES.get(name.lower())


def list_techniques() -> list[dict]:
    """List all available techniques."""
    return [{"key": k, **v} for k, v in TECHNIQUES.items()]


def save_suggestion(seed_text: str, scenario: str, technique: str, notes: str = ""):
    """Save a generated suggestion for later review."""
    from datetime import datetime, timezone
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario,
        "seed_text": seed_text,
        "technique": technique,
        "notes": notes,
        "used": False,
    }
    suggestions = _load_suggestions()
    suggestions.append(entry)
    _save_suggestions(suggestions)
    return entry


def get_saved_suggestions() -> list[dict]:
    return list(reversed(_load_suggestions()))
