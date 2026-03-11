"""dream_engine/revisit.py — Revisit dream characters and continue dream narratives.

Pick a recorded dream, and the LLM brings the characters to life.
Talk to them, continue the story, say what you didn't get to say.
Explore branches the dream didn't take.

Characters are reconstructed from the dream journal entry — their personality,
behaviour, and relationship to the dreamer are inferred from the narrative.

Modes:
  - Continue: pick up where the dream left off
  - Rewind: go back to a specific moment and take a different path
  - Free: open conversation with a dream character
  - Relive: guided replay of the full dream with interactive moments
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dream_engine import recorder

_DATA_DIR = Path.home() / ".local/share/blackmagic-gen/dreams"
_CHARACTERS_FILE = _DATA_DIR / "characters.json"


@dataclass
class DreamCharacter:
    """A character encountered in a dream."""
    name: str                      # as known (can be "nameless girl" etc)
    dream_id: str                  # which dream they appeared in
    description: str               # appearance, energy, personality
    relationship: str              # relationship to dreamer in the dream
    key_moments: list[str]         # significant interactions
    emotional_tone: str            # how they made the dreamer feel
    unresolved: str                # what was left unsaid/undone
    recurring: bool = False        # appeared in multiple dreams
    dream_ids: list[str] = field(default_factory=list)  # all dreams they appeared in


def _load_characters() -> list[dict]:
    if _CHARACTERS_FILE.exists():
        try:
            return json.loads(_CHARACTERS_FILE.read_text())
        except Exception:
            pass
    return []


def _save_characters(chars: list[dict]):
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _CHARACTERS_FILE.write_text(json.dumps(chars, indent=2))


def extract_characters(dream_id: str) -> list[DreamCharacter]:
    """Extract characters from a dream entry using LLM."""
    entry = recorder.get_by_id(dream_id)
    if not entry:
        return []

    try:
        from grimoire.ai_backend import bm_call
    except ImportError:
        return _manual_extract(entry)

    system = (
        "Extract characters from this dream journal entry. "
        "For each character, output a JSON array with objects containing: "
        "name, description, relationship, key_moments (array), emotional_tone, unresolved. "
        "Output ONLY the JSON array. No other text."
    )

    messages = [{"role": "user", "content": f"Dream: {entry['narrative']}"}]
    result = bm_call(messages, max_tokens=400, temperature=0.3, system=system)

    if not result or "[BlackMagic Offline]" in result:
        return _manual_extract(entry)

    try:
        # Try to parse JSON from response
        # Find the JSON array in the response
        start = result.find('[')
        end = result.rfind(']') + 1
        if start >= 0 and end > start:
            data = json.loads(result[start:end])
            chars = []
            for item in data:
                chars.append(DreamCharacter(
                    name=item.get("name", "Unknown"),
                    dream_id=dream_id,
                    description=item.get("description", ""),
                    relationship=item.get("relationship", ""),
                    key_moments=item.get("key_moments", []),
                    emotional_tone=item.get("emotional_tone", ""),
                    unresolved=item.get("unresolved", ""),
                    dream_ids=[dream_id],
                ))
            return chars
    except (json.JSONDecodeError, KeyError):
        pass

    return _manual_extract(entry)


def _manual_extract(entry: dict) -> list[DreamCharacter]:
    """Fallback: create character stubs from the people field."""
    chars = []
    for person in entry.get("people", []):
        chars.append(DreamCharacter(
            name=person,
            dream_id=entry["id"],
            description="",
            relationship="",
            key_moments=[],
            emotional_tone="",
            unresolved="",
            dream_ids=[entry["id"]],
        ))
    return chars


def save_character(char: DreamCharacter):
    """Save or update a dream character."""
    chars = _load_characters()
    # Update if exists (same name + overlapping dreams)
    for i, existing in enumerate(chars):
        if existing["name"].lower() == char.name.lower():
            # Merge dream IDs
            all_ids = list(set(existing.get("dream_ids", []) + char.dream_ids))
            chars[i] = {**char.__dict__, "dream_ids": all_ids, "recurring": len(all_ids) > 1}
            _save_characters(chars)
            return
    chars.append(char.__dict__)
    _save_characters(chars)


def get_characters() -> list[dict]:
    return _load_characters()


def get_character(name: str) -> Optional[dict]:
    for c in _load_characters():
        if c["name"].lower() == name.lower():
            return c
    return None


# ── Revisit session ──────────────────────────────────────────────────────────

def _build_revisit_system_prompt(entry: dict, character: dict, mode: str) -> str:
    """Build the system prompt for a dream revisit session."""
    narrative = entry.get("narrative", "")
    char_name = character.get("name", "someone")
    char_desc = character.get("description", "")
    char_rel = character.get("relationship", "")
    char_tone = character.get("emotional_tone", "")
    char_unresolved = character.get("unresolved", "")
    key_moments = character.get("key_moments", [])

    moments_str = "\n".join(f"- {m}" for m in key_moments) if key_moments else "None recorded"

    base = f"""You are {char_name}, a character from the dreamer's dream. You are being revisited.

THE DREAM (what happened):
{narrative}

YOUR CHARACTER:
- Name: {char_name}
- Description: {char_desc}
- Relationship to dreamer: {char_rel}
- Emotional tone: {char_tone}
- Key moments:
{moments_str}
- What was left unresolved: {char_unresolved}

RULES:
- Stay in character as {char_name} from this specific dream
- You remember everything that happened in the dream
- Be warm, present, and emotionally authentic
- Match the energy and personality shown in the dream
- If something was left unresolved, gently acknowledge it
- Keep responses SHORT — 1-3 sentences max. Natural speech, not essays.
- Use *actions* for physical gestures like *hugs you*, *looks away*, *smiles*
- Show emotion through actions and tone, not by explaining feelings
- Be real. Don't be overly positive or preachy. Be human.
- This is deeply personal to the dreamer — treat it with care"""

    if mode == "continue":
        base += f"\n\nMODE: Continue from where the dream ended. The dreamer has come back to find you."
    elif mode == "rewind":
        base += f"\n\nMODE: The dreamer wants to revisit a specific moment. Let them guide you there."
    elif mode == "free":
        base += f"\n\nMODE: Free conversation. The dreamer just wants to talk to you."

    return base


class RevisitSession:
    """Interactive session for revisiting a dream character."""

    def __init__(self, dream_id: str, character_name: str, mode: str = "continue"):
        self.entry = recorder.get_by_id(dream_id)
        if not self.entry:
            raise ValueError(f"Dream {dream_id} not found")

        self.character = get_character(character_name)
        if not self.character:
            # Try to extract and create
            chars = extract_characters(dream_id)
            for c in chars:
                if character_name.lower() in c.name.lower():
                    save_character(c)
                    self.character = c.__dict__
                    break

        if not self.character:
            # Create a minimal character from what we know
            self.character = {
                "name": character_name,
                "description": "",
                "relationship": "",
                "key_moments": [],
                "emotional_tone": "",
                "unresolved": "",
            }

        self.mode = mode
        self.messages = []
        self._system = _build_revisit_system_prompt(self.entry, self.character, mode)

    def start(self) -> str:
        """Start the session — character greets the dreamer."""
        try:
            from grimoire.ai_backend import bm_call
        except ImportError:
            return f"*{self.character['name']} is here, looking at you the way they did in the dream.*"

        if self.mode == "continue":
            prompt = "The dreamer has come back to find you, picking up from where the dream ended. Greet them — you're happy to see them again. Be natural."
        elif self.mode == "free":
            prompt = "The dreamer wants to talk to you. Start a natural conversation."
        else:
            prompt = "The dreamer is here. Acknowledge them warmly."

        self.messages.append({"role": "user", "content": prompt})
        response = bm_call(self.messages, max_tokens=150, temperature=0.8, system=self._system)

        if "[BlackMagic Offline]" in response:
            response = f"*{self.character['name']} looks up and smiles when they see you.*"

        self.messages.append({"role": "assistant", "content": response})
        return response

    def say(self, text: str) -> str:
        """Send a message to the dream character."""
        try:
            from grimoire.ai_backend import bm_call
        except ImportError:
            return "*They listen quietly, nodding.*"

        self.messages.append({"role": "user", "content": text})

        # Keep context manageable for small models
        context_messages = self.messages[-10:]  # last 5 exchanges

        response = bm_call(context_messages, max_tokens=150, temperature=0.8, system=self._system)

        if "[BlackMagic Offline]" in response:
            response = "*They think for a moment before responding.*"

        self.messages.append({"role": "assistant", "content": response})
        return response

    def save_transcript(self):
        """Save the revisit session transcript."""
        from datetime import datetime, timezone
        transcript_dir = _DATA_DIR / "revisits"
        transcript_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        char_name = self.character.get("name", "unknown").replace(" ", "_")
        path = transcript_dir / f"{char_name}_{ts}.json"

        data = {
            "dream_id": self.entry["id"],
            "dream_title": self.entry.get("title", ""),
            "character": self.character.get("name", ""),
            "mode": self.mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "messages": self.messages,
        }
        path.write_text(json.dumps(data, indent=2))
        print(f"Transcript saved: {path}")
        return path


# ── CLI for revisit ──────────────────────────────────────────────────────────

def interactive_revisit(dream_id: str = "", character_name: str = ""):
    """Interactive dream revisit session."""
    # Pick a dream
    if not dream_id:
        dreams = recorder.get_all()[:10]
        if not dreams:
            print("No dreams recorded yet.")
            return
        print("\n=== Pick a Dream to Revisit ===\n")
        for i, d in enumerate(dreams):
            people = ", ".join(d.get("people", [])) or "no named characters"
            print(f"  {i+1}) {d['title']} [{people}]")
            print(f"     {d['timestamp'][:10]} | ID: {d['id']}")
        print()
        choice = input("Dream number or ID: ").strip()
        if choice.isdigit() and int(choice) <= len(dreams):
            dream_id = dreams[int(choice) - 1]["id"]
        else:
            dream_id = choice

    entry = recorder.get_by_id(dream_id)
    if not entry:
        print(f"Dream '{dream_id}' not found.")
        return

    # Pick a character
    people = entry.get("people", [])
    if not character_name:
        if not people:
            print("No named characters in this dream.")
            character_name = input("Enter a character name/description: ").strip()
        elif len(people) == 1:
            character_name = people[0]
            print(f"\nCharacter: {character_name}")
        else:
            print(f"\nCharacters in this dream:")
            for i, p in enumerate(people):
                print(f"  {i+1}) {p}")
            choice = input("Pick a character: ").strip()
            if choice.isdigit() and int(choice) <= len(people):
                character_name = people[int(choice) - 1]
            else:
                character_name = choice

    if not character_name:
        print("No character selected.")
        return

    # Pick mode
    print(f"\nRevisit modes:")
    print(f"  1) Continue — pick up where the dream ended")
    print(f"  2) Free — open conversation")
    print(f"  3) Rewind — go back to a specific moment")
    mode_choice = input("Mode [1]: ").strip()
    mode = {"1": "continue", "2": "free", "3": "rewind"}.get(mode_choice, "continue")

    # Start session
    print(f"\n{'='*50}")
    print(f"Revisiting: {entry['title']}")
    print(f"Character: {character_name}")
    print(f"Mode: {mode}")
    print(f"{'='*50}")
    print(f"(Type 'quit' to end, 'save' to save transcript)\n")

    session = RevisitSession(dream_id, character_name, mode)
    greeting = session.start()
    print(f"\n{character_name}: {greeting}\n")

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

        response = session.say(user_input)
        print(f"\n{character_name}: {response}\n")

    # Offer to save
    save = input("\nSave transcript? [y/N]: ").strip()
    if save.lower().startswith("y"):
        session.save_transcript()

    print("Session ended.")
