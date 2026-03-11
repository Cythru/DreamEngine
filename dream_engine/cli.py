#!/usr/bin/env python3
"""dream_engine/cli.py — Interactive dream recorder and suggester.

Usage:
    python -m dream_engine.cli                 # interactive menu
    python -m dream_engine.cli record          # quick-record a dream
    python -m dream_engine.cli suggest [scene] # generate a dream seed
    python -m dream_engine.cli journal         # list recent dreams
    python -m dream_engine.cli stats           # journal statistics
    python -m dream_engine.cli search <query>  # search dreams
    python -m dream_engine.cli interpret <id>  # LLM interpretation
    python -m dream_engine.cli techniques      # lucid dreaming techniques
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dream_engine import recorder, suggester, analyzer
from dream_engine.revisit import interactive_revisit, extract_characters, save_character


def _input(prompt: str, default: str = "") -> str:
    try:
        val = input(prompt).strip()
        return val if val else default
    except (EOFError, KeyboardInterrupt):
        print()
        return default


def _input_int(prompt: str, default: int, lo: int, hi: int) -> int:
    raw = _input(f"{prompt} [{lo}-{hi}, default {default}]: ")
    if not raw:
        return default
    try:
        v = int(raw)
        return max(lo, min(hi, v))
    except ValueError:
        return default


def cmd_record():
    """Interactive dream recording."""
    print("\n=== Record a Dream ===\n")
    print("(Write as much or as little as you remember. Every detail helps.)\n")

    title = _input("Title (short name): ")
    if not title:
        print("Cancelled.")
        return

    print("Narrative (describe the dream, press Enter twice to finish):")
    lines = []
    while True:
        line = _input("")
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)
    narrative = "\n".join(lines).strip()

    if not narrative:
        narrative = title

    tags_raw = _input("Tags (comma-separated, e.g. flying,water,chase): ")
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []

    mood = _input("Mood (e.g. anxious, peaceful, confused, excited): ")
    vividness = _input_int("Vividness", 5, 1, 10)
    lucidity = _input_int("Lucidity (0=none, 5=full control)", 0, 0, 5)
    recurring = _input("Recurring dream? [y/N]: ").lower().startswith("y")

    people_raw = _input("People in the dream (comma-separated): ")
    people = [p.strip() for p in people_raw.split(",") if p.strip()] if people_raw else []

    locations_raw = _input("Locations (comma-separated): ")
    locations = [l.strip() for l in locations_raw.split(",") if l.strip()] if locations_raw else []

    sleep_quality = _input_int("Sleep quality", 5, 1, 10)
    notes = _input("Extra notes: ")

    entry = recorder.record(
        title=title, narrative=narrative, tags=tags, mood=mood,
        vividness=vividness, lucidity=lucidity, recurring=recurring,
        people=people, locations=locations, notes=notes,
        sleep_quality=sleep_quality,
    )

    print(f"\nDream recorded! ID: {entry['id']}")
    print(f"  Title: {entry['title']}")
    print(f"  Vividness: {entry['vividness']}/10 | Lucidity: {entry['lucidity']}/5")
    print(f"  Tags: {', '.join(entry['tags']) or 'none'}")


def cmd_suggest(scenario: str = ""):
    """Generate a pre-sleep dream seed."""
    print("\n=== Dream Suggester ===\n")

    if not scenario:
        scenario = _input("What would you like to dream about? (blank for journal-based): ")

    seed_info = suggester.suggest_dream_seed(
        scenario=scenario,
        include_lucidity=True,
        use_journal_patterns=True,
    )

    print(f"\nRecommended technique: {seed_info['technique_info']['name']}")
    for step in seed_info["technique_info"]["steps"]:
        print(f"  - {step}")

    print(f"\nGenerating dream seed with LLM...")
    seed_text = suggester.generate_seed_with_llm(scenario=scenario)
    print(f"\n{'─'*50}")
    print(seed_text)
    print(f"{'─'*50}")

    print("\nTips:")
    for tip in seed_info["tips"]:
        print(f"  - {tip}")

    save = _input("\nSave this suggestion? [y/N]: ")
    if save.lower().startswith("y"):
        suggester.save_suggestion(seed_text, scenario, seed_info["technique"])
        print("Saved!")


def cmd_journal(count: int = 10):
    """Show recent dreams."""
    dreams = recorder.get_all()[:count]
    if not dreams:
        print("\nNo dreams recorded yet. Use 'record' to log your first dream.")
        return

    print(f"\n=== Dream Journal (last {count}) ===\n")
    for e in dreams:
        date = e["timestamp"][:10]
        lucid = f" [LUCID:{e['lucidity']}]" if e.get("lucidity", 0) > 0 else ""
        recurring = " [RECURRING]" if e.get("recurring") else ""
        tags = f" [{', '.join(e.get('tags', []))}]" if e.get("tags") else ""
        print(f"  {date} | {e['id']} | {e['title']}{lucid}{recurring}{tags}")
        if e.get("narrative"):
            preview = e["narrative"][:80].replace("\n", " ")
            print(f"           {preview}...")
        print()


def cmd_stats():
    """Show journal statistics."""
    s = recorder.stats()
    if s["total"] == 0:
        print("\nNo dreams recorded yet.")
        return

    print(f"\n=== Dream Stats ===\n")
    print(f"  Total dreams:       {s['total']}")
    print(f"  Recurring:          {s['recurring']}")
    print(f"  Lucid:              {s['lucid']}")
    print(f"  Avg vividness:      {s['avg_vividness']}/10")
    print(f"  Avg lucidity:       {s['avg_lucidity']}/5")
    print(f"  Avg sleep quality:  {s['avg_sleep_quality']}/10")

    streak_info = analyzer.streak()
    print(f"  Current streak:     {streak_info['current_streak']} days")
    print(f"  Longest streak:     {streak_info['longest_streak']} days")
    print(f"  Total days logged:  {streak_info['total_days']}")

    if s.get("top_tags"):
        print(f"\n  Top themes: {', '.join(f'{t[0]}({t[1]})' for t in s['top_tags'][:5])}")
    if s.get("top_people"):
        print(f"  Top people: {', '.join(f'{p[0]}({p[1]})' for p in s['top_people'][:5])}")
    if s.get("moods"):
        print(f"  Moods:      {', '.join(f'{m[0]}({m[1]})' for m in s['moods'][:5])}")


def cmd_search(query: str):
    results = recorder.search(query)
    if not results:
        print(f"\nNo dreams matching '{query}'.")
        return
    print(f"\n=== {len(results)} dreams matching '{query}' ===\n")
    for e in results[:20]:
        date = e["timestamp"][:10]
        print(f"  {date} | {e['id']} | {e['title']}")


def cmd_interpret(dream_id: str):
    entry = recorder.get_by_id(dream_id)
    if not entry:
        print(f"\nDream '{dream_id}' not found.")
        return
    print(f"\n=== Interpreting: {entry['title']} ===\n")
    result = analyzer.interpret_with_llm(dream_id)
    print(result)


def cmd_techniques():
    print("\n=== Lucid Dreaming Techniques ===\n")
    for tech in suggester.list_techniques():
        print(f"  [{tech['key']}] {tech['name']}")
        print(f"    Best for: {tech['best_for']}")
        for step in tech["steps"]:
            print(f"      - {step}")
        print()


def cmd_revisit():
    """Revisit a dream character."""
    interactive_revisit()


def cmd_menu():
    """Interactive menu."""
    print("\n=== DreamEngine ===")
    print("  1) Record a dream")
    print("  2) Suggest a dream (pre-sleep seed)")
    print("  3) View journal")
    print("  4) Stats")
    print("  5) Search")
    print("  6) Interpret a dream (LLM)")
    print("  7) Lucid dreaming techniques")
    print("  8) Revisit a dream character")
    print("  q) Quit")

    while True:
        choice = _input("\n> ")
        if choice in ("1", "record"):
            cmd_record()
        elif choice in ("2", "suggest"):
            cmd_suggest()
        elif choice in ("3", "journal"):
            cmd_journal()
        elif choice in ("4", "stats"):
            cmd_stats()
        elif choice in ("5", "search"):
            q = _input("Search: ")
            if q:
                cmd_search(q)
        elif choice in ("6", "interpret"):
            dream_id = _input("Dream ID: ")
            if dream_id:
                cmd_interpret(dream_id)
        elif choice in ("7", "techniques"):
            cmd_techniques()
        elif choice in ("8", "revisit"):
            cmd_revisit()
        elif choice in ("q", "quit", "exit"):
            print("Goodnight.")
            break
        else:
            print("Pick 1-8 or q")


def main():
    args = sys.argv[1:]

    if not args:
        cmd_menu()
    elif args[0] == "record":
        cmd_record()
    elif args[0] == "suggest":
        cmd_suggest(" ".join(args[1:]) if len(args) > 1 else "")
    elif args[0] == "journal":
        cmd_journal(int(args[1]) if len(args) > 1 else 10)
    elif args[0] == "stats":
        cmd_stats()
    elif args[0] == "search":
        cmd_search(" ".join(args[1:]) if len(args) > 1 else "")
    elif args[0] == "interpret":
        cmd_interpret(args[1] if len(args) > 1 else "")
    elif args[0] == "techniques":
        cmd_techniques()
    elif args[0] == "revisit":
        dream_id = args[1] if len(args) > 1 else ""
        char_name = args[2] if len(args) > 2 else ""
        interactive_revisit(dream_id, char_name)
    else:
        print(f"Unknown command: {args[0]}")
        print("Commands: record, suggest, journal, stats, search, interpret, techniques, revisit")


if __name__ == "__main__":
    main()
