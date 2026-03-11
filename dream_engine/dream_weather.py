"""dream_engine/dream_weather.py — Predicts tonight's dream weather.

Predicts tonight's dream conditions — what you're likely to dream about,
how vivid it will be, and whether storms (nightmares) are on the horizon.
Built from statistical patterns in your dream journal, calibrated by your
waking state.
"""
from __future__ import annotations

import math
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dream_engine import recorder, analyzer


# ---------------------------------------------------------------------------
# Mood mapping — waking moods to likely dream moods
# ---------------------------------------------------------------------------

_MOOD_MAP: dict[str, list[tuple[str, float]]] = {
    # waking mood -> [(dream_mood, weight), ...]
    "happy": [("peaceful", 0.4), ("excited", 0.3), ("happy", 0.2), ("adventurous", 0.1)],
    "excited": [("excited", 0.4), ("adventurous", 0.3), ("happy", 0.2), ("chaotic", 0.1)],
    "anxious": [("anxious", 0.4), ("confused", 0.2), ("tense", 0.2), ("surreal", 0.2)],
    "stressed": [("anxious", 0.3), ("tense", 0.3), ("chaotic", 0.2), ("dark", 0.2)],
    "sad": [("melancholic", 0.4), ("lonely", 0.3), ("peaceful", 0.2), ("nostalgic", 0.1)],
    "angry": [("tense", 0.3), ("chaotic", 0.3), ("dark", 0.2), ("anxious", 0.2)],
    "calm": [("peaceful", 0.5), ("serene", 0.2), ("nostalgic", 0.2), ("adventurous", 0.1)],
    "peaceful": [("peaceful", 0.5), ("serene", 0.3), ("happy", 0.1), ("nostalgic", 0.1)],
    "tired": [("confused", 0.3), ("surreal", 0.3), ("peaceful", 0.2), ("melancholic", 0.2)],
    "bored": [("adventurous", 0.3), ("surreal", 0.3), ("nostalgic", 0.2), ("confused", 0.2)],
    "confused": [("confused", 0.4), ("surreal", 0.3), ("chaotic", 0.2), ("anxious", 0.1)],
    "nostalgic": [("nostalgic", 0.5), ("melancholic", 0.2), ("peaceful", 0.2), ("happy", 0.1)],
}

_NIGHTMARE_MOODS = {"anxious", "stressed", "angry", "dark", "tense", "chaotic", "scared",
                    "terrified", "panicked", "horrified", "nightmare", "disturbing"}

_POSITIVE_MOODS = {"happy", "peaceful", "calm", "excited", "serene", "adventurous",
                   "joyful", "content", "blissful", "hopeful"}


# ---------------------------------------------------------------------------
# DreamForecast dataclass
# ---------------------------------------------------------------------------

@dataclass
class DreamForecast:
    """Tonight's dream weather prediction."""
    predicted_vividness: float          # 1-10
    predicted_lucidity_chance: float    # 0-1
    predicted_mood: str
    predicted_themes: list[str]         # likely tags/themes
    predicted_characters: list[str]     # people likely to appear
    nightmare_risk: float               # 0-1
    recurring_dream_chance: float       # 0-1
    confidence: float                   # 0-1
    tips: list[str] = field(default_factory=list)
    forecast_text: str = ""


# ---------------------------------------------------------------------------
# DreamWeatherStation
# ---------------------------------------------------------------------------

class DreamWeatherStation:
    """Analyzes dream journal patterns and predicts tonight's dream conditions."""

    def __init__(self):
        self._journal: list[dict] = recorder.get_all()  # newest first
        self._base_rates: dict = {}
        self._calculate_base_rates()

    # -- base rates ---------------------------------------------------------

    def _calculate_base_rates(self):
        """Compute averages and distributions from the full journal."""
        journal = self._journal
        if not journal:
            self._base_rates = {
                "avg_vividness": 5.0,
                "avg_lucidity": 0.0,
                "avg_sleep_quality": 5.0,
                "lucid_rate": 0.0,
                "recurring_rate": 0.0,
                "nightmare_rate": 0.0,
                "total": 0,
                "mood_counts": {},
                "tag_counts": {},
                "people_counts": {},
            }
            return

        vividness_vals = [e.get("vividness", 5) for e in journal]
        lucidity_vals = [e.get("lucidity", 0) for e in journal]
        sleep_vals = [e.get("sleep_quality", 5) for e in journal]

        mood_counts: Counter = Counter()
        tag_counts: Counter = Counter()
        people_counts: Counter = Counter()
        nightmare_count = 0

        for e in journal:
            mood = e.get("mood", "").lower()
            if mood:
                mood_counts[mood] += 1
                if mood in _NIGHTMARE_MOODS:
                    nightmare_count += 1
            for t in e.get("tags", []):
                tag_counts[t.lower()] += 1
            for p in e.get("people", []):
                people_counts[p.lower()] += 1

        n = len(journal)
        self._base_rates = {
            "avg_vividness": sum(vividness_vals) / n,
            "avg_lucidity": sum(lucidity_vals) / n,
            "avg_sleep_quality": sum(sleep_vals) / n,
            "lucid_rate": sum(1 for v in lucidity_vals if v > 0) / n,
            "recurring_rate": sum(1 for e in journal if e.get("recurring")) / n,
            "nightmare_rate": nightmare_count / n if n else 0.0,
            "total": n,
            "mood_counts": dict(mood_counts.most_common()),
            "tag_counts": dict(tag_counts.most_common()),
            "people_counts": dict(people_counts.most_common()),
        }

    # -- recency weighting --------------------------------------------------

    def _recency_weight(self, timestamp: str) -> float:
        """More recent dreams are weighted higher. Returns 0-1 weight.

        Uses exponential decay with a half-life of 14 days.
        """
        try:
            dt = datetime.fromisoformat(timestamp)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return 0.5

        now = datetime.now(timezone.utc)
        days_ago = max(0.0, (now - dt).total_seconds() / 86400)
        half_life = 14.0
        return math.exp(-0.693 * days_ago / half_life)  # ln(2) ~ 0.693

    # -- mood correlation ---------------------------------------------------

    def _mood_correlation(self, input_mood: str) -> str:
        """Map a waking mood to the most likely dream mood.

        Uses journal data when available (recency-weighted), falls back
        to the static mood map.
        """
        input_lower = input_mood.lower().strip() if input_mood else ""

        # If we have journal entries with mood data, use recency-weighted
        # distribution to find what dream moods followed similar waking states.
        # Since we don't track waking mood separately, we use journal mood
        # distribution weighted by recency as a proxy.
        if self._journal and self._base_rates["mood_counts"]:
            weighted_moods: Counter = Counter()
            for e in self._journal:
                m = e.get("mood", "").lower()
                if m:
                    w = self._recency_weight(e.get("timestamp", ""))
                    weighted_moods[m] += w

            # If the input mood appears in our journal, boost similar moods
            if input_lower and weighted_moods:
                # Blend journal distribution with static map
                if input_lower in _MOOD_MAP:
                    for mapped_mood, map_weight in _MOOD_MAP[input_lower]:
                        weighted_moods[mapped_mood] += map_weight * 2.0

                if weighted_moods:
                    return weighted_moods.most_common(1)[0][0]

        # Fall back to static mood map
        if input_lower in _MOOD_MAP:
            # Pick the highest-weight entry
            candidates = _MOOD_MAP[input_lower]
            return max(candidates, key=lambda x: x[1])[0]

        # Completely unknown mood — return it as-is or a default
        return input_lower if input_lower else "neutral"

    # -- theme prediction ---------------------------------------------------

    def _theme_prediction(self, recent_n: int = 10) -> list[str]:
        """Predict likely themes from recent patterns.

        Combines overall frequency with recency weighting.
        """
        if not self._journal:
            return []

        recent = self._journal[:recent_n]
        weighted_tags: Counter = Counter()

        # Recent dreams get strong recency weighting
        for e in recent:
            w = self._recency_weight(e.get("timestamp", ""))
            for tag in e.get("tags", []):
                weighted_tags[tag.lower()] += w * 2.0

        # Add background frequency (lower weight)
        for tag, count in self._base_rates.get("tag_counts", {}).items():
            weighted_tags[tag] += count * 0.3

        if not weighted_tags:
            return []

        # Return top themes, sorted by weighted score
        top = weighted_tags.most_common(8)
        return [tag for tag, _ in top]

    # -- character prediction -----------------------------------------------

    def _character_prediction(self, recent_n: int = 10) -> list[str]:
        """Predict which people are likely to appear tonight."""
        if not self._journal:
            return []

        recent = self._journal[:recent_n]
        weighted_people: Counter = Counter()

        for e in recent:
            w = self._recency_weight(e.get("timestamp", ""))
            for person in e.get("people", []):
                weighted_people[person.lower()] += w * 2.0

        # Background frequency
        for person, count in self._base_rates.get("people_counts", {}).items():
            weighted_people[person] += count * 0.3

        if not weighted_people:
            return []

        top = weighted_people.most_common(5)
        return [person for person, _ in top]

    # -- nightmare risk model -----------------------------------------------

    def _nightmare_risk(self, stress: int, caffeine: bool, sleep_quality: int) -> float:
        """Calculate nightmare risk from 0 to 1.

        Factors: stress level, caffeine intake, poor sleep quality,
        historical nightmare rate.
        """
        risk = self._base_rates.get("nightmare_rate", 0.1)

        # Stress is a strong nightmare predictor (0-10 scale)
        stress_factor = max(0, min(10, stress)) / 10.0
        risk += stress_factor * 0.35

        # Caffeine disrupts sleep architecture, increasing nightmare risk
        if caffeine:
            risk += 0.1

        # Poor sleep quality correlates with nightmares
        sq = max(1, min(10, sleep_quality)) if sleep_quality > 0 else 5
        if sq <= 3:
            risk += 0.15
        elif sq <= 5:
            risk += 0.05

        # Recent nightmares increase short-term risk (momentum)
        recent = self._journal[:5]
        recent_nightmare_count = sum(
            1 for e in recent
            if e.get("mood", "").lower() in _NIGHTMARE_MOODS
        )
        if recent_nightmare_count > 0:
            risk += 0.05 * recent_nightmare_count

        return max(0.0, min(1.0, risk))

    # -- day-of-week patterns -----------------------------------------------

    def _day_of_week_adjustment(self) -> dict:
        """Check if dreams differ by day of week (weekend vs weekday)."""
        if len(self._journal) < 7:
            return {"vividness_adj": 0.0, "lucidity_adj": 0.0}

        weekday_vivid = []
        weekend_vivid = []
        weekday_lucid = []
        weekend_lucid = []

        for e in self._journal:
            try:
                dt = datetime.fromisoformat(e["timestamp"])
                v = e.get("vividness", 5)
                l = e.get("lucidity", 0)
                if dt.weekday() < 5:  # Mon-Fri
                    weekday_vivid.append(v)
                    weekday_lucid.append(l)
                else:
                    weekend_vivid.append(v)
                    weekend_lucid.append(l)
            except (ValueError, KeyError):
                continue

        now = datetime.now(timezone.utc)
        is_weekend = now.weekday() >= 5

        vividness_adj = 0.0
        lucidity_adj = 0.0

        if is_weekend and weekend_vivid and weekday_vivid:
            avg_we = sum(weekend_vivid) / len(weekend_vivid)
            avg_wd = sum(weekday_vivid) / len(weekday_vivid)
            vividness_adj = avg_we - avg_wd
        elif not is_weekend and weekday_vivid and weekend_vivid:
            avg_wd = sum(weekday_vivid) / len(weekday_vivid)
            avg_we = sum(weekend_vivid) / len(weekend_vivid)
            vividness_adj = avg_wd - avg_we

        if is_weekend and weekend_lucid and weekday_lucid:
            avg_we = sum(weekend_lucid) / len(weekend_lucid)
            avg_wd = sum(weekday_lucid) / len(weekday_lucid)
            lucidity_adj = (avg_we - avg_wd) / 5.0  # normalize to ~0-1 range
        elif not is_weekend and weekday_lucid and weekend_lucid:
            avg_wd = sum(weekday_lucid) / len(weekday_lucid)
            avg_we = sum(weekend_lucid) / len(weekend_lucid)
            lucidity_adj = (avg_wd - avg_we) / 5.0

        return {"vividness_adj": vividness_adj, "lucidity_adj": lucidity_adj}

    # -- vividness prediction -----------------------------------------------

    def _predict_vividness(self, sleep_quality: int, exercise: bool,
                           screen_time_hours: float) -> float:
        """Predict dream vividness based on inputs and journal correlation."""
        base = self._base_rates["avg_vividness"]

        # Sleep quality correlation from journal data
        if sleep_quality > 0:
            vs_data = analyzer.vividness_vs_sleep()
            if len(vs_data) >= 3:
                # Simple linear relationship: compute correlation
                sq_vals = [d["sleep_quality"] for d in vs_data]
                vv_vals = [d["vividness"] for d in vs_data]
                sq_mean = sum(sq_vals) / len(sq_vals)
                vv_mean = sum(vv_vals) / len(vv_vals)

                cov = sum((s - sq_mean) * (v - vv_mean)
                          for s, v in zip(sq_vals, vv_vals))
                var_sq = sum((s - sq_mean) ** 2 for s in sq_vals)

                if var_sq > 0:
                    slope = cov / var_sq
                    predicted = vv_mean + slope * (sleep_quality - sq_mean)
                    # Blend journal-derived prediction with base
                    base = 0.6 * predicted + 0.4 * base
                else:
                    # No variance in sleep quality, use difference from mean
                    base += (sleep_quality - sq_mean) * 0.3
            else:
                # Few data points — simple heuristic
                base += (sleep_quality - 5) * 0.3

        # Exercise tends to improve sleep quality and dream vividness
        if exercise:
            base += 0.5

        # High screen time before bed can reduce sleep quality
        if screen_time_hours > 2:
            base -= (screen_time_hours - 2) * 0.2

        # Day-of-week adjustment
        dow = self._day_of_week_adjustment()
        base += dow["vividness_adj"] * 0.5  # dampen the effect

        return max(1.0, min(10.0, round(base, 1)))

    # -- lucidity chance prediction -----------------------------------------

    def _predict_lucidity_chance(self) -> float:
        """Predict chance of lucidity based on recent momentum and base rate."""
        base = self._base_rates["lucid_rate"]

        # Lucidity momentum — recent lucid dreams increase the chance
        recent = self._journal[:7]
        if recent:
            recent_lucid = sum(
                1 for e in recent
                if e.get("lucidity", 0) > 0
            )
            momentum = recent_lucid / len(recent)
            # Blend base rate with recent momentum
            base = 0.5 * base + 0.5 * momentum

        # Day-of-week adjustment
        dow = self._day_of_week_adjustment()
        base += dow["lucidity_adj"] * 0.3

        return max(0.0, min(1.0, round(base, 3)))

    # -- recurring dream chance ---------------------------------------------

    def _predict_recurring_chance(self) -> float:
        """Predict chance of a recurring dream."""
        base = self._base_rates["recurring_rate"]

        # If recent dreams have been recurring, bump the chance
        recent = self._journal[:5]
        if recent:
            recent_recurring = sum(1 for e in recent if e.get("recurring"))
            momentum = recent_recurring / len(recent)
            base = 0.5 * base + 0.5 * momentum

        return max(0.0, min(1.0, round(base, 3)))

    # -- tip generation -----------------------------------------------------

    def _generate_tips(self, forecast: DreamForecast, sleep_quality: int,
                       stress: int, caffeine: bool, exercise: bool,
                       screen_time_hours: float) -> list[str]:
        """Generate personalized tips based on the forecast."""
        tips = []

        if forecast.nightmare_risk > 0.5:
            tips.append("High nightmare risk tonight. Try a calming pre-sleep "
                        "routine: warm drink, no screens, deep breathing.")
        elif forecast.nightmare_risk > 0.3:
            tips.append("Moderate nightmare risk. Consider journaling your "
                        "worries before bed to process them consciously.")

        if caffeine:
            tips.append("Caffeine can fragment sleep and increase nightmare "
                        "frequency. Next time try cutting off caffeine by 2pm.")

        if screen_time_hours > 3:
            tips.append("Heavy screen time before bed suppresses melatonin. "
                        "Try reading a book for the last hour instead.")
        elif screen_time_hours > 1.5:
            tips.append("Consider reducing screen time in the last hour before "
                        "bed — it can improve dream recall and vividness.")

        if not exercise:
            tips.append("Exercise during the day improves sleep quality and "
                        "dream vividness. Even a 20-minute walk helps.")

        if sleep_quality > 0 and sleep_quality <= 4:
            tips.append("Your sleep quality estimate is low. Try keeping your "
                        "bedroom cool, dark, and quiet.")

        if stress > 6:
            tips.append("High stress detected. Progressive muscle relaxation "
                        "before bed can reduce stress-related dreams.")

        if forecast.predicted_lucidity_chance > 0.3:
            tips.append("Good lucidity conditions tonight! Do a reality check "
                        "now and set an intention to recognize you're dreaming.")
        elif forecast.predicted_lucidity_chance < 0.1:
            tips.append("Low lucidity chance tonight. To build the habit, do "
                        "reality checks throughout the day (count your fingers, "
                        "check a clock twice).")

        if forecast.predicted_vividness >= 7:
            tips.append("High vividness predicted. Keep your dream journal "
                        "beside the bed — you'll want to capture the details.")

        if forecast.recurring_dream_chance > 0.3:
            tips.append("Recurring dream likely. Before sleep, visualize how "
                        "you'd change the dream if you could control it.")

        # Always include at least one tip
        if not tips:
            tips.append("Set an intention before falling asleep: tell yourself "
                        "what you'd like to dream about tonight.")

        return tips

    # -- forecast text generation -------------------------------------------

    def _generate_forecast_text(self, forecast: DreamForecast) -> str:
        """Generate a human-readable forecast summary.

        Tries LLM first, falls back to template-based.
        """
        # Try LLM-generated text
        try:
            from grimoire.ai_backend import bm_call

            system = (
                "You are a dream weather forecaster. Given tonight's dream "
                "predictions, write a short, evocative weather-style forecast "
                "(3-5 sentences). Use weather metaphors — clear skies for peaceful "
                "dreams, storms for nightmares, fog for confusion, auroras for "
                "lucid dreams. Be poetic but concise."
            )
            context = (
                f"Predicted vividness: {forecast.predicted_vividness}/10\n"
                f"Lucidity chance: {forecast.predicted_lucidity_chance:.0%}\n"
                f"Predicted mood: {forecast.predicted_mood}\n"
                f"Nightmare risk: {forecast.nightmare_risk:.0%}\n"
                f"Recurring chance: {forecast.recurring_dream_chance:.0%}\n"
                f"Likely themes: {', '.join(forecast.predicted_themes) or 'unknown'}\n"
                f"Likely characters: {', '.join(forecast.predicted_characters) or 'none'}\n"
                f"Confidence: {forecast.confidence:.0%}"
            )
            messages = [{"role": "user", "content": context}]
            result = bm_call(messages, max_tokens=200, temperature=0.8, system=system)
            if result and "[BlackMagic Offline]" not in result:
                return result.strip()
        except (ImportError, Exception):
            pass

        # Template-based fallback
        return self._template_forecast_text(forecast)

    def _template_forecast_text(self, forecast: DreamForecast) -> str:
        """Generate forecast text from templates."""
        parts = []

        # Vividness
        v = forecast.predicted_vividness
        if v >= 8:
            parts.append("Expect exceptionally vivid dreams tonight — "
                         "colours will be saturated and details sharp.")
        elif v >= 6:
            parts.append("Moderate to high vividness expected — "
                         "dreams should be clear and memorable.")
        elif v >= 4:
            parts.append("Average vividness tonight — dreams may be "
                         "somewhat hazy around the edges.")
        else:
            parts.append("Low vividness forecast — dreams may be faint "
                         "and hard to recall in the morning.")

        # Mood
        mood = forecast.predicted_mood
        if mood in _POSITIVE_MOODS:
            parts.append(f"The emotional climate looks {mood} — "
                         f"clear skies in the dreamscape.")
        elif mood in _NIGHTMARE_MOODS:
            parts.append(f"Storm warning: the emotional tone leans {mood}. "
                         f"Rough seas possible.")
        else:
            parts.append(f"Emotional weather: {mood}.")

        # Nightmare risk
        nr = forecast.nightmare_risk
        if nr > 0.6:
            parts.append("Nightmare advisory in effect. "
                         "Consider a calming pre-sleep routine.")
        elif nr > 0.3:
            parts.append("Slight chance of unsettling dreams.")

        # Lucidity
        lc = forecast.predicted_lucidity_chance
        if lc > 0.4:
            parts.append("Lucidity conditions are favourable — "
                         "auroras may appear on the dream horizon.")
        elif lc > 0.15:
            parts.append("Faint chance of lucidity. Reality checks recommended.")

        # Themes
        if forecast.predicted_themes:
            themes_str = ", ".join(forecast.predicted_themes[:4])
            parts.append(f"Themes on the radar: {themes_str}.")

        # Characters
        if forecast.predicted_characters:
            chars_str = ", ".join(forecast.predicted_characters[:3])
            parts.append(f"Likely visitors: {chars_str}.")

        # Confidence
        conf = forecast.confidence
        if conf < 0.3:
            parts.append("(Low confidence — limited journal data available.)")

        return " ".join(parts)

    # -- confidence calculation ---------------------------------------------

    def _calculate_confidence(self) -> float:
        """Confidence scales with journal size and recency.

        0 entries -> 0.1 (we still give a forecast, just low confidence)
        5 entries -> ~0.35
        20 entries -> ~0.6
        50+ entries -> ~0.8+
        """
        n = self._base_rates["total"]
        if n == 0:
            return 0.1

        # Logarithmic scaling — diminishing returns after ~50 entries
        base_conf = min(0.85, 0.1 + 0.3 * math.log(1 + n))

        # Boost if recent entries exist (journal is active)
        if self._journal:
            most_recent_weight = self._recency_weight(
                self._journal[0].get("timestamp", "")
            )
            # Active journal (recent entry) boosts confidence
            base_conf = base_conf * (0.7 + 0.3 * most_recent_weight)

        return max(0.1, min(0.95, round(base_conf, 2)))

    # == main forecast method ===============================================

    def forecast(
        self,
        mood: str = "",
        sleep_quality: int = 0,
        stress_level: int = 0,
        caffeine: bool = False,
        exercise: bool = False,
        screen_time_hours: float = 0.0,
    ) -> DreamForecast:
        """Generate tonight's dream forecast.

        Args:
            mood: Current waking mood (e.g. "anxious", "happy", "calm")
            sleep_quality: Expected sleep quality 1-10 (0 = unknown)
            stress_level: Current stress level 0-10
            caffeine: Whether caffeine was consumed today
            exercise: Whether exercise was done today
            screen_time_hours: Hours of screen time in the evening
        """
        predicted_mood = self._mood_correlation(mood)
        predicted_vividness = self._predict_vividness(
            sleep_quality, exercise, screen_time_hours
        )
        predicted_lucidity = self._predict_lucidity_chance()
        predicted_themes = self._theme_prediction(recent_n=10)
        predicted_characters = self._character_prediction(recent_n=10)
        nm_risk = self._nightmare_risk(stress_level, caffeine, sleep_quality)
        recurring_chance = self._predict_recurring_chance()
        confidence = self._calculate_confidence()

        # Stress also suppresses lucidity
        if stress_level > 6:
            predicted_lucidity *= 0.7

        # Exercise boosts lucidity slightly
        if exercise:
            predicted_lucidity = min(1.0, predicted_lucidity + 0.05)

        predicted_lucidity = round(predicted_lucidity, 3)

        fc = DreamForecast(
            predicted_vividness=predicted_vividness,
            predicted_lucidity_chance=predicted_lucidity,
            predicted_mood=predicted_mood,
            predicted_themes=predicted_themes,
            predicted_characters=predicted_characters,
            nightmare_risk=round(nm_risk, 2),
            recurring_dream_chance=round(recurring_chance, 3),
            confidence=confidence,
        )

        fc.tips = self._generate_tips(
            fc, sleep_quality, stress_level, caffeine, exercise, screen_time_hours
        )
        fc.forecast_text = self._generate_forecast_text(fc)

        return fc

    # == weekly outlook =====================================================

    def weekly_outlook(self) -> list[DreamForecast]:
        """Generate rough forecasts for the next 7 nights.

        Without knowing future waking state, this uses base rates
        and day-of-week patterns with slight random variation seeded
        by the day offset.
        """
        forecasts = []
        now = datetime.now(timezone.utc)

        for day_offset in range(7):
            future = now + timedelta(days=day_offset)
            is_weekend = future.weekday() >= 5

            # Slight adjustments per day based on patterns
            sq_estimate = round(self._base_rates["avg_sleep_quality"])
            if is_weekend:
                sq_estimate = min(10, sq_estimate + 1)  # weekends often better

            # Use a simple seed-based variation to make days feel different
            variation = math.sin(day_offset * 2.17) * 0.5  # deterministic jitter

            fc = self.forecast(
                mood="",
                sleep_quality=sq_estimate,
                stress_level=max(0, min(10, 3 + int(variation * 3))),
                caffeine=False,
                exercise=day_offset % 2 == 0,  # assume alternating
                screen_time_hours=1.5 + variation,
            )

            # Adjust confidence down for future predictions
            distance_penalty = 1.0 - (day_offset * 0.08)
            fc.confidence = round(max(0.05, fc.confidence * distance_penalty), 2)

            # Prepend the day name to the forecast text
            day_name = future.strftime("%A")
            fc.forecast_text = f"[{day_name}] {fc.forecast_text}"

            forecasts.append(fc)

        return forecasts

    # == accuracy check =====================================================

    def accuracy_check(self, dream_id: str) -> dict:
        """Compare a recorded dream against what would have been forecast.

        Returns a dict with accuracy metrics for calibration.
        """
        entry = recorder.get_by_id(dream_id)
        if not entry:
            return {"error": f"Dream '{dream_id}' not found."}

        # Generate a forecast using what we'd have known before that dream
        # (We use current journal, which includes the dream — not perfect,
        # but good enough for calibration.)
        actual_mood = entry.get("mood", "").lower()
        actual_vividness = entry.get("vividness", 5)
        actual_lucidity = entry.get("lucidity", 0)
        actual_tags = {t.lower() for t in entry.get("tags", [])}
        actual_people = {p.lower() for p in entry.get("people", [])}
        actual_recurring = entry.get("recurring", False)
        actual_sq = entry.get("sleep_quality", 5)

        # Get a forecast with average inputs
        fc = self.forecast(sleep_quality=actual_sq)

        # Calculate accuracy scores
        vividness_error = abs(fc.predicted_vividness - actual_vividness)
        vividness_accuracy = max(0.0, 1.0 - vividness_error / 9.0)

        # Mood match — exact or semantic similarity
        mood_match = 1.0 if fc.predicted_mood == actual_mood else 0.0
        if not mood_match:
            # Partial credit: both positive, both negative, etc.
            pred_positive = fc.predicted_mood in _POSITIVE_MOODS
            actual_positive = actual_mood in _POSITIVE_MOODS
            pred_negative = fc.predicted_mood in _NIGHTMARE_MOODS
            actual_negative = actual_mood in _NIGHTMARE_MOODS
            if pred_positive == actual_positive and pred_negative == actual_negative:
                mood_match = 0.5

        # Theme overlap
        predicted_tags = {t.lower() for t in fc.predicted_themes}
        if predicted_tags and actual_tags:
            theme_overlap = len(predicted_tags & actual_tags) / max(
                len(actual_tags), 1
            )
        elif not actual_tags and not predicted_tags:
            theme_overlap = 1.0
        else:
            theme_overlap = 0.0

        # Character overlap
        predicted_people = {p.lower() for p in fc.predicted_characters}
        if predicted_people and actual_people:
            people_overlap = len(predicted_people & actual_people) / max(
                len(actual_people), 1
            )
        elif not actual_people and not predicted_people:
            people_overlap = 1.0
        else:
            people_overlap = 0.0

        # Lucidity prediction accuracy
        was_lucid = actual_lucidity > 0
        lucidity_accuracy = (
            fc.predicted_lucidity_chance if was_lucid
            else 1.0 - fc.predicted_lucidity_chance
        )

        # Recurring prediction accuracy
        recurring_accuracy = (
            fc.recurring_dream_chance if actual_recurring
            else 1.0 - fc.recurring_dream_chance
        )

        # Nightmare prediction
        was_nightmare = actual_mood in _NIGHTMARE_MOODS
        nightmare_accuracy = (
            fc.nightmare_risk if was_nightmare
            else 1.0 - fc.nightmare_risk
        )

        overall = (
            vividness_accuracy * 0.25
            + mood_match * 0.2
            + theme_overlap * 0.2
            + people_overlap * 0.1
            + lucidity_accuracy * 0.1
            + recurring_accuracy * 0.05
            + nightmare_accuracy * 0.1
        )

        return {
            "dream_id": dream_id,
            "dream_title": entry.get("title", ""),
            "overall_accuracy": round(overall, 2),
            "vividness_accuracy": round(vividness_accuracy, 2),
            "mood_match": round(mood_match, 2),
            "theme_overlap": round(theme_overlap, 2),
            "people_overlap": round(people_overlap, 2),
            "lucidity_accuracy": round(lucidity_accuracy, 2),
            "recurring_accuracy": round(recurring_accuracy, 2),
            "nightmare_accuracy": round(nightmare_accuracy, 2),
            "predicted": {
                "vividness": fc.predicted_vividness,
                "mood": fc.predicted_mood,
                "themes": fc.predicted_themes[:5],
                "characters": fc.predicted_characters[:3],
                "lucidity_chance": fc.predicted_lucidity_chance,
                "nightmare_risk": fc.nightmare_risk,
            },
            "actual": {
                "vividness": actual_vividness,
                "mood": actual_mood,
                "tags": list(actual_tags),
                "people": list(actual_people),
                "lucidity": actual_lucidity,
                "recurring": actual_recurring,
            },
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _input(prompt: str, default: str = "") -> str:
    try:
        val = input(prompt).strip()
        return val if val else default
    except (EOFError, KeyboardInterrupt):
        print()
        return default


def cmd_weather():
    """Interactive dream weather forecast."""
    print("\n=== Dream Weather Forecast ===\n")
    print("Answer a few questions about your day to predict tonight's dreams.\n")

    mood = _input("Current mood (e.g. happy, anxious, calm, tired): ", "neutral")

    sq_raw = _input("Expected sleep quality [1-10, default 5]: ", "5")
    try:
        sleep_quality = max(1, min(10, int(sq_raw)))
    except ValueError:
        sleep_quality = 5

    stress_raw = _input("Stress level [0-10, default 3]: ", "3")
    try:
        stress = max(0, min(10, int(stress_raw)))
    except ValueError:
        stress = 3

    caffeine = _input("Had caffeine today? [y/N]: ", "n").lower().startswith("y")
    exercise = _input("Exercised today? [y/N]: ", "n").lower().startswith("y")

    screen_raw = _input("Evening screen time (hours) [default 2]: ", "2")
    try:
        screen_time = max(0.0, float(screen_raw))
    except ValueError:
        screen_time = 2.0

    station = DreamWeatherStation()
    fc = station.forecast(
        mood=mood,
        sleep_quality=sleep_quality,
        stress_level=stress,
        caffeine=caffeine,
        exercise=exercise,
        screen_time_hours=screen_time,
    )

    print(f"\n{'='*55}")
    print(f"  TONIGHT'S DREAM WEATHER")
    print(f"{'='*55}\n")

    print(f"  Vividness:        {fc.predicted_vividness}/10")
    print(f"  Lucidity chance:  {fc.predicted_lucidity_chance:.0%}")
    print(f"  Mood:             {fc.predicted_mood}")
    print(f"  Nightmare risk:   {fc.nightmare_risk:.0%}")
    print(f"  Recurring chance: {fc.recurring_dream_chance:.0%}")
    print(f"  Confidence:       {fc.confidence:.0%}")

    if fc.predicted_themes:
        print(f"\n  Likely themes:    {', '.join(fc.predicted_themes[:5])}")
    if fc.predicted_characters:
        print(f"  Likely visitors:  {', '.join(fc.predicted_characters[:3])}")

    print(f"\n{'~'*55}")
    print(f"  {fc.forecast_text}")
    print(f"{'~'*55}")

    if fc.tips:
        print(f"\n  Tips for tonight:")
        for tip in fc.tips:
            print(f"    - {tip}")

    print()

    # Offer weekly outlook
    weekly = _input("See weekly outlook? [y/N]: ", "n")
    if weekly.lower().startswith("y"):
        print(f"\n{'='*55}")
        print(f"  7-DAY DREAM OUTLOOK")
        print(f"{'='*55}\n")
        for day_fc in station.weekly_outlook():
            print(f"  {day_fc.forecast_text}")
            print(f"    Vividness: {day_fc.predicted_vividness}/10 | "
                  f"Lucidity: {day_fc.predicted_lucidity_chance:.0%} | "
                  f"Nightmare: {day_fc.nightmare_risk:.0%} | "
                  f"Confidence: {day_fc.confidence:.0%}")
            print()


if __name__ == "__main__":
    cmd_weather()
