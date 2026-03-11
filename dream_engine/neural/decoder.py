"""dream_engine/neural/decoder.py — Dream content decoding from EEG.

Decodes neural signals during REM into interpretable dream content:

1. Emotional valence — frontal asymmetry (left > right = positive, right > left = negative)
2. Arousal level — overall beta/gamma power
3. Semantic categories — trained classifier maps EEG patterns to content categories
4. Movement intention — motor cortex activity during REM (despite atonia)
5. Narrative reconstruction — LLM generates dream descriptions from decoded features

The emotional and arousal decoders work with basic 4-channel EEG (Muse 2).
Semantic and movement decoding need 8+ channels (OpenBCI Cyton).

References:
- Davidson (2004) — Frontal EEG asymmetry and emotion
- Kamitani & Tong (2005) — Decoding visual content from brain activity
- Horikawa et al. (2013) — Neural decoding of dream content
- Siclari et al. (2017) — Neural correlates of dreaming
"""
from __future__ import annotations

import json
import time
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from .bci import BCIStream


class Emotion(Enum):
    VERY_NEGATIVE = "very_negative"    # terror, anguish
    NEGATIVE = "negative"              # anxiety, sadness
    NEUTRAL = "neutral"
    POSITIVE = "positive"              # contentment, joy
    VERY_POSITIVE = "very_positive"    # euphoria, bliss


class Arousal(Enum):
    VERY_LOW = "very_low"       # deep calm, near-sleep
    LOW = "low"                 # relaxed, peaceful
    MODERATE = "moderate"       # engaged, normal
    HIGH = "high"               # excited, tense
    VERY_HIGH = "very_high"     # fight-or-flight, nightmare


class SemanticCategory(Enum):
    """Broad content categories decodable from EEG patterns."""
    FACES = "faces"             # fusiform face area activation
    PLACES = "places"           # parahippocampal place area
    OBJECTS = "objects"         # lateral occipital cortex
    MOVEMENT = "movement"       # motor cortex
    LANGUAGE = "language"       # left temporal / Broca's area
    SOCIAL = "social"           # mirror neuron / empathy circuits
    THREAT = "threat"           # amygdala-driven patterns
    UNKNOWN = "unknown"


@dataclass
class DreamFrame:
    """A single decoded snapshot of dream content."""
    timestamp: float
    emotion: Emotion
    arousal: Arousal
    valence_score: float         # -1 (negative) to +1 (positive)
    arousal_score: float         # 0 (calm) to 1 (intense)
    semantic_categories: list[SemanticCategory]
    semantic_confidence: dict[str, float]   # category → confidence
    movement_detected: bool
    movement_type: str           # "still", "walking", "running", "flying", "falling"
    frontal_asymmetry: float     # raw left-right alpha difference
    gamma_coherence: float       # inter-hemispheric gamma sync (consciousness marker)


@dataclass
class DecoderCalibration:
    """Calibration data from waking baseline recordings."""
    baseline_frontal_asymmetry: float = 0.0
    baseline_alpha_power: float = 0.0
    baseline_beta_power: float = 0.0
    category_templates: dict = field(default_factory=dict)
    calibrated: bool = False


class EmotionalDecoder:
    """Decode emotional valence and arousal from EEG.

    Works with 4+ channels. Uses:
    - Frontal alpha asymmetry for valence (Davidson model)
    - Beta/gamma power for arousal
    - Theta coherence for emotional depth
    """

    def __init__(self, stream: BCIStream):
        self.stream = stream
        self.calibration = DecoderCalibration()

    def calibrate_baseline(self, seconds: float = 30.0):
        """Record neutral emotional baseline (eyes closed, relaxed)."""
        print("[Decoder] Calibrating emotional baseline...")
        time.sleep(seconds)

        bands = self.stream.get_bandpower(seconds=min(seconds, 10.0))

        # Frontal asymmetry baseline
        alpha = bands.get("alpha", np.zeros(2))
        if len(alpha) >= 2:
            self.calibration.baseline_frontal_asymmetry = float(
                np.log(alpha[1] + 1e-10) - np.log(alpha[0] + 1e-10)
            )
        self.calibration.baseline_alpha_power = float(np.mean(alpha))
        self.calibration.baseline_beta_power = float(np.mean(bands.get("beta", [0])))
        self.calibration.calibrated = True
        print("[Decoder] Baseline calibrated")

    def decode_emotion(self) -> tuple[Emotion, float]:
        """Decode current emotional valence.

        Returns (Emotion enum, valence score from -1 to +1).
        """
        bands = self.stream.get_bandpower(seconds=2.0)
        alpha = bands.get("alpha", np.zeros(2))

        if len(alpha) < 2:
            return Emotion.NEUTRAL, 0.0

        # Frontal alpha asymmetry
        # More left alpha = more right frontal activation = negative emotion
        # More right alpha = more left frontal activation = positive emotion
        asymmetry = float(np.log(alpha[1] + 1e-10) - np.log(alpha[0] + 1e-10))
        relative = asymmetry - self.calibration.baseline_frontal_asymmetry

        # Map to -1 to +1 scale
        valence = np.tanh(relative * 2)  # sigmoid-ish scaling

        if valence < -0.6:
            emotion = Emotion.VERY_NEGATIVE
        elif valence < -0.2:
            emotion = Emotion.NEGATIVE
        elif valence < 0.2:
            emotion = Emotion.NEUTRAL
        elif valence < 0.6:
            emotion = Emotion.POSITIVE
        else:
            emotion = Emotion.VERY_POSITIVE

        return emotion, round(float(valence), 3)

    def decode_arousal(self) -> tuple[Arousal, float]:
        """Decode current arousal level from beta/gamma power."""
        bands = self.stream.get_bandpower(seconds=2.0)
        beta = float(np.mean(bands.get("beta", [0])))
        gamma = float(np.mean(bands.get("gamma", [0])))

        # Arousal correlates with beta + gamma power
        baseline_beta = self.calibration.baseline_beta_power or 1.0
        arousal_raw = (beta + gamma * 2) / (baseline_beta + 1e-10)

        # Normalize to 0-1
        arousal = float(np.tanh(arousal_raw * 0.5))

        if arousal < 0.15:
            level = Arousal.VERY_LOW
        elif arousal < 0.35:
            level = Arousal.LOW
        elif arousal < 0.55:
            level = Arousal.MODERATE
        elif arousal < 0.75:
            level = Arousal.HIGH
        else:
            level = Arousal.VERY_HIGH

        return level, round(arousal, 3)


class SemanticDecoder:
    """Decode dream content categories from EEG patterns.

    Requires calibration phase where user views images from each category
    while EEG is recorded. The decoder learns category-specific patterns
    and applies them during REM to estimate dream content.

    Needs 8+ channels for spatial resolution.
    """

    CALIBRATION_DIR = Path.home() / ".local/share/blackmagic-gen/dreams/calibration"

    def __init__(self, stream: BCIStream):
        self.stream = stream
        self._templates: dict[str, np.ndarray] = {}
        self._calibrated = False

    def calibrate_category(self, category: SemanticCategory, seconds: float = 10.0):
        """Record EEG template while user views category-specific images.

        Call this for each category during a waking calibration session.
        E.g., show faces for 10s, record pattern. Show places for 10s, record.
        """
        print(f"[SemanticDecoder] Recording template for: {category.value}")
        print(f"  Look at {category.value} images for {seconds}s...")
        time.sleep(seconds)

        bands = self.stream.get_bandpower(seconds=min(seconds, 5.0))
        # Create feature vector from all bands
        features = np.concatenate([
            bands.get("delta", []),
            bands.get("theta", []),
            bands.get("alpha", []),
            bands.get("beta", []),
            bands.get("gamma", []),
        ])
        self._templates[category.value] = features
        print(f"  Template recorded ({len(features)} features)")

    def save_calibration(self):
        """Save calibration templates to disk."""
        self.CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
        data = {k: v.tolist() for k, v in self._templates.items()}
        (self.CALIBRATION_DIR / "semantic_templates.json").write_text(
            json.dumps(data, indent=2)
        )
        print(f"[SemanticDecoder] Calibration saved")

    def load_calibration(self) -> bool:
        """Load calibration templates from disk."""
        path = self.CALIBRATION_DIR / "semantic_templates.json"
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text())
            self._templates = {k: np.array(v) for k, v in data.items()}
            self._calibrated = True
            print(f"[SemanticDecoder] Loaded {len(self._templates)} templates")
            return True
        except Exception:
            return False

    def decode(self) -> list[tuple[SemanticCategory, float]]:
        """Decode current EEG into semantic categories with confidence.

        Returns list of (category, confidence) sorted by confidence.
        """
        if not self._templates:
            return [(SemanticCategory.UNKNOWN, 1.0)]

        bands = self.stream.get_bandpower(seconds=2.0)
        current = np.concatenate([
            bands.get("delta", []),
            bands.get("theta", []),
            bands.get("alpha", []),
            bands.get("beta", []),
            bands.get("gamma", []),
        ])

        # Compare with templates using cosine similarity
        results = []
        for cat_name, template in self._templates.items():
            if len(current) != len(template):
                continue
            sim = float(np.dot(current, template) / (
                np.linalg.norm(current) * np.linalg.norm(template) + 1e-10
            ))
            try:
                cat = SemanticCategory(cat_name)
            except ValueError:
                cat = SemanticCategory.UNKNOWN
            results.append((cat, round(max(0, sim), 3)))

        results.sort(key=lambda x: -x[1])
        return results if results else [(SemanticCategory.UNKNOWN, 1.0)]


class MovementDecoder:
    """Decode movement intentions from motor cortex EEG during REM.

    During REM, the body is paralyzed (atonia) but the motor cortex
    still fires as if moving. C3/C4 electrodes (channels over motor cortex)
    show characteristic patterns for different movement types.

    Needs channels over C3/C4 positions (OpenBCI Cyton).
    """

    # Motor cortex channels (approximate positions in standard montage)
    MOTOR_LEFT = 2   # C3 position
    MOTOR_RIGHT = 3  # C4 position

    def __init__(self, stream: BCIStream):
        self.stream = stream
        self._baseline_mu: float = 0.0  # mu rhythm baseline (8-12Hz over motor)

    def calibrate(self, seconds: float = 10.0):
        """Record motor cortex baseline during rest."""
        time.sleep(seconds)
        bands = self.stream.get_bandpower(seconds=5.0)
        alpha = bands.get("alpha", np.zeros(4))
        if len(alpha) > max(self.MOTOR_LEFT, self.MOTOR_RIGHT):
            self._baseline_mu = float(
                (alpha[self.MOTOR_LEFT] + alpha[self.MOTOR_RIGHT]) / 2
            )

    def decode_movement(self) -> tuple[str, float]:
        """Decode movement type from motor cortex activity.

        Returns (movement_type, confidence).
        Movement types: still, walking, running, flying, falling, fighting
        """
        bands = self.stream.get_bandpower(seconds=2.0)
        beta = bands.get("beta", np.zeros(4))
        alpha = bands.get("alpha", np.zeros(4))

        if len(beta) <= max(self.MOTOR_LEFT, self.MOTOR_RIGHT):
            return "still", 0.0

        # Motor beta suppression = movement intention
        motor_beta = float(
            (beta[self.MOTOR_LEFT] + beta[self.MOTOR_RIGHT]) / 2
        )
        motor_alpha = float(
            (alpha[self.MOTOR_LEFT] + alpha[self.MOTOR_RIGHT]) / 2
        )

        # Mu suppression (alpha over motor cortex drops during movement)
        mu_suppression = 0.0
        if self._baseline_mu > 0:
            mu_suppression = 1.0 - (motor_alpha / (self._baseline_mu + 1e-10))
            mu_suppression = max(0, min(1, mu_suppression))

        # Beta rebound after movement = indicates complex movement
        beta_ratio = motor_beta / (self._baseline_mu + 1e-10)

        # Classify (rough heuristics — would be replaced by trained classifier)
        if mu_suppression < 0.2:
            return "still", round(1.0 - mu_suppression, 3)
        elif mu_suppression < 0.4:
            return "walking", round(mu_suppression, 3)
        elif mu_suppression < 0.6:
            if beta_ratio > 1.5:
                return "fighting", round(mu_suppression, 3)
            return "running", round(mu_suppression, 3)
        elif mu_suppression < 0.8:
            # Asymmetric suppression might indicate flying (unusual motor pattern)
            left_right_diff = abs(
                beta[self.MOTOR_LEFT] - beta[self.MOTOR_RIGHT]
            )
            if left_right_diff > motor_beta * 0.3:
                return "flying", round(mu_suppression, 3)
            return "running", round(mu_suppression, 3)
        else:
            # Extreme suppression + high arousal = falling
            return "falling", round(mu_suppression, 3)


class DreamNarrativeReconstructor:
    """Combine all decoded features into a dream narrative using LLM.

    Takes a sequence of DreamFrames (one per epoch during REM) and
    generates a natural language description of the dream.
    """

    def __init__(self):
        self._frames: list[DreamFrame] = []

    def add_frame(self, frame: DreamFrame):
        self._frames.append(frame)

    def clear(self):
        self._frames.clear()

    def reconstruct(self) -> str:
        """Generate a narrative from accumulated dream frames.

        Uses LLM if available, otherwise builds a structured summary.
        """
        if not self._frames:
            return "No dream data recorded."

        # Build structured summary
        emotions = [f.emotion.value for f in self._frames]
        arousals = [f.arousal_score for f in self._frames]
        movements = [f.movement_type for f in self._frames if f.movement_type != "still"]
        categories = []
        for f in self._frames:
            for cat in f.semantic_categories:
                if cat != SemanticCategory.UNKNOWN:
                    categories.append(cat.value)

        # Emotional arc
        avg_valence = np.mean([f.valence_score for f in self._frames])
        peak_arousal = max(arousals)
        dominant_emotion = max(set(emotions), key=emotions.count)

        # Content summary
        unique_categories = list(set(categories))
        unique_movements = list(set(movements))

        summary = {
            "duration_epochs": len(self._frames),
            "emotional_arc": {
                "dominant": dominant_emotion,
                "avg_valence": round(float(avg_valence), 2),
                "peak_arousal": round(peak_arousal, 2),
                "transitions": self._emotion_transitions(),
            },
            "content": unique_categories or ["unknown"],
            "movement": unique_movements or ["still"],
            "narrative_prompt": self._build_llm_prompt(),
        }

        # Try LLM reconstruction
        try:
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
            from grimoire.ai_backend import bm_call

            result = bm_call(
                [{"role": "user", "content": summary["narrative_prompt"]}],
                max_tokens=300,
                temperature=0.7,
                system=(
                    "You are reconstructing a dream from decoded brain signals. "
                    "Write a vivid second-person narrative. Be evocative but "
                    "acknowledge uncertainty — use phrases like 'it seems', "
                    "'possibly', 'there's a sense of'. 2-3 paragraphs."
                ),
            )
            if result and "[BlackMagic Offline]" not in result:
                return result
        except ImportError:
            pass

        # Fallback: structured summary
        return self._structured_narrative(summary)

    def _emotion_transitions(self) -> list[str]:
        """Find major emotional transitions in the dream."""
        transitions = []
        prev = None
        for f in self._frames:
            if prev and f.emotion != prev:
                transitions.append(f"{prev.value} -> {f.emotion.value}")
            prev = f.emotion
        return transitions[:5]

    def _build_llm_prompt(self) -> str:
        """Build a prompt for LLM narrative reconstruction."""
        frames_desc = []
        for i, f in enumerate(self._frames):
            cats = ", ".join(c.value for c in f.semantic_categories
                           if c != SemanticCategory.UNKNOWN) or "unclear"
            frames_desc.append(
                f"Epoch {i+1}: emotion={f.emotion.value}, "
                f"arousal={f.arousal_score:.1f}, "
                f"content={cats}, "
                f"movement={f.movement_type}"
            )

        return (
            "Reconstruct this dream from decoded brain signals:\n\n" +
            "\n".join(frames_desc) +
            "\n\nWrite what the dreamer likely experienced."
        )

    def _structured_narrative(self, summary: dict) -> str:
        """Generate a basic narrative without LLM."""
        emotion = summary["emotional_arc"]["dominant"]
        content = ", ".join(summary["content"])
        movement = ", ".join(summary["movement"])
        valence = summary["emotional_arc"]["avg_valence"]

        tone = "peaceful" if valence > 0.2 else "unsettling" if valence < -0.2 else "neutral"

        return (
            f"Dream reconstruction ({summary['duration_epochs']} epochs):\n"
            f"A {tone} dream with dominant emotion of {emotion}. "
            f"Content involved: {content}. "
            f"Movement detected: {movement}. "
            f"Peak arousal: {summary['emotional_arc']['peak_arousal']:.1f}."
        )
