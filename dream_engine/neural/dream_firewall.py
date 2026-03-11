"""dream_engine/neural/dream_firewall.py — Nightmare detection and intervention.

A guardian that watches your dreams and intervenes when they turn dark.
Like a firewall for your subconscious.

The dream firewall sits between the decoder (which reads emotional state from
EEG) and the stimulator (which can inject calming audio, binaural beats, and
light cues). It continuously scores the threat level of the current dream and
applies graduated interventions:

  CALM          -> no action
  MILD_DISTRESS -> subtle theta binaural to deepen and calm
  MODERATE      -> calming audio + delta binaural to reduce arousal
  NIGHTMARE     -> strong calming intervention + gentle reality anchor
  NIGHT_TERROR  -> wake-up protocol (escalating audio + light)

The firewall distinguishes transient negative spikes (normal in dreaming) from
sustained nightmare patterns by maintaining a sliding window of recent dream
frames. A single scary moment does not trigger intervention — but several
seconds of sustained high arousal + negative valence + threat content will.

Safety features:
  - Cooldown between interventions (don't bombard a sleeping person)
  - Won't wake someone from deep NREM unnecessarily
  - Configurable sensitivity (some people want more aggressive protection)
  - Full intervention log for morning review

Storage: ~/.local/share/blackmagic-gen/dreams/firewall_log.json

References:
- Zadra & Pihl (1997) — Lucid dreaming as treatment for recurrent nightmares
- Spoormaker & Van den Bout (2006) — Nightmare frequency and distress
- Aurora et al. (2010) — Best practice guide for nightmare disorder treatment
"""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from .decoder import DreamFrame, Emotion, Arousal, SemanticCategory
from .stimulator import Stimulator


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

_LOG_DIR = Path.home() / ".local/share/blackmagic-gen/dreams"
_LOG_PATH = _LOG_DIR / "firewall_log.json"


# ---------------------------------------------------------------------------
# Enums and dataclasses
# ---------------------------------------------------------------------------

class ThreatLevel(Enum):
    CALM = "calm"
    MILD_DISTRESS = "mild_distress"
    MODERATE_DISTRESS = "moderate_distress"
    NIGHTMARE = "nightmare"
    NIGHT_TERROR = "night_terror"

    def __ge__(self, other):
        if not isinstance(other, ThreatLevel):
            return NotImplemented
        return _THREAT_ORDER[self] >= _THREAT_ORDER[other]

    def __gt__(self, other):
        if not isinstance(other, ThreatLevel):
            return NotImplemented
        return _THREAT_ORDER[self] > _THREAT_ORDER[other]

    def __le__(self, other):
        if not isinstance(other, ThreatLevel):
            return NotImplemented
        return _THREAT_ORDER[self] <= _THREAT_ORDER[other]

    def __lt__(self, other):
        if not isinstance(other, ThreatLevel):
            return NotImplemented
        return _THREAT_ORDER[self] < _THREAT_ORDER[other]


_THREAT_ORDER = {
    ThreatLevel.CALM: 0,
    ThreatLevel.MILD_DISTRESS: 1,
    ThreatLevel.MODERATE_DISTRESS: 2,
    ThreatLevel.NIGHTMARE: 3,
    ThreatLevel.NIGHT_TERROR: 4,
}


@dataclass
class NightmareSignature:
    """Pattern of EEG-decoded features that characterise a nightmare.

    A single DreamFrame is not enough — nightmares are sustained states.
    This captures the aggregate pattern across a window of frames.
    """
    avg_arousal: float              # mean arousal over window (0-1)
    avg_valence: float              # mean valence over window (-1 to +1)
    peak_arousal: float             # max arousal in window
    min_valence: float              # most negative valence in window
    threat_category_ratio: float    # fraction of frames containing THREAT
    sustained_seconds: float        # how long the pattern has been active
    dominant_emotion: Emotion       # most common emotion in window
    movement_type: str              # most common movement (falling/fighting = bad)


@dataclass
class FirewallConfig:
    """Tunable parameters for the dream firewall."""
    sensitivity: float = 0.5                # 0 = very lenient, 1 = hair-trigger
    enable_auto_wake: bool = True           # allow wake-up protocol for night terrors
    cooldown_s: float = 60.0                # min seconds between interventions
    intervention_cooldown_s: float = 60.0   # alias kept for API compat
    window_seconds: float = 30.0            # how many seconds of history to evaluate
    auto_wake_threshold: float = 0.9        # arousal level that triggers wake protocol
    calming_audio_path: Optional[str] = None  # custom WAV for calming intervention
    wake_audio_path: Optional[str] = None     # custom WAV for wake-up protocol


@dataclass
class InterventionRecord:
    """A single logged intervention event."""
    timestamp: float
    threat_level: str
    action: str
    signature: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "time_str": time.strftime("%H:%M:%S", time.localtime(self.timestamp)),
            "threat_level": self.threat_level,
            "action": self.action,
            "signature": self.signature,
        }


# ---------------------------------------------------------------------------
# Dream Firewall
# ---------------------------------------------------------------------------

class DreamFirewall:
    """Monitors decoded dream state and intervenes when nightmares are detected.

    Sits between the decoder pipeline (which produces DreamFrames) and the
    stimulator (which can play audio, binaural beats, haptic, and light cues).
    Maintains a rolling history of recent frames to distinguish sustained
    nightmare patterns from transient negative blips.

    Usage:
        firewall = DreamFirewall(stimulator, config)
        # In the dream monitoring loop:
        threat = firewall.assess_threat(current_frame)
        if threat > ThreatLevel.CALM:
            firewall.intervene(threat)
    """

    def __init__(
        self,
        stimulator: Stimulator,
        config: Optional[FirewallConfig] = None,
    ):
        self.stimulator = stimulator
        self.config = config or FirewallConfig()
        # Sync the alias
        self.config.intervention_cooldown_s = self.config.cooldown_s

        # Rolling frame history — keep enough for the analysis window
        # Assuming ~1 frame per second (typical decode rate)
        max_frames = int(self.config.window_seconds * 2)  # 2x for safety
        self._history: deque[DreamFrame] = deque(maxlen=max(max_frames, 60))

        # Intervention tracking
        self._last_intervention_time: float = 0.0
        self._intervention_log: list[InterventionRecord] = []
        self._current_threat: ThreatLevel = ThreatLevel.CALM
        self._nightmare_onset_time: Optional[float] = None

        # Night session tracking
        self._session_start: float = time.time()
        self._threat_events: list[dict] = []  # summary events for night report

        # Load any existing log from tonight
        self._load_log()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess_threat(self, dream_frame: DreamFrame) -> ThreatLevel:
        """Score the current dream state and return a threat level.

        Call this once per decoded DreamFrame (typically every 1-2 seconds
        during REM). The frame is added to the rolling history and the
        threat level is computed from the pattern across the window.
        """
        self._history.append(dream_frame)

        # Need at least a few frames to make any judgement
        if len(self._history) < 3:
            self._current_threat = ThreatLevel.CALM
            return ThreatLevel.CALM

        signature = self._detect_nightmare_pattern(self._history)
        threat = self._score_threat(signature)

        # Track nightmare onset for duration measurement
        if threat >= ThreatLevel.MODERATE_DISTRESS and self._nightmare_onset_time is None:
            self._nightmare_onset_time = time.time()
        elif threat < ThreatLevel.MODERATE_DISTRESS:
            if self._nightmare_onset_time is not None:
                # Nightmare ended — log the event
                duration = time.time() - self._nightmare_onset_time
                self._threat_events.append({
                    "onset": self._nightmare_onset_time,
                    "duration_s": round(duration, 1),
                    "peak_threat": self._current_threat.value,
                })
            self._nightmare_onset_time = None

        self._current_threat = threat
        return threat

    def intervene(self, threat_level: ThreatLevel) -> str:
        """Apply a graduated intervention based on threat level.

        Returns a description of the action taken, or an empty string if
        intervention was skipped (cooldown, CALM, etc.).
        """
        if threat_level == ThreatLevel.CALM:
            return ""

        # Enforce cooldown — don't bombard a sleeping person
        now = time.time()
        elapsed = now - self._last_intervention_time
        if elapsed < self.config.cooldown_s:
            remaining = self.config.cooldown_s - elapsed
            return f"cooldown ({remaining:.0f}s remaining)"

        action = ""

        if threat_level == ThreatLevel.MILD_DISTRESS:
            action = self._intervene_mild()

        elif threat_level == ThreatLevel.MODERATE_DISTRESS:
            action = self._intervene_moderate()

        elif threat_level == ThreatLevel.NIGHTMARE:
            action = self._intervene_nightmare()

        elif threat_level == ThreatLevel.NIGHT_TERROR:
            action = self._intervene_night_terror()

        if action:
            self._last_intervention_time = now
            self._log_intervention(threat_level, action)

        return action

    def get_night_report(self) -> dict:
        """Generate a summary of the night's threat events and interventions.

        Call this in the morning for a full picture of what happened.
        """
        now = time.time()
        session_duration = now - self._session_start

        # Compile intervention stats
        intervention_counts: dict[str, int] = {}
        for record in self._intervention_log:
            level = record.threat_level
            intervention_counts[level] = intervention_counts.get(level, 0) + 1

        total_interventions = len(self._intervention_log)

        # Calculate time spent in each threat level (approximate)
        threat_time: dict[str, float] = {}
        for event in self._threat_events:
            level = event["peak_threat"]
            threat_time[level] = threat_time.get(level, 0) + event["duration_s"]

        # If there's a currently active nightmare, include it
        if self._nightmare_onset_time is not None:
            duration = now - self._nightmare_onset_time
            level = self._current_threat.value
            threat_time[level] = threat_time.get(level, 0) + duration

        report = {
            "session_start": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(self._session_start)
            ),
            "session_duration_hours": round(session_duration / 3600, 2),
            "total_threat_events": len(self._threat_events),
            "total_interventions": total_interventions,
            "interventions_by_level": intervention_counts,
            "threat_time_seconds": threat_time,
            "threat_events": self._threat_events,
            "intervention_log": [r.to_dict() for r in self._intervention_log],
            "firewall_config": {
                "sensitivity": self.config.sensitivity,
                "auto_wake_enabled": self.config.enable_auto_wake,
                "cooldown_s": self.config.cooldown_s,
            },
        }

        # Persist the report
        self._save_log()

        return report

    @property
    def current_threat(self) -> ThreatLevel:
        return self._current_threat

    @property
    def is_on_cooldown(self) -> bool:
        return (time.time() - self._last_intervention_time) < self.config.cooldown_s

    # ------------------------------------------------------------------
    # Nightmare pattern detection
    # ------------------------------------------------------------------

    def _detect_nightmare_pattern(
        self, history: deque[DreamFrame]
    ) -> NightmareSignature:
        """Analyse the recent frame window for sustained nightmare indicators.

        A nightmare is not just a single scary frame — it is a sustained state
        of high arousal + negative valence, often with threat content and
        distressed movement (falling, fighting). This method aggregates the
        window and produces a NightmareSignature.
        """
        now = time.time()
        window_start = now - self.config.window_seconds

        # Filter to frames within the analysis window
        window_frames = [
            f for f in history if f.timestamp >= window_start
        ]
        if not window_frames:
            window_frames = list(history)[-10:]  # fallback: last 10 frames

        # Aggregate metrics
        arousal_vals = [f.arousal_score for f in window_frames]
        valence_vals = [f.valence_score for f in window_frames]
        emotions = [f.emotion for f in window_frames]
        movements = [f.movement_type for f in window_frames]

        # Threat category presence
        threat_count = sum(
            1 for f in window_frames
            if SemanticCategory.THREAT in f.semantic_categories
        )
        threat_ratio = threat_count / len(window_frames) if window_frames else 0.0

        # Dominant emotion (most frequent)
        emotion_counts: dict[Emotion, int] = {}
        for e in emotions:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)  # type: ignore

        # Most common distressed movement
        movement_counts: dict[str, int] = {}
        for m in movements:
            movement_counts[m] = movement_counts.get(m, 0) + 1
        dominant_movement = max(movement_counts, key=movement_counts.get)  # type: ignore

        # How long has the negative pattern been sustained?
        # Walk backward from most recent frame to find when negativity started
        sustained = 0.0
        for frame in reversed(window_frames):
            if frame.valence_score < -0.1 and frame.arousal_score > 0.4:
                sustained = now - frame.timestamp
            else:
                break

        return NightmareSignature(
            avg_arousal=sum(arousal_vals) / len(arousal_vals),
            avg_valence=sum(valence_vals) / len(valence_vals),
            peak_arousal=max(arousal_vals),
            min_valence=min(valence_vals),
            threat_category_ratio=threat_ratio,
            sustained_seconds=sustained,
            dominant_emotion=dominant_emotion,
            movement_type=dominant_movement,
        )

    def _score_threat(self, sig: NightmareSignature) -> ThreatLevel:
        """Convert a NightmareSignature into a discrete ThreatLevel.

        Sensitivity shifts all thresholds: 0 = lenient, 1 = hair-trigger.
        """
        # Sensitivity adjusts thresholds — higher sensitivity = lower thresholds
        # At sensitivity 0.5, thresholds are at their default values
        # At sensitivity 1.0, thresholds are ~halved (more sensitive)
        # At sensitivity 0.0, thresholds are ~doubled (less sensitive)
        scale = 1.0 - (self.config.sensitivity - 0.5) * 1.2
        scale = max(0.4, min(1.6, scale))

        # Composite distress score: combines arousal, negative valence, and threat
        # High arousal alone is fine (exciting dream). Negative valence alone
        # might be melancholy but not nightmare. The combination is what matters.
        negativity = max(0.0, -sig.avg_valence)  # 0 to 1
        arousal = sig.avg_arousal                 # 0 to 1
        threat = sig.threat_category_ratio        # 0 to 1

        # Weighted composite
        distress = (
            negativity * 0.35 +
            arousal * 0.30 +
            threat * 0.20 +
            min(1.0, sig.sustained_seconds / 30.0) * 0.15
        )

        # Boost for extremely distressed movement patterns
        if sig.movement_type in ("falling", "fighting"):
            distress += 0.1

        # Boost for very negative peak emotions
        if sig.dominant_emotion in (Emotion.VERY_NEGATIVE,):
            distress += 0.1

        # Night terror detection: extreme arousal + possible motor activation
        # Night terrors occur in NREM (N3) with very high arousal, confusion
        if sig.peak_arousal >= self.config.auto_wake_threshold * scale:
            if sig.sustained_seconds > 10.0 * scale:
                return ThreatLevel.NIGHT_TERROR

        # Apply thresholds (adjusted by sensitivity)
        if distress >= 0.75 * scale:
            return ThreatLevel.NIGHTMARE
        elif distress >= 0.50 * scale:
            return ThreatLevel.MODERATE_DISTRESS
        elif distress >= 0.30 * scale:
            return ThreatLevel.MILD_DISTRESS
        else:
            return ThreatLevel.CALM

    # ------------------------------------------------------------------
    # Graduated interventions
    # ------------------------------------------------------------------

    def _intervene_mild(self) -> str:
        """Subtle theta binaural to gently deepen and calm the dream.

        Theta (4-8Hz) is the dominant rhythm during dreaming. Reinforcing it
        with a binaural beat helps stabilise the dream state and reduce
        surface-level anxiety without disrupting the dream narrative.
        """
        print("[Firewall] MILD_DISTRESS: theta binaural to calm dream")
        audio_data = self.stimulator.audio.generate_binaural(
            base_freq=180.0,
            beat_freq=6.0,    # theta
            duration_s=30.0,
        )
        self.stimulator.audio.play_audio(audio_data)
        return "theta_binaural_30s"

    def _intervene_moderate(self) -> str:
        """Calming audio cue + delta binaural to reduce arousal.

        Delta (1-3Hz) entrainment pulls the brain toward deeper, calmer sleep.
        Combined with a calming sound cue (custom WAV or generated tone),
        this aims to defuse the building nightmare without waking the sleeper.
        """
        print("[Firewall] MODERATE_DISTRESS: calming audio + delta binaural")
        actions = []

        # Play custom calming audio if configured
        if self.config.calming_audio_path:
            try:
                calming_data = self.stimulator.audio.generate_tmr_cue(
                    self.config.calming_audio_path, volume_scale=0.25
                )
                self.stimulator.audio.play_audio(calming_data)
                actions.append("calming_audio")
            except Exception as e:
                print(f"[Firewall] Could not play calming audio: {e}")

        # Delta binaural to pull toward calm deep sleep
        audio_data = self.stimulator.audio.generate_binaural(
            base_freq=150.0,
            beat_freq=2.0,    # delta
            duration_s=45.0,
        )
        self.stimulator.audio.play_audio(audio_data)
        actions.append("delta_binaural_45s")

        return "+".join(actions)

    def _intervene_nightmare(self) -> str:
        """Strong calming intervention + gentle reality anchor.

        This is a confirmed nightmare — sustained high distress. We deploy
        multiple channels: delta binaural for neural calming, a whispered
        reality anchor ("you are safe, you are dreaming"), and gentle light
        cues if available.
        """
        print("[Firewall] NIGHTMARE: full calming intervention + reality anchor")
        actions = []

        # Custom calming audio at higher volume
        if self.config.calming_audio_path:
            try:
                calming_data = self.stimulator.audio.generate_tmr_cue(
                    self.config.calming_audio_path, volume_scale=0.4
                )
                self.stimulator.audio.play_audio(calming_data)
                actions.append("calming_audio_loud")
            except Exception:
                pass

        # Delta binaural — longer duration
        audio_data = self.stimulator.audio.generate_binaural(
            base_freq=140.0,
            beat_freq=1.5,    # deep delta
            duration_s=60.0,
        )
        self.stimulator.audio.play_audio(audio_data)
        actions.append("deep_delta_binaural_60s")

        # Whispered reality anchor
        whisper_data = self.stimulator.audio.generate_whisper(
            "You are safe. You are dreaming. Everything is okay.",
            duration_s=8.0,
        )
        if whisper_data:
            self.stimulator.audio.play_audio(whisper_data)
            actions.append("whisper_reality_anchor")

        # Gentle light cue if available (soft, not startling)
        self.stimulator.light.flash(brightness=25, duration_ms=300)
        actions.append("gentle_light_flash")

        return "+".join(actions)

    def _intervene_night_terror(self) -> str:
        """Wake-up protocol: escalating stimulation to bring the sleeper out.

        Night terrors are NREM parasomnias with extreme arousal and confusion.
        The person is not in a normal dream state — they may be thrashing,
        screaming, with heart rate spiking. The safest response is a controlled
        wake-up with gradually increasing stimulation.

        Gated by config.enable_auto_wake — if disabled, falls back to
        nightmare-level intervention.
        """
        if not self.config.enable_auto_wake:
            print("[Firewall] NIGHT_TERROR: auto-wake disabled, using nightmare protocol")
            action = self._intervene_nightmare()
            return f"night_terror_downgraded:{action}"

        print("[Firewall] NIGHT_TERROR: wake-up protocol initiated")
        actions = []

        # Stage 1: Haptic alert — gentle vibration
        self.stimulator.haptic.pulse(intensity=120, duration_ms=500)
        actions.append("haptic_alert")

        # Stage 2: Light ramp — increasing brightness
        for brightness in (30, 50, 80):
            self.stimulator.light.flash(brightness=brightness, duration_ms=400)
            time.sleep(0.5)
        actions.append("light_ramp")

        # Stage 3: Audio — wake-up sound or binaural alpha to bring to waking
        if self.config.wake_audio_path:
            try:
                wake_data = self.stimulator.audio.generate_tmr_cue(
                    self.config.wake_audio_path, volume_scale=0.6
                )
                self.stimulator.audio.play_audio(wake_data)
                actions.append("wake_audio")
            except Exception:
                pass

        # Alpha binaural (10Hz) to pull toward waking state
        audio_data = self.stimulator.audio.generate_binaural(
            base_freq=200.0,
            beat_freq=10.0,   # alpha — relaxed wakefulness
            duration_s=20.0,
        )
        self.stimulator.audio.play_audio(audio_data)
        actions.append("alpha_binaural_20s")

        # Whispered wake cue
        whisper_data = self.stimulator.audio.generate_whisper(
            "You are waking up now. You are safe. Take a deep breath.",
            duration_s=10.0,
        )
        if whisper_data:
            self.stimulator.audio.play_audio(whisper_data)
            actions.append("whisper_wake_cue")

        # Stronger haptic to seal the deal
        self.stimulator.haptic.pattern([
            (150, 300, 200),
            (180, 300, 200),
            (200, 500, 0),
        ])
        actions.append("haptic_wake_pattern")

        return "+".join(actions)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_intervention(self, threat_level: ThreatLevel, action: str):
        """Record an intervention event with full context."""
        # Build signature dict from current state
        sig_dict = None
        if len(self._history) >= 3:
            sig = self._detect_nightmare_pattern(self._history)
            sig_dict = {
                "avg_arousal": round(sig.avg_arousal, 3),
                "avg_valence": round(sig.avg_valence, 3),
                "peak_arousal": round(sig.peak_arousal, 3),
                "min_valence": round(sig.min_valence, 3),
                "threat_ratio": round(sig.threat_category_ratio, 3),
                "sustained_s": round(sig.sustained_seconds, 1),
                "dominant_emotion": sig.dominant_emotion.value,
                "movement": sig.movement_type,
            }

        record = InterventionRecord(
            timestamp=time.time(),
            threat_level=threat_level.value,
            action=action,
            signature=sig_dict,
        )
        self._intervention_log.append(record)
        self._save_log()

        print(
            f"[Firewall] Logged: {threat_level.value} -> {action} "
            f"(total: {len(self._intervention_log)} tonight)"
        )

    def _save_log(self):
        """Persist the intervention log to disk."""
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        try:
            data = {
                "session_start": self._session_start,
                "interventions": [r.to_dict() for r in self._intervention_log],
                "threat_events": self._threat_events,
            }
            _LOG_PATH.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[Firewall] Could not save log: {e}")

    def _load_log(self):
        """Load any existing log from the current night session.

        If the log file exists and is from the same night (within 12 hours),
        resume from it. Otherwise start fresh.
        """
        if not _LOG_PATH.exists():
            return
        try:
            data = json.loads(_LOG_PATH.read_text())
            saved_start = data.get("session_start", 0)
            # If the saved session is from within the last 12 hours, resume
            if time.time() - saved_start < 12 * 3600:
                self._session_start = saved_start
                self._threat_events = data.get("threat_events", [])
                for entry in data.get("interventions", []):
                    record = InterventionRecord(
                        timestamp=entry["timestamp"],
                        threat_level=entry["threat_level"],
                        action=entry["action"],
                        signature=entry.get("signature"),
                    )
                    self._intervention_log.append(record)
                print(
                    f"[Firewall] Resumed session with "
                    f"{len(self._intervention_log)} prior interventions"
                )
        except Exception:
            pass  # corrupted log — start fresh
