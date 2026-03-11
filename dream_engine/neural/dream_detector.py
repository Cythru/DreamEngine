"""dream_engine/neural/dream_detector.py — Dream onset detection and monitoring.

Builds on sleep staging to detect:
  - Dream onset (REM entry with sustained theta + eye movement)
  - Dream intensity (power fluctuations during REM)
  - Lucid moments (gamma bursts during REM)
  - Dream end (REM → N2 transition)
  - Optimal intervention windows (stable REM for cue delivery)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .sleep_staging import SleepStager, SleepStage, SleepEpoch


@dataclass
class DreamEvent:
    """A detected dream period with metadata."""
    start_time: float
    end_time: Optional[float] = None
    duration_minutes: float = 0.0
    peak_intensity: float = 0.0
    lucid_moments: int = 0
    avg_theta: float = 0.0
    avg_gamma: float = 0.0
    eye_movement_score: float = 0.0
    rem_period_number: int = 0
    interventions_delivered: int = 0


@dataclass
class DreamState:
    """Current dream monitoring state."""
    is_dreaming: bool = False
    dream_stable: bool = False         # stable enough for intervention
    stability_seconds: float = 0.0
    intensity: float = 0.0            # 0-1, derived from theta/gamma/eye movement
    lucidity_score: float = 0.0       # 0-1, gamma power during REM
    current_dream: Optional[DreamEvent] = None
    all_dreams: list[DreamEvent] = field(default_factory=list)


class DreamDetector:
    """Monitors EEG stream for dream events and optimal intervention windows."""

    # How long REM must sustain before we call it a dream
    MIN_REM_FOR_DREAM_S = 60.0
    # How long REM must be stable for safe cue delivery
    MIN_STABLE_FOR_INTERVENTION_S = 120.0
    # Gamma threshold for lucidity detection
    GAMMA_LUCID_THRESHOLD = 0.08

    def __init__(self, stager: SleepStager):
        self.stager = stager
        self.state = DreamState()
        self._rem_start: Optional[float] = None
        self._callbacks_dream_start: list[Callable] = []
        self._callbacks_dream_end: list[Callable] = []
        self._callbacks_lucid: list[Callable] = []
        self._callbacks_stable: list[Callable] = []

    def on_dream_start(self, callback: Callable[[DreamEvent], None]):
        """Register callback for when a dream begins."""
        self._callbacks_dream_start.append(callback)

    def on_dream_end(self, callback: Callable[[DreamEvent], None]):
        """Register callback for when a dream ends."""
        self._callbacks_dream_end.append(callback)

    def on_lucid_detected(self, callback: Callable[[DreamEvent], None]):
        """Register callback for gamma burst during REM (possible lucidity)."""
        self._callbacks_lucid.append(callback)

    def on_dream_stable(self, callback: Callable[[DreamEvent], None]):
        """Register callback when dream is stable enough for intervention."""
        self._callbacks_stable.append(callback)

    def update(self, epoch: SleepEpoch):
        """Process a new sleep epoch and update dream state."""
        now = epoch.timestamp

        if epoch.stage == SleepStage.REM:
            self._handle_rem(epoch, now)
        else:
            self._handle_non_rem(epoch, now)

    def _handle_rem(self, epoch: SleepEpoch, now: float):
        """Handle a REM epoch."""
        if self._rem_start is None:
            self._rem_start = now

        rem_duration = now - self._rem_start
        self.state.stability_seconds = rem_duration

        # Calculate dream intensity from theta + eye movement + gamma
        intensity = (
            epoch.theta_power * 0.4 +
            epoch.rem_eye_movement * 0.4 +
            epoch.gamma_power * 0.2
        )
        self.state.intensity = min(1.0, intensity * 2)

        # Lucidity score from gamma
        self.state.lucidity_score = min(1.0, epoch.gamma_power / 0.10)

        # Dream onset
        if (not self.state.is_dreaming and
                rem_duration >= self.MIN_REM_FOR_DREAM_S):
            self._start_dream(now, epoch)

        # Update current dream
        if self.state.current_dream:
            dream = self.state.current_dream
            dream.duration_minutes = (now - dream.start_time) / 60.0
            dream.peak_intensity = max(dream.peak_intensity, self.state.intensity)
            dream.avg_theta = (dream.avg_theta + epoch.theta_power) / 2
            dream.avg_gamma = (dream.avg_gamma + epoch.gamma_power) / 2
            dream.eye_movement_score = (
                dream.eye_movement_score + epoch.rem_eye_movement
            ) / 2

            # Lucid moment detection
            if epoch.gamma_power > self.GAMMA_LUCID_THRESHOLD:
                dream.lucid_moments += 1
                for cb in self._callbacks_lucid:
                    cb(dream)

        # Stable dream detection
        if (self.state.is_dreaming and
                not self.state.dream_stable and
                rem_duration >= self.MIN_STABLE_FOR_INTERVENTION_S):
            self.state.dream_stable = True
            if self.state.current_dream:
                for cb in self._callbacks_stable:
                    cb(self.state.current_dream)

    def _handle_non_rem(self, epoch: SleepEpoch, now: float):
        """Handle a non-REM epoch (dream may be ending)."""
        if self.state.is_dreaming:
            self._end_dream(now)

        self._rem_start = None
        self.state.is_dreaming = False
        self.state.dream_stable = False
        self.state.stability_seconds = 0.0
        self.state.intensity = 0.0
        self.state.lucidity_score = 0.0

    def _start_dream(self, now: float, epoch: SleepEpoch):
        """Mark dream onset."""
        self.state.is_dreaming = True
        dream = DreamEvent(
            start_time=self._rem_start or now,
            rem_period_number=self.stager.session.rem_periods,
        )
        self.state.current_dream = dream
        for cb in self._callbacks_dream_start:
            cb(dream)

    def _end_dream(self, now: float):
        """Mark dream end and archive."""
        if self.state.current_dream:
            dream = self.state.current_dream
            dream.end_time = now
            dream.duration_minutes = (now - dream.start_time) / 60.0
            self.state.all_dreams.append(dream)
            for cb in self._callbacks_dream_end:
                cb(dream)
        self.state.current_dream = None

    def get_dream_summary(self) -> dict:
        """Get summary of all detected dreams this session."""
        dreams = self.state.all_dreams
        if self.state.current_dream:
            dreams = dreams + [self.state.current_dream]

        return {
            "total_dreams": len(dreams),
            "total_dream_minutes": sum(d.duration_minutes for d in dreams),
            "lucid_moments": sum(d.lucid_moments for d in dreams),
            "peak_intensity": max((d.peak_intensity for d in dreams), default=0),
            "dreams": [
                {
                    "number": i + 1,
                    "duration_min": round(d.duration_minutes, 1),
                    "intensity": round(d.peak_intensity, 2),
                    "lucid_moments": d.lucid_moments,
                    "start": d.start_time,
                }
                for i, d in enumerate(dreams)
            ],
        }
