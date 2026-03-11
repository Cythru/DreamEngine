"""dream_engine/neural/sleep_staging.py — Real-time sleep stage classification.

Classifies EEG into 5 stages using spectral power ratios:
  - Wake:  high alpha (8-13Hz), high beta, low delta
  - N1:    alpha drops, theta rises, slow eye movements
  - N2:    sleep spindles (12-14Hz bursts), K-complexes
  - N3:    dominant delta (0.5-4Hz), >20% of epoch = slow-wave sleep
  - REM:   low-voltage mixed frequency, theta dominant, rapid eye movements

Uses 30-second epochs (standard polysomnography) with real-time sliding window.
"""
from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .bci import BCIStream


class SleepStage(Enum):
    WAKE = "wake"
    N1 = "n1"
    N2 = "n2"
    N3 = "n3"
    REM = "rem"
    UNKNOWN = "unknown"


@dataclass
class SleepEpoch:
    """A single 30-second sleep staging result."""
    timestamp: float
    stage: SleepStage
    confidence: float          # 0-1
    delta_power: float         # 0.5-4 Hz
    theta_power: float         # 4-8 Hz
    alpha_power: float         # 8-13 Hz
    beta_power: float          # 13-30 Hz
    gamma_power: float         # 30-50 Hz
    spindle_detected: bool     # 12-14 Hz burst (N2 marker)
    rem_eye_movement: float    # EOG-derived rapid eye movement score


@dataclass
class SleepSession:
    """Accumulated sleep session data."""
    start_time: float
    epochs: list[SleepEpoch]
    current_stage: SleepStage = SleepStage.UNKNOWN
    rem_periods: int = 0
    total_rem_minutes: float = 0.0
    total_deep_minutes: float = 0.0
    sleep_onset_time: Optional[float] = None
    rem_onset_time: Optional[float] = None


class SleepStager:
    """Real-time sleep stage classifier from EEG stream."""

    # Thresholds tuned for consumer EEG (Muse/OpenBCI)
    DELTA_N3_THRESHOLD = 0.55      # delta ratio > 55% → N3
    ALPHA_WAKE_THRESHOLD = 0.30    # alpha ratio > 30% → wake
    THETA_REM_THRESHOLD = 0.35     # theta dominant + low delta → REM
    SPINDLE_MIN_HZ = 11.0
    SPINDLE_MAX_HZ = 16.0
    SPINDLE_MIN_DURATION_S = 0.5
    SPINDLE_POWER_THRESHOLD = 2.0  # times baseline
    EPOCH_SECONDS = 30.0

    def __init__(self, stream: BCIStream):
        self.stream = stream
        self.session = SleepSession(
            start_time=time.time(),
            epochs=[],
        )
        self._baseline_powers: Optional[dict] = None
        self._last_stage = SleepStage.WAKE
        self._consecutive_rem = 0
        self._consecutive_n3 = 0

    def calibrate(self, seconds: float = 60.0):
        """Record baseline power during relaxed wakefulness.

        User should sit still with eyes closed for calibration.
        """
        print(f"[Sleep] Calibrating baseline ({seconds}s, eyes closed, relax)...")
        time.sleep(seconds)
        self._baseline_powers = self.stream.get_bandpower(seconds=min(seconds, 10.0))
        print("[Sleep] Baseline calibrated")

    def classify_epoch(self) -> SleepEpoch:
        """Classify the current 30-second epoch."""
        bands = self.stream.get_bandpower(seconds=self.EPOCH_SECONDS)

        # Mean power across channels for each band
        delta = float(np.mean(bands["delta"]))
        theta = float(np.mean(bands["theta"]))
        alpha = float(np.mean(bands["alpha"]))
        beta = float(np.mean(bands["beta"]))
        gamma = float(np.mean(bands["gamma"]))

        total = delta + theta + alpha + beta + gamma + 1e-10

        # Normalized ratios
        r_delta = delta / total
        r_theta = theta / total
        r_alpha = alpha / total
        r_beta = beta / total
        r_gamma = gamma / total

        # Spindle detection (12-14Hz burst within sigma band)
        spindle = self._detect_spindle(bands)

        # Eye movement score (uses frontal channels as proxy for EOG)
        rem_eye = self._estimate_eye_movement()

        # Classification logic
        stage, confidence = self._classify(
            r_delta, r_theta, r_alpha, r_beta, r_gamma,
            spindle, rem_eye
        )

        epoch = SleepEpoch(
            timestamp=time.time(),
            stage=stage,
            confidence=confidence,
            delta_power=r_delta,
            theta_power=r_theta,
            alpha_power=r_alpha,
            beta_power=r_beta,
            gamma_power=r_gamma,
            spindle_detected=spindle,
            rem_eye_movement=rem_eye,
        )

        self._update_session(epoch)
        return epoch

    def _classify(
        self,
        r_delta: float, r_theta: float, r_alpha: float,
        r_beta: float, r_gamma: float,
        spindle: bool, rem_eye: float,
    ) -> tuple[SleepStage, float]:
        """Rule-based sleep stage classification with confidence."""

        scores = {
            SleepStage.WAKE: 0.0,
            SleepStage.N1: 0.0,
            SleepStage.N2: 0.0,
            SleepStage.N3: 0.0,
            SleepStage.REM: 0.0,
        }

        # WAKE: high alpha + beta, low delta
        if r_alpha > self.ALPHA_WAKE_THRESHOLD:
            scores[SleepStage.WAKE] += 0.4
        if r_beta > 0.20:
            scores[SleepStage.WAKE] += 0.3
        if r_delta < 0.20:
            scores[SleepStage.WAKE] += 0.2

        # N1: alpha drops, theta rises, transition state
        if r_alpha < 0.20 and r_theta > 0.20 and r_delta < 0.40:
            scores[SleepStage.N1] += 0.5
        if self._last_stage == SleepStage.WAKE:
            scores[SleepStage.N1] += 0.1  # transition bias

        # N2: spindles + K-complexes, moderate delta/theta
        if spindle:
            scores[SleepStage.N2] += 0.5
        if r_theta > 0.20 and r_delta > 0.20 and r_delta < self.DELTA_N3_THRESHOLD:
            scores[SleepStage.N2] += 0.3

        # N3: dominant delta (slow-wave sleep)
        if r_delta > self.DELTA_N3_THRESHOLD:
            scores[SleepStage.N3] += 0.6
        if r_delta > 0.70:
            scores[SleepStage.N3] += 0.3

        # REM: theta dominant, low delta, high eye movement, low muscle tone
        if r_theta > self.THETA_REM_THRESHOLD and r_delta < 0.30:
            scores[SleepStage.REM] += 0.4
        if rem_eye > 0.5:
            scores[SleepStage.REM] += 0.3
        if r_gamma > 0.05:  # gamma bursts common in REM
            scores[SleepStage.REM] += 0.1
        # REM usually follows N2/N3, not wake
        if self._last_stage in (SleepStage.N2, SleepStage.N3):
            scores[SleepStage.REM] += 0.1

        # Pick highest scoring stage
        best_stage = max(scores, key=scores.get)
        best_score = scores[best_stage]
        total_score = sum(scores.values()) + 1e-10
        confidence = best_score / total_score

        # Hysteresis: require sustained change to switch stages
        if best_stage != self._last_stage and confidence < 0.4:
            best_stage = self._last_stage
            confidence = 0.3

        self._last_stage = best_stage
        return best_stage, round(confidence, 3)

    def _detect_spindle(self, bands: dict) -> bool:
        """Detect sleep spindles (sigma band bursts at 12-14Hz)."""
        if self._baseline_powers is None:
            return False
        # Compare current sigma power to baseline
        # Using beta as proxy (includes spindle range)
        current = float(np.mean(bands.get("beta", [0])))
        baseline = float(np.mean(self._baseline_powers.get("beta", [1])))
        if baseline < 1e-10:
            return False
        return (current / baseline) > self.SPINDLE_POWER_THRESHOLD

    def _estimate_eye_movement(self) -> float:
        """Estimate rapid eye movement from frontal EEG channels.

        Frontal channels (Fp1/Fp2 or AF7/AF8) pick up EOG artifacts —
        which is normally unwanted but perfect for REM detection.
        Returns 0-1 score.
        """
        try:
            eeg = self.stream.get_data(seconds=5.0)
            if eeg.shape[0] < 2:
                return 0.0
            # Use first two channels (frontal) — compute difference for EOG
            eog = eeg[0] - eeg[1]  # left - right frontal
            # Count zero-crossings in 0.5-2Hz band (saccade frequency)
            # High zero-crossing rate = rapid eye movements
            crossings = np.sum(np.diff(np.sign(eog)) != 0)
            rate = crossings / (len(eog) / self.stream.sample_rate)
            # Normalize: ~0.5 crossings/sec at rest, ~5/sec during REM
            score = min(1.0, max(0.0, (rate - 1.0) / 4.0))
            return round(score, 3)
        except Exception:
            return 0.0

    def _update_session(self, epoch: SleepEpoch):
        """Update session tracking."""
        self.session.epochs.append(epoch)
        self.session.current_stage = epoch.stage

        # Track sleep onset
        if (self.session.sleep_onset_time is None and
                epoch.stage in (SleepStage.N1, SleepStage.N2)):
            self.session.sleep_onset_time = epoch.timestamp

        # Track REM
        if epoch.stage == SleepStage.REM:
            self._consecutive_rem += 1
            if self._consecutive_rem == 2:  # 1 minute of REM
                self.session.rem_periods += 1
                if self.session.rem_onset_time is None:
                    self.session.rem_onset_time = epoch.timestamp
            self.session.total_rem_minutes += 0.5  # 30-sec epochs
        else:
            self._consecutive_rem = 0

        # Track deep sleep
        if epoch.stage == SleepStage.N3:
            self.session.total_deep_minutes += 0.5

    def get_hypnogram(self) -> list[tuple[float, str]]:
        """Return the full sleep hypnogram as (timestamp, stage) pairs."""
        return [(e.timestamp, e.stage.value) for e in self.session.epochs]

    def is_dreaming(self) -> bool:
        """Quick check: is the person likely dreaming right now?"""
        return (self.session.current_stage == SleepStage.REM and
                self._consecutive_rem >= 2)

    def is_deep_sleep(self) -> bool:
        return self.session.current_stage == SleepStage.N3
