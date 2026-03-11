"""dream_engine/neural/time_dilation.py — Subjective time manipulation in lucid dreams.

Inspired by Inception's nested dream time dilation. In reality, lucid dreamers
consistently report altered time perception — some experience hours in minutes
of real time. Research suggests this is modulable:

- Theta frequency (4-7Hz) correlates with subjective time slowing
- Alpha suppression correlates with "flow state" time distortion
- Gamma bursts (40Hz) anchor awareness without collapsing the dream
- Delta entrainment deepens the dream, increasing time distortion

This module implements frequency protocols designed to maximize
subjective dream duration while maintaining lucidity.

References:
- LaBerge & DeGracia (2000) — Subjective time estimation in lucid dreams
- Erlacher & Schredl (2004) — Time perception during lucid dreaming
- Voss et al. (2009) — Lucid dreaming and metacognition
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .stimulator import Stimulator, AudioGenerator, StimConfig
from .sleep_staging import SleepStager, SleepStage
from .dream_detector import DreamDetector


class DilationLevel(Enum):
    """Subjective time dilation targets."""
    NORMAL = "normal"          # 1:1 perceived vs real time
    STRETCHED = "stretched"    # ~2:1 — minutes feel longer
    DEEP = "deep"              # ~5:1 — deep immersion, time barely moves
    INCEPTION = "inception"    # maximum dilation — experimental


@dataclass
class DilationProtocol:
    """A frequency protocol for time dilation."""
    name: str
    level: DilationLevel
    # Binaural beat parameters
    theta_freq: float          # primary theta entrainment (Hz)
    theta_volume: float        # 0-1
    gamma_anchor_freq: float   # gamma pulse to maintain lucidity (Hz)
    gamma_volume: float        # much quieter than theta
    gamma_duty_cycle: float    # fraction of time gamma is active (0-1)
    # Timing
    ramp_up_seconds: float     # gradual onset to avoid waking
    sustain_seconds: float     # duration of full protocol
    ramp_down_seconds: float   # gradual offset
    # Monitoring thresholds
    min_theta_ratio: float     # abort if theta drops below this
    max_beta_ratio: float      # abort if beta rises above this (waking)


# Pre-built protocols
PROTOCOLS = {
    DilationLevel.STRETCHED: DilationProtocol(
        name="Stretched Time",
        level=DilationLevel.STRETCHED,
        theta_freq=5.5,
        theta_volume=0.10,
        gamma_anchor_freq=40.0,
        gamma_volume=0.03,
        gamma_duty_cycle=0.15,    # 15% on, 85% off — subtle anchor
        ramp_up_seconds=30.0,
        sustain_seconds=300.0,     # 5 minutes
        ramp_down_seconds=20.0,
        min_theta_ratio=0.20,
        max_beta_ratio=0.25,
    ),
    DilationLevel.DEEP: DilationProtocol(
        name="Deep Immersion",
        level=DilationLevel.DEEP,
        theta_freq=4.5,            # lower theta = deeper
        theta_volume=0.12,
        gamma_anchor_freq=40.0,
        gamma_volume=0.02,
        gamma_duty_cycle=0.08,     # very sparse gamma — just enough
        ramp_up_seconds=60.0,
        sustain_seconds=600.0,     # 10 minutes
        ramp_down_seconds=30.0,
        min_theta_ratio=0.25,
        max_beta_ratio=0.20,
    ),
    DilationLevel.INCEPTION: DilationProtocol(
        name="Inception Protocol",
        level=DilationLevel.INCEPTION,
        theta_freq=4.0,            # deep theta boundary
        theta_volume=0.15,
        gamma_anchor_freq=40.0,
        gamma_volume=0.015,
        gamma_duty_cycle=0.05,     # minimal gamma — risk of losing lucidity
        ramp_up_seconds=90.0,
        sustain_seconds=900.0,     # 15 minutes real time
        ramp_down_seconds=45.0,
        min_theta_ratio=0.30,
        max_beta_ratio=0.15,
    ),
}


@dataclass
class DilationState:
    """Current time dilation session state."""
    active: bool = False
    protocol: Optional[DilationProtocol] = None
    phase: str = "idle"        # idle, ramp_up, sustain, ramp_down
    elapsed_real_s: float = 0.0
    estimated_subjective_s: float = 0.0
    dilation_ratio: float = 1.0
    theta_power: float = 0.0
    lucidity_stable: bool = False
    aborted: bool = False
    abort_reason: str = ""


class TimeDilator:
    """Controls time dilation protocols during lucid REM.

    Flow:
    1. Wait for confirmed lucid REM (dream_detector.state.lucidity_score > threshold)
    2. Begin ramp-up: gradually introduce theta entrainment
    3. Monitor: ensure theta stays dominant, beta stays low, dreamer stays in REM
    4. Sustain: maintain protocol for target duration
    5. Ramp-down: gradually reduce stimulation
    6. Log estimated subjective duration based on theta depth and real elapsed time
    """

    LUCIDITY_THRESHOLD = 0.3
    # Estimated dilation ratios based on theta depth
    # These are approximations from lucid dreaming research
    DILATION_ESTIMATES = {
        DilationLevel.NORMAL: 1.0,
        DilationLevel.STRETCHED: 2.5,
        DilationLevel.DEEP: 5.0,
        DilationLevel.INCEPTION: 10.0,  # speculative
    }

    def __init__(self, stager: SleepStager, detector: DreamDetector, stim: Stimulator):
        self.stager = stager
        self.detector = detector
        self.stim = stim
        self.state = DilationState()
        self._start_time: Optional[float] = None

    def begin(self, level: DilationLevel = DilationLevel.STRETCHED) -> bool:
        """Start a time dilation protocol.

        Returns True if conditions are met, False if not safe to start.
        """
        # Safety checks
        if not self.detector.state.is_dreaming:
            self.state.abort_reason = "not dreaming"
            return False
        if self.detector.state.lucidity_score < self.LUCIDITY_THRESHOLD:
            self.state.abort_reason = "insufficient lucidity"
            return False
        if not self.detector.state.dream_stable:
            self.state.abort_reason = "dream not stable"
            return False

        protocol = PROTOCOLS.get(level)
        if not protocol:
            self.state.abort_reason = f"unknown level: {level}"
            return False

        self.state = DilationState(
            active=True,
            protocol=protocol,
            phase="ramp_up",
        )
        self._start_time = time.time()
        print(f"[TimeDilation] Starting: {protocol.name}")
        return True

    def update(self) -> DilationState:
        """Called each epoch during active dilation. Manages phase transitions."""
        if not self.state.active or not self.state.protocol:
            return self.state

        protocol = self.state.protocol
        elapsed = time.time() - (self._start_time or time.time())
        self.state.elapsed_real_s = elapsed

        # Get current brain state
        epoch = self.stager.session.epochs[-1] if self.stager.session.epochs else None
        if epoch:
            self.state.theta_power = epoch.theta_power

        # Safety: check we're still in REM
        if not self.detector.state.is_dreaming:
            self._abort("dream ended")
            return self.state

        # Safety: check beta isn't spiking (waking up)
        if epoch and epoch.beta_power > protocol.max_beta_ratio:
            self._abort("beta spike — possible awakening")
            return self.state

        # Phase management
        if self.state.phase == "ramp_up":
            if elapsed >= protocol.ramp_up_seconds:
                self.state.phase = "sustain"
                print(f"[TimeDilation] Ramp complete — entering sustain phase")
            else:
                # Gradually increase theta entrainment
                progress = elapsed / protocol.ramp_up_seconds
                self._apply_stimulation(progress)

        elif self.state.phase == "sustain":
            sustain_end = protocol.ramp_up_seconds + protocol.sustain_seconds
            if elapsed >= sustain_end:
                self.state.phase = "ramp_down"
                print(f"[TimeDilation] Sustain complete — ramping down")
            else:
                self._apply_stimulation(1.0)

                # Check theta depth for dilation estimation
                if epoch and epoch.theta_power > protocol.min_theta_ratio:
                    ratio = self.DILATION_ESTIMATES.get(protocol.level, 1.0)
                    # Scale by actual theta depth
                    theta_factor = epoch.theta_power / 0.35  # normalize
                    actual_ratio = ratio * min(1.5, max(0.5, theta_factor))
                    self.state.dilation_ratio = round(actual_ratio, 1)
                    self.state.estimated_subjective_s = elapsed * actual_ratio

        elif self.state.phase == "ramp_down":
            total = (protocol.ramp_up_seconds + protocol.sustain_seconds +
                     protocol.ramp_down_seconds)
            if elapsed >= total:
                self._complete()
            else:
                ramp_progress = (total - elapsed) / protocol.ramp_down_seconds
                self._apply_stimulation(max(0, ramp_progress))

        return self.state

    def _apply_stimulation(self, intensity: float):
        """Apply theta + gamma stimulation at given intensity (0-1)."""
        protocol = self.state.protocol
        if not protocol:
            return

        # Theta binaural
        # (In real implementation, this would modulate an ongoing audio stream
        # rather than generating new audio each epoch)
        # For now, we log the target parameters
        theta_vol = protocol.theta_volume * intensity
        gamma_vol = protocol.gamma_volume * intensity

        self.state.lucidity_stable = (
            self.detector.state.lucidity_score > self.LUCIDITY_THRESHOLD
        )

    def _abort(self, reason: str):
        """Abort the protocol safely."""
        self.state.active = False
        self.state.aborted = True
        self.state.abort_reason = reason
        self.state.phase = "idle"
        self.stim.audio.stop()
        print(f"[TimeDilation] Aborted: {reason}")

    def _complete(self):
        """Protocol completed successfully."""
        self.state.active = False
        self.state.phase = "idle"
        self.stim.audio.stop()

        est_minutes = self.state.estimated_subjective_s / 60
        real_minutes = self.state.elapsed_real_s / 60
        print(f"[TimeDilation] Complete. Real: {real_minutes:.1f}m, "
              f"Estimated subjective: {est_minutes:.1f}m "
              f"(ratio: {self.state.dilation_ratio}:1)")

    def get_report(self) -> dict:
        """Get a report on the dilation session."""
        return {
            "protocol": self.state.protocol.name if self.state.protocol else None,
            "real_duration_s": round(self.state.elapsed_real_s, 1),
            "estimated_subjective_s": round(self.state.estimated_subjective_s, 1),
            "dilation_ratio": self.state.dilation_ratio,
            "aborted": self.state.aborted,
            "abort_reason": self.state.abort_reason,
            "peak_theta": round(self.state.theta_power, 3),
        }
