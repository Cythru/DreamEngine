"""dream_engine/neural/communicator.py — Two-way lucid dream communication.

Implements the Konkoly et al. (2021) protocol for communicating with
lucid dreamers in real-time:

OUTBOUND (dreamer → machine):
  - Detect pre-agreed eye movement patterns (LRLR = Left-Right-Left-Right)
  - Decode morse-like eye signals for simple messages
  - Detect intentional muscle twitches (EMG) as binary signals

INBOUND (machine → dreamer):
  - Audio cues (whispered questions, tones)
  - Haptic patterns (coded vibrations)
  - Light signals through closed eyelids

PROTOCOL:
  1. Before sleep, agree on signal patterns (e.g., LRLR = "I'm lucid")
  2. System detects REM + gamma burst (lucidity likely)
  3. System sends gentle audio: "Are you dreaming? Signal with your eyes."
  4. System monitors for LRLR pattern
  5. On detection: "Confirmed. You're dreaming. [optional prompt]"
  6. Dreamer can answer yes/no questions with eye patterns
"""
from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from .bci import BCIStream
from .stimulator import Stimulator


class EyeSignal(Enum):
    NONE = "none"
    LEFT = "left"
    RIGHT = "right"
    LRLR = "lrlr"        # standard lucid confirmation
    LR = "lr"             # yes
    LL = "ll"             # no
    RLRL = "rlrl"         # request to wake up
    CUSTOM = "custom"


@dataclass
class SignalDetection:
    signal: EyeSignal
    timestamp: float
    confidence: float
    raw_pattern: list[str]


@dataclass
class CommSession:
    """Tracks a communication session with a lucid dreamer."""
    start_time: float
    lucid_confirmed: bool = False
    messages_sent: list[str] = field(default_factory=list)
    signals_received: list[SignalDetection] = field(default_factory=list)
    qa_log: list[dict] = field(default_factory=list)


class EyePatternDetector:
    """Detect deliberate eye movement patterns from frontal EEG/EOG."""

    # Minimum amplitude for eye movement detection (microvolts)
    EOG_THRESHOLD_UV = 50.0
    # Window for pattern detection
    PATTERN_WINDOW_S = 8.0
    # Minimum time between movements in a pattern
    MIN_MOVEMENT_GAP_S = 0.3
    MAX_MOVEMENT_GAP_S = 2.0

    def __init__(self, stream: BCIStream):
        self.stream = stream
        self._pattern_buffer: list[tuple[str, float]] = []

    def detect_movement(self) -> Optional[str]:
        """Detect a single eye movement direction from EOG.

        Returns "left", "right", or None.
        """
        try:
            eeg = self.stream.get_data(seconds=1.0)
            if eeg.shape[0] < 2:
                return None

            # EOG from frontal channel difference
            eog = eeg[0] - eeg[1]

            # Find peaks (deliberate eye movements are large, slow deflections)
            peak = np.max(eog)
            trough = np.min(eog)

            if peak > self.EOG_THRESHOLD_UV and peak > abs(trough):
                return "right"
            elif abs(trough) > self.EOG_THRESHOLD_UV and abs(trough) > peak:
                return "left"
            return None
        except Exception:
            return None

    def update(self) -> Optional[EyeSignal]:
        """Check for eye movement and update pattern buffer.

        Returns detected pattern or None.
        """
        now = time.time()
        direction = self.detect_movement()

        if direction:
            # Don't add if too close to last movement
            if (self._pattern_buffer and
                    now - self._pattern_buffer[-1][1] < self.MIN_MOVEMENT_GAP_S):
                return None
            self._pattern_buffer.append((direction, now))

        # Clean old entries
        self._pattern_buffer = [
            (d, t) for d, t in self._pattern_buffer
            if now - t < self.PATTERN_WINDOW_S
        ]

        # Check for patterns
        return self._match_pattern()

    def _match_pattern(self) -> Optional[EyeSignal]:
        """Match buffered movements against known patterns."""
        if len(self._pattern_buffer) < 2:
            return None

        dirs = [d for d, t in self._pattern_buffer]
        pattern = "".join(d[0] for d in dirs)  # "l", "r" shorthand

        # LRLR — lucid confirmation (4 movements)
        if len(dirs) >= 4:
            last4 = "".join(d[0] for d in dirs[-4:])
            if last4 == "lrlr":
                self._pattern_buffer.clear()
                return EyeSignal.LRLR
            if last4 == "rlrl":
                self._pattern_buffer.clear()
                return EyeSignal.RLRL

        # LR — yes (2 movements)
        if len(dirs) >= 2:
            last2 = "".join(d[0] for d in dirs[-2:])
            times = [t for _, t in self._pattern_buffer[-2:]]
            gap = times[1] - times[0]

            # Only match 2-movement patterns if enough time has passed
            # (to avoid false positives while waiting for 4-movement patterns)
            if gap > self.MAX_MOVEMENT_GAP_S:
                if last2 == "lr":
                    self._pattern_buffer.clear()
                    return EyeSignal.LR
                if last2 == "ll":
                    self._pattern_buffer.clear()
                    return EyeSignal.LL

        return None


class DreamCommunicator:
    """Two-way communication interface with a lucid dreamer."""

    def __init__(self, stream: BCIStream, stimulator: Stimulator):
        self.stream = stream
        self.stim = stimulator
        self.eye_detector = EyePatternDetector(stream)
        self.session: Optional[CommSession] = None
        self._signal_callbacks: list[Callable] = []

    def on_signal(self, callback: Callable[[SignalDetection], None]):
        self._signal_callbacks.append(callback)

    def start_session(self):
        """Begin a communication session."""
        self.session = CommSession(start_time=time.time())
        print("[Comm] Session started — waiting for lucid confirmation")

    def send_lucidity_check(self):
        """Ask the dreamer if they're lucid."""
        msg = "You are dreaming. If you can hear this, look left right left right."
        self.stim.whisper_cue(msg)
        if self.session:
            self.session.messages_sent.append(msg)

    def send_message(self, text: str):
        """Whisper a message into the dream."""
        self.stim.whisper_cue(text)
        if self.session:
            self.session.messages_sent.append(text)

    def ask_yes_no(self, question: str, timeout_s: float = 30.0) -> Optional[bool]:
        """Ask a yes/no question. LR = yes, LL = no.

        Returns True (yes), False (no), or None (no response).
        """
        prompt = f"{question} Look left-right for yes, left-left for no."
        self.send_message(prompt)

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            signal = self.eye_detector.update()
            if signal == EyeSignal.LR:
                if self.session:
                    self.session.qa_log.append({
                        "question": question,
                        "answer": "yes",
                        "time": time.time(),
                    })
                return True
            elif signal == EyeSignal.LL:
                if self.session:
                    self.session.qa_log.append({
                        "question": question,
                        "answer": "no",
                        "time": time.time(),
                    })
                return False
            time.sleep(0.5)

        return None

    def poll_signals(self) -> Optional[SignalDetection]:
        """Check for eye signals. Call this in the main loop."""
        signal = self.eye_detector.update()
        if signal and signal != EyeSignal.NONE:
            detection = SignalDetection(
                signal=signal,
                timestamp=time.time(),
                confidence=0.8,
                raw_pattern=[d for d, _ in self.eye_detector._pattern_buffer],
            )
            if self.session:
                self.session.signals_received.append(detection)

                # Handle LRLR as lucid confirmation
                if signal == EyeSignal.LRLR and not self.session.lucid_confirmed:
                    self.session.lucid_confirmed = True
                    self.send_message("Confirmed. You are lucid. The dream is yours.")
                    print("[Comm] LUCID CONFIRMED via LRLR!")

                # Handle RLRL as wake request
                elif signal == EyeSignal.RLRL:
                    self.send_message("Waking you up now. Open your eyes slowly.")
                    print("[Comm] Wake request received!")

            for cb in self._signal_callbacks:
                cb(detection)

            return detection
        return None
