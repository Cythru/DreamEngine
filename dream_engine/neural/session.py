"""dream_engine/neural/session.py — Full sleep session orchestrator.

Ties everything together into a single overnight sleep session:
  1. Calibrate EEG during pre-sleep relaxation
  2. Monitor sleep stages in real-time
  3. During N3: deliver TMR cues to plant dream content
  4. During REM: detect dreams, attempt lucid induction
  5. During lucid REM: open two-way communication
  6. On wake: auto-generate dream journal entries from detected periods
  7. Morning debrief: LLM asks questions about detected dreams

Usage:
    session = SleepSession(config)
    session.set_dream_scenario("flying over mountains")
    session.add_tmr_cue("/path/to/mountain_wind.wav")
    session.start()  # blocks until morning or interrupted
"""
from __future__ import annotations

import json
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .bci import BCIStream, BCIConfig, BoardType
from .sleep_staging import SleepStager, SleepStage
from .dream_detector import DreamDetector, DreamEvent
from .stimulator import Stimulator, StimConfig
from .communicator import DreamCommunicator


@dataclass
class SessionConfig:
    """Configuration for an overnight session."""
    # Hardware
    board_type: BoardType = BoardType.SYNTHETIC
    serial_port: str = ""
    mac_address: str = ""

    # Dream induction
    enable_lucid_induction: bool = True
    enable_tmr: bool = True
    enable_communication: bool = True

    # TMR
    tmr_cue_paths: list[str] = field(default_factory=list)
    tmr_scenario: str = ""

    # Timing
    calibration_seconds: float = 60.0
    epoch_seconds: float = 30.0

    # Safety
    max_stimulations_per_hour: int = 6
    min_gap_between_stims_s: float = 300.0
    quiet_hours_start: float = 0.0    # hours after session start to begin stim
    quiet_hours_end: float = 1.0      # minimum 1hr natural sleep before stim

    # Output
    data_dir: str = str(Path.home() / ".local/share/blackmagic-gen/dreams/sessions")


class SleepSession:
    """Full overnight sleep session orchestrator."""

    def __init__(self, config: SessionConfig):
        self.config = config
        self._running = False
        self._session_start: Optional[float] = None
        self._last_stim_time: float = 0.0
        self._stim_count: int = 0
        self._stim_hour_start: float = 0.0

        # Initialize components
        bci_config = BCIConfig(
            board_type=config.board_type,
            serial_port=config.serial_port,
            mac_address=config.mac_address,
        )
        self.stream = BCIStream(bci_config)
        self.stager = SleepStager(self.stream)
        self.detector = DreamDetector(self.stager)
        self.stim = Stimulator(StimConfig())
        self.comm = DreamCommunicator(self.stream, self.stim)

        # Register callbacks
        self.detector.on_dream_start(self._on_dream_start)
        self.detector.on_dream_end(self._on_dream_end)
        self.detector.on_lucid_detected(self._on_lucid)
        self.detector.on_dream_stable(self._on_dream_stable)

        # Session log
        self._log: list[dict] = []

    def _log_event(self, event_type: str, data: dict = None):
        entry = {
            "time": time.time(),
            "elapsed_min": round((time.time() - (self._session_start or time.time())) / 60, 1),
            "type": event_type,
            **(data or {}),
        }
        self._log.append(entry)
        print(f"[{entry['elapsed_min']:6.1f}m] {event_type}: {data or ''}")

    def _can_stimulate(self) -> bool:
        """Check if we're allowed to deliver stimulation."""
        now = time.time()
        elapsed_h = (now - (self._session_start or now)) / 3600

        # Respect quiet hours (natural sleep period)
        if elapsed_h < self.config.quiet_hours_end:
            return False

        # Rate limit
        if now - self._last_stim_time < self.config.min_gap_between_stims_s:
            return False

        # Hourly limit
        if now - self._stim_hour_start > 3600:
            self._stim_hour_start = now
            self._stim_count = 0
        if self._stim_count >= self.config.max_stimulations_per_hour:
            return False

        return True

    def _record_stim(self):
        self._last_stim_time = time.time()
        self._stim_count += 1

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _on_dream_start(self, dream: DreamEvent):
        self._log_event("dream_start", {"rem_period": dream.rem_period_number})

    def _on_dream_end(self, dream: DreamEvent):
        self._log_event("dream_end", {
            "duration_min": round(dream.duration_minutes, 1),
            "lucid_moments": dream.lucid_moments,
            "intensity": round(dream.peak_intensity, 2),
        })

    def _on_lucid(self, dream: DreamEvent):
        self._log_event("lucid_moment", {
            "gamma": round(dream.avg_gamma, 3),
            "lucid_count": dream.lucid_moments,
        })
        # If communication is enabled, try to talk to the dreamer
        if self.config.enable_communication and self._can_stimulate():
            self.comm.send_lucidity_check()
            self._record_stim()

    def _on_dream_stable(self, dream: DreamEvent):
        self._log_event("dream_stable", {"duration_min": round(dream.duration_minutes, 1)})

        if self.config.enable_lucid_induction and self._can_stimulate():
            self._log_event("lucid_induction", {"method": "40hz_gamma"})
            self.stim.induce_lucidity()
            self._record_stim()

    # ── Main loop ────────────────────────────────────────────────────────────

    def start(self):
        """Start the overnight session. Blocks until stopped."""
        print("=" * 60)
        print("DreamEngine Neural Session")
        print("=" * 60)
        print(f"Board: {self.config.board_type.value}")
        print(f"Lucid induction: {'ON' if self.config.enable_lucid_induction else 'OFF'}")
        print(f"TMR: {'ON' if self.config.enable_tmr else 'OFF'}")
        print(f"Communication: {'ON' if self.config.enable_communication else 'OFF'}")
        print()

        self.stream.start()
        self.stim.setup()
        self._session_start = time.time()
        self._stim_hour_start = time.time()
        self._running = True

        # Calibrate
        print("Calibrating — close your eyes and relax...")
        self.stager.calibrate(self.config.calibration_seconds)
        print("Calibration done. Goodnight.\n")

        try:
            while self._running:
                # Classify current epoch
                epoch = self.stager.classify_epoch()

                # Update dream detector
                self.detector.update(epoch)

                # TMR during N3
                if (epoch.stage == SleepStage.N3 and
                        self.config.enable_tmr and
                        self._can_stimulate() and
                        self.config.tmr_cue_paths):
                    cue = self.config.tmr_cue_paths[
                        self._stim_count % len(self.config.tmr_cue_paths)
                    ]
                    self._log_event("tmr_delivery", {"cue": cue})
                    self.stim.deliver_tmr_cue(cue)
                    self._record_stim()

                # Poll for dreamer signals during REM
                if (epoch.stage == SleepStage.REM and
                        self.config.enable_communication):
                    signal = self.comm.poll_signals()
                    if signal:
                        self._log_event("dreamer_signal", {
                            "signal": signal.signal.value,
                            "confidence": signal.confidence,
                        })

                # Log stage transitions
                if len(self.stager.session.epochs) >= 2:
                    prev = self.stager.session.epochs[-2].stage
                    if epoch.stage != prev:
                        self._log_event("stage_change", {
                            "from": prev.value,
                            "to": epoch.stage.value,
                            "confidence": epoch.confidence,
                        })

                # Wait for next epoch
                time.sleep(self.config.epoch_seconds)

        except KeyboardInterrupt:
            print("\n[Session] Interrupted — saving data...")
        finally:
            self._running = False
            self.stream.stop()
            self._save_session()

    def stop(self):
        """Stop the session (call from another thread)."""
        self._running = False

    def _save_session(self):
        """Save session data to disk."""
        data_dir = Path(self.config.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        session_file = data_dir / f"session_{ts}.json"

        session_data = {
            "start": datetime.fromtimestamp(
                self._session_start or time.time(), tz=timezone.utc
            ).isoformat(),
            "duration_hours": round(
                (time.time() - (self._session_start or time.time())) / 3600, 2
            ),
            "config": {
                "board": self.config.board_type.value,
                "lucid_induction": self.config.enable_lucid_induction,
                "tmr": self.config.enable_tmr,
                "communication": self.config.enable_communication,
                "scenario": self.config.tmr_scenario,
            },
            "sleep_summary": {
                "rem_periods": self.stager.session.rem_periods,
                "total_rem_min": round(self.stager.session.total_rem_minutes, 1),
                "total_deep_min": round(self.stager.session.total_deep_minutes, 1),
                "sleep_onset_latency_min": round(
                    ((self.stager.session.sleep_onset_time or time.time()) -
                     (self._session_start or time.time())) / 60, 1
                ),
            },
            "dreams": self.detector.get_dream_summary(),
            "communication": {
                "lucid_confirmed": self.comm.session.lucid_confirmed if self.comm.session else False,
                "signals": len(self.comm.session.signals_received) if self.comm.session else 0,
                "qa_log": self.comm.session.qa_log if self.comm.session else [],
            },
            "hypnogram": self.stager.get_hypnogram(),
            "event_log": self._log,
            "stimulations_delivered": self._stim_count,
        }

        session_file.write_text(json.dumps(session_data, indent=2, default=str))
        print(f"\n[Session] Data saved: {session_file}")

        # Auto-generate dream journal entries
        self._auto_journal(session_data)

    def _auto_journal(self, session_data: dict):
        """Auto-create dream journal entries from detected dream periods."""
        from dream_engine import recorder

        dreams = session_data.get("dreams", {}).get("dreams", [])
        for dream in dreams:
            recorder.record(
                title=f"Dream #{dream['number']} (auto-detected)",
                narrative=(
                    f"Auto-detected dream during REM period {dream['number']}. "
                    f"Duration: {dream['duration_min']} minutes. "
                    f"Peak intensity: {dream['intensity']}. "
                    f"{'Lucid moments detected!' if dream['lucid_moments'] > 0 else 'Non-lucid.'}"
                ),
                tags=["auto-detected", "neural"],
                mood="",
                vividness=max(1, min(10, int(dream["intensity"] * 10))),
                lucidity=min(5, dream["lucid_moments"]),
                notes=f"Session: {session_data['start']}",
                sleep_quality=7 if session_data["sleep_summary"]["total_deep_min"] > 30 else 5,
            )
            print(f"  [Journal] Auto-logged dream #{dream['number']}")
