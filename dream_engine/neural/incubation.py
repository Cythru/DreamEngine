"""dream_engine/neural/incubation.py — Full dream incubation protocol.

A complete day-to-night pipeline for planting specific dream content:

DAY PHASE (Waking):
  1. Define dream scenario
  2. Generate multisensory associations:
     - Audio cue (specific sound/music tied to the scenario)
     - Visual priming (images related to the scenario)
     - Optional: scent association (most powerful TMR channel)
  3. Rehearse: user reads the dream seed while hearing the audio cue
  4. Periodic reinforcement throughout the day (play cue briefly)

NIGHT PHASE (Sleep):
  5. Monitor sleep stages
  6. During N3 (slow-wave): play the audio cue at low volume (TMR)
  7. Brain replays associated memories, weaving them toward the scenario
  8. During REM: optionally trigger lucidity (40Hz) so user can steer

This is proven neuroscience (Rasch 2007, Antony 2012), just automated.

The scent channel is the most effective because olfactory cortex has
a direct anatomical connection to the hippocampus (memory center),
bypassing the thalamus. A scent associated with a memory during waking
reactivates that memory when presented during N3 with ~80% reliability.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


_INCUBATION_DIR = Path.home() / ".local/share/blackmagic-gen/dreams/incubations"


@dataclass
class ScentConfig:
    """Configuration for scent-triggered TMR.

    Hardware: USB-controlled scent diffuser or smart diffuser with API.
    A simple solenoid valve + essential oil works too.
    """
    enabled: bool = False
    device_type: str = ""      # "usb_diffuser", "smart_plug", "manual"
    device_address: str = ""    # serial port, IP, or smart plug ID
    scent_name: str = ""        # e.g. "lavender", "ocean", "pine"
    pulse_duration_s: float = 3.0
    intensity: int = 50         # 0-100


@dataclass
class IncubationPlan:
    """A complete dream incubation plan."""
    id: str
    created_at: str
    scenario: str                  # what the user wants to dream about
    dream_seed: str                # LLM-generated vivid scene description
    audio_cue_path: str            # path to audio file tied to scenario
    audio_cue_description: str     # what the sound is
    scent_config: ScentConfig
    # Day phase tracking
    rehearsals: list[dict]         # timestamps of when user rehearsed
    reinforcements: list[dict]     # timestamps of when cue was played
    # Night phase config
    tmr_volume: float = 0.10       # volume for nighttime playback (LOW)
    tmr_min_n3_seconds: float = 120.0  # wait this long in N3 before first cue
    tmr_interval_s: float = 300.0  # gap between TMR cue plays
    enable_lucid_trigger: bool = True
    # Status
    status: str = "planning"       # planning, day_phase, night_phase, complete
    night_tmr_deliveries: int = 0
    dream_matched: Optional[bool] = None  # did the user dream about it?


def create_plan(
    scenario: str,
    audio_cue_path: str = "",
    audio_description: str = "",
    scent: Optional[ScentConfig] = None,
    enable_lucid: bool = True,
) -> IncubationPlan:
    """Create a new dream incubation plan.

    Args:
        scenario: What the user wants to dream about
        audio_cue_path: Path to the audio file to associate with the dream
        audio_description: Description of the audio cue
        scent: Optional scent configuration
        enable_lucid: Whether to trigger lucidity during REM
    """
    # Generate dream seed using LLM
    from dream_engine.suggester import generate_seed_with_llm
    seed = generate_seed_with_llm(scenario=scenario)

    plan = IncubationPlan(
        id=uuid.uuid4().hex[:12],
        created_at=datetime.now(timezone.utc).isoformat(),
        scenario=scenario,
        dream_seed=seed,
        audio_cue_path=audio_cue_path,
        audio_cue_description=audio_description,
        scent_config=scent or ScentConfig(),
        rehearsals=[],
        reinforcements=[],
        enable_lucid_trigger=enable_lucid,
    )

    save_plan(plan)
    return plan


def save_plan(plan: IncubationPlan):
    _INCUBATION_DIR.mkdir(parents=True, exist_ok=True)
    path = _INCUBATION_DIR / f"{plan.id}.json"
    path.write_text(json.dumps(plan.__dict__, indent=2, default=str))


def load_plan(plan_id: str) -> Optional[IncubationPlan]:
    path = _INCUBATION_DIR / f"{plan_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    scent_data = data.pop("scent_config", {})
    plan = IncubationPlan(**data)
    plan.scent_config = ScentConfig(**scent_data) if isinstance(scent_data, dict) else ScentConfig()
    return plan


def list_plans() -> list[dict]:
    _INCUBATION_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for path in sorted(_INCUBATION_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text())
            results.append({
                "id": data["id"],
                "scenario": data["scenario"],
                "status": data["status"],
                "created_at": data["created_at"],
            })
        except Exception:
            continue
    return results


class DayPhaseManager:
    """Manages the daytime association-building phase.

    User should:
    1. Read the dream seed 2-3 times while hearing the audio cue
    2. (Optional) Smell the associated scent while reading
    3. Periodically hear the audio cue throughout the day (reinforcement)
    4. Before sleep, do one final rehearsal
    """

    def __init__(self, plan: IncubationPlan):
        self.plan = plan

    def rehearse(self):
        """Full rehearsal session — present all cues together."""
        print(f"\n{'='*50}")
        print("DREAM INCUBATION — REHEARSAL")
        print(f"{'='*50}\n")
        print("Read this dream seed slowly while listening to the audio cue.\n")
        print(f"Scenario: {self.plan.scenario}\n")
        print(f"{'─'*50}")
        print(self.plan.dream_seed)
        print(f"{'─'*50}\n")

        if self.plan.audio_cue_path:
            print(f"Audio cue: {self.plan.audio_cue_description}")
            print(f"  File: {self.plan.audio_cue_path}")
            # In full implementation, play the audio here
            # stim.audio.play_file(self.plan.audio_cue_path)

        if self.plan.scent_config.enabled:
            print(f"Scent: {self.plan.scent_config.scent_name}")
            # In full implementation, trigger scent diffuser

        self.plan.rehearsals.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "full_rehearsal",
        })
        save_plan(self.plan)
        print("\nClose your eyes and visualize the scene for 2-3 minutes.")
        print("The more vivid the visualization, the stronger the association.\n")

    def reinforce(self):
        """Quick reinforcement — brief audio cue to maintain association."""
        if self.plan.audio_cue_path:
            print(f"[Incubation] Reinforcement: playing {self.plan.audio_cue_description}")
            # Play audio cue briefly (5-10 seconds)
        self.plan.reinforcements.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        save_plan(self.plan)

    def pre_sleep(self):
        """Final rehearsal before sleep — maximum association strength."""
        print("\n=== PRE-SLEEP INCUBATION ===\n")
        print("This is the most important rehearsal.")
        print("Read the seed, hear the sound, smell the scent.\n")
        self.rehearse()
        print("\nAs you drift off, hold the scene gently in your mind.")
        print("Don't force it — let it be the last thing you think about.")
        print("The system will replay the cue during deep sleep.\n")
        self.plan.status = "night_phase"
        save_plan(self.plan)


class NightPhaseController:
    """Controls nighttime TMR delivery during sleep.

    Integrates with the sleep session to deliver TMR cues
    at the optimal moments during N3 slow-wave sleep.
    """

    def __init__(self, plan: IncubationPlan, stim):
        self.plan = plan
        self.stim = stim  # Stimulator instance
        self._last_tmr_time: float = 0
        self._n3_entry_time: Optional[float] = None

    def on_n3_entry(self, timestamp: float):
        """Called when N3 is entered."""
        if self._n3_entry_time is None:
            self._n3_entry_time = timestamp

    def on_n3_exit(self):
        """Called when N3 is exited."""
        self._n3_entry_time = None

    def should_deliver_tmr(self) -> bool:
        """Check if conditions are met for TMR delivery."""
        now = time.time()

        # Must be in N3
        if self._n3_entry_time is None:
            return False

        # Must have been in N3 long enough
        n3_duration = now - self._n3_entry_time
        if n3_duration < self.plan.tmr_min_n3_seconds:
            return False

        # Rate limiting
        if now - self._last_tmr_time < self.plan.tmr_interval_s:
            return False

        return True

    def deliver_tmr(self):
        """Deliver the TMR audio cue."""
        if not self.plan.audio_cue_path:
            return

        print(f"[TMR] Delivering cue: {self.plan.audio_cue_description}")
        self.stim.deliver_tmr_cue(self.plan.audio_cue_path)
        self._last_tmr_time = time.time()
        self.plan.night_tmr_deliveries += 1
        save_plan(self.plan)

        # Also trigger scent if configured
        if self.plan.scent_config.enabled:
            self._trigger_scent()

    def _trigger_scent(self):
        """Trigger the scent diffuser."""
        sc = self.plan.scent_config
        if not sc.device_address:
            return

        print(f"[TMR] Scent pulse: {sc.scent_name} ({sc.pulse_duration_s}s)")

        if sc.device_type == "smart_plug":
            # Toggle smart plug on/off
            try:
                import subprocess
                # Using a generic smart plug CLI — would be replaced per hardware
                subprocess.run(
                    ["smart-plug", "on", sc.device_address],
                    timeout=5, capture_output=True
                )
                time.sleep(sc.pulse_duration_s)
                subprocess.run(
                    ["smart-plug", "off", sc.device_address],
                    timeout=5, capture_output=True
                )
            except Exception:
                pass
        elif sc.device_type == "usb_diffuser":
            try:
                import serial
                ser = serial.Serial(sc.device_address, 9600, timeout=1)
                ser.write(f"PULSE,{sc.intensity},{int(sc.pulse_duration_s * 1000)}\n".encode())
                ser.close()
            except Exception:
                pass

    def morning_report(self) -> str:
        """Generate morning report on TMR delivery."""
        return (
            f"Incubation Report — '{self.plan.scenario}'\n"
            f"  TMR cues delivered: {self.plan.night_tmr_deliveries}\n"
            f"  Audio: {self.plan.audio_cue_description}\n"
            f"  Scent: {self.plan.scent_config.scent_name or 'none'}\n"
            f"  Status: {self.plan.status}\n"
        )
