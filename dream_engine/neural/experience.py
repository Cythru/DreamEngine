"""dream_engine/neural/experience.py — Dream experience recording and replay.

Inspired by Braindance (Cyberpunk 2077) and SQUID (Strange Days, 1995).

Records a complete dream experience as a multi-channel timeline:
  - EEG raw data (brainwaves)
  - Decoded emotional arc (valence + arousal over time)
  - Decoded content categories
  - Sleep stage transitions
  - Stimulation events (what cues were delivered)
  - Dreamer signals (eye patterns)
  - Audio environment

The recording can be:
  1. Replayed as stimulation — re-induce a similar dream state in the same
     or different person by replaying the associated TMR cues, binaural
     frequencies, and emotional arc as stimulation.
  2. Visualized as a "dream film" — timeline view of the emotional and
     content arc, with LLM-generated narrative.
  3. Shared — export the experience file for another DreamEngine user.

File format: .dreamxp (JSON + optional binary EEG data)
"""
from __future__ import annotations

import json
import time
import gzip
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .decoder import DreamFrame, Emotion, Arousal, SemanticCategory


_XP_DIR = Path.home() / ".local/share/blackmagic-gen/dreams/experiences"


@dataclass
class ExperienceFrame:
    """Single frame in a dream experience recording."""
    timestamp: float
    epoch_index: int
    # Decoded content
    emotion: str
    valence: float
    arousal: float
    categories: list[str]
    movement: str
    # Raw features
    delta_power: float
    theta_power: float
    alpha_power: float
    beta_power: float
    gamma_power: float
    frontal_asymmetry: float
    # Context
    sleep_stage: str
    lucidity_score: float
    stimulation_active: str       # what stim was playing, if any
    dreamer_signal: str            # eye signal detected, if any


@dataclass
class DreamExperience:
    """A complete recorded dream experience."""
    id: str
    recorded_at: str
    duration_seconds: float
    dreamer: str                   # identifier
    title: str
    frames: list[ExperienceFrame]
    # Metadata
    hardware: str
    sample_rate: int
    channels: int
    # Summaries
    emotional_arc: list[dict]      # simplified arc for quick viz
    content_summary: list[str]     # dominant categories
    peak_intensity: float
    lucid: bool
    lucid_duration_s: float
    # TMR cues used (for replay)
    tmr_cues_used: list[str]
    binaural_frequencies: list[dict]
    # Optional raw EEG (large, stored separately)
    raw_eeg_path: Optional[str] = None


class ExperienceRecorder:
    """Records dream experiences during a sleep session."""

    def __init__(self, dreamer: str = "default", hardware: str = "unknown"):
        self._dreamer = dreamer
        self._hardware = hardware
        self._frames: list[ExperienceFrame] = []
        self._recording = False
        self._start_time: Optional[float] = None
        self._raw_eeg_buffer: list[np.ndarray] = []
        self._tmr_cues: list[str] = []
        self._binaural_log: list[dict] = []

    def start_recording(self):
        self._recording = True
        self._start_time = time.time()
        self._frames.clear()
        self._raw_eeg_buffer.clear()
        print("[XP] Recording started")

    def stop_recording(self) -> DreamExperience:
        self._recording = False
        xp = self._build_experience()
        print(f"[XP] Recording stopped: {len(self._frames)} frames, "
              f"{xp.duration_seconds:.0f}s")
        return xp

    def add_frame(
        self,
        dream_frame: Optional[DreamFrame],
        sleep_stage: str,
        lucidity_score: float,
        stimulation: str = "",
        signal: str = "",
        raw_eeg: Optional[np.ndarray] = None,
    ):
        """Add a frame to the recording."""
        if not self._recording:
            return

        if dream_frame:
            frame = ExperienceFrame(
                timestamp=dream_frame.timestamp,
                epoch_index=len(self._frames),
                emotion=dream_frame.emotion.value,
                valence=dream_frame.valence_score,
                arousal=dream_frame.arousal_score,
                categories=[c.value for c in dream_frame.semantic_categories],
                movement=dream_frame.movement_type,
                delta_power=0.0,  # filled from epoch if available
                theta_power=0.0,
                alpha_power=0.0,
                beta_power=0.0,
                gamma_power=0.0,
                frontal_asymmetry=dream_frame.frontal_asymmetry,
                sleep_stage=sleep_stage,
                lucidity_score=lucidity_score,
                stimulation_active=stimulation,
                dreamer_signal=signal,
            )
        else:
            frame = ExperienceFrame(
                timestamp=time.time(),
                epoch_index=len(self._frames),
                emotion="neutral",
                valence=0.0,
                arousal=0.0,
                categories=["unknown"],
                movement="still",
                delta_power=0.0,
                theta_power=0.0,
                alpha_power=0.0,
                beta_power=0.0,
                gamma_power=0.0,
                frontal_asymmetry=0.0,
                sleep_stage=sleep_stage,
                lucidity_score=lucidity_score,
                stimulation_active=stimulation,
                dreamer_signal=signal,
            )

        self._frames.append(frame)

        if raw_eeg is not None:
            self._raw_eeg_buffer.append(raw_eeg)

    def log_tmr_cue(self, cue_path: str):
        self._tmr_cues.append(cue_path)

    def log_binaural(self, freq: float, duration: float, purpose: str):
        self._binaural_log.append({
            "freq": freq,
            "duration": duration,
            "purpose": purpose,
            "time": time.time(),
        })

    def _build_experience(self) -> DreamExperience:
        """Build a DreamExperience from recorded frames."""
        import uuid

        duration = (time.time() - self._start_time) if self._start_time else 0

        # Build emotional arc (sampled, not every frame)
        arc = []
        step = max(1, len(self._frames) // 30)  # ~30 points
        for i in range(0, len(self._frames), step):
            f = self._frames[i]
            arc.append({
                "t": round(f.timestamp - (self._start_time or 0), 1),
                "v": f.valence,
                "a": f.arousal,
                "e": f.emotion,
            })

        # Content summary
        all_cats = []
        for f in self._frames:
            all_cats.extend(c for c in f.categories if c != "unknown")
        from collections import Counter
        cat_counts = Counter(all_cats)
        top_cats = [c for c, _ in cat_counts.most_common(5)]

        # Lucidity
        lucid_frames = [f for f in self._frames if f.lucidity_score > 0.3]
        lucid = len(lucid_frames) > 2
        lucid_duration = len(lucid_frames) * 30  # ~30s per epoch

        # Peak intensity
        peak = max((f.arousal for f in self._frames), default=0)

        xp_id = uuid.uuid4().hex[:12]

        return DreamExperience(
            id=xp_id,
            recorded_at=datetime.now(timezone.utc).isoformat(),
            duration_seconds=round(duration, 1),
            dreamer=self._dreamer,
            title=f"Dream Experience {xp_id[:6]}",
            frames=[f.__dict__ for f in self._frames],
            hardware=self._hardware,
            sample_rate=250,
            channels=8,
            emotional_arc=arc,
            content_summary=top_cats,
            peak_intensity=round(peak, 3),
            lucid=lucid,
            lucid_duration_s=lucid_duration,
            tmr_cues_used=list(set(self._tmr_cues)),
            binaural_frequencies=self._binaural_log,
        )


def save_experience(xp: DreamExperience) -> Path:
    """Save a dream experience to disk."""
    _XP_DIR.mkdir(parents=True, exist_ok=True)
    path = _XP_DIR / f"{xp.id}.dreamxp"

    data = {
        "version": 1,
        "id": xp.id,
        "recorded_at": xp.recorded_at,
        "duration_seconds": xp.duration_seconds,
        "dreamer": xp.dreamer,
        "title": xp.title,
        "hardware": xp.hardware,
        "sample_rate": xp.sample_rate,
        "channels": xp.channels,
        "emotional_arc": xp.emotional_arc,
        "content_summary": xp.content_summary,
        "peak_intensity": xp.peak_intensity,
        "lucid": xp.lucid,
        "lucid_duration_s": xp.lucid_duration_s,
        "tmr_cues_used": xp.tmr_cues_used,
        "binaural_frequencies": xp.binaural_frequencies,
        "frames": xp.frames,
    }

    # Compress — dream recordings can be large
    compressed = gzip.compress(json.dumps(data, default=str).encode())
    path.write_bytes(compressed)
    print(f"[XP] Saved: {path} ({len(compressed)} bytes)")
    return path


def load_experience(path: str | Path) -> DreamExperience:
    """Load a dream experience from disk."""
    p = Path(path)
    raw = gzip.decompress(p.read_bytes())
    data = json.loads(raw)

    return DreamExperience(
        id=data["id"],
        recorded_at=data["recorded_at"],
        duration_seconds=data["duration_seconds"],
        dreamer=data["dreamer"],
        title=data["title"],
        frames=data["frames"],
        hardware=data["hardware"],
        sample_rate=data["sample_rate"],
        channels=data["channels"],
        emotional_arc=data["emotional_arc"],
        content_summary=data["content_summary"],
        peak_intensity=data["peak_intensity"],
        lucid=data["lucid"],
        lucid_duration_s=data["lucid_duration_s"],
        tmr_cues_used=data["tmr_cues_used"],
        binaural_frequencies=data["binaural_frequencies"],
    )


def list_experiences() -> list[dict]:
    """List all saved dream experiences."""
    if not _XP_DIR.exists():
        return []
    results = []
    for path in sorted(_XP_DIR.glob("*.dreamxp"), reverse=True):
        try:
            xp = load_experience(path)
            results.append({
                "id": xp.id,
                "title": xp.title,
                "recorded_at": xp.recorded_at,
                "duration_s": xp.duration_seconds,
                "lucid": xp.lucid,
                "peak_intensity": xp.peak_intensity,
                "content": xp.content_summary,
                "path": str(path),
            })
        except Exception:
            continue
    return results


class ExperienceReplayer:
    """Replay a recorded dream experience as stimulation.

    Takes a .dreamxp file and converts its emotional arc + content
    back into stimulation parameters:
      - Binaural beats matching the original theta/gamma profile
      - TMR cues from the original session
      - Emotional tone via frequency selection
      - Lucidity triggers at the same relative timepoints

    This is the "Braindance" — re-experiencing someone else's dream.
    Won't be identical, but the emotional/thematic contour should transfer.
    """

    def __init__(self, stim):
        self.stim = stim  # Stimulator instance

    def build_replay_protocol(self, xp: DreamExperience) -> list[dict]:
        """Convert a dream experience into a replay stimulation protocol.

        Returns a list of timed stimulation events.
        """
        protocol = []
        total_duration = xp.duration_seconds

        for point in xp.emotional_arc:
            t = point["t"]
            valence = point["v"]
            arousal = point["a"]

            # Map emotional state to binaural frequency
            if arousal > 0.7:
                # High arousal: beta-range binaural (exciting)
                beat_freq = 18.0 + arousal * 10  # 18-28 Hz
            elif arousal > 0.4:
                # Moderate: alpha-theta border
                beat_freq = 8.0 + arousal * 6  # 8-14 Hz
            else:
                # Low arousal: deep theta
                beat_freq = 4.0 + arousal * 4  # 4-8 Hz

            # Valence affects base frequency (lower = warmer)
            base_freq = 180 + valence * 40  # 140-220 Hz

            protocol.append({
                "time_offset_s": t,
                "type": "binaural",
                "base_freq": round(base_freq, 1),
                "beat_freq": round(beat_freq, 1),
                "duration_s": 30.0,
                "volume": 0.08 + arousal * 0.07,  # louder when more intense
            })

        # Add TMR cues at intervals if available
        if xp.tmr_cues_used:
            cue_interval = total_duration / (len(xp.tmr_cues_used) + 1)
            for i, cue in enumerate(xp.tmr_cues_used):
                protocol.append({
                    "time_offset_s": cue_interval * (i + 1),
                    "type": "tmr_cue",
                    "path": cue,
                    "volume": 0.15,
                })

        # Add lucidity triggers at moments where original was lucid
        if xp.lucid:
            for point in xp.emotional_arc:
                # Proxy: high gamma moments in the original
                if point.get("a", 0) > 0.6:
                    protocol.append({
                        "time_offset_s": point["t"],
                        "type": "gamma_pulse",
                        "duration_s": 15.0,
                        "volume": 0.03,
                    })

        protocol.sort(key=lambda x: x["time_offset_s"])
        return protocol

    def replay(self, xp: DreamExperience):
        """Execute a replay protocol. Should be called during stable REM."""
        protocol = self.build_replay_protocol(xp)
        print(f"[Replay] Playing back dream experience: {xp.title}")
        print(f"  {len(protocol)} stimulation events over {xp.duration_seconds:.0f}s")

        start = time.time()
        event_idx = 0

        while event_idx < len(protocol):
            elapsed = time.time() - start
            event = protocol[event_idx]

            if elapsed >= event["time_offset_s"]:
                if event["type"] == "binaural":
                    audio = self.stim.audio.generate_binaural(
                        base_freq=event["base_freq"],
                        beat_freq=event["beat_freq"],
                        duration_s=event["duration_s"],
                    )
                    self.stim.audio.play_audio(audio)
                elif event["type"] == "tmr_cue":
                    self.stim.deliver_tmr_cue(event["path"])
                elif event["type"] == "gamma_pulse":
                    audio = self.stim.audio.generate_gamma_pulse(
                        duration_s=event["duration_s"]
                    )
                    self.stim.audio.play_audio(audio)

                event_idx += 1
            else:
                time.sleep(1.0)
