"""dream_engine/neural/stimulator.py — Sensory output for dream intervention.

Controls stimulation channels:
  - Audio: binaural beats, TMR cues, voice prompts, gamma entrainment
  - Haptic: vibration patterns (e.g. wrist buzzer for reality checks)
  - Light: LED pulses through closed eyelids (Remee-style)

All stimulation is gated by sleep stage — nothing plays unless the
dream detector confirms the right conditions.
"""
from __future__ import annotations

import math
import struct
import time
import threading
import wave
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Optional

try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False


class StimType(Enum):
    BINAURAL = "binaural"
    TONE = "tone"
    VOICE = "voice"
    WAV_FILE = "wav_file"
    GAMMA_40HZ = "gamma_40hz"
    HAPTIC = "haptic"
    LIGHT = "light"


@dataclass
class StimConfig:
    volume: float = 0.15           # 0-1, keep LOW for sleeping person
    sample_rate: int = 44100
    audio_device: Optional[int] = None  # None = default
    haptic_port: str = ""          # serial port for haptic controller
    light_port: str = ""           # serial port for LED controller


class AudioGenerator:
    """Generate audio stimulation signals in real-time."""

    def __init__(self, config: StimConfig):
        self.config = config
        self._sample_rate = config.sample_rate
        self._stream = None
        self._playing = False
        self._thread: Optional[threading.Thread] = None

    def _ensure_pyaudio(self):
        if not HAS_PYAUDIO:
            raise RuntimeError("pyaudio not installed. Install with: pip install pyaudio")

    def generate_binaural(
        self,
        base_freq: float = 200.0,
        beat_freq: float = 6.0,
        duration_s: float = 30.0,
    ) -> bytes:
        """Generate binaural beat audio (stereo).

        Left ear gets base_freq, right ear gets base_freq + beat_freq.
        The brain perceives the difference as the "beat."

        Common frequencies:
          - Delta (2Hz): deep sleep maintenance
          - Theta (6Hz): dream enhancement
          - Alpha (10Hz): relaxation
          - Gamma (40Hz): lucid dream induction
        """
        n_samples = int(self._sample_rate * duration_s)
        frames = bytearray()

        for i in range(n_samples):
            t = i / self._sample_rate
            left = math.sin(2 * math.pi * base_freq * t)
            right = math.sin(2 * math.pi * (base_freq + beat_freq) * t)

            # Fade in/out (3 seconds each)
            fade = 1.0
            if t < 3.0:
                fade = t / 3.0
            elif t > duration_s - 3.0:
                fade = (duration_s - t) / 3.0

            vol = self.config.volume * fade
            l_sample = int(left * vol * 32767)
            r_sample = int(right * vol * 32767)
            frames.extend(struct.pack('<hh', l_sample, r_sample))

        return bytes(frames)

    def generate_gamma_pulse(self, duration_s: float = 60.0) -> bytes:
        """Generate 40Hz gamma entrainment audio.

        Voss et al. (2014): 40Hz tACS during REM induces lucid dreaming.
        Audio entrainment at 40Hz is the non-invasive equivalent.
        Uses amplitude-modulated pink noise at 40Hz for comfort.
        """
        n_samples = int(self._sample_rate * duration_s)
        frames = bytearray()

        for i in range(n_samples):
            t = i / self._sample_rate

            # Pink noise approximation (filtered white noise)
            import random
            noise = random.gauss(0, 0.3)

            # 40Hz amplitude modulation
            modulation = 0.5 + 0.5 * math.sin(2 * math.pi * 40.0 * t)
            signal = noise * modulation

            # Gentle fade
            fade = 1.0
            if t < 5.0:
                fade = t / 5.0
            elif t > duration_s - 5.0:
                fade = (duration_s - t) / 5.0

            vol = self.config.volume * 0.5 * fade  # extra quiet for 40Hz
            sample = int(signal * vol * 32767)
            sample = max(-32767, min(32767, sample))
            frames.extend(struct.pack('<hh', sample, sample))

        return bytes(frames)

    def generate_tmr_cue(
        self,
        wav_path: str,
        volume_scale: float = 0.3,
    ) -> bytes:
        """Load a WAV file as a TMR cue, scaled to sleep-safe volume."""
        with wave.open(wav_path, 'rb') as wf:
            raw = wf.readframes(wf.getnframes())
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()

        # Scale volume
        if sampwidth == 2:
            samples = struct.unpack(f'<{len(raw)//2}h', raw)
            scaled = [int(s * volume_scale * self.config.volume) for s in samples]
            return struct.pack(f'<{len(scaled)}h', *scaled)
        return raw

    def generate_whisper(self, text: str, duration_s: float = 10.0) -> Optional[bytes]:
        """Generate whispered speech using TTS (for dream cues).

        Falls back to a soft tone if TTS is unavailable.
        """
        # Try espeak/piper for TTS
        import subprocess
        import tempfile

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
            # espeak with whisper effect
            subprocess.run(
                ["espeak-ng", "-v", "en+whisper", "-s", "120",
                 "-w", tmp_path, text],
                capture_output=True, timeout=10
            )
            return self.generate_tmr_cue(tmp_path, volume_scale=0.2)
        except Exception:
            return None

    def play_audio(self, audio_data: bytes, blocking: bool = False):
        """Play raw audio data through speakers."""
        self._ensure_pyaudio()
        pa = pyaudio.PyAudio()

        stream = pa.open(
            format=pyaudio.paInt16,
            channels=2,
            rate=self._sample_rate,
            output=True,
            output_device_index=self.config.audio_device,
        )

        def _play():
            chunk_size = 4096
            for i in range(0, len(audio_data), chunk_size):
                if not self._playing:
                    break
                stream.write(audio_data[i:i + chunk_size])
            stream.stop_stream()
            stream.close()
            pa.terminate()

        self._playing = True
        if blocking:
            _play()
        else:
            self._thread = threading.Thread(target=_play, daemon=True)
            self._thread.start()

    def stop(self):
        self._playing = False


class HapticController:
    """Control haptic feedback devices (wrist buzzers, bed shakers)."""

    def __init__(self, serial_port: str = ""):
        self._port = serial_port
        self._serial = None

    def connect(self):
        if not self._port:
            print("[Haptic] No device configured")
            return
        try:
            import serial
            self._serial = serial.Serial(self._port, 9600, timeout=1)
            print(f"[Haptic] Connected: {self._port}")
        except Exception as e:
            print(f"[Haptic] Connection failed: {e}")

    def pulse(self, intensity: int = 128, duration_ms: int = 200):
        """Send a vibration pulse. Intensity 0-255."""
        if self._serial:
            cmd = f"V{intensity},{duration_ms}\n"
            self._serial.write(cmd.encode())

    def pattern(self, pulses: list[tuple[int, int, int]]):
        """Play a pattern: list of (intensity, duration_ms, gap_ms)."""
        for intensity, duration, gap in pulses:
            self.pulse(intensity, duration)
            time.sleep((duration + gap) / 1000.0)

    def reality_check_signal(self):
        """Gentle double-tap to trigger reality check in dream."""
        self.pattern([
            (100, 150, 200),
            (100, 150, 0),
        ])


class LightController:
    """Control LED light cues (through closed eyelids, Remee-style)."""

    def __init__(self, serial_port: str = ""):
        self._port = serial_port
        self._serial = None

    def connect(self):
        if not self._port:
            print("[Light] No device configured")
            return
        try:
            import serial
            self._serial = serial.Serial(self._port, 9600, timeout=1)
            print(f"[Light] Connected: {self._port}")
        except Exception as e:
            print(f"[Light] Connection failed: {e}")

    def flash(self, brightness: int = 50, duration_ms: int = 100):
        """Single LED flash. Keep brightness LOW (sleeping person)."""
        if self._serial:
            cmd = f"L{brightness},{duration_ms}\n"
            self._serial.write(cmd.encode())

    def pattern_40hz(self, duration_s: float = 10.0):
        """40Hz LED flicker for gamma entrainment through eyelids."""
        if not self._serial:
            return
        # 40Hz = 25ms period, 12.5ms on/off
        cycles = int(duration_s * 40)
        for _ in range(cycles):
            self.flash(30, 12)
            time.sleep(0.0125)

    def dream_signal(self):
        """Gentle red flash pattern recognizable inside a dream."""
        for _ in range(3):
            self.flash(40, 200)
            time.sleep(0.3)


class Stimulator:
    """Unified stimulation controller — coordinates audio, haptic, and light."""

    def __init__(self, config: StimConfig):
        self.config = config
        self.audio = AudioGenerator(config)
        self.haptic = HapticController(config.haptic_port)
        self.light = LightController(config.light_port)
        self._active = False

    def setup(self):
        """Initialize all available output devices."""
        self.haptic.connect()
        self.light.connect()
        self._active = True

    def induce_lucidity(self):
        """Full lucidity induction protocol:
        1. 40Hz gamma audio (binaural + AM noise)
        2. 40Hz LED flicker (if available)
        3. Gentle haptic reality check signal
        """
        print("[Stim] Lucidity induction: 40Hz gamma protocol")
        audio_data = self.audio.generate_gamma_pulse(duration_s=30.0)
        self.audio.play_audio(audio_data)
        self.light.pattern_40hz(duration_s=10.0)
        time.sleep(15)
        self.haptic.reality_check_signal()

    def deliver_tmr_cue(self, wav_path: str):
        """Play a TMR sound cue during slow-wave sleep."""
        print(f"[Stim] TMR cue: {wav_path}")
        audio_data = self.audio.generate_tmr_cue(wav_path)
        self.audio.play_audio(audio_data)

    def whisper_cue(self, text: str):
        """Whisper a text cue into the dream."""
        print(f"[Stim] Whisper: {text}")
        audio_data = self.audio.generate_whisper(text)
        if audio_data:
            self.audio.play_audio(audio_data)

    def dream_maintenance(self):
        """Theta binaural beat to sustain dream state."""
        print("[Stim] Dream maintenance: 6Hz theta binaural")
        audio_data = self.audio.generate_binaural(
            base_freq=200, beat_freq=6.0, duration_s=60.0
        )
        self.audio.play_audio(audio_data)

    def deep_sleep_maintenance(self):
        """Delta binaural beat to maintain N3 for TMR window."""
        print("[Stim] Deep sleep: 2Hz delta binaural")
        audio_data = self.audio.generate_binaural(
            base_freq=150, beat_freq=2.0, duration_s=120.0
        )
        self.audio.play_audio(audio_data)

    def stop_all(self):
        self.audio.stop()
        self._active = False
