"""dream_engine/neural/bci.py — Brain-Computer Interface abstraction layer.

Wraps brainflow to provide a clean interface for EEG data acquisition
across multiple hardware backends. Handles board setup, channel mapping,
streaming, and signal quality monitoring.

Supported boards:
  - OpenBCI Cyton (8ch, 250Hz) — best for full sleep staging
  - OpenBCI Ganglion (4ch, 200Hz) — good budget option
  - Muse 2 (4ch, 256Hz via BLE) — easiest to wear during sleep
  - NeuroSky MindWave (1ch, 512Hz) — minimal, basic REM detection only
  - Synthetic (8ch, 250Hz) — testing without hardware
"""
from __future__ import annotations

import time
import threading
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
    HAS_BRAINFLOW = True
except ImportError:
    HAS_BRAINFLOW = False


class BoardType(Enum):
    SYNTHETIC = "synthetic"
    OPENBCI_CYTON = "openbci_cyton"
    OPENBCI_GANGLION = "openbci_ganglion"
    MUSE_2 = "muse_2"
    MUSE_S = "muse_s"
    NEUROSKY = "neurosky"


# Standard 10-20 electrode positions mapped per board
CHANNEL_MAPS = {
    BoardType.OPENBCI_CYTON: {
        "Fp1": 0, "Fp2": 1, "C3": 2, "C4": 3,
        "O1": 4, "O2": 5, "F3": 6, "F4": 7,
    },
    BoardType.OPENBCI_GANGLION: {
        "Fp1": 0, "Fp2": 1, "O1": 2, "O2": 3,
    },
    BoardType.MUSE_2: {
        "TP9": 0, "AF7": 1, "AF8": 2, "TP10": 3,
    },
    BoardType.MUSE_S: {
        "TP9": 0, "AF7": 1, "AF8": 2, "TP10": 3,
    },
    BoardType.NEUROSKY: {
        "Fp1": 0,
    },
    BoardType.SYNTHETIC: {
        "Ch1": 0, "Ch2": 1, "Ch3": 2, "Ch4": 3,
        "Ch5": 4, "Ch6": 5, "Ch7": 6, "Ch8": 7,
    },
}

_BOARD_IDS = {
    BoardType.SYNTHETIC: "SYNTHETIC_BOARD",
    BoardType.OPENBCI_CYTON: "CYTON_BOARD",
    BoardType.OPENBCI_GANGLION: "GANGLION_BOARD",
    BoardType.MUSE_2: "MUSE_2_BOARD",
    BoardType.MUSE_S: "MUSE_S_BOARD",
    BoardType.NEUROSKY: "MINDWAVE_BOARD",
}


@dataclass
class BCIConfig:
    board_type: BoardType = BoardType.SYNTHETIC
    serial_port: str = ""          # e.g. /dev/ttyUSB0 for OpenBCI
    mac_address: str = ""          # for Muse BLE
    buffer_seconds: int = 30       # ring buffer size
    notch_filter_hz: float = 50.0  # mains frequency (50Hz UK, 60Hz US)


@dataclass
class SignalQuality:
    """Per-channel signal quality metrics."""
    channel: str
    impedance_ok: bool
    noise_uv: float          # RMS noise in microvolts
    artifact_ratio: float    # 0-1, fraction of samples with artifacts
    usable: bool


class BCIStream:
    """Manages a live EEG data stream from any supported board."""

    def __init__(self, config: BCIConfig):
        if not HAS_BRAINFLOW:
            raise RuntimeError(
                "brainflow not installed. Install with: pip install brainflow"
            )
        self.config = config
        self._board: Optional[BoardShim] = None
        self._params = BrainFlowInputParams()
        self._running = False
        self._callbacks: list[Callable] = []
        self._thread: Optional[threading.Thread] = None

        # Configure board params
        if config.serial_port:
            self._params.serial_port = config.serial_port
        if config.mac_address:
            self._params.mac_address = config.mac_address

        # Resolve board ID
        board_id_name = _BOARD_IDS.get(config.board_type, "SYNTHETIC_BOARD")
        self._board_id = getattr(BoardIds, board_id_name)

        # Get board info
        self.sample_rate = BoardShim.get_sampling_rate(self._board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(self._board_id)
        self.num_channels = len(self.eeg_channels)
        self.channel_map = CHANNEL_MAPS.get(config.board_type, {})

    def start(self):
        """Start the EEG data stream."""
        if self._running:
            return

        BoardShim.enable_dev_board_logger()
        self._board = BoardShim(self._board_id, self._params)
        self._board.prepare_session()
        self._board.start_stream(
            self.sample_rate * self.config.buffer_seconds
        )
        self._running = True
        print(f"[BCI] Stream started: {self.config.board_type.value} "
              f"({self.num_channels}ch @ {self.sample_rate}Hz)")

    def stop(self):
        """Stop the stream and release the board."""
        if not self._running:
            return
        self._running = False
        if self._board:
            try:
                self._board.stop_stream()
                self._board.release_session()
            except Exception:
                pass
        print("[BCI] Stream stopped")

    def get_data(self, seconds: float = 1.0) -> np.ndarray:
        """Get the last N seconds of EEG data.

        Returns: numpy array of shape (n_channels, n_samples)
        """
        if not self._board or not self._running:
            raise RuntimeError("Stream not running")

        n_samples = int(self.sample_rate * seconds)
        raw = self._board.get_current_board_data(n_samples)
        eeg = raw[self.eeg_channels]

        # Apply notch filter (remove mains hum) and detrend
        for ch in range(eeg.shape[0]):
            if eeg[ch].shape[0] > 20:  # need enough samples for filter
                try:
                    DataFilter.perform_bandstop(
                        eeg[ch], self.sample_rate,
                        self.config.notch_filter_hz, 4.0,
                        4, FilterTypes.BUTTERWORTH, 0
                    )
                except Exception:
                    pass  # skip filter if params don't suit this board
                DataFilter.detrend(eeg[ch], DetrendOperations.LINEAR)

        return eeg

    def get_bandpower(self, seconds: float = 2.0) -> dict[str, np.ndarray]:
        """Get power in standard frequency bands for all channels.

        Returns dict with keys: delta, theta, alpha, beta, gamma
        Each value is an array of power values per channel.
        """
        eeg = self.get_data(seconds)
        n_samples = eeg.shape[1]

        bands = {
            "delta": (0.5, 4.0),    # deep sleep (N3)
            "theta": (4.0, 8.0),    # light sleep (N1/N2), dreaming
            "alpha": (8.0, 13.0),   # relaxed wakefulness
            "beta": (13.0, 30.0),   # active thinking, alertness
            "gamma": (30.0, 50.0),  # lucidity, higher cognition
        }

        result = {}
        for band_name, (lo, hi) in bands.items():
            powers = []
            for ch in range(eeg.shape[0]):
                # Bandpass filter then compute RMS power
                filtered = eeg[ch].copy()
                try:
                    DataFilter.perform_bandpass(
                        filtered, self.sample_rate,
                        (lo + hi) / 2, hi - lo,
                        4, FilterTypes.BUTTERWORTH, 0
                    )
                except Exception:
                    pass  # use unfiltered if params don't suit
                power = float(np.sqrt(np.mean(filtered ** 2)))
                powers.append(power)
            result[band_name] = np.array(powers)

        return result

    def signal_quality(self) -> list[SignalQuality]:
        """Check signal quality for each channel."""
        try:
            eeg = self.get_data(2.0)
        except Exception:
            return []

        qualities = []
        for i, ch_idx in enumerate(self.eeg_channels):
            ch_name = list(self.channel_map.keys())[i] if i < len(self.channel_map) else f"Ch{i}"
            signal = eeg[i] if i < eeg.shape[0] else np.zeros(10)

            rms = float(np.sqrt(np.mean(signal ** 2)))
            # Good EEG is typically 10-100 uV. >200 uV suggests artifact
            artifact_mask = np.abs(signal) > 200
            artifact_ratio = float(np.mean(artifact_mask))

            qualities.append(SignalQuality(
                channel=ch_name,
                impedance_ok=rms > 1.0 and rms < 300,
                noise_uv=round(rms, 1),
                artifact_ratio=round(artifact_ratio, 3),
                usable=rms > 1.0 and artifact_ratio < 0.3,
            ))

        return qualities

    @property
    def is_running(self) -> bool:
        return self._running

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
