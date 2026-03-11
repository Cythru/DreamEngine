"""dream_engine.neural — Brain-computer interface for dream capture and injection.

Hardware support (via brainflow):
  - OpenBCI Cyton/Ganglion (best: 8-16 channel EEG)
  - Muse 2/S (4 channel EEG + PPG + accelerometer)
  - NeuroSky MindWave (1 channel, basic)
  - Synthetic board (testing without hardware)

Capabilities:
  - Real-time sleep stage classification (Wake/N1/N2/N3/REM)
  - Dream onset detection with confidence scoring
  - Lucid dream induction via 40Hz gamma entrainment
  - Targeted Memory Reactivation (TMR) for dream content planting
  - Two-way lucid dream communication (LRLR eye signal detection)
  - Emotional valence/arousal decoding from EEG during REM
  - Auto-journaling of dream periods with physiological data
"""
