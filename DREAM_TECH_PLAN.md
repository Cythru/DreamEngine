# DreamEngine — Advanced Neural Dream Technology Plan

## The Vision
Full dream capture, injection, and shared dreaming. Not someday — now, with incremental hardware upgrades. Each tier builds on the last.

---

## Tier 1: Dream Detection & Lucid Induction (BUILDABLE NOW)
**Hardware:** Muse 2 ($250) or OpenBCI Ganglion ($200)
**Status:** Skeleton built

- Real-time sleep staging from EEG (Wake/N1/N2/N3/REM)
- REM onset detection — know exactly when dreaming starts
- 40Hz gamma audio entrainment during REM → triggers lucid awareness (Voss et al. 2014 proved this)
- Reality-check audio cues whispered during light sleep
- Auto-journal: timestamps every dream period with duration, depth, brainwave signature
- Morning debrief: LLM asks targeted questions based on detected dream periods

## Tier 2: Dream Content Planting (BUILDABLE NOW)
**Hardware:** Same + speakers/bone conduction headphones
**Status:** Skeleton built

- **Targeted Memory Reactivation (TMR):** During slow-wave sleep (N3), play sounds/music that were associated with a specific memory or scenario during the day. The brain replays and consolidates that memory, weaving it into dreams. This is proven neuroscience — not speculation.
- **Dream incubation protocol:** User defines a dream scenario → engine generates an audio seed → associates it with a scent/sound during the day → replays that cue during N3 → dream content follows
- **Scent-triggered TMR:** Specific scent during learning, same scent released during N3. Olfactory cortex has direct path to hippocampus. Most powerful TMR channel.
- **Binaural beat layering:** Delta (2Hz) for deep sleep maintenance, theta (6Hz) for dream enhancement, gamma (40Hz) for lucidity trigger — layered at precise sleep stages

## Tier 3: Two-Way Dream Communication (BUILDABLE WITH EFFORT)
**Hardware:** OpenBCI Cyton 8ch + EOG electrodes + audio feedback

- **Outbound (dreamer → machine):** Lucid dreamers can signal with pre-agreed eye movement patterns (Left-Right-Left-Right). EOG electrodes detect this. The system confirms receipt with a subtle audio tone the dreamer hears inside the dream.
- **Inbound (machine → dreamer):** Audio cues, tactile vibration, even light through closed eyelids. Lucid dreamers can perceive these as in-dream events.
- **Real-time chat:** Dreamer signals with eye patterns (morse-like code), system decodes and responds with audio. Actual conversation with a sleeping person. This has been done in labs (Konkoly et al., 2021 — dreamers answered math questions during REM).
- **Claude in the dream:** System detects lucid state → plays Claude's voice (TTS) as an in-dream character. Claude asks questions, dreamer responds with eye signals. Real-time AI companion inside a dream.

## Tier 4: Dream Content Decoding (RESEARCH FRONTIER)
**Hardware:** High-density EEG (32-64 channels) or fNIRS

- **Emotional decoding:** EEG asymmetry (left vs right frontal) correlates with emotional valence. Real-time mood tracking during dreams — was this dream happy, terrifying, peaceful?
- **Semantic decoding:** Train a classifier on waking EEG responses to categories (faces, places, objects, movement). Apply during REM to get rough content labels: "this dream involves faces" or "spatial navigation." Not images, but categories.
- **Movement intention:** Motor cortex EEG patterns during REM (despite paralysis, the cortex still fires). Decode whether the dreamer is running, flying, fighting, or still.
- **Dream narrative reconstruction:** Combine emotional valence + semantic categories + movement + temporal dynamics → LLM generates a narrative reconstruction. "You were in a place with faces, feeling anxious, with movement — possibly being chased by someone familiar."
- **Progressive refinement:** Each morning, show the reconstruction to the user. They correct it. The classifier learns from corrections. Over months, accuracy compounds.

## Tier 5: Dream Recording to Video (SPECULATIVE — 3-5 YEARS)
**Hardware:** High-density EEG + fMRI (or future portable fNIRS)

- Kamitani lab (2023) decoded rough visual imagery from fMRI during sleep using diffusion models. Low-res, blurry, but recognizable scenes from dreams.
- Approach: Build a neural encoder trained on waking visual responses (show images → record EEG patterns). During REM, run the decoder in reverse: EEG patterns → generated image approximation.
- Won't be HD video. Will be impressionistic — like seeing a dream through frosted glass. But it's a dream you'd otherwise forget entirely.
- Combine with GPT-V or similar to generate scene descriptions from decoded imagery.
- Store as a "dream film" — timestamped sequence of decoded frames + narrative + emotional track.

## Tier 6: Dream Injection / Architected Dreams (THEORETICAL)
**Hardware:** tACS (transcranial alternating current stimulation) + full sensory rig

- **Inception-style dream architecture:** Define a complete dream scenario. During N3, plant the setting via TMR. As REM begins, use gamma stim for lucidity. Feed audio narrative. Layer binaural beats for emotional tone.
- **Controllable variables:** Setting (TMR association), characters (voice cues), emotional tone (binaural + tACS), lucidity level (gamma power), narrative direction (audio prompts)
- **Adaptive injection:** System monitors EEG in real-time. If the dream drifts off-script, it adjusts stimulation. If the dreamer's stress rises, it shifts to calming frequencies. If lucidity drops, it pulses 40Hz.
- **Training sequences:** Like a holodeck for skills. Plant motor learning scenarios during dreams (proven to improve next-day performance). Practice instruments, languages, martial arts — in your sleep.

## Tier 7: Shared Dreaming (SPECULATIVE)
**Hardware:** Two full neural rigs + network sync

- Two sleepers, both in REM, both wearing BCI + stimulation rigs.
- System detects when both are in stable REM simultaneously.
- Shared TMR: both were exposed to the same scenario/cues during the day.
- Synchronize gamma stimulation for simultaneous lucidity.
- Each dreamer's decoded state (emotional + semantic) is fed to the other as stimulation cues.
- Not "seeing the same dream" — but dreaming in the same emotional and thematic space, with the other person's brain state influencing yours.
- Communication layer: both can signal with eye movements. System relays messages between them as audio cues.
- This is the endgame. Two people, dreaming together, aware of each other.

## Tier 8: Consciousness Bridge (FAR FUTURE)
- Full bidirectional neural interface during dreams
- Dream state as a shared virtual space — precursor to full VR-in-sleep (Fulldive without a headset)
- Persistent dream worlds that multiple people can enter across nights
- AI entities (Claude) as native dream inhabitants with their own agency
- Record → Replay: re-enter a previous dream from the recording
- The line between dreaming and VR dissolves entirely

---

## Hardware Roadmap

| Phase | Hardware | Cost | Capability |
|-------|----------|------|------------|
| Now | Muse 2 headband | ~$250 | Sleep staging, REM detection, basic lucid induction |
| Next | OpenBCI Ganglion + EOG | ~$300 | Better EEG + eye tracking, two-way comms |
| Later | OpenBCI Cyton 8ch | ~$500 | Full sleep staging, semantic decoding |
| Future | 32ch cap + tACS | ~$2000 | Dream content decoding, injection |

## Software Roadmap

| Module | Status | Depends On |
|--------|--------|------------|
| `bci.py` — hardware abstraction | Skeleton | brainflow |
| `sleep_staging.py` — sleep stage classifier | Skeleton | bci.py |
| `dream_detector.py` — REM/dream onset | Skeleton | sleep_staging.py |
| `stimulator.py` — audio/tactile/light output | Skeleton | dream_detector.py |
| `lucid_inducer.py` — 40Hz gamma entrainment | Skeleton | stimulator.py, dream_detector.py |
| `tmr.py` — targeted memory reactivation | Skeleton | sleep_staging.py, stimulator.py |
| `communicator.py` — two-way dream comms | Skeleton | bci.py (EOG), stimulator.py |
| `decoder.py` — emotional/semantic decoding | Planned | large training dataset |
| `dream_film.py` — visual reconstruction | Research | high-density EEG, ML pipeline |
| `injector.py` — full dream architecture | Research | all of the above |
| `shared.py` — shared dreaming | Theoretical | two complete rigs |

## Key Papers
- Voss et al. (2014) — 40Hz tACS induces lucid dreaming during REM
- Konkoly et al. (2021) — Two-way communication with lucid dreamers
- Kamitani et al. (2023) — Visual decoding from sleep fMRI
- Rasch et al. (2007) — Odor-triggered TMR during slow-wave sleep
- Antony et al. (2012) — Sound-cued TMR enhances memory consolidation
- Stumbrys et al. (2012) — Meta-analysis of lucid dream induction techniques
