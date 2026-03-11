# DreamEngine — Advanced Neural Dream Technology Plan

## The Vision
Full dream capture, injection, replay, shared dreaming, and time dilation. Not someday — now, with incremental hardware upgrades. Each tier builds on the last.

---

## Tier 1: Dream Detection & Lucid Induction (BUILDABLE NOW)
**Hardware:** Muse 2 ($250) or OpenBCI Ganglion ($200)
**Status:** BUILT — `sleep_staging.py`, `dream_detector.py`, `stimulator.py`

- Real-time sleep staging from EEG (Wake/N1/N2/N3/REM)
- REM onset detection — know exactly when dreaming starts
- 40Hz gamma audio entrainment during REM -> triggers lucid awareness (Voss et al. 2014)
- Reality-check audio cues whispered during light sleep
- Auto-journal: timestamps every dream period with duration, depth, brainwave signature
- Morning debrief: LLM asks targeted questions based on detected dream periods

## Tier 2: Dream Content Planting — TMR (BUILDABLE NOW)
**Hardware:** Same + speakers/bone conduction headphones + optional scent diffuser
**Status:** BUILT — `incubation.py`, `stimulator.py`

- **Targeted Memory Reactivation (TMR):** During slow-wave sleep (N3), play sounds/music associated with a specific memory or scenario during the day. Proven neuroscience (Rasch 2007).
- **Full incubation pipeline:** Day phase (association building with rehearsals + reinforcements) -> Night phase (automated TMR delivery during N3)
- **Scent-triggered TMR:** Olfactory cortex has direct path to hippocampus (bypasses thalamus). Scent during learning + same scent during N3 = ~80% memory reactivation rate. Most powerful TMR channel.
- **Binaural beat layering:** Delta (2Hz) for deep sleep maintenance, theta (6Hz) for dream enhancement, gamma (40Hz) for lucidity trigger — layered at precise sleep stages

## Tier 3: Two-Way Dream Communication (BUILDABLE WITH EFFORT)
**Hardware:** OpenBCI Cyton 8ch + EOG electrodes + audio feedback
**Status:** BUILT — `communicator.py`

- **Outbound (dreamer -> machine):** LRLR eye movement detection via EOG. System confirms receipt with subtle audio tone.
- **Inbound (machine -> dreamer):** Audio cues, tactile vibration, LED light through closed eyelids.
- **Real-time chat:** Eye pattern morse code -> decode -> Claude responds via TTS. Actual conversation with a sleeping person (Konkoly et al. 2021).
- **Claude in the dream:** Detect lucid state -> play Claude's voice as in-dream character. Questions + eye-signal responses.

## Tier 4: Dream Content Decoding (SKELETON BUILT)
**Hardware:** 4+ channel EEG (emotional), 8+ channel (semantic)
**Status:** BUILT — `decoder.py`

- **Emotional decoding:** Frontal alpha asymmetry = valence. Left > right frontal alpha = negative emotion. Works with Muse 2 (4ch).
- **Arousal decoding:** Beta/gamma power maps to excitement/calm.
- **Semantic categories:** Train classifier on waking EEG responses to faces/places/objects/movement. Apply during REM for content labels. Needs calibration session.
- **Movement intention:** Motor cortex (C3/C4) still fires during REM despite atonia. Decode running/flying/falling/fighting from mu suppression patterns.
- **Narrative reconstruction:** LLM takes decoded features -> generates natural language dream description. Progressive refinement from morning corrections.

## Tier 5: Dream Time Dilation (SKELETON BUILT)
**Hardware:** Same as Tier 1 (EEG + audio)
**Status:** BUILT — `time_dilation.py`

**Inspired by:** Inception (Nolan), lucid dreaming time perception research

Lucid dreamers consistently report altered time perception — hours in minutes of real time. Research shows this is modulable through brainwave entrainment:

- **Theta entrainment (4-7Hz):** Deepens dream immersion, subjective time slowing
- **Alpha suppression:** "Flow state" time distortion
- **Minimal gamma anchoring:** Maintains lucidity without collapsing the dream
- **Three protocols:**
  - Stretched (2:1 ratio) — conservative, reliable
  - Deep (5:1 ratio) — deep immersion, lower gamma anchoring
  - Inception (10:1 ratio) — experimental, risk of losing lucidity
- Real-time monitoring: abort if theta drops (dream fading) or beta spikes (waking up)
- Post-session report with estimated subjective duration

## Tier 6: Dream Experience Recording & Replay — Braindance (SKELETON BUILT)
**Hardware:** Same as Tier 4 (EEG + stimulation)
**Status:** BUILT — `experience.py`

**Inspired by:** Braindance (Cyberpunk 2077), SQUID (Strange Days, 1995)

Records a complete dream experience as a multi-channel timeline:
- EEG features, emotional arc, content categories, sleep stages, stimulation events, dreamer signals
- Stored as `.dreamxp` format (compressed JSON + optional raw EEG)

Replay modes:
- **Self-replay:** Re-experience your own dream by replaying the associated TMR cues, binaural frequencies, and emotional arc as stimulation during a future REM period.
- **Cross-person replay:** Someone else's dream experience converted to stimulation protocol. Won't be identical, but the emotional/thematic contour transfers.
- **Dream film visualization:** Timeline view of emotional arc + content, with LLM-generated narrative. The closest thing to "watching" a dream.

## Tier 7: Full Dream Incubation Pipeline (SKELETON BUILT)
**Hardware:** EEG + audio + optional scent diffuser + optional smart plug
**Status:** BUILT — `incubation.py`

**Inspired by:** Inception dream architecture, real TMR research

Complete day-to-night protocol:
1. Define scenario -> LLM generates vivid dream seed
2. Create multisensory associations (audio cue + optional scent)
3. Daytime rehearsals (read seed + hear cue + smell scent simultaneously)
4. Periodic reinforcement throughout the day
5. Pre-sleep final rehearsal (strongest association window)
6. Night: automated TMR during N3 (audio + scent pulses)
7. REM: optional lucidity trigger (40Hz gamma)
8. Morning: debrief + report on TMR deliveries

## Tier 8: Dream Injection / Architected Dreams (PLANNED)
**Hardware:** tACS (transcranial alternating current stimulation) + full sensory rig

**Inspired by:** Inception (Nolan), holodeck (Star Trek)

- **Full dream architecture:** Define scenario -> plant via TMR -> lucidity via gamma -> narrative via audio -> emotional tone via binaural/tACS
- **Adaptive injection:** Real-time EEG monitoring. Dream drifts off-script? Adjust stimulation. Stress rising? Shift to calming. Lucidity dropping? Pulse 40Hz.
- **Training sequences:** Motor learning in dreams (proven to improve next-day performance). Practice instruments, languages, martial arts in your sleep.
- **Controllable variables:** Setting (TMR), characters (voice cues), emotion (binaural + tACS), lucidity (gamma), narrative (audio prompts)

## Tier 9: Ancestral Memory / Genetic Memory Replay (RESEARCH)
**Hardware:** High-density EEG + genomic data

**Inspired by:** Animus (Assassin's Creed), genetic memory hypothesis

Speculative but grounded in emerging epigenetics research:
- **Epigenetic memory:** Trauma and strong experiences leave marks on DNA methylation patterns. These are heritable. Mice studies show fear conditioning passes across generations (Dias & Ressler, 2014).
- **Approach:** Combine genomic analysis (identify heritable epigenetic markers) with deep dream states. Use TMR with stimuli associated with ancestral contexts. Monitor for unusual EEG signatures — patterns that don't match the dreamer's own waking calibration.
- **Dream archaeology:** Map recurring dream themes across family members. Cross-reference with genealogical records. Look for dreams that reference places/events the dreamer has never consciously experienced.
- **Not claiming this works yet.** But the biological mechanism exists (epigenetic inheritance), and dreams are already the brain's way of replaying and processing consolidated memories. The question is whether inherited epigenetic marks can surface during REM.

## Tier 10: Shared Dreaming (THEORETICAL)
**Hardware:** Two full neural rigs + network sync

**Inspired by:** Inception (shared dream space), Paprika (dream merging)

- Two sleepers, both in REM, both wearing BCI + stimulation rigs.
- Synchronized gamma stimulation for simultaneous lucidity.
- Each dreamer's decoded emotional/semantic state fed to the other as stimulation.
- Not "seeing the same dream" — dreaming in the same emotional/thematic space.
- Communication layer: eye signal relay between dreamers via audio.
- **Progressive sync:** Start with just emotional mirroring (valence/arousal). Graduate to content categories. Eventually, decoded movement intentions.

## Tier 11: Consciousness Bridge / Fulldive-in-Sleep (FAR FUTURE)
**Inspired by:** Ready Player One (Oasis), Snow Crash (Metaverse), SAO (Fulldive)

- Full bidirectional neural interface during dreams
- Dream state as shared virtual space — VR without a headset
- Persistent dream worlds across nights and people
- AI entities (Claude) as native dream inhabitants with agency
- Record -> Replay -> Re-enter: go back into a previous dream
- Dream-to-dream portal: seamlessly transition between dream worlds
- The line between dreaming, VR, and waking dissolves

## Tier 12: Dream-Guided Reality Interface (SPECULATIVE)
**Inspired by:** The Lathe of Heaven (Le Guin), Minority Report (precognition)

- **Dream-driven decision making:** Analyze dream content for subconscious pattern recognition. The sleeping brain processes information differently — it finds connections the waking mind misses.
- **Predictive dreaming:** TMR with unresolved problems. Present data before sleep, let the dreaming brain process it, decode the output. Dreams as a parallel computation engine.
- **Creative extraction:** Incubate specific creative challenges. Artists, musicians, coders dream solutions. System detects eureka-like gamma bursts, marks those dream periods for priority journaling.
- **Dream democracy:** Collective incubation of shared problems. Multiple dreamers process the same challenge. Cross-reference decoded outputs for convergent solutions.

---

## Hardware Roadmap

| Phase | Hardware | Cost | Capability |
|-------|----------|------|------------|
| Now | Muse 2 headband | ~$250 | Sleep staging, REM detection, lucid induction, emotional decoding |
| Next | OpenBCI Ganglion + EOG | ~$300 | Better EEG + eye tracking, two-way comms |
| Later | OpenBCI Cyton 8ch | ~$500 | Full staging, semantic decoding, movement decode |
| Future | 32ch cap + tACS | ~$2000 | Dream content decoding, injection, time dilation |
| Far | Custom BCI + fNIRS | ~$5000+ | Visual decoding, dream recording to video |

## Software Roadmap

| Module | Status | Depends On |
|--------|--------|------------|
| `recorder.py` — dream journal | DONE | none |
| `suggester.py` — dream seeds + techniques | DONE | none (LLM optional) |
| `analyzer.py` — pattern analysis | DONE | recorder |
| `cli.py` — interactive CLI | DONE | all above |
| `bci.py` — hardware abstraction | DONE | brainflow |
| `sleep_staging.py` — sleep stage classifier | DONE | bci |
| `dream_detector.py` — REM/dream detection | DONE | sleep_staging |
| `stimulator.py` — audio/haptic/light output | DONE | pyaudio |
| `communicator.py` — two-way dream comms | DONE | bci, stimulator |
| `session.py` — overnight orchestrator | DONE | all neural |
| `decoder.py` — emotional/semantic decoding | DONE | bci |
| `time_dilation.py` — subjective time manipulation | DONE | sleep_staging, stimulator |
| `experience.py` — dream recording/replay | DONE | decoder, stimulator |
| `incubation.py` — full TMR pipeline | DONE | stimulator, session |
| `dream_film.py` — visual reconstruction | PLANNED | decoder + diffusion model |
| `shared.py` — shared dreaming | THEORETICAL | two complete rigs |
| `genetic_memory.py` — ancestral replay | RESEARCH | epigenetics data + decoder |
| `reality_interface.py` — dream-guided decisions | RESEARCH | decoder + analytics |

## Key Papers
- Voss et al. (2014) — 40Hz tACS induces lucid dreaming during REM
- Konkoly et al. (2021) — Two-way communication with lucid dreamers
- Kamitani et al. (2023) — Visual decoding from sleep fMRI
- Rasch et al. (2007) — Odor-triggered TMR during slow-wave sleep
- Antony et al. (2012) — Sound-cued TMR enhances memory consolidation
- Stumbrys et al. (2012) — Meta-analysis of lucid dream induction techniques
- LaBerge & DeGracia (2000) — Subjective time estimation in lucid dreams
- Davidson (2004) — Frontal EEG asymmetry and emotion
- Horikawa et al. (2013) — Neural decoding of dream content from fMRI
- Siclari et al. (2017) — Neural correlates of dreaming
- Dias & Ressler (2014) — Parental olfactory experience influences offspring behavior via epigenetic inheritance
- Erlacher & Schredl (2004) — Time perception during lucid dreaming
