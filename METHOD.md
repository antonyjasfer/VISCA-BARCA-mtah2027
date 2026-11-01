# Method: v1 HMM possession model

Branch `v1-hmm`. Reproduce with

```
pip install -r requirements.txt -r requirements-model.txt
python -m src.possession.pipeline --out evaluation/results/v1_hmm.csv
```

The run is deterministic (no randomness, Viterbi ties go to the lower state index) and takes under
a minute. `--cache PATH` stores the extracted features as an `.npz` for faster reruns; the cached
arrays are the exact arrays the extractor returns, so the output file is byte-identical.

## Principles

* **No labels, no tuning on ground truth.** Every constant below is chosen from football reasoning
  or from label-free, whole-match distributions of the tracking itself (for example the quantiles
  of collective speed). The public 5-minute ground truth was used once, after the pipeline was
  frozen, to confirm the file is valid and aligned. The parameters were not changed afterwards.
* **Collective, not per-player.** Track slots are not people (DATA.md), so only team-level
  quantities are used.
* **Abstain nowhere that is graded.** Every graded frame is H, A or D. `X` is written only on the
  ungraded frames (before kickoff, halftime, after the whistle), taken from
  `games.half_video_blocks`.

## 1. Features (`features.py`)

Raw ATD is streamed with the same line format and team lookup as `starter/baseline.py`
(`team_of_each_track`); `frame_id - 1` is the video frame. Coordinates are scaled to metres with
105.3 x 68.0 on both axes before any distance is taken.

| feature | definition |
|---|---|
| `ball_x`, `ball_y` | metres; gaps of at most 10 frames (0.4 s) bridged linearly, longer gaps stay NaN |
| `ball_speed` | central difference over +/-4 frames (0.32 s), m/s |
| `d_home`, `d_away` | distance from the ball to the nearest tracked player of each team |
| `ball_in_pitch` | 1 inside the lines, 0 more than 0.3 m beyond a line, NaN without a ball |
| `mean_speed` | mean speed of tracked players, m/s |
| `n_tracked` | tracked slots with a position |

* **10-frame interpolation limit.** Short drop-outs are occlusions by a body; over 0.4 s the ball
  can have been kicked, so a straight line would be invented data. Gaps are filled whole or not at
  all (pandas' `limit=` would fill the first part of a long gap).
* **0.3 m margin** keeps tracker jitter on a line from reading as "ball out".
* **+/-4-frame speed baseline.** A one-frame difference multiplies position jitter by 25 and
  makes a resting ball look like it moves; 0.32 s is short enough to still see a pass.
* Sanity checks run once on whole-match aggregates (never on labelled frames): collective speed
  has a graded median of about 1.5 m/s, so units are m/s; 3.9% of observed ball positions are
  outside the lines, so 0-1 maps to the pitch lines; `n_tracked` has a median of 22.

## 2. Dead ball (`dead_ball.py`)

A per-frame score in [0, 1] is a noisy-OR of five pieces of evidence, `1 - prod(1 - w_i c_i)`.
Weights are below the 0.5 threshold for soft cues, so those need a second cue to agree.

| evidence `c_i` | weight | reasoning |
|---|---|---|
| ball outside the pitch for >= 0.5 s, state carried across tracking gaps for up to 10 s | 0.90 | The laws of the game: a ball over a line is dead. A lost tracker usually stays lost until the restart, hence the hold. Runs under 0.5 s are noise. |
| continuous ball absence ramping from 2 s to 6 s, times collective slowness | 0.70 | The ball vanishes in aerial duels and crowded boxes while players run; vanishing while everyone walks is a stoppage. |
| collective slowness: 3 s median of `mean_speed`, 1 at <= 0.8 m/s, 0 at >= 1.6 m/s | 0.45 | Walking is ~1.4 m/s. Whole-match quantiles of the mean are 1.0 / 1.5 / 2.3 m/s (25th / 50th / 75th), so "everyone walks" sits clearly below the median. Slow build-up also looks like this, so alone it stays under the threshold. |
| ball motionless: speed < 0.8 m/s for the whole +/-0.5 s | 0.40 | A ball waiting for a restart. A goalkeeper holding it also looks like this, so it needs a partner. |
| Smart Tagging `SET PIECES` start within the next 8 s | 0.40 | A restart is about to happen, so the ball was dead beforehand. A tag is evidence, not an answer (DATA.md). |

* The score is smoothed with a 1 s centred mean.
* **Pre-kickoff rule.** From a half's first graded frame up to its `Kick Off (Start)` tag, the
  score is 1: the ball cannot be live before the kickoff. This is derived from the tag, not from
  a hard-coded frame. In the first half the graded block starts at the kickoff, so the rule covers
  no frames there; the second graded block starts about two minutes before the kickoff, so it
  matters there.
* **Anchors use `start_frame`**, as specified. The tag windows are 35 s for throw-ins, corners,
  goal kicks and free kicks and 5 s for kickoffs, so the exact restart instant inside a window is
  uncertain. Anchoring on the start with an 8 s lead is a deliberate, documented assumption.
  `Team` tags are not used in v1 (every throw-in is `N/A` anyway).
* **D segments.** Score >= 0.5, gaps under 1 s merged, runs under 2 s dropped. Two seconds is
  shorter than any real stoppage including a quick throw-in, so what it removes is flicker.

## 3. HMM (`hmm.py`)

States H, A, D. Viterbi per graded half, uniform initial distribution, no training.

**Transitions (per frame).** A possession lasts 10-20 s and ends mostly in a hand-over; a
stoppage lasts 5-15 s.

| from | to H | to A | to D |
|---|---|---|---|
| H | 0.9975 | 0.0015 | 0.0010 |
| A | 0.0015 | 0.9975 | 0.0010 |
| D | 0.0020 | 0.0020 | 0.9960 |

Alive states exit with 0.0025/frame (mean 16 s); D with 0.004/frame (mean 10 s). A team switch
costs -ln(0.0015) = 6.5 nats.

**Ownership emission.** Per-frame log-likelihood ratio (home vs away)

`LLR = 0.25 * clip((d_away - d_home) / 3 m, -1, 1) * near * grounded`

* The closer team is more likely to own the ball; 3 m closer is "clearly" closer.
* `near` = 1 when the nearest player is within 2.5 m and falls to 0 at 8 m: a ball nobody is near
  is loose or in flight and says nothing about who controls it.
* `grounded` = 1 below 8 m/s and falls to 0 at 16 m/s: a ball faster than a sprinting dribbler is
  being passed or kicked.
* With full-strength evidence the model needs 6.5 / 0.25 = 26 frames, about 1 s, to switch owner.
  This implements "control is roughly 0.5-1 s of sustained proximity" (CHALLENGE.md's own
  proxy for controlled possession, "a player keeps the ball for a few seconds", is stricter; the
  weaker evidence of a contested ball takes proportionally longer).
* **No ball, no evidence.** Where the ball is NaN the LLR is 0 and the HMM carries the last
  owner. This is the model's own version of an abstention inside a possession.

**Dead emission.** `P(dead) = max(score, 0.9)` inside a D segment; outside one, `min(score, 0.3)`;
clipped to [0.01, 0.99]. The segment step therefore decides *whether* a stoppage exists and the
continuous score only shades the rest. `H` and `A` get `log(1 - P(dead)) +/- LLR/2` and `D`
gets `log P(dead)`.

## 4. Pipeline (`pipeline.py`)

Features -> pre-kickoff mask and anchors -> score -> segments -> Viterbi -> CSV. Asserts that every
graded frame is H, A or D and every ungraded frame is X. Prints only counts, class shares and
runtime.

## Known limitations of v1

* Ownership is nearest-player proximity; there is no pass/receive logic and no player identity.
* The 35 s set-piece windows are anchored on their start only.
* The dead-ball segmenter has no notion of the ball being back in play after an out; it relies on
  the score dropping.
* Possession counts on the public window are far from the truth's, in the direction of too few
  and too long. That was observed after freezing and is not tuned away; it is a lead for v2
  (the self-transition and evidence scale are the first things to revisit, on a basis of more
  than one 5-minute window).
