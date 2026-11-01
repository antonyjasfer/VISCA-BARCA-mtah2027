# Method: v2 control-based possession model

Branch `v2-control`. Reproduce with

```
pip install -r requirements.txt -r requirements-model.txt
python -m src.possession.pipeline --out evaluation/results/v2_control.csv
python -m src.possession.diagnostics evaluation/results/v2_control.csv
python -m pytest tests -q
```

The run is deterministic (no randomness, Viterbi ties go to the lower state index). `--cache PATH`
stores the extracted features as an `.npz`; a cached and an uncached run produce a byte-identical
file. A cache written by v1 lacks `rel_home`/`rel_away` and fails loudly.

## Principles

* **No labels, no tuning on ground truth.** Every constant comes from football priors or from
  label-free aggregates of the tracking and tagging: whole-match quantiles, the set-piece profile
  below and `diagnostics.py`. The public ground truth was scored once, after the parameters were
  frozen, and nothing changed afterwards.
* **Collective, not per-player.** Track slots are not people. A control event belongs to a team, and
  a slot is followed only across the 0.32 s of one velocity estimate.
* **Possession changes at control.** CHALLENGE.md says a touch, a deflection or a duel is not a change
  of possession. v2 makes that a structural rule rather than a smoothing effect.

## Why v2 (diagnosis of v1)

v1 merged possessions and under-detected stoppages: 73.4 live minutes against the 52-58 of a typical
LaLiga match, and 226 possessions. Merged possessions cost twice in spellF1, once as a miss and once
through the wrong boundaries of the merge. Label-free checks show the causes:

* **Stoppage timing.** An aggregate profile of collective speed and ball visibility around the
  `SET PIECES` tag starts bottoms out from about -5 s to +5 s, with the stoppage building from about
  -15 s and play back by +10 s. So the tag start falls *inside* the stoppage. v1 ended its dead-ball
  evidence at the tag start and missed the seconds after it. The same profile shows kickoffs preceded
  by a long slow period.
* **Team switches.** Proximity evidence was spread over every frame against a 6.5-nat switch cost, so
  short possessions never accumulated enough to switch.

## 1. Features (`features.py`)

Raw ATD is streamed with the same line format and team lookup as `starter/baseline.py`;
`frame_id - 1` is the video frame. Coordinates are scaled to metres (105.3 x 68.0).

| feature | definition |
|---|---|
| `ball_x`, `ball_y` | metres; gaps of at most 10 frames (0.4 s) bridged linearly, longer gaps stay NaN |
| `ball_speed` | central difference over +/-4 frames (0.32 s), m/s |
| `d_home`, `d_away` | distance from the ball to the nearest tracked player of each team |
| `rel_home`, `rel_away` | **new.** Ball velocity minus that player's velocity, same 0.32 s baseline, slot held fixed |
| `ball_in_pitch` | 1 inside the lines, 0 more than 0.3 m beyond a line, NaN without a ball |
| `mean_speed` | mean speed of tracked players, m/s |
| `n_tracked` | tracked slots with a position |

Only 108 frames of the match are actually bridged by interpolation: the ball's gaps are mostly long.

## 2. Dead ball (`dead_ball.py`)

The per-frame score is a noisy-OR, `1 - prod(1 - w_i c_i)`, smoothed with a 1 s centred mean. Weights
below the 0.5 threshold need a second cue to agree.

| evidence `c_i` | weight | reasoning |
|---|---|---|
| ball outside the pitch for >= 0.5 s, held across tracking gaps for up to 10 s | 0.90 | The laws of the game. Runs under 0.5 s are noise. |
| **new** run-up to a located restart: tag start - 10 s (- 30 s after a goal) to the restart | 0.85 | The ball is dead until the restart. 30 s covers a celebration and the walk back. |
| **new** ball still (< 0.8 m/s) for >= 1.5 s, times collective slowness | 0.80 | Replaces v1's 1 s still-ball cue. A goalkeeper's hold is the known false positive, and the slowness term limits it. |
| **new** ball within 1 m of a line (or over it), then lost for >= 1 s (held at most 15 s) or stopped for >= 1 s | 0.75 | A ball that vanishes at a line has usually left play, and the tracker loses it before it is seen outside. |
| continuous ball absence, ramp 2 s to 6 s, times collective slowness | 0.70 | The ball vanishes in duels too, but not while everyone walks. |
| collective slowness: 3 s median of `mean_speed`, 1 at <= 1.0 m/s, 0 at >= 1.8 m/s | 0.45 | See below. Slow build-up looks the same, so it needs a partner. |
| tag with no restart found: start - 10 s to start + 5 s | 0.40 | Weak fallback. Needs a partner. |

**Slowness ramp (changed from 0.8-1.6).** 52-58 live minutes of 106 is a dead share near 47%, and
whole-match quantiles of the signal are 1.0 / 1.5 / 2.3 m/s at the 25th / 50th / 75th percentiles.
Stoppages in the set-piece profile sit at 1.0-1.3 m/s and resumed play at 1.7-2.4 m/s. The ramp is
therefore centred at 1.4 m/s.

**Restart finder (`find_restart`).** Searches from the tag start to 15 s after it, the span in which
the profile shows play returning. A restart is the first sharp acceleration after a slow period:
speed crosses 6 m/s (softer than any real restart) from below 2 m/s (a ball being placed) within
0.5 s, after at least 1 s of slow or *unseen* ball, because the tracker usually loses the ball in a
stoppage. D runs up to that frame, and the score is forced to 0 for the 3 s after it. 47 of the 59
tags are located; the other 12 use the weak fallback.

**Restart team.** The tag's `Team` when it names a club (assumed to be the taking team). `N/A`, which
includes every throw-in, falls back to the team nearest the ball from 12 frames before to 2 frames
after the restart, by median distance. A D segment with no located restart within 2 s of its end gets
a restart at its end, with the team nearest during the next second.

**Pre-kickoff rule.** Score 1 from a half's first graded frame to its `Kick Off (Start)` tag.

**D segments.** Score >= 0.5, gaps under 1 s merged, runs under 1 s dropped (v1: 2 s).

## 3. Control (`control.py`)

A **control frame** for a team: its nearest player is within 1.5 m, the ball's velocity is within
3 m/s of that player's, and it is the closer team. A **control event** is at least 12 frames (0.5 s)
of control frames for one team, with flickers of up to 3 frames bridged.

| constant | value | reasoning |
|---|---|---|
| `CONTROL_RADIUS_M` | 1.5 | A stride plus the ball: the player can play it without moving to it. |
| `MATCH_SPEED_MS` | 3.0 | Dribble touches sway the ball 1-3 m/s around the carrier; a pass goes by at 10+ m/s. |
| `MIN_CONTROL_FRAMES` | 12 | 0.5 s. A touch or deflection is over well before this. |
| `BRIDGE_FRAMES` | 3 | 0.12 s is tracker noise. |
| `LLR_CONTROL` | 0.5 nats/frame | See below. |
| `MAX_SILENT_GAP` | 50 frames | The 2 s rule for writing X. |

In a duel only the closer team can hold a frame, so a real contest fragments and produces no event.

## 4. HMM (`hmm.py`)

States H, A, D. Viterbi per graded half, uniform initial distribution, no training.

**Gate.** A direct H -> A exists only on frames of an away control event or in the 1 s after an away
restart, and the mirror for A -> H. D is unreachable outside a D segment: when it was reachable,
Viterbi used a one-frame H -> D -> A to get round a closed gate (caught by a test).

**Transitions (per frame).**

| | v1 | v2 | why |
|---|---|---|---|
| H <-> A | 0.0015 on every frame | 0.02 on open frames, 0 otherwise | The gate does the filtering. The switch cost falls from 6.5 to 3.9 nats. |
| H/A -> D | 0.0010 | 0.0015 | About 110 stoppages in 55 live minutes: one per 750 frames. |
| D -> H, D -> A | 0.0020 each | 0.00075 each | About 50 dead minutes over 110 stoppages: 27 s each, 680 frames. |
| self (H/A) | 0.9975 | 0.9785 on open frames, 0.9985 on closed | Lower exactly where a switch is possible. |

**Evidence.** `LLR_CONTROL` per control frame, and the same over the 1 s after a restart. A minimum
event earns 6 nats: more than one switch (3.9), less than the two needed to flip and flip back (7.8).
So a lone minimum event between two controls of the same team stays a touch, and the same event with
no answer changes the owner. Outside events and restarts the evidence is 0 and the owner persists.

**Dead emission.** P(dead) = clip(score, 0.9, 0.99) inside a segment, and D is impossible outside one.
Segments therefore decide stoppages and Viterbi decides ownership.

## 5. Abstentions (`control.abstain_gaps`)

A ball gap over 2 s strictly inside one possession, with no opposing control event in it, becomes `X`.
The gap must not touch either end of the possession, so the scorer's fill reproduces the decoded labels
exactly; the pipeline asserts this. `X` therefore never crosses a switch.

This changes nothing on spellF1 and costs macroF1, because `X` is a miss there. The decoded label would
have filled to the same value. It is kept because the brief asks for it, but it is an obvious thing to
reconsider, since macroF1 is the first tie-break.

## 6. Diagnostics (`diagnostics.py`)

Label-free, with no ground-truth access. Per half: live minutes, D share, possession count, duration
percentiles, count under 2 s. Overall: agreement with team-tagged attacking phases, the mirror check
for `DEFENDING` tags, set-piece coverage and X share. X is filled as the scorer fills it.

## Dropped from v1, and why

* **Per-frame proximity LLR with `near`/`grounded` weights.** Replaced by control events; proximity
  without matched velocity counts touches and passes going by.
* **Still ball as a soft cue (1 s, weight 0.40).** Replaced by the 1.5 s cue tied to slowness.
* **`SET PIECES` anchor ending at the tag start.** The profile shows the stoppage runs past it.
* **`P_DEAD_OUTSIDE_CAP` and the P(dead) floor.** Removed with the rule that D exists only in segments.

## Known limitations of v2

* **Live time is still high.** 61.3 minutes against the 52-58 prior. No further change was made
  without a football reason for it.
* **Restart team unknown.** 22 restarts have no team (ball unseen). D -> H and D -> A are ungated, so
  the owner after those stoppages is a tie that goes to H, a small home bias until the next control
  event. A D state that remembers the previous owner would fix it.
* **False restarts.** Any fast ball after a calm second inside the 15 s search counts, so an open-play
  pass can be taken for the restart. Restart evidence totals 12.5 nats, twice a minimum control event,
  so a wrong team or false restart flips possession alone.
* **1 s minimum stoppage.** A false D segment splits one possession in two; 31 possessions are under 2 s.
* **Ball-dependent control.** With the ball missing on 41% of graded frames, no control event is
  possible there, so a hand-over inside a gap is dated when the ball comes back.
* **Tag semantics.** That `Team` on a set piece is the taker, and that `DEFENDING` tags name the
  defender, are assumptions.
