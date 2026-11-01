# Method: v3 rules on the v2 control model

Branch `v3-rules`. Reproduce with

```
pip install -r requirements.txt -r requirements-model.txt
python -m src.possession.pipeline --out evaluation/results/v3_rules.csv
python -m src.possession.diagnostics evaluation/results/v3_rules.csv
python -m pytest tests -q
```

The run is deterministic (no randomness, Viterbi ties go to the lower state index). `--cache PATH`
stores the extracted features as an `.npz`; a cached and an uncached run produce a byte-identical
file. A cache written by v1 lacks `rel_home`/`rel_away` and fails loudly. Source is kept with
`ruff check --fix` and `ruff format`, both at `--line-length 100`.

## What v3 changed from v2

1. **No graded X.** Graded frames always carry the decoded label.
2. **Restart team without a home default** (section 2).
3. **Touch merge** (section 5).
4. **Goal and standstill stoppages** (section 2).
5. **Code hygiene.** Ruff; the labels were confirmed byte-identical before and after.

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
| **v3** standstill: 3 s median speed < 1.0 m/s and ball still or missing, for >= 8 s, nudges under 1 s bridged | 0.95 | Injury, VAR, substitution. 1.0 m/s is under a walk (~1.4 m/s) and is the whole-match 25th percentile. A keeper may hold the ball about 6 s with outfield players moving into shape, so both conditions must hold for longer than that. |
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

**Restart team (v3).** A D segment with no located restart within 2 s of its end gets a restart at its
end. Each restart then takes the first rule that gives an answer, and an exact distance tie is no
answer, so no rule defaults to a side:

| rule | restarts | definition |
|---|---|---|
| `tag` | 23 | The set piece's `Team` when it names a club (assumed to be the taking team). |
| `out` | 39 | The ball left the pitch in this stoppage: seen beyond a line for >= 0.5 s, or lost within 1 m of one for >= 1 s. The search runs from 2 s before the D segment, or 15 s back without one, and takes the earliest crossing. The team that came closest in the 1 s before it touched it last, so the other team restarts. Skipped for kickoffs: the ball has not left play before a half opens, and after a goal the nearest player at the line is often the conceding keeper. |
| `nearest` | 68 | The team nearest at the restart, by median distance from 12 frames before to 2 after. For a D-segment restart, over the next second. |
| `phase` | 14 | The team on the ball in the next team-tagged phase, mirrored for `DEFENDING` tags. |
| unknown | 2 | Carries the owner from before the stoppage through a second decode, so there is no tie to break. |

**Restart team is not validated.** On the 23 tagged restarts the other rules agree with the tag 9 times,
and tag vs nearest at the restart is 13 of 23. Every rule agrees with the next team-tagged phase about
half the time, but that phase is a median 20 s away, so it is too late to judge a restart. No label-free
signal here confirms or refutes these rules, and that aggregate check changed no parameter.

**Goal stoppages (v3).** D from the goal to the kickoff that follows. The `GOALS` tag is not the goal: an
aggregate profile around the 4 tag starts shows active play at +10 to +30 s and the ball gone from
+30 s, and the tag runs to the kickoff (70-250 s). So the goal is the start of the longest ball absence
inside the tag, which must last at least 5 s (a celebration hides the ball far longer than a dropout).
Without one, nothing is forced, because the tag start would turn live play into D. All 4 goals resolve,
forcing 9.6 minutes.

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
| `MAX_TOUCH_FRAMES` | 37 | Touch merge, under 1.5 s. See section 5. |

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

## 5. Touch merge (`control.merge_touches`)

CHALLENGE.md: a touch or a deflection is not a change of possession. After decoding, in a live sequence
X-Y-X with no D between, a Y of under 1.5 s (at most 37 frames) is relabelled X. It repeats left to right
until nothing changes, so a ping-pong collapses into the surrounding possession. X-Y-D is left alone, as
is a run at the edge of a half, and nothing merges across D: a stoppage ends the sequence.

**Why 1.5 s.** A touch, a block and the ball coming straight back fit inside it, while really winning the
ball takes a receive plus one action. It also sits above the 0.5 s minimum control event and the longest
event Viterbi's own flip-and-flip-back cost can absorb (about 0.6 s). The cost is that a genuine turnover
lost again inside 1.5 s disappears. It relabelled 131 frames over the match.

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
* **Graded X (v2, dropped in v3).** The scorer fills an X with the previous label, so spellF1 could not
  gain and macroF1 lost those frames.

## Result, scored once with parameters frozen

Public window, 5-95% bootstrap interval on spellF1 (1000 draws, 17 possessions):

| submission | macroF1 | spellF1 | spellF1 interval | public poss | frag |
|---|---|---|---|---|---|
| `v3_rules` | 0.8129 | 0.4857 | 0.2857 .. 0.6798 | 11 | 0.67 |
| `v2_control` | 0.7820 | 0.5333 | 0.3467 .. 0.7333 | 13 | 0.76 |
| `v1_hmm` | 0.7072 | 0.3360 | 0.1440 .. 0.5321 | 8 | 0.52 |
| `baseline_smooth50` | 0.6207 | 0.3256 | 0.2140 .. 0.4465 | 26 | 1.43 |

v3 and v2 are not separable: the intervals overlap almost entirely and the board puts the noise band
near 0.39 here. Nothing was changed after this table.

## Known limitations of v3

* **Under-segmentation on the public window.** 11 possessions against 17, fewer than v2, which fits the
  touch merge and goal stoppages removing runs. It was seen after freezing and not tuned.
* **Live time is still high.** 60.2 minutes against the 52-58 prior. No further change was made
  without a football reason for it.
* **Restart team.** Unvalidated (section 2). The 2 unknown restarts carry the owner from the first
  decode, one pass only; with no owner before them they would stay a tie.
* **Earliest crossing.** The `out` rule takes the first crossing in its window, which for a restart with
  no D segment behind it can be an older, unrelated one.
* **False restarts.** Any fast ball after a calm second inside the 15 s search counts, so an open-play
  pass can be taken for the restart. Restart evidence totals 12.5 nats, twice a minimum control event,
  so a wrong team or false restart flips possession alone.
* **1 s minimum stoppage.** A false D segment splits one possession in two; 31 possessions are under 2 s.
* **Ball-dependent control.** With the ball missing on 41% of graded frames, no control event is
  possible there, so a hand-over inside a gap is dated when the ball comes back.
* **Tag semantics.** That `Team` on a set piece is the taker, and that `DEFENDING` tags name the
  defender, are assumptions.
