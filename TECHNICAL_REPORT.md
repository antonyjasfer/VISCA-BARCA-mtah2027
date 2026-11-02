# VISCA BARCA: technical report

**Team:** VISCA BARCA
· **Members:** A. Antony Selva Jasfer, C. Akash Narayanan, IV. Pranav Moorthy
· **Repository:** <https://github.com/antonyjasfer/VISCA-BARCA-mtah2027>
· **Commit:** `dd91b93` (the last commit that changes code; later commits change documentation only)

More than a Hack 2027, Phase 1. Aggregate numbers only; this document contains no data rows.

## Approach and why

The model answers one question for each of the 199,319 frames: which team has the ball, or is it dead?
It is a rule-based pipeline with no learned parameters:

1. **Features.** Per-frame ball position and speed, the distance to the nearest player of each team, and
   the ball's velocity relative to that player.
2. **Dead ball.** A score combining independent cues, turned into D segments.
3. **Restarts.** The restart instant inside each Smart Tagging set-piece window, and the team that takes
   the ball.
4. **Control events.** A team-level event meaning a player really has the ball.
5. **Gated Viterbi.** A three-state model (H, A, D) in which a direct H-A switch is possible only at a
   control event or a restart.

We chose this over a nearest-player vote for three reasons. The ranking metric treats a possession as one
object, so one false switch both creates a spurious possession and damages a real one; a structural rule
for when a switch may happen serves that better than smoothing. The rulebook defines a change of
possession as *controlled* possession by the other team, which a gate on control expresses directly. And
we had no labels we could legally train on (see Training data), so every component had to be explainable
from football reasoning.

The design moved through four versions. v1 used per-frame proximity evidence in a sticky HMM, which
under-segmented and left 73.4 live minutes against a typical 52-58. v2 added control events, the switch
gate, restart finding and stronger dead-ball cues (61.3 live minutes). v3 added goal and standstill
stoppages and a restart-team rule chain. v4 added the first-control restart rule, switches for the
optional parts, and a label-free ablation to set them.

## Assets used and dropped

| asset | status | how |
|---|---|---|
| Tactical ATD (tracking and metadata) | used | Streamed from the raw file with the starter's line format and team lookup; `kloppy` was not needed |
| Smart Tagging | used | `SET PIECES` windows locate restarts and name teams; `GOALS` locates goal stoppages; team-tagged phases are a last-resort fallback and a validation signal |
| Homography | not used | Tracking is already in pitch coordinates |
| Tactical video | not used | Everything comes from tracking and tagging |
| Broadcast video and ATD | not used | Not aligned to the tactical frames; no mapping was attempted |

Dropped along the way:

- **Per-frame proximity evidence (v1).** It counts touches and passing balls as ownership.
- **A bare still-ball cue (v1).** It reads a keeper holding the ball as a stoppage, so it now requires low
  collective speed for at least 1.5 s.
- **Set-piece anchors ending at the tag start (v1).** An aggregate profile shows the stoppage runs past it.
- **X on graded frames (v2).** The scorer fills X with the previous label, so it cannot help spellF1 and
  costs macroF1.
- **Touch merge as a default (v3).** See Validation methodology. The code remains behind a switch.

## Training data

**None.** The pipeline has no trained component and uses no labelled dataset from any game.

- No hand labels were created or used, on the provided game or any other.
- No ground truth was used for training or tuning. No parameter was fitted to any label.
- The private ground truth was never available to us. No human-produced event or tracking data was used.
- Constants come from football reasoning and from label-free aggregates of the provided tracking, such as
  whole-match quantiles of collective speed.

## Use of LLMs

An AI coding assistant (Anthropic's Claude, via Claude Code) was used to write, test and review the code
and documentation. In line with the terms of use, no match data, video or extract of them was uploaded
to or shown to it: the data files were only read by our own scripts, run locally, and the assistant saw
only aggregate outputs such as scores, counts, shares and error messages. No LLM is part of the pipeline
that produces the submission.

## Possession rules implemented

**Loose balls.** Possession stays with the last owner. With no control event or restart there is no
evidence, so the owner cannot change. A ball nobody controls, or one in flight, changes nothing.

**Control.** A control frame needs the team's nearest player within 1.5 m of the ball, the ball's velocity
within 3 m/s of that player's, and that team to be the closer one. A control event is at least 12 frames
(0.5 s) of these, bridging flickers of up to 3 frames. In a duel only the closer team can hold a frame, so
a real contest fragments and produces no event.

**Touches.** A touch under 0.5 s is not an event and cannot change the owner. An event of minimum length
earns 6 nats of evidence: more than one switch (3.9 nats), less than the two needed to flip and flip back
(7.8). So a lone short event between two controls of the same team stays a touch, and the same event with
no answer changes the owner.

**Restarts.** The restart instant is the first sharp acceleration after a slow period, searched from the
tag start to 15 s after it: from under 2 m/s to at least 6 m/s within 0.5 s, after at least 1 s of slow or
unseen ball. 47 of 59 set-piece tags are located. Each stoppage with no located restart gets one at its
end, giving 146 restarts. The restart team is the team of the first control event starting within 4 s
(74 restarts, median latency 1.66 s). Otherwise: the set-piece tag's team (13); the ball left the pitch,
so the team that did not touch it last (15); the nearest team at the restart (28); the next team-tagged
phase (14). The remaining 2 carry the previous owner.

**Dead ball.** A noisy-OR of weighted cues: ball over a line for at least 0.5 s; the run-up to a located
restart; a still ball with slow players; a ball lost or stopped at a line; long ball absence with slow
players; collective speed alone; and a tag with no located restart. Cues weighted under the 0.5 threshold
need a second cue to agree. D exists only inside D segments of at least 1 s. Otherwise a one-frame stay in
D could get round the switch gate, which a unit test caught during development. A long standstill counts
as dead: under 1.0 m/s with the ball still or missing for at least 8 s (injury, VAR, substitution). The
ball is also dead before each half's kickoff tag.

**Goals.** The `GOALS` tag starts about 30 s before the goal and runs to the kickoff (70-250 s), so the
tag start would turn a live attack into D. The goal is taken as the start of the longest ball absence in
the tag, which must last at least 5 s. It is dead from there to the kickoff. All 4 goals resolve.

## Handling missing and noisy tracking

The ball is missing on about 41% of graded frames, and most gaps are long: interpolation of gaps up to
10 frames fills only 108 frames in the whole match, and longer gaps stay missing.

- **Missing ball.** No evidence, so the owner persists. A hand-over inside a gap is dated when the ball
  returns. For dead-ball logic, a ball that vanishes at a line or while everyone walks is itself a signal.
- **Identity switches.** Track slots are not people, so all features are team-level and a slot is followed
  only across the 0.32 s of one velocity estimate.
- **Position jitter.** Speeds use a central difference over +/-4 frames, and a 0.3 m margin keeps jitter
  on a line from reading as out of play.
- **Automatic tags.** A tag is evidence, not an answer: restarts are located from the ball, and a set-piece
  tag alone cannot declare a stoppage.

## Validation methodology

**Label-free diagnostics** (`src/possession/diagnostics.py`, no ground-truth access): live minutes and D
share per half against the 52-58 minutes of typical effective playing time; possession counts and
durations; agreement with team-tagged phases; set-piece coverage.

**Timing profiles.** `scripts/set_piece_profile.py` and `scripts/goal_profile.py` print collective speed and
ball visibility by offset from the tag start, as aggregates across all tags. They are what place the
stoppage around a set-piece tag and show that a `GOALS` tag starts before the goal.

**Blind restart validation.** Each restart rule is compared with the team of the 23 located set pieces
naming a club, with tags hidden from the rule, since the rules otherwise use the tag and would agree by
construction. The 4 such set pieces with no located restart cannot be tested.

**Ablation** (`src/possession/ablation.py`). All four combinations of restart rule and touch merge run in
memory, are not written to disk and are not scored. The decision rules were set before running it:

| restart rule | touch merge | live min | poss | <2 s | attack | defend | set piece | blind tag |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v3 chain | on | 60.2 | 251 | 27 | 0.826 | 0.839 | 0.966 | 9/23 |
| v3 chain | off | 60.2 | 261 | 33 | 0.825 | 0.839 | 0.966 | 9/23 |
| first control | on | 60.2 | 237 | 18 | 0.828 | 0.840 | 0.966 | 11/23 |
| **first control** | **off** | 60.2 | 245 | 23 | 0.828 | 0.840 | 0.966 | 11/23 |

The higher blind agreement wins, with first control preferred on a gap under 3 of 23; it wins on both. The
merge stays only if turning it off raises the under-2 s count by more than 50%; it rose 28%, so it is off.

**One scoring run per version, with parameters frozen first.** No parameter was changed after a version
was scored. For v4, the chosen defaults and their reasoning were committed before scoring.

**Use of the public window.** It was used to confirm the pipeline is correct: frames aligned, file valid,
labels meaning what we think. It was not used for fitting or for choosing between the v4 variants. In the
interest of accuracy: scores were seen between versions, and they did inform what we worked on next. v1's
under-segmentation on the public window was part of the diagnosis that led to v2, and v3's result is part
of the context in which the v4 selection rules were written. We note this so a reader can weigh it.

**Bootstrap.** `run_evaluation.py --bootstrap 1000`, on the public window.

**Tests.** 100 unit tests on synthetic data only.

## Results

Public window (5 minutes, 17 possessions), with 5-95% bootstrap intervals:

| submission | macroF1 | spellF1 | spellF1 interval | poss (truth 17) | frag |
|---|---|---:|---:|---:|---:|
| **v4_restart (submitted)** | 0.8108 | 0.5333 | 0.3467 .. 0.7333 | 13 | 0.76 |
| v3_rules | 0.8129 | 0.4857 | 0.2857 .. 0.6798 | 11 | 0.67 |
| v2_control | 0.7820 | 0.5333 | 0.3467 .. 0.7333 | 13 | 0.76 |
| v1_hmm | 0.7072 | 0.3360 | 0.1440 .. 0.5321 | 8 | 0.52 |
| baseline_smooth50 | 0.6207 | 0.3256 | 0.2140 .. 0.4465 | 26 | 1.43 |

**These figures cannot rank anything.** 17 possessions is too small a sample: the scorer puts the noise
band near 0.39 here, and v2, v3 and v4 are not separable. v4 has the same spellF1 as v2 on this window, so
these 5 minutes do not exercise the difference in restart rule. The organisers show that public and
full-match order can differ even for trivial baselines.

Full match, label-free: 60.2 live minutes (D 43.1% of graded frames), 245 possessions, mean 14.7 s,
median about 9 s, 23 possessions under 2 s. Agreement with team-tagged phases is 0.828 attacking and
0.840 defending. 96.6% of set pieces have at least 1 s of D within 10 s of the tag start.

## Limitations

- **Restart team is the weakest part.** The best blind agreement with the set-piece tag is 11 of 23, close
  to chance. A control event, when one exists, agrees 7 of 10 times; half of the restarts have none. What
  the tag's `Team` means is itself an assumption.
- **First control is not always the taker.** After a contested corner it names who secured the ball.
- **Live time is high.** 60.2 minutes against the 52-58 prior.
- **Under-segmentation.** 13 possessions against 17 on the public window.
- **Control depends on the ball.** With the ball missing on 41% of graded frames, hand-overs in gaps are
  dated late or missed.
- **False restarts.** Any fast ball after a calm second in the search window counts, and a false restart
  can change the owner on its own.
- **Short stoppages.** A false D segment of 1 s splits one possession in two.
- **Score ties.** Viterbi breaks exact score ties toward the lower state index (H); we expect this to be
  rare but did not measure it.

With more time we would recover ball position in gaps from player movement, use the video to confirm
restarts, and make restart evidence depend on how much it can be trusted.

## Reproduction

```bash
python -m venv .venv
pip install -r requirements-model.txt
python make_submission.py
```

Python 3.11, with the data in `data/tactical/` (see README.md). The run is deterministic, takes under a
minute, writes `submissions/VISCA BARCA.csv` with LF line endings, validates it and prints its SHA-256.
A fresh clone with a fresh environment reproduced the identical file on Windows; the expected hash is in
README.md.
