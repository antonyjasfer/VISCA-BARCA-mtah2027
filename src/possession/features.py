"""Per-frame features from the raw ATD. Every array is indexed by video frame (frame_id - 1).

Streams the raw tracking file with the same line format and team lookup as starter/baseline.py
(kloppy's to_df() on 199319 x 46 slots costs minutes and gigabytes; the raw file does not).

Distances and speeds are in metres and m/s: the 0-1 coordinates are scaled by the pitch size in
games.py on both axes before anything is compared.
"""

import os
import sys
from dataclasses import dataclass

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402
from starter.baseline import team_of_each_track  # noqa: E402

FPS = 25

# --- parameters (football reasoning, not tuned on any ground truth; see METHOD.md) -------------
# Ball gaps up to 10 frames (0.4 s) are bridged linearly: a ball cannot change direction much in
# 0.4 s except by being kicked, and short drop-outs are occlusions by a player's body. Anything
# longer is left NaN rather than invented.
MAX_INTERP_GAP = 10
# Ball is "outside the pitch" only if it is more than this far beyond a line, so tracker jitter
# on the line itself is not read as the ball leaving play. 0.3 m is under half a ball diameter
# of margin plus typical jitter; the ball is out when it has wholly crossed the line.
PITCH_MARGIN_M = 0.3
# Ball speed is a central difference over +/-4 frames (0.32 s). A single-frame difference
# multiplies position jitter by 25; this baseline is short enough to still see a pass.
SPEED_HALF_SPAN = 4


@dataclass
class Features:
    """One float32 array per feature, length n_frames. NaN means "not observable"."""
    ball_x: np.ndarray            # m, after short-gap interpolation
    ball_y: np.ndarray
    ball_observed: np.ndarray     # bool: ball seen by the tracker on this exact frame
    ball_speed: np.ndarray        # m/s
    d_home: np.ndarray            # m, ball to nearest tracked home player
    d_away: np.ndarray            # m, ball to nearest tracked away player
    ball_in_pitch: np.ndarray     # 1.0 inside, 0.0 outside (beyond margin), NaN if no ball
    mean_speed: np.ndarray        # m/s, mean over tracked players
    n_tracked: np.ndarray         # tracked slots with a position

    def as_dict(self):
        return dict(self.__dict__)


def parse_raw(raw_path, n_frames):
    """Raw ATD -> (player_xy [n,46,2] in 0-1, player_speed [n,46], ball_xy [n,2] in 0-1).

    Raw line: "<frame_id>:<x,y,s>;<x,y,s>;...:<x,y,s>" with NaN for a missing observation.
    """
    n_slots = None
    px = py = ps = None
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            tokens = parts[1].replace(";", ",").split(",")
            if n_slots is None:
                if len(tokens) % 3:
                    raise ValueError("player block is not a whole number of x,y,s triples")
                n_slots = len(tokens) // 3
                px = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
                py = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
                ps = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
            values = np.array(tokens, dtype=np.float64).reshape(n_slots, 3)
            px[frame], py[frame], ps[frame] = values[:, 0], values[:, 1], values[:, 2]

            ball_tokens = parts[2].strip().split(",")
            ball[frame, 0] = float(ball_tokens[0])
            ball[frame, 1] = float(ball_tokens[1])
    return px, py, ps, ball


def interpolate_short_gaps(series, max_gap=MAX_INTERP_GAP):
    """Linear fill of NaN runs of at most `max_gap` frames that have a real value on both sides.

    pandas' interpolate(limit=...) would fill the first `max_gap` frames of a longer gap, which
    invents a trajectory the tracker never saw; here a longer gap stays NaN in full.
    """
    out = series.astype(np.float64).copy()
    observed = np.flatnonzero(~np.isnan(out))
    if len(observed) < 2:
        return out
    missing = np.flatnonzero(np.isnan(out))
    right = np.searchsorted(observed, missing)
    inside = (right > 0) & (right < len(observed))
    missing, right = missing[inside], right[inside]
    gap = observed[right] - observed[right - 1] - 1
    fill = missing[gap <= max_gap]
    out[fill] = np.interp(fill, observed, out[observed])
    return out


def ball_speed_ms(x, y, half_span=SPEED_HALF_SPAN):
    """|p(t+h) - p(t-h)| / (2h / fps), NaN wherever either end is missing."""
    ahead_x, behind_x = np.roll(x, -half_span), np.roll(x, half_span)
    ahead_y, behind_y = np.roll(y, -half_span), np.roll(y, half_span)
    speed = np.hypot(ahead_x - behind_x, ahead_y - behind_y) / (2.0 * half_span / FPS)
    speed[:half_span] = np.nan
    speed[-half_span:] = np.nan
    return speed


def nearest_team_distance(px_m, py_m, ball_x, ball_y, mask):
    """Distance from the ball to the closest tracked player of the slots in `mask`; NaN if none."""
    dist = np.hypot(px_m[:, mask] - ball_x[:, None], py_m[:, mask] - ball_y[:, None])
    dist = np.where(np.isnan(dist), np.inf, dist)
    nearest = dist.min(axis=1)
    nearest[np.isinf(nearest)] = np.nan
    nearest[np.isnan(ball_x)] = np.nan
    return nearest


def compute(raw_path, meta_path, n_frames, pitch):
    length, width = pitch
    px, py, ps, ball = parse_raw(raw_path, n_frames)
    team = team_of_each_track(meta_path)
    if len(team) != px.shape[1]:
        raise ValueError("metadata lists {} slots but the raw file has {}".format(
            len(team), px.shape[1]))

    px_m = px.astype(np.float64) * length
    py_m = py.astype(np.float64) * width
    observed_x = ball[:, 0].astype(np.float64) * length
    observed_y = ball[:, 1].astype(np.float64) * width

    ball_x = interpolate_short_gaps(observed_x)
    ball_y = interpolate_short_gaps(observed_y)
    ball_y[np.isnan(ball_x)] = np.nan
    ball_x[np.isnan(ball_y)] = np.nan

    d_home = nearest_team_distance(px_m, py_m, ball_x, ball_y, team == 0)
    d_away = nearest_team_distance(px_m, py_m, ball_x, ball_y, team == 1)

    inside = ((ball_x >= -PITCH_MARGIN_M) & (ball_x <= length + PITCH_MARGIN_M)
              & (ball_y >= -PITCH_MARGIN_M) & (ball_y <= width + PITCH_MARGIN_M))
    in_pitch = np.where(np.isnan(ball_x), np.nan, inside.astype(np.float64))

    tracked = ~np.isnan(px)
    n_tracked = tracked.sum(axis=1)
    speed_ok = ~np.isnan(ps)
    mean_speed = np.where(speed_ok.sum(axis=1) > 0,
                          np.where(speed_ok, ps, 0.0).sum(axis=1) / np.maximum(speed_ok.sum(axis=1), 1),
                          np.nan)

    f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
    return Features(
        ball_x=f32(ball_x), ball_y=f32(ball_y),
        ball_observed=~np.isnan(observed_x),
        ball_speed=f32(ball_speed_ms(ball_x, ball_y)),
        d_home=f32(d_home), d_away=f32(d_away),
        ball_in_pitch=f32(in_pitch),
        mean_speed=f32(mean_speed),
        n_tracked=n_tracked.astype(np.int16),
    )


def load_features(game=GAME, data_dir=None, cache=None):
    """Features for the whole tactical video. `cache` is an optional .npz path; the cached arrays
    are the exact float32/bool/int16 arrays compute() returns, so the output does not depend on it."""
    if cache and os.path.isfile(cache):
        with np.load(cache) as archive:
            return Features(**{k: archive[k] for k in archive.files})
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    feats = compute(raw_path, meta_path, source_frames("tactical", game), GAMES[game]["pitch"])
    if cache:
        os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
        np.savez(cache, **feats.as_dict())
    return feats
