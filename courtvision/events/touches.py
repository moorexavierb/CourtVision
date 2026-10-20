"""Touch-by-touch table: the main data product of the pipeline.

One row per touch, where a touch is one player's continuous control of the
ball (a possession segment). Each row says:

* who: ``player_id``, team, and jersey / name when a roster mapping is given
* how the touch began (``start_action``): catch, steal, rebound, recovery ...
* what the player did with it (``action``): pass, shot, turnover, or the clip
  ended, with ``outcome``, ``target_id`` and shot details where they apply,
  plus the dribbles taken during the touch
* where: the ball handler's court position at the start and at the action,
  speed, distance covered, distance to the basket and to the nearest defender
* everyone else on the floor at the action frame, as wide columns: teammates
  ``tm1..tm4`` and opponents ``op1..op5``, each nearest-to-the-handler first,
  with id, x, y, speed and distance to the handler
* ``confidence``: 0-1, how much to trust the row (see ``_confidence``), and
  ``id_confidence``: 0-1, how sure we are that ``player_id`` is one person
  across the clip (low when a re-join after an occlusion or pan was a close
  call between teammates)

Positions are court metres (origin at a corner, x along the length), speeds
are km/h. Rows with no court calibration keep their ids and actions, with the
positional columns left empty.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..court.geometry import CourtSpec
from ..types import Event, PipelineResult, Possession, bbox_height

TEAMMATE_SLOTS = 4
OPPONENT_SLOTS = 5
SLOT_FIELDS = ("id", "x_m", "y_m", "speed_kmh", "dist_m")
CONTEST_RADIUS_M = 2.0  # a defender this close to the handler counts as contesting

START_ACTIONS = {"pass": "catch", "interception": "steal", "turnover": "recovery",
                 "offensive_rebound": "offensive_rebound", "defensive_rebound": "defensive_rebound"}


def _at(series: dict, tid: int, f: int):
    track = series.get(tid)
    return track[f] if track is not None and 0 <= f < len(track) else None


def _r(v, nd=2):
    return None if v is None else round(float(v), nd)


def count_dribbles(seg: Possession, result: PipelineResult, low: float = 0.75, high: float = 0.6) -> int:
    """Bounces during a touch: the ball drops to the bottom of the holder's box (near the floor)
    after being up at the hands. Image space, so it works without a court calibration."""
    count, armed = 0, False
    for f in range(seg.start_frame, seg.end_frame + 1):
        p, b = result.players[f].get(seg.player_id), result.ball[f]
        if p is None or b is None:
            continue
        y1 = p["bbox"][1]
        frac = ((b[1] + b[3]) / 2 - y1) / max(bbox_height(p["bbox"]), 1.0)
        if frac < high:
            armed = True
        elif frac > low and armed:
            count, armed = count + 1, False
    return count


def _index_events(events: list[Event]):
    """Map events onto the touches they start and end, keyed by (player, frame)."""
    starts, ends = {}, {}
    for e in events:
        catch, release = e.extra.get("catch_frame"), e.extra.get("release_frame")
        if e.type == "pass":
            ends[(e.player_id, release)] = e
            starts[(e.target_id, catch)] = e
        elif e.type == "interception":
            starts[(e.player_id, catch)] = e
            ends[(e.target_id, release)] = e
        elif e.type == "turnover":
            ends[(e.player_id, release)] = e
            starts[(e.target_id, catch)] = e
        elif e.type in ("offensive_rebound", "defensive_rebound"):
            starts[(e.player_id, catch)] = e
        elif e.type == "shot" and e.player_id is not None:
            ends[(e.player_id, release)] = e  # a shot wins over anything else ending the touch
    return starts, ends


def _end_action(e: Optional[Event], seg: Possession, nxt: Optional[Possession]) -> dict:
    if e is None:
        if nxt is None:
            return {"action": "end_of_clip", "outcome": None, "target_id": None}
        if nxt.player_id == seg.player_id:
            return {"action": "fumble", "outcome": "kept", "target_id": None}
        return {"action": "loose_ball", "outcome": None, "target_id": nxt.player_id}
    if e.type == "shot":
        pts = e.extra.get("points")
        return {"action": "shot", "outcome": "made" if e.extra.get("made") else "missed", "target_id": None,
                "shot_points": pts, "shot_distance_m": e.extra.get("distance_m")}
    if e.type == "pass":
        return {"action": "pass", "outcome": "completed", "target_id": e.target_id,
                "pass_air_s": e.extra.get("air_frames")}
    if e.type == "interception":
        return {"action": "pass", "outcome": "intercepted", "target_id": e.player_id,
                "pass_air_s": e.extra.get("air_frames")}
    return {"action": "turnover", "outcome": "lost_ball", "target_id": e.target_id}


def _confidence(seg: Possession, result: PipelineResult, start_known: bool, end_known: bool,
                min_frames: int = 10) -> float:
    """Product of four 0-1 factors: detector confidence on the handler, share of the touch with a court
    position, length of the touch (very short touches are often a ball passing in front of someone),
    and whether its start and end were explained by a detected event."""
    frames = range(seg.start_frame, seg.end_frame + 1)
    confs = [result.players[f][seg.player_id]["conf"] for f in frames if seg.player_id in result.players[f]]
    det = float(np.mean(confs)) if confs else 0.0
    if result.court_positions:
        located = sum(_at(result.court_positions, seg.player_id, f) is not None for f in frames) / len(frames)
    else:
        located = 0.0
    length = min(1.0, len(frames) / min_frames)
    explained = 0.7 + 0.15 * start_known + 0.15 * end_known
    return round(det * (0.5 + 0.5 * located) * length * explained, 3)


def touches_table(possessions: list[Possession], result: PipelineResult, team_names: list[str],
                  court: Optional[CourtSpec] = None, player_info: Optional[dict[int, dict]] = None) -> pd.DataFrame:
    """One row per touch; see the module docstring for the columns.

    ``player_info`` maps track id -> {"jersey": ..., "name": ...}, e.g. from a roster table joined on
    team + jersey number. Without it those columns are left empty.
    """
    info = player_info or {}
    starts, ends = _index_events(result.events)
    fps = result.fps
    rows = []
    team_poss, last_team = 0, None
    for i, seg in enumerate(possessions):
        prev = possessions[i - 1] if i else None
        nxt = possessions[i + 1] if i + 1 < len(possessions) else None
        se = starts.get((seg.player_id, seg.start_frame))
        ee = ends.get((seg.player_id, seg.end_frame))
        if seg.team != last_team:
            team_poss, last_team = team_poss + 1, seg.team

        if se is not None:
            start_action = START_ACTIONS[se.type]
        else:
            start_action = "start_of_clip" if prev is None else "recovery"
        end = _end_action(ee, seg, nxt)
        if end.get("pass_air_s") is not None:
            end["pass_air_s"] = round(end["pass_air_s"] / fps, 2)

        f0, f = seg.start_frame, seg.end_frame
        pid = seg.player_id
        start_xy = _at(result.court_positions, pid, f0)
        xy = _at(result.court_positions, pid, f)
        d0, d1 = _at(result.distance_m, pid, f0), _at(result.distance_m, pid, f)
        speeds = [s for s in (_at(result.speed_kmh, pid, g) for g in range(f0, f + 1)) if s is not None]
        me = info.get(pid, {})
        target = end.get("target_id")

        row = {
            "touch_id": i, "possession_id": team_poss,
            "start_frame": f0, "end_frame": f,
            "start_s": round(f0 / fps, 2), "end_s": round(f / fps, 2),
            "duration_s": round((f - f0 + 1) / fps, 2),
            "player_id": pid, "jersey": me.get("jersey"), "name": me.get("name"),
            "team": team_names[seg.team] if seg.team in (0, 1) else None,
            "start_action": start_action, "action": end["action"], "outcome": end["outcome"],
            "target_id": target, "target_name": info.get(target, {}).get("name") if target is not None else None,
            "dribbles": count_dribbles(seg, result),
            "shot_points": end.get("shot_points"), "shot_distance_m": _r(end.get("shot_distance_m")),
            "pass_air_s": end.get("pass_air_s"),
            "start_x_m": _r(start_xy[0]) if start_xy else None, "start_y_m": _r(start_xy[1]) if start_xy else None,
            "x_m": _r(xy[0]) if xy else None, "y_m": _r(xy[1]) if xy else None,
            "speed_kmh": _r(_at(result.speed_kmh, pid, f), 1),
            "max_speed_kmh": _r(max(speeds), 1) if speeds else None,
            "distance_covered_m": _r(d1 - d0) if d0 is not None and d1 is not None else None,
            "dist_to_basket_m": _r(court.shot_distance(*xy)) if court is not None and xy else None,
        }

        # Everyone else on the floor at the action frame, nearest to the handler first.
        mates, opps = [], []
        for tid in result.players[f]:
            if tid == pid:
                continue
            team = result.team_of.get(tid, -1)
            pos = _at(result.court_positions, tid, f)
            dist = float(np.hypot(pos[0] - xy[0], pos[1] - xy[1])) if pos and xy else None
            entry = {"id": tid, "x_m": _r(pos[0]) if pos else None, "y_m": _r(pos[1]) if pos else None,
                     "speed_kmh": _r(_at(result.speed_kmh, tid, f), 1), "dist_m": _r(dist)}
            if team == seg.team and team in (0, 1):
                mates.append(entry)
            elif team in (0, 1):
                opps.append(entry)
        order = lambda e: (e["dist_m"] is None, e["dist_m"] or 0.0, e["id"])  # noqa: E731
        mates.sort(key=order)
        opps.sort(key=order)

        near = opps[0] if opps and opps[0]["dist_m"] is not None else None
        row.update({
            "nearest_defender_id": near["id"] if near else None,
            "nearest_defender_m": near["dist_m"] if near else None,
            "defenders_within_2m": sum(o["dist_m"] is not None and o["dist_m"] <= CONTEST_RADIUS_M for o in opps)
            if xy else None,
            "players_tracked": 1 + len(mates) + len(opps),
        })
        for prefix, group, n in (("tm", mates, TEAMMATE_SLOTS), ("op", opps, OPPONENT_SLOTS)):
            for k in range(n):
                e = group[k] if k < len(group) else {}
                for field in SLOT_FIELDS:
                    row[f"{prefix}{k + 1}_{field}"] = e.get(field)
        row["confidence"] = _confidence(seg, result, se is not None, ee is not None or nxt is None)
        # how sure we are this id is one person throughout (1 = never re-joined on a close call)
        row["id_confidence"] = round(result.identity_conf.get(pid, 1.0), 3)
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        # Nullable ints, so ids stay 7 rather than 7.0 when some rows are empty.
        ids = ["target_id", "shot_points", "nearest_defender_id", "defenders_within_2m"]
        ids += [c for c in df.columns if c[:2] in ("tm", "op") and c.endswith("_id")]
        df[ids] = df[ids].astype("Int64")
    return df
