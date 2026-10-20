"""Re-link player tracks that the tracker split in two.

ByteTrack matches boxes in image space, so a camera pan, an occlusion in the
paint or a missed detection often ends a player's track and starts a new ID a
few frames later. Downstream that reads as a player vanishing and a new one
appearing (a pop in the 3D view, a broken touch). Here a track that ends is
joined to a track that starts shortly after, when both are on the same team,
never on screen together, and the court distance between the end and the
start is reachable at a running speed. Candidates are taken cheapest first,
so each track gets at most one predecessor and one successor.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..types import PlayerTracks

CourtPositions = dict[int, list[Optional[tuple[float, float]]]]


def _ends(track: list[Optional[tuple[float, float]]], k: int = 5):
    idx = [f for f, p in enumerate(track) if p is not None]
    if not idx:
        return None
    first, last = idx[0], idx[-1]
    start = np.median([track[f] for f in idx[:k]], axis=0)
    end = np.median([track[f] for f in idx[-k:]], axis=0)
    return first, last, start, end


def stitch_tracks(tracks: PlayerTracks, court_pos: CourtPositions, team_of: dict[int, int], fps: float,
                  max_gap_s: float = 1.0, max_speed_mps: float = 6.0, slack_m: float = 1.0) -> dict[int, int]:
    """Return {track_id: id it should be renamed to}; ids not in the map keep their own."""
    seen: dict[int, list[int]] = {}
    for f, ft in enumerate(tracks):
        for tid in ft:
            seen.setdefault(tid, []).append(f)
    spans = {tid: (fs[0], fs[-1]) for tid, fs in seen.items()}
    ends = {tid: e for tid in spans if tid in court_pos and (e := _ends(court_pos[tid])) is not None}

    max_gap = max_gap_s * fps
    cands = []
    for a, (_, a_last, _, a_end) in ends.items():
        for b, (b_first, _, b_start, _) in ends.items():
            if a == b or team_of.get(a, -1) != team_of.get(b, -1):
                continue
            if spans[b][0] <= spans[a][1]:  # on screen together at some point (or b starts first)
                continue
            gap = b_first - a_last
            if gap <= 0 or gap > max_gap:
                continue
            reach = slack_m + max_speed_mps * gap / fps
            dist = float(np.hypot(*(b_start - a_end)))
            if dist <= reach:
                cands.append((dist / reach + gap / max_gap, a, b))

    nxt: dict[int, int] = {}
    prv: dict[int, int] = {}
    for _, a, b in sorted(cands):
        if a not in nxt and b not in prv:
            nxt[a], prv[b] = b, a

    remap = {}
    for tid in prv:
        root = tid
        while root in prv:
            root = prv[root]
        remap[tid] = root
    return remap


def apply_remap(tracks: PlayerTracks, remap: dict[int, int]) -> PlayerTracks:
    if not remap:
        return tracks
    return [{remap.get(tid, tid): d for tid, d in ft.items()} for ft in tracks]


def assign_identities(tracks: PlayerTracks, court_pos: CourtPositions, team_of: dict[int, int], fps: float,
                      max_per_team: int = 5, max_gap_s: float = 10.0, max_speed_mps: float = 8.0,
                      typical_speed_mps: float = 3.0, slack_m: float = 1.0
                      ) -> tuple[dict[int, int], dict[int, float], set[int]]:
    """Give every track one of at most ``max_per_team`` identities per team.

    A team has five players on the floor, so however often the tracker splits them, a clip holds at
    most five identities per team (substitutions aside). Tracks are taken in order of their first
    frame, the way a live stream would deliver them, and each is matched to an identity that is
    free (its last track has ended), reachable from where that player was last seen at
    ``max_speed_mps``, and within ``max_gap_s``; or it opens a new identity while the team has
    fewer than five. Tracks that start together (after a cut, every track restarts) are matched
    jointly. The cost of a link is the distance over what a player covers at ``typical_speed_mps``,
    and opening a new identity costs 1, so a track far from every lost player starts a new identity
    while the team still has room, and joins the most plausible one once all five are known. A
    player who left the picture during a pan and comes back seconds later gets his old id back.

    Returns (remap, confidence, extras): remap is {track_id: identity id} (the identity's first
    track), confidence is {identity id: 0-1}, the weakest link in its chain (1 with no links), and
    extras are tracks that fit no identity (a sixth player on a team at once: bench players,
    a wrong team call, a coach), which keep their own ids.
    """
    from scipy.optimize import linear_sum_assignment

    seen: dict[int, list[int]] = {}
    for f, ft in enumerate(tracks):
        for tid in ft:
            seen.setdefault(tid, []).append(f)
    ends = {tid: e for tid in seen if tid in court_pos and (e := _ends(court_pos[tid])) is not None}
    max_gap = max_gap_s * fps
    INF, EXTRA = 1e6, 1e3

    remap: dict[int, int] = {}
    conf: dict[int, float] = {}
    extras: set[int] = set()
    for team in sorted({team_of.get(t) for t in ends} - {None, -1}):
        slots: list[dict] = []  # {"root", "last", "end"}
        order = sorted((t for t in ends if team_of.get(t) == team), key=lambda t: seen[t][0])
        groups: dict[int, list[int]] = {}
        for t in order:
            groups.setdefault(seen[t][0], []).append(t)
        for first, group in sorted(groups.items()):
            free = [s for s in slots if s["last"] < first]
            room = max(max_per_team - len(slots), 0)
            cost = np.full((len(group), len(free) + room + len(group)), INF)
            for i, t in enumerate(group):
                start = ends[t][2]
                for j, s in enumerate(free):
                    gap = (first - s["last"]) / fps
                    dist = float(np.hypot(*(start - s["end"])))
                    if gap * fps <= max_gap and dist <= slack_m + max_speed_mps * gap:
                        cost[i, j] = dist / (slack_m + typical_speed_mps * gap)
                cost[i, len(free):len(free) + room] = 1.0
                cost[i, len(free) + room + i] = EXTRA
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                t = group[i]
                _, last, _, end = ends[t]
                if j < len(free):
                    s = free[j]
                    others = np.delete(cost[i], j)
                    second = others.min() if len(others) else INF
                    margin = 1.0 if second >= EXTRA else float(np.clip((second - cost[i, j]) / max(second, 1e-6), 0, 1))
                    remap[t] = s["root"]
                    conf[s["root"]] = min(conf[s["root"]], margin)
                    s["last"], s["end"] = last, end
                elif j < len(free) + room:
                    slots.append({"root": t, "last": last, "end": end})
                    conf[t] = 1.0
                else:
                    extras.add(t)
    return remap, conf, extras
