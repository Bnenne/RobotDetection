from collections import defaultdict


def iou(box_a: tuple, box_b: tuple) -> float:
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def suppress_duplicate_boxes(
    frame_robots: list[dict],
    iou_threshold: float = 0.5,
) -> list[dict]:
    """
    Remove boxes that overlap significantly with another box in the same frame.
    When two boxes overlap above the threshold, the smaller one is suppressed
    (larger box is more likely to be the full robot).
    """
    suppressed = set()

    for i, robot_i in enumerate(frame_robots):
        if i in suppressed:
            continue
        for j, robot_j in enumerate(frame_robots):
            if j <= i or j in suppressed:
                continue
            if iou(robot_i["box"], robot_j["box"]) > iou_threshold:
                bi, bj = robot_i["box"], robot_j["box"]
                area_i = (bi[2] - bi[0]) * (bi[3] - bi[1])
                area_j = (bj[2] - bj[0]) * (bj[3] - bj[1])
                suppressed.add(j if area_i >= area_j else i)

    return [r for i, r in enumerate(frame_robots) if i not in suppressed]


def _total_score(track_votes: dict, tid: int) -> float:
    return sum(track_votes[tid].values())


def _top_team(track_votes: dict, tid: int) -> int | None:
    votes = track_votes.get(tid, {})
    return max(votes, key=votes.get) if votes else None


def merge_lost_tracks(
    track_votes: dict[int, dict[int, float]],
    frame_data: dict[int, list[dict]],
) -> dict[int, int]:
    """
    Detect track_ids that are likely the same physical robot (tracker lost it
    and re-found it with a new ID) and group them.

    Two tracks are merged when:
      1. They never appear in the same frame simultaneously.
      2. Their top-voted team is the same.

    Returns a mapping { old_track_id -> canonical_track_id } where the
    canonical is whichever member of the group accumulated the most total score.
    """
    track_ids = list(track_votes.keys())

    # frame -> set of track_ids present in that frame
    frame_presence: dict[int, set[int]] = {
        frame_num: {r["track_id"] for r in robots if r["track_id"] != -1}
        for frame_num, robots in frame_data.items()
    }

    # Union-Find
    parent = {tid: tid for tid in track_ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px == py:
            return
        # canonical = the one with higher total score
        if _total_score(track_votes, px) >= _total_score(track_votes, py):
            parent[py] = px
        else:
            parent[px] = py

    for i, tid_a in enumerate(track_ids):
        for tid_b in track_ids[i + 1:]:
            co_exists = any(
                tid_a in presence and tid_b in presence
                for presence in frame_presence.values()
            )
            if co_exists:
                continue
            if (
                _top_team(track_votes, tid_a) is not None
                and _top_team(track_votes, tid_a) == _top_team(track_votes, tid_b)
            ):
                union(tid_a, tid_b)

    return {tid: find(tid) for tid in track_ids}


def apply_track_merge(
    track_votes: dict[int, dict[int, float]],
    mapping: dict[int, int],
) -> dict[int, dict[int, float]]:
    """
    Sum vote histograms for all tracks that share a canonical ID.
    Returns a new track_votes dict keyed only by canonical IDs.
    """
    merged: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for tid, votes in track_votes.items():
        canonical = mapping.get(tid, tid)
        for team, score in votes.items():
            merged[canonical][team] += score
    return dict(merged)