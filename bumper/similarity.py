from __future__ import annotations

# Perceptual digit-shape similarity, flat lookup table
# Pre-compute everything into a single flat tuple indexed by (ord(d1)-48)*10 + (ord(d2)-48).
# Avoids dict lookups entirely on the hot path.

def _build_similarity_table() -> tuple[float, ...]:
    segments: dict[int, frozenset[str]] = {
        0: frozenset("abcdef"),
        1: frozenset("bc"),
        2: frozenset("abdeg"),
        3: frozenset("abcdg"),
        4: frozenset("bcfg"),
        5: frozenset("acdfg"),
        6: frozenset("acdefg"),
        7: frozenset("abc"),
        8: frozenset("abcdefg"),
        9: frozenset("abcdfg"),
    }

    perceptual_floor: dict[tuple[int, int], float] = {
        (0, 6): 0.60, (0, 8): 0.55, (0, 9): 0.55,
        (1, 7): 0.50, (3, 8): 0.50, (5, 9): 0.65,
        (6, 8): 0.65, (6, 9): 0.60, (8, 9): 0.70,
    }

    table = [0.0] * 100
    for a in range(10):
        sa = segments[a]
        for b in range(10):
            if a == b:
                sim = 1.0
            else:
                sb = segments[b]
                sim = len(sa & sb) / len(sa | sb)
            # Apply perceptual floor (check both orderings)
            key = (min(a, b), max(a, b))
            floor = perceptual_floor.get(key, 0.0)
            table[a * 10 + b] = max(sim, floor)
    return tuple(table)


_SIM_TABLE: tuple[float, ...] = _build_similarity_table()
_ORD0 = 48  # ord('0')

def digit_similarity(d1: str, d2: str) -> float:
    """Return perceptual similarity [0, 1] between two digit characters."""
    return _SIM_TABLE[(ord(d1) - _ORD0) * 10 + (ord(d2) - _ORD0)]


# Positional alignment score
# Precompute digit counts via a fixed-size array [0..9] instead of Counter.

def _positional_score(s1: str, s2: str) -> float:
    len1, len2 = len(s1), len(s2)
    max_len = max(len1, len2)

    # Right-align by zero-padding
    if len1 < max_len:
        p1 = s1.zfill(max_len)
        p2 = s2
    elif len2 < max_len:
        p1 = s1
        p2 = s2.zfill(max_len)
    else:
        p1 = s1
        p2 = s2

    # Positional similarity via direct table lookup
    sim_table = _SIM_TABLE
    ord0 = _ORD0
    positional_total = 0.0
    pad1 = len1 < len2
    pad2 = len2 < len1

    for i in range(max_len):
        c1 = p1[i]
        c2 = p2[i]
        # Padded zeros score 0
        if pad1 and i < max_len - len1 and c1 == '0':
            continue
        if pad2 and i < max_len - len2 and c2 == '0':
            continue
        positional_total += sim_table[(ord(c1) - ord0) * 10 + (ord(c2) - ord0)]

    positional = positional_total / max_len

    # Anagram bonus for same-length numbers using fixed array
    if len1 == len2:
        counts1 = [0] * 10
        counts2 = [0] * 10
        for ch in s1:
            counts1[ord(ch) - ord0] += 1
        for ch in s2:
            counts2[ord(ch) - ord0] += 1
        shared = 0
        for k in range(10):
            a, b = counts1[k], counts2[k]
            if a < b:
                shared += a
            else:
                shared += b
        bonus = 0.20 * (shared / max_len)
        result = positional + bonus
        return result if result <= 1.0 else 1.0

    return positional


# Length penalty

def _length_score(len1: int, len2: int) -> float:
    if len1 == len2:
        return 1.0
    if len1 < len2:
        ratio = len1 / len2
    else:
        ratio = len2 / len1
    return ratio * (ratio ** 0.5)  # equivalent to ratio ** 1.5


# Structural pattern detection
# Encode patterns as bit flags for fast comparison instead of set operations.

_PAT_ALL_SAME   = 1
_PAT_PALINDROME = 2
_PAT_ASCENDING  = 4
_PAT_DESCENDING = 8
_PAT_REPEAT     = 16  # any repeating block

def _detect_patterns(s: str) -> int:
    flags = 0
    n = len(s)

    # All same digit
    ch0 = s[0]
    all_same = True
    for i in range(1, n):
        if s[i] != ch0:
            all_same = False
            break
    if all_same:
        flags |= _PAT_ALL_SAME

    # Palindrome (length > 1)
    if n > 1:
        is_pal = True
        half = n >> 1
        for i in range(half):
            if s[i] != s[n - 1 - i]:
                is_pal = False
                break
        if is_pal:
            flags |= _PAT_PALINDROME

    # Ascending / descending (3+ distinct digits)
    if n >= 3:
        distinct = len(set(s))
        if distinct >= 3:
            asc = True
            desc = True
            for i in range(1, n):
                if s[i] < s[i - 1]:
                    asc = False
                if s[i] > s[i - 1]:
                    desc = False
                if not asc and not desc:
                    break
            if asc:
                flags |= _PAT_ASCENDING
            if desc:
                flags |= _PAT_DESCENDING

    # Repeating block
    for bl in range(1, n // 2 + 1):
        if n % bl != 0:
            continue
        block = s[:bl]
        match = True
        for start in range(bl, n, bl):
            if s[start:start + bl] != block:
                match = False
                break
        if match:
            flags |= _PAT_REPEAT
            break  # found shortest repeating block, that's enough

    return flags


def _popcount(x: int) -> int:
    c = 0
    while x:
        c += 1
        x &= x - 1
    return c


def _structural_score(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0

    p1 = _detect_patterns(s1)
    p2 = _detect_patterns(s2)

    if p1 == 0 and p2 == 0:
        return 0.45  # neither has special structure
    if p1 == 0 or p2 == 0:
        return 0.25  # one structured, one not

    shared = _popcount(p1 & p2)
    union = _popcount(p1 | p2)
    jaccard = shared / union if union else 0.5
    return 0.35 + 0.45 * jaccard


# Combined similarity function

_W_POSITIONAL  = 0.60
_W_LENGTH      = 0.25
_W_STRUCTURAL  = 0.15


def visual_similarity(a: int | str, b: int | str) -> float:
    """
    Return a visual similarity score in [0.0, 1.0] between two non-negative
    integers (or their string representations).

    1.0 = identical, 0.0 = maximally dissimilar.
    """
    s1 = str(int(str(a)))
    s2 = str(int(str(b)))

    if s1 == s2:
        return 1.0

    pos   = _positional_score(s1, s2)
    leng  = _length_score(len(s1), len(s2))
    struc = _structural_score(s1, s2)

    pos_squashed = pos * pos  # pos ** 2.0 but faster for floats

    score = _W_POSITIONAL * pos_squashed + _W_LENGTH * leng + _W_STRUCTURAL * struc

    if score < 0.0:
        score = 0.0
    elif score > 1.0:
        score = 1.0

    # Round to 4 decimal places without calling round()
    score = int(score * 10000 + 0.5) / 10000
    return score