"""Boolean operations (union / intersection / difference) on two simple
line-only contours with rational coordinates.

The nonzero-winding fill rule is preserved: a region cell is "inside" a
contour whenever its signed winding number is non-zero, so input contours
may be oriented either clockwise or counter-clockwise.

Scope (the mathematical subset this module contracts for):

* exactly **two** contours;
* each contour is **simple**: one closed loop, no holes, no
  self-intersections, no self-touching vertices, no overlapping edges of
  its own (collinear extra vertices on a straight boundary are allowed);
* line segments only -- contours containing quadratic Beziers are
  rejected with :class:`InvalidInputError` (a quadratic arc is not a
  rational-coordinate polygon boundary and flattening it would be an
  approximation, which this library never does silently);
* coordinates are rational numbers (every accepted int/float is an exact
  binary rational); all construction below uses :class:`fractions.Fraction`
  end to end, so tangencies, vertex-on-edge contacts and coincident
  collinear edges are resolved *exactly*, never with an epsilon.

Algorithm (implemented from scratch, no geometry library used):

1. Validate both inputs fully (type, line-only, simplicity) before any
   result object is built.
2. Split every edge at every intersection point with every other edge
   (proper crossings, T-junctions, endpoints of collinear overlaps).
3. Merge coincident subdivided edges (possibly carrying one directed
   contribution from each input contour) into a planar arrangement of
   half-edges.
4. Walk the arrangement faces with an exact angular ordering; propagate
   the pair of winding numbers ``(wa, wb)`` from the exterior face(s)
   using the per-edge winding jump (crossing a contour edge on the left
   changes that contour's winding by +1).
5. Select faces by the operation and trace their boundary loops, keeping
   the result region on the left (outer loops CCW, holes CW).  Degenerate
   zero-area contacts drop out automatically.

The result is a tuple of :class:`Contour` objects ready to feed into
:func:`winding_fill.rasterize`; an empty tuple means the empty region.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cmp_to_key
from typing import Dict, List, Sequence, Tuple

from .errors import InvalidInputError
from .geometry import Contour, Line, Point, Quad, Segment

F = Fraction
P2 = Tuple[Fraction, Fraction]

UNION = "union"
INTERSECTION = "intersection"
DIFFERENCE = "difference"
_OPS = (UNION, INTERSECTION, DIFFERENCE)


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BooleanResult:
    """Outcome of one boolean operation.

    Attributes:
        op: one of ``"union"`` / ``"intersection"`` / ``"difference"``;
        contours: boundary loops of the result region, outer loops
            oriented CCW and hole loops CW (an empty tuple means the
            empty region -- e.g. intersecting two disjoint polygons);
        area: exact signed area of the region (``Fraction``); with the
            loop orientation above it equals
            ``sum(signed_area(c) for c in contours)`` and is >= 0.
    """

    op: str
    contours: Tuple[Contour, ...]
    area: Fraction

    def __iter__(self):
        # Allow ``contours, area = boolean_union(...)`` style use while
        # keeping attribute access as the primary interface.
        yield self.contours
        yield self.area


# ---------------------------------------------------------------------------
# Exact predicates
# ---------------------------------------------------------------------------


def _sub(a: P2, b: P2) -> P2:
    return (a[0] - b[0], a[1] - b[1])


def _cross(a: P2, b: P2) -> Fraction:
    return a[0] * b[1] - a[1] * b[0]


def _cross3(o: P2, a: P2, b: P2) -> Fraction:
    return _cross(_sub(a, o), _sub(b, o))


def _dot(a: P2, b: P2) -> Fraction:
    return a[0] * b[0] + a[1] * b[1]


def _segment_intersections(a: P2, b: P2, c: P2, d: P2) -> List[P2]:
    """All contact points of segment [a,b] with [c,d], as Fractions.

    Returns [] (no contact), [p] (a proper crossing, an endpoint touch or
    a zero-length overlap) or [p, q] (a collinear overlap interval).
    Endpoint touches are included.
    """
    r = _sub(b, a)
    s = _sub(d, c)
    rxs = _cross(r, s)
    cma = _sub(c, a)
    if rxs != 0:
        t = _cross(cma, s) / rxs
        u = _cross(cma, r) / rxs
        if 0 <= t <= 1 and 0 <= u <= 1:
            return [(a[0] + t * r[0], a[1] + t * r[1])]
        return []
    # Parallel: collinear iff (c - a) is parallel to r.
    if _cross(cma, r) != 0:
        return []
    rr = _dot(r, r)
    tc = _dot(_sub(c, a), r) / rr
    td = _dot(_sub(d, a), r) / rr
    lo, hi = (tc, td) if tc <= td else (td, tc)
    lo = max(F(0), lo)
    hi = min(F(1), hi)
    if lo > hi:
        return []
    p = (a[0] + lo * r[0], a[1] + lo * r[1])
    if lo == hi:
        return [p]
    q = (a[0] + hi * r[0], a[1] + hi * r[1])
    return [p, q]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _line_edges_of(contour: Contour, where: str) -> List[Tuple[P2, P2]]:
    edges: List[Tuple[P2, P2]] = []
    for si, seg in enumerate(contour.segments):
        if isinstance(seg, Quad):
            raise InvalidInputError(
                where,
                f"segments[{si}] is a quadratic Bezier: boolean operations "
                "support simple line-only polygons (curved contours are not "
                "approximated)",
            )
        if not isinstance(seg, Line):  # pragma: no cover - defensive
            raise InvalidInputError(
                f"{where}.segments[{si}]",
                f"unsupported segment type {type(seg).__name__}",
            )
        edges.append(
            (
                (F(seg.p0.x), F(seg.p0.y)),
                (F(seg.p1.x), F(seg.p1.y)),
            )
        )
    return edges


def _assert_simple(
    edges: Sequence[Tuple[P2, P2]], where: str
) -> None:
    """Reject anything that is not one simple closed polygon loop."""
    n = len(edges)

    # Closure was already enforced on Contour, but re-check cheaply; the
    # boolean algorithm additionally needs every joint to be an actual
    # shared-vertex joint.
    for i, (a, b) in enumerate(edges):
        c = edges[(i + 1) % n][0]
        if b != c:
            raise InvalidInputError(
                f"{where}.segments[{i}]", "contour is not a closed vertex chain"
            )
        if a == b:
            raise InvalidInputError(
                f"{where}.segments[{i}]", "zero-length edge in contour"
            )

    # Repeated vertex (other than closure) makes a pinch or a degenerate
    # loop, both outside the "simple" contract.
    verts = [e[0] for e in edges]
    if len(set(verts)) != n:
        raise InvalidInputError(
            where, "contour repeats a vertex (not a simple polygon)"
        )

    for i in range(n):
        a, b = edges[i]
        for j in range(i + 1, n):
            c, d = edges[j]
            hits = _segment_intersections(a, b, c, d)
            if not hits:
                continue
            adjacent = (j == i + 1) or (i == 0 and j == n - 1)
            if adjacent:
                shared = b if j == i + 1 else a
                # Adjacent edges may meet at the shared joint and nowhere
                # else.  A backtracking/collinear overlap is rejected.
                if len(hits) == 1 and hits[0] == shared:
                    continue
                kind = (
                    "overlapping collinear adjacent edges"
                    if len(hits) == 2
                    else "adjacent edges meeting away from their joint"
                )
                raise InvalidInputError(
                    f"{where}.segments[{j}]", f"non-simple contour: {kind}"
                )
            raise InvalidInputError(
                f"{where}.segments[{j}]",
                "non-simple contour: edge intersects/touches "
                f"segments[{i}] at "
                f"({float(hits[0][0]):g}, {float(hits[0][1]):g})",
            )

    if polygon_signed_area2(edges) == 0:
        raise InvalidInputError(where, "contour has zero signed area")


def _validate(contour: object, where: str) -> List[Tuple[P2, P2]]:
    if not isinstance(contour, Contour):
        raise InvalidInputError(
            where,
            f"expected Contour, got {type(contour).__name__}; raw data goes "
            "through parse_contours() or boolean_op_raw()",
        )
    edges = _line_edges_of(contour, where)
    _assert_simple(edges, where)
    return edges


# ---------------------------------------------------------------------------
# Area helpers (public + internal)
# ---------------------------------------------------------------------------


def polygon_signed_area2(edges: Sequence[Tuple[P2, P2]]) -> Fraction:
    """Twice the signed area of a closed edge list (CCW positive)."""
    total = F(0)
    for (x0, y0), (x1, y1) in edges:
        total += x0 * y1 - x1 * y0
    return total


def signed_area(contour: Contour) -> Fraction:
    """Exact signed shoelace area of one line-only contour.

    Positive for CCW traversal, negative for CW.  Contours containing
    quadratic Beziers are rejected (their area needs curve integration
    and is outside this helper's contract).
    """
    edges = _line_edges_of(contour, "contour")
    return polygon_signed_area2(edges) / 2


# ---------------------------------------------------------------------------
# Planar arrangement
# ---------------------------------------------------------------------------


def _split_edges(
    edge_sets: Sequence[Sequence[Tuple[P2, P2]]],
) -> List[Tuple[P2, P2, int]]:
    """Split all edges at all mutual contact points.

    Returns directed subdivided edges tagged with their owner index.
    """
    tagged: List[Tuple[P2, P2, int]] = []
    for owner, edges in enumerate(edge_sets):
        for a0, b0 in edges:
            cuts: List[Tuple[Fraction, P2]] = [(F(0), a0), (F(1), b0)]
            r = _sub(b0, a0)
            rr = _dot(r, r)
            for other in edge_sets:
                for c, d in other:
                    for p in _segment_intersections(a0, b0, c, d):
                        t = _dot(_sub(p, a0), r) / rr
                        cuts.append((t, p))
            cuts.sort(key=lambda z: (z[0], z[1]))
            uniq: List[Tuple[Fraction, P2]] = []
            for t, p in cuts:
                if not uniq or p != uniq[-1][1]:
                    uniq.append((t, p))
            for k in range(len(uniq) - 1):
                tagged.append((uniq[k][1], uniq[k + 1][1], owner))
    return tagged


@dataclass
class _HE:
    u: int
    v: int
    twin: int


def _build_arrangement(
    edge_sets: Sequence[Sequence[Tuple[P2, P2]]],
):
    pieces = _split_edges(edge_sets)

    vmap: Dict[P2, int] = {}
    coords: List[P2] = []

    def vid(p: P2) -> int:
        i = vmap.get(p)
        if i is None:
            i = len(coords)
            vmap[p] = i
            coords.append(p)
        return i

    # canonical undirected geometric edge -> delta per owner along (lo->hi)
    geo: Dict[Tuple[int, int], List[int]] = {}
    for a, b, owner in pieces:
        u, v = vid(a), vid(b)
        key = (u, v) if u < v else (v, u)
        slot = geo.setdefault(key, [0, 0])
        slot[owner] += 1 if (u, v) == key else -1

    half: List[_HE] = []
    out: List[List[int]] = [[] for _ in coords]
    for (u, v), d in geo.items():
        h0 = len(half)
        h1 = h0 + 1
        half.append(_HE(u, v, h1))
        half.append(_HE(v, u, h0))
        out[u].append(h0)
        out[v].append(h1)

    # Exact CCW angular ordering of outgoing rays around each vertex.
    def angle_cmp(h1: int, h2: int) -> int:
        e1 = half[h1]
        e2 = half[h2]
        vx1, vy1 = _sub(coords[e1.v], coords[e1.u])
        vx2, vy2 = _sub(coords[e2.v], coords[e2.u])

        def upper(vx: Fraction, vy: Fraction) -> int:
            # +x axis starts the CCW sweep; upper half (y > 0) and +x.
            return 0 if (vy > 0 or (vy == 0 and vx > 0)) else 1

        up1, up2 = upper(vx1, vy1), upper(vx2, vy2)
        if up1 != up2:
            return -1 if up1 < up2 else 1
        cr = vx1 * vy2 - vy1 * vx2
        if cr > 0:
            return -1
        if cr < 0:
            return 1
        return 0

    order: List[List[int]] = []
    for v in range(len(coords)):
        lst = sorted(out[v], key=cmp_to_key(angle_cmp))
        order.append(lst)

    # Enumerate face cycles (the face is always on the left of each
    # directed half-edge): arriving at h's end vertex, take the outgoing
    # ray immediately clockwise of the twin ray.  Every cycle is simple:
    # it follows one angular sector at each vertex, so a face that only
    # touches itself at a point yields separate cycles.  A face with holes
    # yields several cycles (outer CCW + inner CW) -- they all receive the
    # same winding label below, so they never need to be merged.
    face_of = [-1] * len(half)
    cycles: List[List[int]] = []
    for h0 in range(len(half)):
        if face_of[h0] != -1:
            continue
        fid = len(cycles)
        cyc: List[int] = []
        h = h0
        while True:
            face_of[h] = fid
            cyc.append(h)
            e = half[h]
            v = e.v
            lst = order[v]
            pos = {hh: k for k, hh in enumerate(lst)}
            h = lst[(pos[e.twin] - 1) % len(lst)]
            if h == h0:
                break
        cycles.append(cyc)

    # Every geometric segment of the arrangement (undirected), used to
    # place an exact probe point strictly inside each face.
    all_segments = list(geo.keys())

    def label_cycle(cyc: List[int]) -> Tuple[int, int]:
        h = cyc[0]
        e = half[h]
        u, v = coords[e.u], coords[e.v]
        m = ((u[0] + v[0]) / 2, (u[1] + v[1]) / 2)
        d = (-(v[1] - u[1]), v[0] - u[0])  # exact left normal, into face
        dd = _dot(d, d)

        # First parameter t > 0 where the ray m + t*d meets ANY boundary
        # feature: an edge interior, an endpoint, or the start of a
        # collinear overlap.  The open ray before it lies strictly inside
        # the face, so its midpoint is an exact interior probe point --
        # vertex-grazing rays included (no epsilon).
        tmin = None

        def consider(t: Fraction) -> None:
            nonlocal tmin
            if t > 0 and (tmin is None or t < tmin):
                tmin = t

        for gu, gv in all_segments:
            a, b = coords[gu], coords[gv]
            r = _sub(b, a)
            den = _cross(r, d)
            am = _sub(a, m)
            if den != 0:
                t = -_cross(am, r) / den  # solve m + t d = a + s r
                s = -_cross(am, d) / den
                if 0 <= s <= 1:
                    consider(t)
            elif _cross(am, d) == 0:
                # Collinear: project the segment onto the ray.
                t0 = _dot(am, d) / dd
                t1 = _dot(_sub(b, m), d) / dd
                lo, hi = (t0, t1) if t0 <= t1 else (t1, t0)
                if hi > 0:
                    consider(max(F(0), lo))
        if tmin is None:
            q = (m[0] + d[0], m[1] + d[1])  # ray escapes: exterior face
        else:
            q = (m[0] + tmin / 2 * d[0], m[1] + tmin / 2 * d[1])
        return (
            _winding_of_point(edge_sets[0], q),
            _winding_of_point(edge_sets[1], q),
        )

    labels = [label_cycle(cyc) for cyc in cycles]
    wa = [w[0] for w in labels]
    wb = [w[1] for w in labels]

    return coords, half, order, face_of, wa, wb


def _winding_of_point(
    edges: Sequence[Tuple[P2, P2]], q: P2
) -> int:
    """Exact signed winding of q against one simple polygon edge list.

    Same contractual rule as the rasterizer: horizontal edges ignored,
    half-open span ``lo <= qy < hi``, upward edge +1 / downward -1.
    """
    qx, qy = q
    total = 0
    for (x0, y0), (x1, y1) in edges:
        if y1 > y0:
            lo, hi, direction = y0, y1, 1
        elif y1 < y0:
            lo, hi, direction = y1, y0, -1
        else:
            continue
        if lo <= qy < hi:
            x = x0 + (qy - y0) * (x1 - x0) / (y1 - y0)
            if x >= qx:
                total += direction
    return total


def _selected(op: str, wa: int, wb: int) -> bool:
    ina, inb = wa != 0, wb != 0
    if op == UNION:
        return ina or inb
    if op == INTERSECTION:
        return ina and inb
    # difference: A under the fill, B erased (nonzero rule preserved)
    return ina and not inb


# ---------------------------------------------------------------------------
# Boundary extraction
# ---------------------------------------------------------------------------


def _merge_collinear(pts: List[P2]) -> List[P2]:
    work = list(pts)
    while len(work) >= 3:
        out: List[P2] = []
        for i, p in enumerate(work):
            prev = work[i - 1]
            nxt = work[(i + 1) % len(work)]
            if _cross3(prev, p, nxt) != 0:
                out.append(p)
        if len(out) == len(work) or len(out) < 3:
            return out if len(out) >= 3 else []
        work = out
    return []


def _split_at_pinches(pts: List[P2]) -> List[List[P2]]:
    """Split a walk that revisits a vertex into separate loops.

    Two regions touching at a single point trace as one figure-eight
    walk; the pinch vertex is the boundary of two distinct components
    and must be reported as two simple loops.
    """
    first_at: Dict[P2, int] = {}
    loops: List[List[P2]] = []
    cur: List[P2] = []
    for p in pts:
        if p in first_at:
            k = first_at[p]
            sub = cur[k:]
            if len(sub) >= 3:
                loops.append(sub)
            cur = cur[:k]
            # The prefix cannot revisit the pinch again in a two-region
            # arrangement; rebuild the index for the shortened walk.
            first_at = {q: i for i, q in enumerate(cur)}
        first_at[p] = len(cur)
        cur.append(p)
    if len(cur) >= 3:
        loops.append(cur)
    return loops


def _trace_boundary(coords, half, order, face_of, wa, wb, op) -> List[List[P2]]:
    sel = [_selected(op, wa[f], wb[f]) for f in range(len(wa))]

    def is_boundary(h: int) -> bool:
        return sel[face_of[h]] and not sel[face_of[half[h].twin]]

    seen = [False] * len(half)
    walks: List[List[P2]] = []
    for h0 in range(len(half)):
        if seen[h0] or not is_boundary(h0):
            continue
        walk: List[P2] = []
        h = h0
        while True:
            seen[h] = True
            e = half[h]
            walk.append(coords[e.u])
            lst = order[e.v]
            pos = {hh: k for k, hh in enumerate(lst)}
            j = pos[e.twin]
            k = (j - 1) % len(lst)
            # Sectors immediately clockwise of the twin ray belong to the
            # result region while selected; leave along the ray at the far
            # clockwise end of that selected wedge (region stays left).
            while sel[face_of[lst[k]]]:
                k = (k - 1) % len(lst)
            h = lst[(k + 1) % len(lst)]
            if h == h0:
                break
            if seen[h] or not is_boundary(h):  # pragma: no cover - defensive
                raise RuntimeError("internal error: degenerate boundary trace")
        walks.append(walk)

    loops: List[List[P2]] = []
    for walk in walks:
        for sub in _split_at_pinches(walk):
            sub = _merge_collinear(sub)
            if not sub:
                continue
            if _loop_area2(sub) != 0:
                loops.append(sub)
    # Deterministic order: outer (CCW) loops first by decreasing area,
    # then hole (CW) loops; ties broken by the first vertex.
    loops.sort(key=lambda lp: (0 if _loop_area2(lp) > 0 else 1,
                               -abs(_loop_area2(lp)), lp[0]))
    return loops


def _loop_area2(pts: List[P2]) -> Fraction:
    return sum(
        pts[i][0] * pts[(i + 1) % len(pts)][1]
        - pts[(i + 1) % len(pts)][0] * pts[i][1]
        for i in range(len(pts))
    )


def _to_contour(pts: List[P2]) -> Contour:
    # Rotate the simple CCW/CW loop so it starts at its lexicographically
    # smallest vertex: makes the output independent of which input edge
    # the arrangement walk happened to start from.
    k = min(range(len(pts)), key=lambda i: pts[i])
    pts = pts[k:] + pts[:k]
    segs: List[Segment] = []
    for i in range(len(pts)):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % len(pts)]
        segs.append(
            Line(
                Point(float(x0), float(y0)),
                Point(float(x1), float(y1)),
            )
        )
    return Contour(tuple(segs))


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _boolean_loops(a: object, b: object, op: str) -> List[List[P2]]:
    """Validate both inputs and return result loops in exact Fractions."""
    if op not in _OPS:
        raise InvalidInputError(
            "op", f"unknown operation {op!r} (want one of {', '.join(_OPS)})"
        )
    # Full validation of both inputs happens before any arrangement work.
    edges_a = _validate(a, "contours[0]")
    edges_b = _validate(b, "contours[1]")
    coords, half, order, face_of, wa, wb = _build_arrangement([edges_a, edges_b])
    return _trace_boundary(coords, half, order, face_of, wa, wb, op)


def _boolean(a: object, b: object, op: str) -> BooleanResult:
    loops = _boolean_loops(a, b, op)
    contours = tuple(_to_contour(pts) for pts in loops)
    area = sum((_loop_area2(pts) for pts in loops), F(0)) / 2
    if area < 0:  # pragma: no cover - defensive
        raise RuntimeError("internal error: negative result area")
    return BooleanResult(op=op, contours=contours, area=area)


def boolean_union(a: Contour, b: Contour) -> BooleanResult:
    """Region covered by either contour (nonzero rule)."""
    return _boolean(a, b, UNION)


def boolean_intersection(a: Contour, b: Contour) -> BooleanResult:
    """Region covered by both contours at once."""
    return _boolean(a, b, INTERSECTION)


def boolean_difference(a: Contour, b: Contour) -> BooleanResult:
    """Region covered by ``a`` but not by ``b``."""
    return _boolean(a, b, DIFFERENCE)


def boolean_op_raw(raw_a: object, raw_b: object, op: str) -> BooleanResult:
    """Validate raw dict geometry and run one boolean operation.

    Each argument accepts either a glyph (a list containing exactly one
    contour, as :func:`winding_fill.parse_contours` does) or a bare
    segment list for one contour.  Both contours are fully parsed and
    validated before any computation, so a bad input never yields a
    partial result.
    """
    from .geometry import parse_contours

    def one(raw: object, where: str) -> Contour:
        if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], dict):
            raw = [raw]
        contours = parse_contours(raw)
        if len(contours) != 1:
            raise InvalidInputError(
                where,
                f"boolean operations take exactly one simple contour per "
                f"argument, got {len(contours)}",
            )
        return contours[0]

    contour_a = one(raw_a, "contours[0]")
    contour_b = one(raw_b, "contours[1]")
    return _boolean(contour_a, contour_b, op)
