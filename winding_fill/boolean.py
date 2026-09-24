"""Exact-rational boolean set operations on simple line-only contours.

This module adds union / intersection / difference to the library while
keeping the nonzero-winding fill rule as the single source of truth:

* inputs are *simple* (Jordan), hole-free contours made exclusively of
  :class:`~winding_fill.geometry.Line` segments with rational
  coordinates (every accepted float is an exact binary rational, turned
  into a :class:`fractions.Fraction`);
* each operand may be one contour or a list of them (1..MAX_CONTOURS),
  just like :func:`~winding_fill.core.rasterize`;
* the result is again a list of contours whose nonzero-winding fill is
  exactly the requested set, naturally oriented (outer boundaries CCW,
  holes CW), so it can be fed straight back into the rasterizer.

Everything geometric here is computed **exactly** -- no tolerances, no
float comparisons -- in three stages:

1. **Validate.** Every input contour is proved simple: non-adjacent edges
   may not meet at all (even a touch is illegal for a Jordan curve),
   adjacent edges may meet only at their common endpoint, and no vertex
   may repeat.  A self-intersecting contour raises
   :class:`~winding_fill.errors.SelfIntersectionError` *before* anything
   is computed; operands are never mutated.

2. **Build a planar arrangement (``O(E^2)`` edge pairs).** Every pair of
   edges is intersected exactly (proper crossings, endpoint contacts and
   full/partial collinear overlap share the same rational code).  Edges
   are split at every contact parameter and coincident sub-edges are
   merged, producing a proper planar straight-line graph whose faces are
   open regions of constant winding pair ``(w_a, w_b)``.

3. **Classify faces and trace boundaries with symbolic perturbation.**
   Winding numbers are evaluated exactly at ``midpoint + epsilon * n``
   using dual numbers ``a + b*eps`` (``eps`` a positive infinitesimal,
   ``eps^2 = 0``): a representative sitting exactly on an edge or on a
   vertex y-level is resolved by the ``eps`` component, so tangencies
   need no tolerance.  Faces are selected by the operation predicate
   (union: ``w_a != 0 or w_b != 0``; intersection: both nonzero;
   difference: ``w_a != 0 and w_b == 0``); the arrangement edges
   separating a selected face from a non-selected one are assembled into
   closed loops, traversed with the kept region on the left.

The exact area (sum of the oriented loop areas, holes negative) is
reported alongside the contours, which gives an independent hand-check
target in addition to rasterization.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cmp_to_key
from typing import Callable, Dict, List, Sequence, Tuple

from .errors import InvalidInputError, SelfIntersectionError
from .geometry import MAX_CONTOURS, Contour, Line, Point, parse_contours

# Public operation names.
UNION = "union"
INTERSECTION = "intersection"
DIFFERENCE = "difference"
_OPERATIONS = (UNION, INTERSECTION, DIFFERENCE)

# A rational point; a tagged rational edge ``p0 -> p1`` (tag = operand).
_RPoint = Tuple[Fraction, Fraction]
_REdge = Tuple[_RPoint, _RPoint, int]

_PREDICATES: Dict[str, Callable[[int, int], bool]] = {
    UNION: lambda wa, wb: wa != 0 or wb != 0,
    INTERSECTION: lambda wa, wb: wa != 0 and wb != 0,
    DIFFERENCE: lambda wa, wb: wa != 0 and wb == 0,
}


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BooleanResult:
    """Outcome of one boolean set operation.

    Attributes:
        operation: one of ``"union"`` / ``"intersection"`` /
            ``"difference"``.
        contours: closed, naturally oriented result contours (outer loops
            CCW, holes CW) as ordinary library
            :class:`~winding_fill.geometry.Contour` objects, ready for
            :func:`~winding_fill.core.rasterize`; the empty tuple when
            the resulting set has zero area (e.g. a disjoint
            intersection).  Vertices that arise from intersections are
            rounded to ``float`` here (see the error-budget section of
            the README); the unrounded values live in :attr:`polygons`.
        polygons: the same loops as exact rational vertex chains
            (tuples of :class:`fractions.Fraction` pairs, open chains,
            no repeated closing point).
        area: exact area as a :class:`fractions.Fraction` -- the signed
            sum over the result loops, hence holes subtract.
        contour_areas: exact signed area of each result loop.
    """

    operation: str
    contours: Tuple[Contour, ...]
    polygons: Tuple[Tuple[_RPoint, ...], ...]
    area: Fraction
    contour_areas: Tuple[Fraction, ...]

    @property
    def is_empty(self) -> bool:
        """True when the resulting set has zero area."""
        return not self.contours or self.area == 0

    @property
    def area_float(self) -> float:
        """The exact area converted to ``float``."""
        return float(self.area)

    def rasterize(self, width: int, height: int):
        """Rasterize the result with the nonzero-winding fill rule.

        An empty (zero-area) result yields an all-zero canvas instead of
        being rejected for having no contours.  Otherwise this is
        :func:`~winding_fill.core.rasterize` applied to :attr:`contours`.
        """
        from .core import CoverageResult, rasterize
        from .geometry import canvas_dims

        width, height = canvas_dims(width, height)
        if not self.contours:
            zero_w = tuple(0 for _ in range(16))
            return CoverageResult(
                width=width,
                height=height,
                levels=tuple(tuple(0 for _ in range(width))
                             for _ in range(height)),
                windings=tuple(
                    tuple(zero_w for _ in range(width))
                    for _ in range(height)
                ),
            )
        return rasterize(list(self.contours), width, height)


# ---------------------------------------------------------------------------
# Validation and conversion to rational edges
# ---------------------------------------------------------------------------


def polygon_area(contour: Contour) -> Fraction:
    """Exact signed shoelace area of one line-only contour.

    Positive for CCW traversal (coordinate system y up), negative for CW.
    Useful as an independent check value; accepts the same contour kind
    the boolean operations accept.  Raises
    :class:`~winding_fill.errors.InvalidInputError` for a non-contour or
    a contour containing quadratic Beziers, whose area this does not
    compute.
    """
    if not isinstance(contour, Contour):
        raise InvalidInputError(
            "contour", f"expected Contour, got {type(contour).__name__}"
        )
    if not contour.segments:
        raise InvalidInputError("contour", "contour has no segments")
    for si, seg in enumerate(contour.segments):
        if not isinstance(seg, Line):
            raise InvalidInputError(
                f"contour.segments[{si}]",
                "polygon_area computes line-only contours (no quadratic "
                "Beziers)",
            )
    total = Fraction(0)
    first = (
        Fraction(contour.segments[0].p0.x),
        Fraction(contour.segments[0].p0.y),
    )
    prev = first
    for seg in contour.segments[1:]:
        cur = (Fraction(seg.p0.x), Fraction(seg.p0.y))
        total += prev[0] * cur[1] - prev[1] * cur[0]
        prev = cur
    total += prev[0] * first[1] - prev[1] * first[0]
    return total / 2


def _normalize_operand(value: object, where: str) -> List[Contour]:
    if isinstance(value, Contour):
        return [value]
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise InvalidInputError(where, "at least one contour is required")
        if len(value) > MAX_CONTOURS:
            raise InvalidInputError(
                where, f"too many contours: {len(value)} > {MAX_CONTOURS}"
            )
        for i, con in enumerate(value):
            if not isinstance(con, Contour):
                raise InvalidInputError(
                    f"{where}[{i}]",
                    f"expected Contour, got {type(con).__name__}",
                )
        return list(value)
    raise InvalidInputError(
        where,
        "expected a Contour or a list/tuple of Contours; raw data goes "
        "through parse_contours()/boolean_raw()",
    )


def _intersection_params(
    p: _RPoint, q: _RPoint, r: _RPoint, s: _RPoint
) -> Tuple[List[Fraction], List[Fraction]]:
    """Exact contact parameters of segments ``p->q`` and ``r->s``.

    Returns ``(ts, us)``: parameters on each segment where the other
    geometry meets it (subset of ``[0, 1]``); empty tuples when the
    segments are disjoint.  Collinear overlap reports the endpoints of
    the overlap interval: a point tangency yields one parameter per
    side, a real overlap yields two.
    """
    px, py = p
    qx, qy = q
    rx, ry = r
    sx, sy = s
    dx, dy = qx - px, qy - py
    ex, ey = sx - rx, sy - ry
    fx, fy = rx - px, ry - py

    den = dx * ey - dy * ex
    if den != 0:
        t = (fx * ey - fy * ex) / den
        u = (fx * dy - fy * dx) / den
        if 0 <= t <= 1 and 0 <= u <= 1:
            return [t], [u]
        return [], []

    # Parallel: collinear only if r lies on p->q's supporting line.
    if fx * dy - fy * dx != 0:
        return [], []

    dd = dx * dx + dy * dy
    ee = ex * ex + ey * ey
    t_r = ((rx - px) * dx + (ry - py) * dy) / dd
    t_s = ((sx - px) * dx + (sy - py) * dy) / dd
    u_p = ((px - rx) * ex + (py - ry) * ey) / ee
    u_q = ((qx - rx) * ex + (qy - ry) * ey) / ee
    ts = [t for t in (t_r, t_s) if 0 <= t <= 1]
    us = [u for u in (u_p, u_q) if 0 <= u <= 1]
    return ts, us


def _assert_simple(verts: List[_RPoint], cwhere: str) -> None:
    """Reject a closed polygon walk that is not a simple Jordan curve.

    ``verts`` holds ``m + 1`` points with ``verts[m] == verts[0]``.
    """
    m = len(verts) - 1

    seen: Dict[_RPoint, int] = {}
    for i in range(m):
        prev = seen.get(verts[i])
        if prev is not None:
            raise SelfIntersectionError(
                cwhere,
                f"vertex {i} ({float(verts[i][0])}, {float(verts[i][1])}) "
                f"duplicates vertex {prev}; a boolean input must be a "
                "simple contour without repeated vertices or "
                "self-contact",
            )
        seen[verts[i]] = i

    for i in range(m):
        p, q = verts[i], verts[(i + 1) % m]
        for j in range(i + 1, m):
            r, s = verts[j], verts[(j + 1) % m]
            ts, us = _intersection_params(p, q, r, s)
            if not ts:
                continue
            adjacent = (j == i + 1) or (i == 0 and j == m - 1)
            if not adjacent:
                raise SelfIntersectionError(
                    cwhere,
                    f"non-adjacent segments {i} and {j} meet (cross or "
                    "touch); a boolean input must be a simple contour "
                    "without self-intersection",
                )
            # Adjacent pair: the only legal contact is the shared vertex.
            if i == 0 and j == m - 1:
                allowed_t, allowed_u = Fraction(0), Fraction(1)
            else:
                allowed_t, allowed_u = Fraction(1), Fraction(0)
            if any(t != allowed_t for t in ts) or any(
                u != allowed_u for u in us
            ):
                raise SelfIntersectionError(
                    cwhere,
                    f"adjacent segments {i} and {j} overlap or cross "
                    "away from their shared endpoint; the contour is "
                    "not a simple Jordan curve",
                )


def _operand_edges(
    contours: Sequence[Contour], side: int, where: str
) -> List[_REdge]:
    edges: List[_REdge] = []
    for ci, con in enumerate(contours):
        cwhere = f"{where}[{ci}]"
        verts: List[_RPoint] = []
        for si, seg in enumerate(con.segments):
            if not isinstance(seg, Line):
                raise InvalidInputError(
                    f"{cwhere}.segments[{si}]",
                    "boolean operations accept line-only contours "
                    "(quadratic Beziers are not part of this iteration)",
                )
            if si == 0:
                verts.append((Fraction(seg.p0.x), Fraction(seg.p0.y)))
            verts.append((Fraction(seg.p1.x), Fraction(seg.p1.y)))
        _assert_simple(verts, cwhere)
        m = len(verts) - 1
        for i in range(m):
            edges.append((verts[i], verts[i + 1], side))
    return edges


def _prepare(
    a: object, b: object, op: object
) -> Tuple[str, List[_REdge], List[_REdge]]:
    if not isinstance(op, str):
        raise InvalidInputError(
            "op", f"expected a string, got {type(op).__name__}"
        )
    if op not in _PREDICATES:
        raise InvalidInputError(
            "op", f"unknown operation {op!r} (want one of {_OPERATIONS})"
        )
    contours_a = _normalize_operand(a, "a")
    contours_b = _normalize_operand(b, "b")
    # Side a is validated completely before side b is touched; on any
    # failure nothing is returned and inputs are left untouched.
    edges_a = _operand_edges(contours_a, 0, "a")
    edges_b = _operand_edges(contours_b, 1, "b")
    return op, edges_a, edges_b


# ---------------------------------------------------------------------------
# Arrangement construction
# ---------------------------------------------------------------------------


def _split_edges(all_edges: List[_REdge]) -> List[Tuple[_RPoint, _RPoint]]:
    """Split edges at every contact parameter; merge coincident pieces."""
    n = len(all_edges)
    params: List[set] = [set() for _ in range(n)]
    for ei in range(n):
        params[ei].add(Fraction(0))
        params[ei].add(Fraction(1))

    for i in range(n):
        p, q, _ = all_edges[i]
        for j in range(i + 1, n):
            r, s, _ = all_edges[j]
            ts, us = _intersection_params(p, q, r, s)
            if ts:
                params[i].update(ts)
                params[j].update(us)

    unique: Dict[Tuple[_RPoint, _RPoint], None] = {}
    for ei, (p, q, _) in enumerate(all_edges):
        px, py = p
        dx, dy = q[0] - px, q[1] - py
        pts = [
            (px + t * dx, py + t * dy)
            for t in sorted(params[ei])
        ]
        for k in range(len(pts) - 1):
            u, v = pts[k], pts[k + 1]
            if u == v:
                continue
            key = (u, v) if u <= v else (v, u)
            unique[key] = None
    return list(unique.keys())


# ---------------------------------------------------------------------------
# Symbolic-perturbation winding numbers (dual numbers a + b*eps)
# ---------------------------------------------------------------------------


def _winding_dual(
    edges: Sequence[_REdge],
    x0: Fraction, x1: Fraction,
    y0: Fraction, y1: Fraction,
) -> int:
    """Signed winding at ``(x0 + x1*eps, y0 + y1*eps)``, eps > 0 tiny.

    Same contract as the rasterizer: half-open y interval, horizontal
    edges ignored, crossings with ``x_cross >= sample_x`` counted with
    the traversal direction.  Comparisons are evaluated in the dual ring
    so an equality on the primary coordinates is broken by the
    infinitesimal component -- never by a tolerance.
    """
    total = 0
    for p0, p1, _side in edges:
        ax, ay = p0
        bx, by = p1
        if by == ay:
            continue
        if by > ay:
            lo, hi, direction = ay, by, 1
        else:
            lo, hi, direction = by, ay, -1
        # lo <= y0 + y1*eps ?
        if not (y0 > lo or (y0 == lo and y1 >= 0)):
            continue
        # y0 + y1*eps < hi ?
        if not (y0 < hi or (y0 == hi and y1 < 0)):
            continue
        dy = by - ay
        dx = bx - ax
        # (x_cross - sx) * dy, with eps coefficient separated:
        num = (ax - x0) * dy + (y0 - ay) * dx
        num_eps = -x1 * dy + y1 * dx
        if dy > 0:
            to_the_right = num > 0 or (num == 0 and num_eps >= 0)
        else:
            to_the_right = num < 0 or (num == 0 and num_eps <= 0)
        if to_the_right:
            total += direction
    return total


# ---------------------------------------------------------------------------
# Half-edge structure, face classification, boundary loop extraction
# ---------------------------------------------------------------------------


def _angle_half(d: _RPoint) -> int:
    dx, dy = d
    return 0 if (dy > 0 or (dy == 0 and dx > 0)) else 1


def _angle_cmp(u: _RPoint, v: _RPoint) -> int:
    """Compare direction vectors by CCW angle, +x first (exact)."""
    hu, hv = _angle_half(u), _angle_half(v)
    if hu != hv:
        return -1 if hu < hv else 1
    cross = u[0] * v[1] - u[1] * v[0]
    if cross > 0:
        return -1
    if cross < 0:
        return 1
    return 0


def _build_arrangement(segments):
    """Half-edge planar embedding of the merged arrangement segments.

    Half-edge ``2k`` walks segments[k][0] -> segments[k][1]; ``2k+1`` is
    its reverse.  Faces are cycled with the face on the LEFT: on arrival
    at a vertex, continue on the outgoing edge immediately clockwise
    (cyclic predecessor in CCW order) of the reverse edge.

    Returns ``(face_of, face_point, n_faces, h_start, h_end)`` where
    ``face_point[fid]`` is the representative ``(x0,x1,y0,y1)`` for the
    face used by the dual-number winding query.
    """
    vid: Dict[_RPoint, int] = {}
    coords: List[_RPoint] = []
    h_start: List[int] = []
    h_end: List[int] = []

    def vertex(p: _RPoint) -> int:
        if p not in vid:
            vid[p] = len(coords)
            coords.append(p)
        return vid[p]

    for p, q in segments:
        u, v = vertex(p), vertex(q)
        h_start.extend((u, v))
        h_end.extend((v, u))

    nhe = len(h_start)

    def hvec(h: int) -> _RPoint:
        p, q = segments[h >> 1]
        if h & 1:
            return (p[0] - q[0], p[1] - q[1])
        return (q[0] - p[0], q[1] - p[1])

    sort_key = cmp_to_key(lambda h1, h2: _angle_cmp(hvec(h1), hvec(h2)))
    outgoing: Dict[int, List[int]] = {}
    for h in range(nhe):
        outgoing.setdefault(h_start[h], []).append(h)
    for hs in outgoing.values():
        hs.sort(key=sort_key)

    face_of = [-1] * nhe
    face_point: List[Tuple[Fraction, Fraction, Fraction, Fraction]] = []

    for start in range(nhe):
        if face_of[start] != -1:
            continue
        fid = len(face_point)
        h = start
        while True:
            face_of[h] = fid
            v = h_end[h]
            hs = outgoing[v]
            idx = hs.index(h ^ 1)
            h = hs[(idx - 1) % len(hs)]
            if h == start:
                break
        # Representative: midpoint of the first arc, nudged by an
        # infinitesimal step along its left normal, which points into
        # the face on this arc's left.
        p, q = segments[start >> 1]
        if start & 1:
            (ax, ay), (bx, by) = q, p
        else:
            (ax, ay), (bx, by) = p, q
        mx, my = (ax + bx) / 2, (ay + by) / 2
        dx, dy = bx - ax, by - ay
        face_point.append((mx, -dy, my, dx))

    return face_of, face_point, len(face_point), h_start, h_end


def _halfedge_dir(h: int, segments) -> _RPoint:
    """Travel direction vector of half-edge ``h``."""
    p, q = segments[h >> 1]
    if h & 1:
        return (p[0] - q[0], p[1] - q[1])
    return (q[0] - p[0], q[1] - p[1])


def _extract_boundary_loops(
    segments, face_of, n_faces, kept, h_start, h_end
) -> List[List[_RPoint]]:
    """Trace result boundary loops with the kept region on the left.

    Each geometric segment contributes exactly one directed boundary
    half-edge: the one whose left face is kept and whose right face is
    not.  At a vertex the continuation is purely angular: from the
    reverse of the incoming arc, walk clockwise through the complete ring
    of outgoing arcs and take the first boundary arc encountered.  This is
    the standard boundary-of-a-face-set rule -- it rounds convex and
    concave corners alike even when the unkept side switches between two
    distinct faces, and at a point tangency it keeps two merely-touching
    loops apart without any straightest-turn heuristic.
    """
    boundary: List[int] = []
    for k in range(len(segments)):
        left_kept = kept[face_of[2 * k]]
        right_kept = kept[face_of[2 * k + 1]]
        if left_kept and not right_kept:
            boundary.append(2 * k)
        elif right_kept and not left_kept:
            boundary.append(2 * k + 1)
    boundary_set = set(boundary)

    # Full outgoing ring per vertex (every half-edge), sorted CCW from +x.
    all_out: Dict[int, List[int]] = {}
    nhe = len(segments) * 2
    for h in range(nhe):
        all_out.setdefault(h_start[h], []).append(h)
    angle_key = cmp_to_key(
        lambda g1, g2: _angle_cmp(
            _halfedge_dir(g1, segments), _halfedge_dir(g2, segments)
        )
    )
    for hs in all_out.values():
        hs.sort(key=angle_key)

    remaining = set(boundary)
    raw_loops: List[List[_RPoint]] = []

    while remaining:
        start = next(iter(remaining))
        start_vertex = h_start[start]
        verts: List[_RPoint] = []
        h = start
        while True:
            remaining.remove(h)
            k = h >> 1
            p, q = segments[k]
            verts.append(q if h & 1 else p)
            v = h_end[h]
            ring = all_out[v]
            # Locate the twin (reverse incoming) in the CCW ring, then
            # scan its clockwise (cyclic-predecessor) side for the next
            # still-unused boundary arc.  Already consumed arcs -- the
            # incoming one included -- are skipped automatically.
            idx = ring.index(h ^ 1)
            nxt = None
            for step in range(1, len(ring) + 1):
                cand = ring[(idx - step) % len(ring)]
                if cand in remaining:
                    nxt = cand
                    break
            if nxt is None:
                if v != start_vertex:  # pragma: no cover - defensive
                    raise RuntimeError(
                        "internal error: boundary chain did not close"
                    )
                break
            h = nxt
        raw_loops.append(verts)

    # A walk that revisits a vertex (internal tangency, e.g. a hole
    # touching the outer boundary at one point) is a "lasso" of simple
    # cycles sharing that vertex; split it so every returned contour is a
    # simple loop.  The cycles share the vertex geometrically, which is
    # exactly what a point-tangent result looks like.
    loops: List[List[_RPoint]] = []
    for verts in raw_loops:
        loops.extend(_split_pinch_walk(verts))
    return loops


def _split_pinch_walk(verts: List[_RPoint]) -> List[List[_RPoint]]:
    """Split a closed vertex walk at repeated vertices into simple cycles."""
    cycles: List[List[_RPoint]] = []
    prefix: List[_RPoint] = []
    at: Dict[_RPoint, int] = {}
    for z in verts:
        if z in at:
            i = at[z]
            cycles.append(prefix[i:])
            del prefix[i:]
            for w in list(at):
                if at[w] >= i:
                    del at[w]
        at[z] = len(prefix)
        prefix.append(z)
    if prefix:
        cycles.append(prefix)
    return cycles


def _simplify(verts: List[_RPoint]) -> List[_RPoint]:
    """Drop vertices that sit collinearly (same direction) on a straight run.

    A shared-edge junction leaves a redundant vertex on a straight side;
    removing it makes a merged rectangle a true 4-vertex rectangle.  A
    pinch reversal (incoming/outgoing collinear but opposite) is kept,
    since it is a genuine corner of a zero-width contact.
    """
    m = len(verts)
    keep = [True] * m
    for i in range(m):
        p = verts[i - 1]
        q = verts[i]
        r = verts[(i + 1) % m]
        ux, uy = q[0] - p[0], q[1] - p[1]
        vx, vy = r[0] - q[0], r[1] - q[1]
        if ux * vy - uy * vx == 0 and ux * vx + uy * vy > 0:
            keep[i] = False
    out = [v for v, k in zip(verts, keep) if k]
    return out if len(out) >= 3 else verts


def _loops_to_contours(
    loops: List[List[_RPoint]],
) -> Tuple[
    Tuple[Contour, ...],
    Tuple[Tuple[_RPoint, ...], ...],
    Fraction,
    Tuple[Fraction, ...],
]:
    built: List[Tuple[Fraction, Tuple[_RPoint, ...], Contour]] = []
    for raw_verts in loops:
        verts = _simplify(raw_verts)
        m = len(verts)
        twice = Fraction(0)
        for i in range(m):
            x0, y0 = verts[i]
            x1, y1 = verts[(i + 1) % m]
            twice += x0 * y1 - y0 * x1
        area = twice / 2
        segs = tuple(
            Line(
                Point(float(verts[i][0]), float(verts[i][1])),
                Point(float(verts[(i + 1) % m][0]),
                      float(verts[(i + 1) % m][1])),
            )
            for i in range(m)
        )
        built.append((area, tuple(verts), Contour(segs)))

    # Deterministic, natural reading order: outer (CCW, positive) loops
    # first by decreasing area, then holes (CW, negative) by decreasing
    # area; equal areas break on the first vertex.
    built.sort(key=lambda item: (
        0 if item[0] > 0 else 1,
        -item[0] if item[0] > 0 else item[0],
        item[1][0],
    ))
    contours = tuple(item[2] for item in built)
    polygons = tuple(item[1] for item in built)
    areas = tuple(item[0] for item in built)
    return contours, polygons, sum(areas, Fraction(0)), areas


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def boolean(a: object, b: object, op: object) -> BooleanResult:
    """Set operation on two simple line-only contour operands.

    ``a`` and ``b`` are each either one :class:`Contour` or a list of
    them; ``op`` is ``"union"``, ``"intersection"`` or ``"difference"``
    (``a`` minus ``b``).  Every operand contour must be a simple Jordan
    polygon of line segments or :class:`SelfIntersectionError` is raised
    before any result is built.  Quadratic Beziers are rejected with
    :class:`InvalidInputError` in this iteration.

    Returns a :class:`BooleanResult` whose contours reproduce exactly the
    requested set under the nonzero-winding fill rule.
    """
    op, edges_a, edges_b = _prepare(a, b, op)
    segments = _split_edges(edges_a + edges_b)
    face_of, face_point, n_faces, h_start, h_end = _build_arrangement(
        segments
    )
    predicate = _PREDICATES[op]
    kept = [False] * n_faces
    for fid, (x0, x1, y0, y1) in enumerate(face_point):
        wa = _winding_dual(edges_a, x0, x1, y0, y1)
        wb = _winding_dual(edges_b, x0, x1, y0, y1)
        kept[fid] = predicate(wa, wb)

    loops = _extract_boundary_loops(
        segments, face_of, n_faces, kept, h_start, h_end
    )
    contours, polygons, area, areas = _loops_to_contours(loops)
    return BooleanResult(
        operation=op,
        contours=contours,
        polygons=polygons,
        area=area,
        contour_areas=areas,
    )


def union(a: object, b: object) -> BooleanResult:
    """``a`` union ``b`` -- see :func:`boolean`."""
    return boolean(a, b, UNION)


def intersection(a: object, b: object) -> BooleanResult:
    """``a`` intersect ``b`` -- see :func:`boolean`."""
    return boolean(a, b, INTERSECTION)


def difference(a: object, b: object) -> BooleanResult:
    """``a`` minus ``b`` -- see :func:`boolean`."""
    return boolean(a, b, DIFFERENCE)


def boolean_raw(raw_a: object, raw_b: object, op: object) -> BooleanResult:
    """Validate raw dict/tuple geometry and apply ``op`` in one call.

    Each operand uses the same format as
    :func:`winding_fill.geometry.parse_contours` (a list of contours).
    Both operands are fully parsed before the operation runs, so bad
    input never yields a partial result.  Error locations are prefixed
    with the operand side (``a.`` / ``b.``), matching the typed API's
    ``a[i]`` / ``b[i]`` scheme.
    """
    def parse_side(raw: object, side: str) -> List[Contour]:
        try:
            return parse_contours(raw)
        except InvalidInputError as exc:
            raise InvalidInputError(
                f"{side}.{exc.location}", exc.message
            ) from None

    contours_a = parse_side(raw_a, "a")
    contours_b = parse_side(raw_b, "b")
    return boolean(contours_a, contours_b, op)
