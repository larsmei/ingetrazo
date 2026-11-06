# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""Array-backed display mesh for dense imports (GLB and the like).

A large glTF is a triangle soup. Turning every triangle into a
:class:`core.mesh.Face` plus three edges makes orbiting, hovering and the
garbage collector slow long before the GPU does. A :class:`ReferenceMesh`
keeps positions and indices in contiguous arrays, draws from those, and
only builds Python faces when somebody asks to edit — and then only a
clustered, coplanar-merged working set.

Instances share one object. The viewport reads ``mesh._reference`` and
never walks ``mesh.faces`` while it is set.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Display and snap keep creases and borders, not every triangle seam.
CREASE_DOT = 0.985
# Hard edges uploaded for drawing and inference. Past this, coplanar
# seams are dropped first (see the viewport) and feature edges are capped.
FEATURE_EDGE_CAP = 80_000
# Entering a reference, or saving one, builds at most this many faces.
PROMOTE_MAX_TRIANGLES = 40_000


@dataclass
class ReferenceMesh:
    """Welded triangles plus the few edges worth drawing and snapping to.

    ``positions`` are float32, local to the prototype. ``triangles`` indexes
    them. ``colors`` is per triangle RGB in 0..1, or ``None`` for the
    viewport default. ``uvs`` is per corner ``(T, 3, 2)`` when ``texture``
    is set; those triangles are drawn textured and left out of the flat
    colour buffer.
    """

    positions: np.ndarray
    triangles: np.ndarray
    colors: np.ndarray | None = None
    texture: str | None = None
    uvs: np.ndarray | None = None
    feature_edges: np.ndarray = field(default_factory=lambda: np.zeros((0, 2, 3), np.float32))

    @property
    def triangle_count(self) -> int:
        return int(len(self.triangles))

    def transformed(self, matrix) -> "ReferenceMesh":
        """A copy with positions and feature edges mapped by a QMatrix4x4."""
        from core.group import np_affine
        rot, trans = np_affine(matrix)
        pos = (self.positions.astype(np.float64) @ rot.T + trans).astype(np.float32)
        edges = self.feature_edges
        if len(edges):
            flat = edges.reshape(-1, 3).astype(np.float64) @ rot.T + trans
            edges = flat.astype(np.float32).reshape(-1, 2, 3)
        return ReferenceMesh(pos, self.triangles, self.colors, self.texture,
                             self.uvs, edges)


def weld_triangles(corners: np.ndarray):
    """``(N, 3, 3)`` corner soup → ``(positions, triangles, keep)``.

    ``keep`` is the mask of input triangles that survived (degenerate
    triangles, collapsed by the weld, are dropped). The weld key matches
    :data:`core.mesh._KEY_SCALE`.
    """
    from core.mesh import _KEY_SCALE
    corners = np.asarray(corners, dtype=np.float64).reshape(-1, 3, 3)
    n = len(corners)
    if not n:
        return (np.zeros((0, 3), np.float32),
                np.zeros((0, 3), np.int32),
                np.zeros(0, dtype=bool))
    flat = np.ascontiguousarray(corners.reshape(-1, 3))
    keys = np.round(flat.astype(np.float32).astype(np.float64) * _KEY_SCALE)
    keys = np.ascontiguousarray(keys.astype(np.int64)).view(
        [("x", "<i8"), ("y", "<i8"), ("z", "<i8")]).reshape(-1)
    _u, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    inverse = rank[inverse].reshape(n, 3)
    keep = ((inverse[:, 0] != inverse[:, 1]) & (inverse[:, 1] != inverse[:, 2])
            & (inverse[:, 0] != inverse[:, 2]))
    pos = flat[first[order]].astype(np.float32)
    return pos, inverse[keep].astype(np.int32), keep


def from_batches(batches) -> ReferenceMesh:
    """Batches as :func:`formats.glb.read_triangles` yields them, one mesh.

    Each batch is ``(tris (N,3,3), uvs (N,3,2)|None, (tex_path, rgba, opacity))``.
    A single texture is kept when every triangle carries it; mixed materials
    fall back to each triangle's base colour.
    """
    corners = []
    cols = []
    uv_parts = []
    tex = None
    textured = True
    for tris, uv, recipe in batches:
        tex_path, rgba, _opacity = recipe
        tris = np.asarray(tris, dtype=np.float64)
        if tris.ndim != 3 or not len(tris):
            continue
        col = np.asarray(rgba[:3] if rgba else (1.0, 1.0, 1.0), dtype=np.float32)
        corners.append(tris)
        cols.append(np.broadcast_to(col, (len(tris), 3)).copy())
        if tex_path and uv is not None and len(uv) == len(tris):
            if tex is None:
                tex = tex_path
            elif tex_path != tex:
                textured = False
            uv_parts.append(np.asarray(uv, dtype=np.float32))
        else:
            textured = False
            uv_parts.append(np.zeros((len(tris), 3, 2), dtype=np.float32))
    if not corners:
        return ReferenceMesh(np.zeros((0, 3), np.float32),
                             np.zeros((0, 3), np.int32))
    all_c = np.concatenate(corners)
    all_col = np.concatenate(cols).astype(np.float32)
    pos, tris, keep = weld_triangles(all_c)
    cols_kept = all_col[keep]
    uvs = None
    if textured and tex is not None and len(uv_parts) == len(corners):
        uvs = np.concatenate(uv_parts)[keep]
    else:
        tex = None
    return ReferenceMesh(pos, tris, cols_kept, tex, uvs,
                         feature_edges(pos, tris))


def _colors_after_reweld(corners, cols):
    """Kept for callers that already hold a corner soup and a colour per triangle."""
    _pos, _tris, keep = weld_triangles(corners)
    return np.asarray(cols, dtype=np.float32)[keep]


def feature_edges(pos, tris, crease=CREASE_DOT, cap=FEATURE_EDGE_CAP):
    """Boundary edges, creases (dihedral below ``crease``) and junctions.

    ``(E, 2, 3)`` float32. Coplanar triangle seams — the usual cost of a
    GLB — are not included, so snap and the line pass stay on the shape.
    """
    pos = np.asarray(pos, dtype=np.float32)
    tris = np.asarray(tris, dtype=np.int64)
    tcount = len(tris)
    if tcount == 0:
        return np.zeros((0, 2, 3), np.float32)
    e = np.stack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=1)
    a = np.minimum(e[..., 0], e[..., 1]).astype(np.uint64)
    b = np.maximum(e[..., 0], e[..., 1]).astype(np.uint64)
    key = (a << np.uint64(32)) | b
    flat = key.reshape(-1)
    face_of = np.repeat(np.arange(tcount, dtype=np.int64), 3)
    order = np.argsort(flat, kind="mergesort")
    flat_s = flat[order]
    face_s = face_of[order]
    cuts = np.flatnonzero(np.r_[True, flat_s[1:] != flat_s[:-1]])
    ends = np.r_[cuts[1:], len(flat_s)]
    counts = ends - cuts
    p = pos[tris]
    nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]).astype(np.float64)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.maximum(ln, 1e-20)
    chosen = [np.flatnonzero(counts == 1), np.flatnonzero(counts > 2)]
    paired = np.flatnonzero(counts == 2)
    if len(paired):
        f0 = face_s[cuts[paired]]
        f1 = face_s[cuts[paired] + 1]
        dots = np.abs(np.einsum("ij,ij->i", nrm[f0], nrm[f1]))
        chosen.append(paired[dots < crease])
    if not any(len(c) for c in chosen):
        return np.zeros((0, 2, 3), np.float32)
    chosen = np.concatenate(chosen)
    if len(chosen) > cap:
        # Borders and junctions first (they were concatenated first).
        chosen = chosen[:cap]
    keys = flat_s[cuts[chosen]]
    ia = (keys >> np.uint64(32)).astype(np.int64)
    ib = (keys & np.uint64(0xFFFFFFFF)).astype(np.int64)
    return np.stack([pos[ia], pos[ib]], axis=1).astype(np.float32)


def cluster_simplify(pos, tris, target: int):
    """Vertex clustering until ``len(tris) <= target`` (or the grid bottoms out).

    Keeps the outer shape of a scan well enough to edit; it is not a
    quadric decimator. ``target`` at or above the input returns it unchanged.
    """
    pos = np.asarray(pos, dtype=np.float32)
    tris = np.asarray(tris, dtype=np.int64)
    if target <= 0 or len(tris) <= target:
        return pos, tris
    extent = pos.max(axis=0) - pos.min(axis=0)
    longest = float(extent.max()) if len(pos) else 0.0
    if longest <= 0:
        return pos, tris
    cell = longest / max(4.0, target ** (1.0 / 3.0))
    for _ in range(12):
        pos, tris = _cluster_once(pos, tris, cell)
        if len(tris) <= target or cell > longest:
            break
        cell *= 1.6
    return pos, tris


def _cluster_once(pos, tris, cell: float):
    if cell <= 0 or not len(tris):
        return pos, tris
    keys = np.floor(pos.astype(np.float64) / cell).astype(np.int64)
    view = np.ascontiguousarray(keys).view(
        [("x", "<i8"), ("y", "<i8"), ("z", "<i8")]).reshape(-1)
    _u, first, inverse = np.unique(view, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    inverse = rank[inverse]
    acc = np.zeros((len(first), 3), dtype=np.float64)
    cnt = np.zeros(len(first), dtype=np.float64)
    np.add.at(acc, inverse, pos)
    np.add.at(cnt, inverse, 1.0)
    pos = (acc / cnt[:, None]).astype(np.float32)
    tris = inverse[tris]
    keep = ((tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2])
            & (tris[:, 0] != tris[:, 2]))
    tris = tris[keep]
    used = np.unique(tris)
    remap = np.full(len(pos), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return pos[used], remap[tris].astype(np.int64)


def coplanar_loops(pos, tris, crease=CREASE_DOT):
    """Merge coplanar neighbouring triangles into boundary loops.

    Returns a list of vertex-index loops (no holes). A group that does not
    trace as simple loops is emitted as its original triangles, so nothing
    disappears.
    """
    pos = np.asarray(pos, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int64)
    if not len(tris):
        return []
    p = pos[tris]
    nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    unit = nrm / np.maximum(ln, 1e-20)
    flip = np.where(np.abs(unit[:, 2]) > 0.5, unit[:, 2] < 0, unit[:, 1] < 0)
    unit = np.where(flip[:, None], -unit, unit)
    d = np.einsum("ij,ij->i", unit, p[:, 0])
    q = np.column_stack([
        np.round(unit * 200.0).astype(np.int64),
        np.round(d * 200.0).astype(np.int64),
    ])
    _u, inverse = np.unique(np.ascontiguousarray(q).view(
        [("a", "<i8"), ("b", "<i8"), ("c", "<i8"), ("d", "<i8")]).reshape(-1),
        return_inverse=True)
    loops: list = []
    for g in range(int(inverse.max()) + 1 if len(inverse) else 0):
        sel = tris[inverse == g]
        if len(sel) == 1:
            loops.append(sel[0].tolist())
            continue
        traced = _trace_boundary(sel)
        if traced is None:
            loops.extend(sel.tolist())
        else:
            loops.extend(traced)
    return loops


def _trace_boundary(tris: np.ndarray):
    """Closed boundary loops of a triangle set, or ``None`` if non-simple."""
    a = np.minimum(tris[:, [0, 1, 2]], tris[:, [1, 2, 0]])
    b = np.maximum(tris[:, [0, 1, 2]], tris[:, [1, 2, 0]])
    key = (a.astype(np.uint64) << np.uint64(32)) | b.astype(np.uint64)
    flat = key.reshape(-1)
    ends = np.stack([tris[:, [0, 1, 2]], tris[:, [1, 2, 0]]], axis=-1).reshape(-1, 2)
    order = np.argsort(flat, kind="mergesort")
    flat_s = flat[order]
    ends_s = ends[order]
    cuts = np.flatnonzero(np.r_[True, flat_s[1:] != flat_s[:-1]])
    counts = np.diff(np.r_[cuts, len(flat_s)])
    border = counts == 1
    if not border.any():
        return None
    be = ends_s[cuts[border]]
    adj: dict = {}
    for u, v in be.tolist():
        adj.setdefault(int(u), []).append(int(v))
        adj.setdefault(int(v), []).append(int(u))
    if any(len(n) != 2 for n in adj.values()):
        return None
    pending = {frozenset(pair) for pair in be.tolist()}
    loops = []
    while pending:
        u, v = tuple(next(iter(pending)))
        pending.discard(frozenset((u, v)))
        loop = [u, v]
        prev, cur = u, v
        guard = 0
        while cur != loop[0] or guard == 0:
            guard += 1
            if guard > len(be) + 2:
                return None
            nbrs = adj[cur]
            nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]
            pending.discard(frozenset((cur, nxt)))
            if nxt == loop[0]:
                break
            loop.append(nxt)
            prev, cur = cur, nxt
        if len(loop) >= 3 and len(set(loop)) == len(loop):
            loops.append(loop)
        else:
            return None
    return loops or None


def attach(mesh, ref: ReferenceMesh) -> None:
    """Hang ``ref`` on an empty mesh. Drawing reads it; faces stay unbuilt."""
    mesh.__dict__["_reference"] = ref


def is_reference(mesh) -> bool:
    return mesh is not None and mesh.__dict__.get("_reference") is not None \
        and not mesh.faces and not mesh.edges


def promote_mesh(mesh, *, max_triangles: int = PROMOTE_MAX_TRIANGLES,
                 merge: bool = True) -> int:
    """Build editable faces from ``mesh._reference`` and drop the arrays.

    Over ``max_triangles``, vertices are clustered first. Coplanar seams
    become single faces when ``merge`` is set. Returns the face count, or
    0 when the mesh was not a reference.
    """
    ref = None if mesh is None else mesh.__dict__.get("_reference")
    if ref is None:
        return 0
    pos, tris = ref.positions, ref.triangles
    cols = ref.colors
    if len(tris) > max_triangles:
        # Colours are per original triangle; clustering drops that mapping.
        pos, tris = cluster_simplify(pos, tris, max_triangles)
        cols = None
    if merge and len(tris):
        loops = coplanar_loops(pos, tris)
    else:
        loops = tris.tolist()
    if not loops:
        mesh.__dict__.pop("_reference", None)
        return 0
    from core.mesh import Mesh
    assert isinstance(mesh, Mesh)
    flat = [pos[np.asarray(loop, dtype=np.int64)] for loop in loops]
    sizes = [len(loop) for loop in loops]
    pts = np.concatenate(flat).astype(np.float64)
    attrs = None
    if cols is not None and len(cols) == len(tris) and not merge:
        attrs = [{"color": [float(c) for c in cols[i]], "back": True}
                 for i in range(len(tris))]
    elif cols is not None and len(cols) and not (len(tris) > 0 and cols is None):
        # Merged: one colour, the mean, when every source triangle agreed
        # closely; otherwise leave the default paint.
        mean = cols.mean(axis=0)
        if float(np.abs(cols - mean).max()) < 1e-3:
            attrs = [{"color": [float(c) for c in mean], "back": True}] * len(loops)
    mesh.add_faces_bulk(pts, np.asarray(sizes, dtype=np.int64),
                        np.ones(len(loops), dtype=np.int64), attrs)
    mesh.__dict__.pop("_reference", None)
    mesh._chunk_dirty = True
    return len(mesh.faces)


def promote_scene(scene, **kwargs) -> int:
    """:func:`promote_mesh` on every placement. Used when saving, because
    ``.igz`` stores faces, not the display arrays."""
    if scene is None or not hasattr(scene, "groups"):
        return 0
    from core.group import iter_placements
    done: set = set()
    total = 0
    for g in scene.groups:
        for place, _m in iter_placements(g):
            mesh = place.mesh
            if mesh is None or id(mesh) in done:
                continue
            done.add(id(mesh))
            total += promote_mesh(mesh, **kwargs)
    return total


class _RefPick:
    """The one pick entity of a reference chunk: a click resolves to the
    group that owns it (the viewport pairs this with the placement). It is
    not a face of the mesh, so promoting stays explicit."""

    __slots__ = ("attrs", "ref")

    def __init__(self, ref: ReferenceMesh):
        self.attrs = {}
        self.ref = ref

    def normal(self):
        from PySide6.QtGui import QVector3D
        return QVector3D(0.0, 0.0, 1.0)

    def area(self) -> float:
        return 0.0

    @property
    def vertices(self):
        return []

    def centroid(self):
        from PySide6.QtGui import QVector3D
        if not len(self.ref.positions):
            return QVector3D()
        c = self.ref.positions.mean(axis=0)
        return QVector3D(float(c[0]), float(c[1]), float(c[2]))


def chunk_for(vp, ref: ReferenceMesh, mesh, msig, vkey, uid) -> dict:
    """A viewport chunk built from arrays: shaded triangles, optional
    texture, feature edges only. ``faces`` has one pick proxy so the index
    treats the whole reference as a single entity."""
    from core.style import DEFAULT_FACE_COLOR
    pos = np.asarray(ref.positions, dtype=np.float32)
    tris = np.asarray(ref.triangles, dtype=np.int64)
    ntri = len(tris)
    if ntri:
        p = pos[tris]                                   # (T, 3, 3)
        e1 = p[:, 1] - p[:, 0]
        e2 = p[:, 2] - p[:, 0]
        nrm = np.cross(e1.astype(np.float64), e2.astype(np.float64))
        ln = np.linalg.norm(nrm, axis=1, keepdims=True)
        unit = nrm / np.maximum(ln, 1e-20)
        light = np.array([vp._LIGHT.x(), vp._LIGHT.y(), vp._LIGHT.z()],
                         dtype=np.float64)
        d = np.abs(unit @ light)
        shade = np.rint((0.62 + 0.38 * d) * 32.0) / 32.0
        if ref.colors is not None and len(ref.colors) == ntri:
            base = ref.colors.astype(np.float64)
        else:
            base = np.array(DEFAULT_FACE_COLOR, dtype=np.float64)
        rgb = np.minimum(1.0, base * shade[:, None]).astype(np.float32)
        textured = (ref.texture and ref.uvs is not None
                    and len(ref.uvs) == ntri)
        vcol = np.empty((ntri, 3, 6), dtype=np.float32)
        vcol[:, :, :3] = p
        vcol[:, :, 3:] = rgb[:, None, :]
        if textured:
            # Textured triangles live in by_texture, not the flat colour pass.
            by_texture = {}
            stream = np.empty((ntri, 3, 5), dtype=np.float32)
            stream[:, :, :3] = p
            stream[:, :, 3:] = np.asarray(ref.uvs, dtype=np.float32)
            by_texture[ref.texture] = stream.reshape(-1).tobytes()
            vcol_bytes = b""
        else:
            by_texture = {}
            vcol_bytes = vcol.reshape(-1).tobytes()
        v0 = np.ascontiguousarray(p[:, 0], dtype=np.float64)
        e1d = np.ascontiguousarray(e1, dtype=np.float64)
        e2d = np.ascontiguousarray(e2, dtype=np.float64)
        corners = np.concatenate([v0, v0 + e1d, v0 + e2d])
        blo, bhi = corners.min(axis=0), corners.max(axis=0)
        bbox = (tuple(float(x) for x in blo), tuple(float(x) for x in bhi))
    else:
        vcol_bytes = b""
        by_texture = {}
        v0 = e1d = e2d = None
        bbox = None
    edges = np.asarray(ref.feature_edges, dtype=np.float32).reshape(-1, 3)
    nv = len(pos)
    samples = []
    if nv:
        idxs = sorted({k * max(nv - 1, 0) // 31 for k in range(min(32, nv))})
        samples = [(i, (float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])))
                   for i in idxs]
    empty_f = np.zeros(0, dtype=np.float64)
    empty_b = np.zeros(0, dtype=bool)
    return {
        "fp": (nv, 0, ntri, 0, 0, 0, 0),
        "vkey": vkey, "rev": 0, "uid": uid, "msig": msig,
        "serial": getattr(mesh, "_mut_serial", None),
        "nv": nv, "ne": len(edges) // 2, "nf": 1 if ntri else 0,
        "samples": samples, "coordsum": 0.0, "bbox": bbox, "obb": None,
        "edges": edges.tobytes() if len(edges) else b"",
        "vcol": vcol_bytes,
        "by_texture": by_texture,
        "tcol": {}, "ttex": {},
        "back_vcol": b"", "back_tex": {}, "back_tcol": {}, "back_ttex": {},
        "fvcol": b"", "dback": b"",
        "faces": [_RefPick(ref)] if ntri else [],
        "areas": np.array([float(ntri)], dtype=np.float64) if ntri else empty_f,
        "v0": v0, "e1": e1d, "e2": e2d,
        "tri_ent": np.zeros(ntri, dtype=np.int64) if ntri else None,
        "soft_pts": None,
        "soft_n0": empty_f, "soft_c0": empty_f,
        "soft_n1": empty_f, "soft_c1": empty_f,
        "soft_single": empty_b,
        "reference": ref,
        "mesh": mesh,
    }
