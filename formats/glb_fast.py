# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""Fast path for large glTF/GLB imports.

:func:`formats.glb.load_glb` delegates here. A file whose expanded triangle
count is at or under :data:`REFERENCE_TRIANGLES` is handed back to the
eager welder, unchanged. Above that:

* shared glTF meshes become component instances (one weld per prototype);
* a prototype that is itself over the cap is a
  :class:`core.reference_mesh.ReferenceMesh`: arrays for drawing and
  snapping, Python faces only when :func:`core.reference_mesh.promote_mesh`
  is asked.

Parsing and the array build do not touch a ``Scene``, so
:func:`prepare_glb` can run on a worker thread.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from formats import glb as glb_format

#: Expanded triangles at or under this stay one editable welded group, as
#: before. Above it a prototype is a full-fidelity reference: every triangle
#: is drawn, and it is not editable until Make editable is asked for.
REFERENCE_TRIANGLES = 10_000


def _route(n_unique: int, n_expanded: int) -> str:
    if n_expanded <= REFERENCE_TRIANGLES:
        return "legacy"
    if n_unique <= REFERENCE_TRIANGLES:
        return "instances"
    return "reference"


def read_structure(path: Path):
    """Shared prototypes in Z-up metres, copies kept as matrices.

    ``matrix`` is row-major. Prototype positions are already Z-up, and
    ``matrix`` satisfies ``p_world = p_local @ matrix[:3,:3].T + matrix[:3,3]``,
    which is what :func:`formats.glb.read_triangles` bakes per node.
    """
    path = Path(path)
    js, buffers = glb_format.parse_container(path)
    for ext in js.get("extensionsRequired", []):
        if "draco" in ext.lower():
            raise ValueError(
                "Draco-compressed GLB — re-export/download uncompressed")
    image_paths = glb_format._extract_images(js, buffers, path)
    zup = np.array([[1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0]])
    z4 = np.eye(4)
    z4[:3, :3] = zup
    meshes = js.get("meshes", [])
    nodes = js.get("nodes", [])
    proto_of: dict = {}
    prototypes: list = []
    placed = []
    for node_idx, world in glb_format._node_transforms(js):
        nd = nodes[node_idx]
        mi = nd.get("mesh")
        if mi is None or mi >= len(meshes):
            continue
        slot = proto_of.get(mi)
        if slot is None:
            batches = []
            n = 0
            for prim in meshes[mi].get("primitives", []):
                if prim.get("mode", 4) != 4:
                    continue
                attr = prim.get("attributes", {})
                if "POSITION" not in attr:
                    continue
                pos = glb_format._accessor(js, buffers, attr["POSITION"])[:, :3]
                if "indices" in prim:
                    idx = glb_format._accessor(js, buffers, prim["indices"]).reshape(-1)
                else:
                    idx = np.arange(len(pos), dtype=np.int64)
                idx = idx.astype(np.int64)[:(len(idx) // 3) * 3].reshape(-1, 3)
                tex_path, uvt, rgba, uv_set, opacity = glb_format._material_recipe(
                    js, prim.get("material"), image_paths)
                local = pos @ zup.T
                tris = local[idx]
                uvs = None
                uv_key = f"TEXCOORD_{uv_set}"
                if tex_path is not None and uv_key in attr:
                    uv = glb_format._accessor(js, buffers, attr[uv_key])[:, :2]
                    ox, oy, sx, sy = uvt
                    uv = uv * (sx, sy) + (ox, oy)
                    uvs = uv[idx]
                elif tex_path is not None:
                    tex_path = None
                n += len(tris)
                batches.append((tris, uvs, (tex_path, rgba, opacity)))
            if not n:
                continue
            slot = len(prototypes)
            proto_of[mi] = slot
            prototypes.append({"batches": batches, "n": n})
        # p_world = Z @ W @ p_y and p_local = Z @ p_y, so
        # p_world = (Z @ W @ Z.T) @ p_local.
        placed.append((slot, z4 @ world @ z4.T,
                       (nd.get("name") or meshes[mi].get("name") or "")[:60]))
    return {"prototypes": prototypes, "nodes": placed, "js": js}


def _structure_bounds(structure):
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    corners_of = []
    for proto in structure["prototypes"]:
        pts = [b[0].reshape(-1, 3) for b in proto["batches"] if len(b[0])]
        if not pts:
            corners_of.append(None)
            continue
        p = np.concatenate(pts)
        plo, phi = p.min(axis=0), p.max(axis=0)
        bits = [(plo[i], phi[i]) for i in range(3)]
        corners_of.append(np.array(
            [[bits[0][a], bits[1][b], bits[2][c]]
             for a in (0, 1) for b in (0, 1) for c in (0, 1)],
            dtype=np.float64))
    for slot, matrix, _name in structure["nodes"]:
        corners = corners_of[slot]
        if corners is None:
            continue
        world = corners @ matrix[:3, :3].T + matrix[:3, 3]
        lo = np.minimum(lo, world.min(axis=0))
        hi = np.maximum(hi, world.max(axis=0))
    return lo, hi


def _scale_of(structure) -> float:
    ours = ((structure["js"].get("asset", {}) or {}).get("generator")
            == "IngeTrazo")
    if ours:
        return 1.0
    lo, hi = _structure_bounds(structure)
    if not np.isfinite(lo).all():
        return 1.0
    dim = float((hi - lo).max())
    if dim <= 0:
        return 1.0
    for s in (1.0, 0.0254, 0.01, 0.001):
        if 0.2 <= dim * s <= 40.0:
            return s
    return 1.0


def _scaled_batches(batches, scale: float):
    if scale == 1.0:
        return batches
    return [(tris * scale, uvs, recipe) for tris, uvs, recipe in batches]


def _matrix_scaled(matrix, scale: float):
    m = np.array(matrix, dtype=np.float64, copy=True)
    if scale != 1.0:
        m[:3, 3] *= scale
    return m


def _shift_matrix(matrix, offset):
    m = np.array(matrix, dtype=np.float64, copy=True)
    if offset:
        m[:3, 3] -= np.asarray(offset, dtype=np.float64)
    return m


def _qt_matrix(matrix):
    from PySide6.QtGui import QMatrix4x4
    flat = [float(x) for x in np.asarray(matrix, dtype=np.float64).reshape(-1)]
    return QMatrix4x4(*flat)


def _title(structure, path) -> str:
    title = (structure["js"].get("asset", {}).get("extras", {}) or {}).get("title")
    return (title or Path(path).stem)[:60]


def _offset_of(lo, hi, scale: float):
    from formats.recenter import Bounds
    box = Bounds()
    if np.isfinite(lo).all():
        box.add_array(np.stack([lo * scale, hi * scale]))
    return box.offset()


def prepare_glb(path, progress=None):
    """``{"offset", "groups"}`` for ``path``. Does not touch a scene."""
    import gc
    was = gc.isenabled()
    gc.disable()
    try:
        return _prepare(path, progress=progress)
    finally:
        if was:
            gc.enable()


def _prepare(path, progress=None):
    def tick(frac, text):
        if progress is not None:
            progress(frac, text)

    tick(0.05, "Reading file…")
    structure = read_structure(path)
    if not structure["nodes"]:
        raise ValueError("No triangle geometry found in the glTF file")
    n_unique = sum(p["n"] for p in structure["prototypes"])
    n_expanded = sum(structure["prototypes"][slot]["n"]
                     for slot, _m, _n in structure["nodes"])
    route = _route(n_unique, n_expanded)
    if route == "legacy":
        return _prepare_legacy(path, progress)
    if route == "instances":
        return _prepare_instances(structure, path, tick)
    return _prepare_reference(structure, path, tick)


def _prepare_legacy(path, progress):
    """Eager weld into one group, without appending it to a scene."""
    from core.group import Group
    from formats.recenter import Bounds

    def tick(frac, text):
        if progress is not None:
            progress(frac, text)

    batches, js = glb_format.read_triangles(Path(path))
    ours = (js.get("asset", {}) or {}).get("generator") == "IngeTrazo"
    scale = 1.0 if ours else glb_format._guess_scale(batches)
    if scale != 1.0:
        batches = [(tris * scale, uvs, r) for tris, uvs, r in batches]
    box = Bounds()
    for tris, _uvs, _r in batches:
        box.add_array(tris)
    offset = box.offset()
    if offset:
        batches = [(tris - offset, uvs, r) for tris, uvs, r in batches]
    mesh = glb_format.build_mesh(batches, progress=progress)
    title = (js.get("asset", {}).get("extras", {}) or {}).get("title")
    name = (title or Path(path).stem)[:60]
    return {"offset": offset, "groups": [Group(mesh, name=name)]}


def _prepare_instances(structure, path, tick):
    from core.group import Group

    scale = _scale_of(structure)
    lo, hi = _structure_bounds(structure)
    offset = _offset_of(lo, hi, scale)
    protos = []
    total = len(structure["prototypes"]) or 1
    for i, proto in enumerate(structure["prototypes"]):
        tick(0.2 + 0.6 * i / total, "Building shared meshes…")
        batches = _scaled_batches(proto["batches"], scale)
        protos.append(glb_format.build_mesh(batches, progress=None))
    children = []
    for slot, matrix, name in structure["nodes"]:
        g = Group(protos[slot], name=name or _title(structure, path))
        g.xform = _qt_matrix(_shift_matrix(_matrix_scaled(matrix, scale), offset))
        g.component = True
        children.append(g)
    root = Group(name=_title(structure, path))
    root.adopt(children)
    root.component = True
    tick(0.95, "Placing instances…")
    return {"offset": offset, "groups": [root]}


def _prepare_reference(structure, path, tick):
    from core.group import Group
    from core.mesh import Mesh
    from core.reference_mesh import attach, from_batches

    scale = _scale_of(structure)
    lo, hi = _structure_bounds(structure)
    offset = _offset_of(lo, hi, scale)
    refs = []
    total = len(structure["prototypes"]) or 1
    for i, proto in enumerate(structure["prototypes"]):
        tick(0.2 + 0.55 * i / total, "Packing reference geometry…")
        ref = from_batches(_scaled_batches(proto["batches"], scale))
        mesh = Mesh()
        attach(mesh, ref)
        refs.append(mesh)
    children = []
    shared = len(structure["nodes"]) > 1
    for slot, matrix, name in structure["nodes"]:
        label = name or _title(structure, path)
        g = Group(refs[slot], name=label if shared else label)
        g.xform = _qt_matrix(_shift_matrix(_matrix_scaled(matrix, scale), offset))
        g.component = shared
        children.append(g)
    tick(0.9, "Placing reference…")
    if len(children) == 1:
        only = children[0]
        if "reference" not in (only.name or ""):
            only.name = (only.name or _title(structure, path)) + " · reference"
        return {"offset": offset, "groups": children}
    root = Group(name=_title(structure, path))
    root.adopt(children)
    root.component = True
    return {"offset": offset, "groups": [root]}


def apply_glb(scene, payload):
    for group in payload["groups"]:
        scene.groups.append(group)
    return payload.get("offset")


def load_into(scene, path, progress=None):
    return apply_glb(scene, prepare_glb(path, progress=progress))
