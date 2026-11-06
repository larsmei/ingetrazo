# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""Dense GLB imports stay arrays: shared instances, no face per triangle."""
from __future__ import annotations

import json
import os
import struct

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtGui import QMatrix4x4, QVector3D
from PySide6.QtWidgets import QApplication

if QApplication.instance() is None:
    QApplication([])

from core.mesh import Mesh
from core.reference_mesh import (cluster_simplify, coplanar_loops,
                                 feature_edges, from_batches, promote_mesh,
                                 weld_triangles)
from core.scene import Scene
from formats import glb_fast


def _grid(n=4):
    """``n`` by ``n`` quads on z=0, as a corner soup ``(T, 3, 3)``."""
    tris = []
    for y in range(n):
        for x in range(n):
            a = (x, y, 0.0)
            b = (x + 1, y, 0.0)
            c = (x + 1, y + 1, 0.0)
            d = (x, y + 1, 0.0)
            tris.append((a, b, c))
            tris.append((a, c, d))
    return np.asarray(tris, dtype=np.float64)


def test_coplanar_grid_is_one_loop_and_drops_seams():
    corners = _grid(3)
    pos, tris, _keep = weld_triangles(corners)
    assert len(tris) == 18
    edges = feature_edges(pos, tris)
    # 3×3 quads: 12 boundary edges. Interior vertices are not on the loop,
    # and the coplanar seams between quads are not edges we draw.
    assert len(edges) == 12
    loops = coplanar_loops(pos, tris)
    assert len(loops) == 1
    assert len(loops[0]) == 12


def test_promote_builds_one_face_and_forgets_the_arrays():
    corners = _grid(2)
    ref = from_batches([(corners, None, (None, (0.2, 0.4, 0.6, 1.0), None))])
    mesh = Mesh()
    mesh._reference = ref
    n = promote_mesh(mesh, max_triangles=10_000, merge=True)
    assert n == 1
    assert len(mesh.faces) == 1
    assert "_reference" not in mesh.__dict__
    assert mesh.faces[0].attrs["color"] == pytest.approx([0.2, 0.4, 0.6])


def test_cluster_shrinks_a_dense_grid():
    corners = _grid(20)                 # 800 triangles
    pos, tris, _ = weld_triangles(corners)
    pos2, tris2 = cluster_simplify(pos, tris, target=40)
    assert len(tris2) <= 40
    assert len(tris2) > 0


def test_transformed_reference_moves_and_stays_a_reference():
    from core.group import transformed_mesh
    corners = _grid(1)
    ref = from_batches([(corners, None, (None, None, None))])
    mesh = Mesh()
    mesh._reference = ref
    m = QMatrix4x4()
    m.translate(5.0, 0.0, 0.0)
    copied = transformed_mesh(mesh, m)
    assert not copied.faces
    moved = copied._reference.positions
    assert float(moved[:, 0].min()) == pytest.approx(5.0, abs=1e-4)
    # The prototype did not move with the copy.
    assert float(mesh._reference.positions[:, 0].min()) == pytest.approx(0.0, abs=1e-5)


def _write_glb(path, positions, indices, translations):
    """One triangle mesh, one node per translation, all sharing it."""
    pos = np.asarray(positions, dtype="<f4")
    idx = np.asarray(indices, dtype="<u4")
    blob = pos.tobytes() + idx.tobytes()
    pos_len = pos.nbytes
    js = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(translations)))}],
        "nodes": [
            {"mesh": 0, "translation": [float(x), float(y), float(z)]}
            for x, y, z in translations
        ],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": 0},
            "indices": 1,
            "mode": 4,
        }]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(pos),
             "type": "VEC3"},
            {"bufferView": 1, "componentType": 5125, "count": int(idx.size),
             "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": pos_len},
            {"buffer": 0, "byteOffset": pos_len, "byteLength": idx.nbytes},
        ],
        "buffers": [{"byteLength": len(blob)}],
    }
    raw = json.dumps(js, separators=(",", ":")).encode()
    raw += b" " * ((4 - len(raw) % 4) % 4)
    blob += b"\x00" * ((4 - len(blob) % 4) % 4)
    total = 12 + 8 + len(raw) + 8 + len(blob)
    out = struct.pack("<III", 0x46546C67, 2, total)
    out += struct.pack("<II", len(raw), 0x4E4F534A) + raw
    out += struct.pack("<II", len(blob), 0x004E4942) + blob
    path.write_bytes(out)


def test_repeated_mesh_under_the_cap_is_one_editable_prototype(tmp_path, monkeypatch):
    """Copies share a welded mesh instead of one face soup per copy."""
    monkeypatch.setattr(glb_fast, "REFERENCE_TRIANGLES", 4)
    path = tmp_path / "chairs.glb"
    pos = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    idx = [0, 1, 2, 0, 2, 3]
    _write_glb(path, pos, idx, [(0, 0, 0), (3, 0, 0), (6, 0, 0)])
    scene = Scene()
    glb_fast.load_into(scene, path)
    kids = scene.groups[0].children
    assert len(kids) == 3
    assert kids[0].mesh is kids[1].mesh is kids[2].mesh
    assert kids[0].mesh.faces                 # still editable
    assert "_reference" not in kids[0].mesh.__dict__
    assert len(kids[0].mesh.faces) == 2       # welded once, not 2×3
    lo, hi = scene.bounds()
    assert hi.x() - lo.x() == pytest.approx(7.0, abs=1e-2)


def test_a_small_glb_is_still_one_editable_group(tmp_path):
    path = tmp_path / "small.glb"
    pos = [(0, 0, 0), (1, 0, 0), (1, 1, 0)]
    _write_glb(path, pos, [0, 1, 2], [(0, 0, 0)])
    scene = Scene()
    glb_fast.load_into(scene, path)
    group = scene.groups[0]
    assert group.mesh.faces
    assert "_reference" not in group.mesh.__dict__


def test_a_dense_shared_mesh_is_one_reference_placed_twice(tmp_path, monkeypatch):
    monkeypatch.setattr(glb_fast, "REFERENCE_TRIANGLES", 8)
    path = tmp_path / "dense.glb"
    corners = _grid(3)                      # 18 triangles, over the cap
    pos, tris, _ = weld_triangles(corners)
    _write_glb(path, pos, tris.reshape(-1), [(0, 0, 0), (10, 0, 0)])
    scene = Scene()
    glb_fast.load_into(scene, path)
    assert len(scene.groups) == 1
    kids = scene.groups[0].children
    assert len(kids) == 2
    assert kids[0].mesh is kids[1].mesh
    ref = kids[0].mesh._reference
    assert ref.triangle_count == 18
    assert not kids[0].mesh.faces          # nothing was built
    lo, hi = scene.bounds()
    # Second copy sits 10 m along X, in Z-up (the grid's X is glTF X).
    assert hi.x() - lo.x() == pytest.approx(13.0, abs=1e-2)
    assert lo.x() == pytest.approx(0.0, abs=1e-2)


def test_reference_chunk_draws_triangles_but_not_every_seam():
    """The viewport chunk is triangles for the GPU and border edges only."""
    from types import SimpleNamespace
    corners = _grid(2)
    ref = from_batches([(corners, None, (None, (0.8, 0.2, 0.2, 1), None))])
    vp = SimpleNamespace(_LIGHT=QVector3D(0.3, 0.5, 0.8))
    from core.reference_mesh import chunk_for
    mesh = Mesh()
    entry = chunk_for(vp, ref, mesh, None, (0, 0), 1)
    assert entry["v0"] is not None and len(entry["v0"]) == ref.triangle_count
    # 2x2 quads → 8 border edges, not 8*3/2 = 12 triangle edges.
    n_edges = len(entry["edges"]) // (2 * 3 * 4)   # two xyz float32
    assert n_edges == 8
    assert len(entry["faces"]) == 1
    assert entry["tri_ent"].shape == (ref.triangle_count,)


def test_selecting_a_reference_uses_its_real_box(tmp_path, monkeypatch):
    """The active cue is the model's box, not a point at the origin."""
    monkeypatch.setattr(glb_fast, "REFERENCE_TRIANGLES", 8)
    path = tmp_path / "dense.glb"
    pos, tris, _ = weld_triangles(_grid(3))
    _write_glb(path, pos, tris.reshape(-1), [(0, 0, 0), (10, 0, 0)])
    scene = Scene()
    glb_fast.load_into(scene, path)
    scene.selection.add(scene.groups[0])
    lo, hi = scene.selection_bounds()
    assert lo is not None
    assert hi.x() - lo.x() == pytest.approx(13.0, abs=1e-2)
    # The grid is written flat and arrives in the XZ plane (Z-up).
    assert hi.z() - lo.z() == pytest.approx(3.0, abs=1e-2)


def test_saving_a_reference_keeps_every_triangle(tmp_path, monkeypatch):
    monkeypatch.setattr(glb_fast, "REFERENCE_TRIANGLES", 8)
    path = tmp_path / "dense.glb"
    pos, tris, _ = weld_triangles(_grid(3))
    _write_glb(path, pos, tris.reshape(-1), [(0, 0, 0)])
    scene = Scene()
    glb_fast.load_into(scene, path)
    from formats import igz
    out = tmp_path / "kept.igz"
    igz.save_scene(scene, out)
    mesh = scene.groups[0].mesh
    assert not mesh.faces
    assert mesh._reference.triangle_count == 18
    again = Scene()
    igz.load_into(again, out)
    loaded = again.groups[0].mesh
    assert not loaded.faces
    assert loaded._reference.triangle_count == 18


def test_make_editable_builds_the_real_triangles_not_a_cluster():
    from core.reference_mesh import MakeEditable, attach, from_batches
    corners = _grid(3)
    ref = from_batches([(corners, None, (None, (0.2, 0.4, 0.9, 1), None))])
    mesh = Mesh()
    attach(mesh, ref)
    scene = Scene()
    cmd = MakeEditable([mesh])
    cmd.do(scene)
    assert len(mesh.faces) == 18
    assert "_reference" not in mesh.__dict__
    cmd.undo(scene)
    assert not mesh.faces
    assert mesh._reference.triangle_count == 18

