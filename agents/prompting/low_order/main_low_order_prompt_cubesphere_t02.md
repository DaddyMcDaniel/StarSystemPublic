Summary: Main low-order prompt (Claude Code executor) — T02 Cube‑sphere primitive (single‑res) in Agent D. Deterministic, seam‑aware, shared vertices.

Prompt to paste:

Assume T01 complete.
Goal: Create a uniform cube‑sphere generator (positions, uvs, indices), seam‑aware, shared vertices.
Do this:

Generate 6 face grids at face_res (N×N), store shared edge vertices.

Project each vertex to unit sphere; compute UVs per face (seam‑safe, no polar pinch).

Build triangle indices (two tris per quad).

Export buffers (no displacement yet) using the manifest format.
Deliverables: agent_d/mesh/cubesphere.cpp|.py + sample manifest.


