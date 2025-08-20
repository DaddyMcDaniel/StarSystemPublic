Summary: Main low-order prompt (Claude Code executor) — T01 Replace GLUT sphere with mesh ingestion (viewer). Deterministic, no cloud keys, no global state leaks.

Prompt to paste:

Context: Project “StarSystem + The Forge.” PCC emits deterministic terrain specs; Agent D will produce baked mesh buffers. The viewer currently ignores terrain and calls glutSolidSphere.
Goal: Remove GLUT sphere; add a manifest‑driven mesh ingestion path (VAO/VBO/EBO) that draws triangle indices.
Do this (3–4 steps):

1) Add LoadMeshFromManifest(jsonPath) to map "positions"|"normals"|"tangents"|"uv0"|"indices" binary buffers into VBO/EBO and build a VAO.
2) Implement RenderMeshVAO(vao, indexCount) → glDrawElements(GL_TRIANGLES, ...).
3) Replace every glutSolidSphere() call with the new path, using a test manifest+buffers (dummy grid).
4) Add debug toggles: wireframe on/off, AABB draw.

Constraints: no cloud keys; deterministic; no global state leaks.
Deliverables: viewer: VAO/VBO/EBO loader, draw path, debug toggles.


