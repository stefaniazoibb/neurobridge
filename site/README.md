# NeuroBridge website

The public face of NeuroBridge: a scroll-driven WebGL journey followed by sections on the science, data, agents, validation, platform and research. The research application itself is the Streamlit app in `../app/`, and the website does not change it.

## Files

| File | Purpose |
|---|---|
| `story.src.html` | Source of the page (HTML, CSS and JavaScript in one file) |
| `spatial_data.js` | Real Visium spots (positions, domains, five expression layers), base64-packed, about 42 KB |
| `assets/app_*.jpg` | Screenshots of the running Streamlit app (dark theme, live run on public data) |
| `build.py` | Inlines the spatial data and writes the two outputs below |
| `index.html` | Standalone page for GitHub Pages or any static host |
| `neurobridge-story.html` | Page fragment as published to claude.ai |
| `story.v1.src.html`, `story.v2.src.html` | Earlier designs (fluorescence scrollytelling; light editorial plate), kept for reference |

## Run and deploy

```bash
python site/build.py          # after editing story.src.html
open site/index.html          # view locally (needs internet for the fonts and three.js)
```

GitHub Pages: Settings → Pages → Deploy from a branch → `main` → `/site`. If only `/` and `/docs` are offered, add a Pages workflow that uploads `site/`.

## How the journey works

- **Stage.** `#journey` is 900vh tall and holds a sticky 100vh stage. Scroll position maps to a progress value `s` from 0 to 7 (eight stops). Progress is damped for a cinematic feel, and reduced motion removes the damping.
- **Particle field.** One `THREE.Points` geometry with 16,000 points (7,000 on mobile). It morphs between eight precomputed 3D target shapes on the CPU, and only when progress changes:
  1. cell
  2. brain, built from a Perlin-ridged ellipsoid with hemispheric fissure, cerebellum and brainstem
  3. tumour environment: tumour cells, neuron trees, vessels
  4. cell
  5. neuron with axon to the cell
  6. gene and protein network
  7. five data modalities, the spatial plate using the real Visium layout
  8. evidence graph
  
  Each shape holds for the middle 64% of its segment. The point shader adds per-point jitter and a depth-of-field effect: size grows and alpha falls away from the focus distance.
- **The glioblastoma cell.** An icosphere with GPU simplex-noise displacement and normals recomputed in the vertex shader.
  - **Protrusions.** Four directional bulges where tumour microtubes leave the cell.
  - **Shading.** Fresnel-weighted translucency with front and back passes, plus a soft specular highlight.
  - **Interior.** A noise-displaced nucleus with nucleoli, and instanced organelles.
  - **Interaction.** Lighting follows the mouse, and the membrane ripples where the pointer meets it (ray–sphere test).
- **Flowing lines.** Blood vessels, the axon, data links and evidence edges are `LineSegments` with a travelling pulse in the fragment shader. Signal quanta travel along the axon in scene 04.
- **Camera.** Keyframes per scene (position, target, horizontal view offset so the subject clears the text), eased between stops, with mouse parallax and slow drift. Mobile pulls the camera back and centres the subject above the text.
- **Annotations.** HTML labels projected from 3D anchors every frame, hidden when off-screen. Each carries a status tag: established structure, conceptual, hypothesis in patients, or real data and results.

## Performance and accessibility

- **Rendering cost.** Three draw-call families: points, cell meshes and lines. Device pixel ratio is capped at 1.75 (1.5 on mobile). Rendering pauses when the stage is off-screen or the tab is hidden. Headless Chrome on Apple silicon held 60 fps.
- **Reduced motion.** `prefers-reduced-motion` stops ambient motion, camera drift and smoothing. Scenes still follow the scroll.
- **No WebGL.** Without WebGL the page shows a static illustration and keeps every scene's text.
- **Keyboard and screen readers.** Annotations, the scene rail, platform tabs and chart toggles are keyboard operable. Charts and maps have text alternatives.
- **Dependencies.** One script dependency: three.js r158 UMD from cdnjs, which logs a deprecation notice. Moving to the ES-module build would remove it.

## Scientific labelling

Biological scenes are conceptual renderings unless labelled otherwise. The spatial plate, hex map, forest plots and figures in the text are real results from the reference run, documented in `../docs/`.
