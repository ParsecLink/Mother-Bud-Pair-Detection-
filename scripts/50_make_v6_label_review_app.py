#!/usr/bin/env python
"""Build a self-contained browser app for reviewing/editing v6 labels."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V6_DIR = PROJECT_ROOT / "v6_ml_detection"
DEFAULT_MANIFEST = V6_DIR / "annotations" / "image_manifest.csv"
DEFAULT_LABELS = V6_DIR / "pseudo_labels" / "labels_pseudo_rgb.csv"
DEFAULT_CLASSES = V6_DIR / "classes.txt"
DEFAULT_OUTPUT = V6_DIR / "review" / "label_editor" / "index.html"
DEFAULT_REFERENCES = V6_DIR / "references" / "tif_reference_manifest.csv"
LABEL_COLUMNS = ["image_id", "class_name", "x1", "y1", "x2", "y2", "source", "review_status", "notes"]
CLASS_COLORS = {"single_cell": "#00ffff", "mother_bud_pair": "#ffd200", "early_bud_pair": "#ff50dc"}


def read_manifest(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_path = PROJECT_ROOT / row["image_path"]
            suffix = image_path.suffix.lower().lstrip(".")
            mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
            data_uri = "data:image/" + mime + ";base64," + base64.b64encode(image_path.read_bytes()).decode("ascii")
            rows.append(
                {
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                    "split": row.get("split", ""),
                    "data_uri": data_uri,
                }
            )
    return rows


def read_references(path: Path) -> dict[str, list[dict[str, object]]]:
    if not path.exists():
        return {}
    references: dict[str, list[dict[str, object]]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for idx, row in enumerate(csv.DictReader(handle), start=1):
            image_id = row.get("image_id", "")
            ref_path_text = row.get("reference_path", "")
            if not image_id or not ref_path_text:
                continue
            ref_path = PROJECT_ROOT / ref_path_text
            if not ref_path.exists():
                continue
            suffix = ref_path.suffix.lower().lstrip(".")
            mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
            references.setdefault(image_id, []).append(
                {
                    "reference_id": f"ref_{idx}",
                    "name": row.get("reference_name", "reference") or "reference",
                    "reference_path": ref_path_text,
                    "width": int(float(row.get("width", 0) or 0)),
                    "height": int(float(row.get("height", 0) or 0)),
                    "source_tif": row.get("source_tif", ""),
                    "frame_index": row.get("frame_index", ""),
                    "data_uri": "data:image/" + mime + ";base64," + base64.b64encode(ref_path.read_bytes()).decode("ascii"),
                }
            )
    return references


def attach_references(manifest: list[dict[str, object]], references: dict[str, list[dict[str, object]]]) -> None:
    for row in manifest:
        row["references"] = references.get(str(row["image_id"]), [])


def read_labels(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for idx, row in enumerate(csv.DictReader(handle), start=1):
            if not row.get("image_id"):
                continue
            rows.append(
                {
                    "label_id": idx,
                    "image_id": row.get("image_id", ""),
                    "class_name": row.get("class_name", ""),
                    "x1": float(row.get("x1", 0) or 0),
                    "y1": float(row.get("y1", 0) or 0),
                    "x2": float(row.get("x2", 0) or 0),
                    "y2": float(row.get("y2", 0) or 0),
                    "source": row.get("source", ""),
                    "review_status": row.get("review_status", ""),
                    "notes": row.get("notes", ""),
                }
            )
    return rows


def read_classes(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def html_document(
    manifest: list[dict[str, object]],
    labels: list[dict[str, object]],
    classes: list[str],
    class_draw_buttons: bool = False,
) -> str:
    payload = json.dumps({"manifest": manifest, "labels": labels, "classes": classes}, separators=(",", ":"))
    editor_variant = "class_draw_buttons" if class_draw_buttons else "default"
    if class_draw_buttons:
        draw_controls_html = "\n".join(
            (
                f'      <button class="draw-class-btn" data-class="{html.escape(class_name)}" '
                f'style="border-color: {CLASS_COLORS.get(class_name, "#888")};">'
                f'Draw {html.escape(class_name)}</button>'
            )
            for class_name in classes
        )
    else:
        draw_controls_html = """      <button id="drawBtn">Draw box</button>
      <select id="newClassSelect" title="Class for new boxes"></select>"""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>v6 Label Review</title>
<style>
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: Arial, sans-serif; background: #111; color: #eee; }}
.app {{ display: grid; grid-template-columns: 260px minmax(0, 1fr) 320px; height: 100vh; min-height: 100vh; overflow: hidden; }}
aside, .panel {{ background: #1b1b1b; border-color: #333; border-style: solid; }}
aside {{ border-width: 0 1px 0 0; display: grid; grid-template-rows: auto auto minmax(0, 1fr); height: 100vh; overflow: hidden; }}
main {{ min-width: 0; display: grid; grid-template-rows: auto minmax(0, 1fr); overflow: hidden; }}
.panel {{ border-width: 0 0 0 1px; padding: 12px; overflow: auto; }}
.top {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; padding: 10px; background: #181818; border-bottom: 1px solid #333; }}
button, select, input, textarea {{ background: #252525; color: #eee; border: 1px solid #444; border-radius: 4px; padding: 7px; }}
button {{ cursor: pointer; }}
button:disabled {{ color: #777; cursor: not-allowed; }}
button.active {{ outline: 2px solid #ffd200; }}
.draw-class-btn {{ font-weight: 700; }}
.sidebar-tools {{ display: grid; gap: 8px; padding: 10px; background: #1b1b1b; border-bottom: 1px solid #333; }}
.sidebar-tools input[type="search"] {{ width: 100%; }}
#imageList {{ min-height: 0; overflow-y: auto; }}
.thumb {{ padding: 8px 10px; border-bottom: 1px solid #2b2b2b; cursor: pointer; display: grid; grid-template-columns: 1fr auto; gap: 8px; }}
.thumb.active {{ background: #313131; }}
.thumb small {{ color: #aaa; }}
.canvas-wrap {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px; align-items: center; padding: 12px; overflow: auto; min-height: 0; }}
.viewer {{ display: grid; gap: 6px; justify-items: center; min-width: 0; }}
.viewer-title {{ font-size: 12px; color: #aaa; min-height: 14px; text-align: center; overflow-wrap: anywhere; }}
.ref-viewer.hidden {{ display: none; }}
canvas {{ image-rendering: pixelated; background: #000; max-width: 100%; max-height: calc(100vh - 120px); }}
.label-row {{ border: 1px solid #333; margin-bottom: 8px; padding: 8px; border-radius: 6px; cursor: pointer; }}
.label-row.selected {{ border-color: #ffd200; background: #2a2717; }}
.field {{ display: grid; gap: 4px; margin-bottom: 10px; }}
.muted {{ color: #aaa; }}
.counts {{ padding: 10px; font-size: 13px; border-bottom: 1px solid #333; }}
.save-status {{ color: #aaa; font-size: 12px; min-width: 104px; }}
</style>
</head>
<body>
<div class="app">
  <aside>
    <div class="counts" id="counts"></div>
    <div class="sidebar-tools">
      <input id="imageSearch" type="search" placeholder="Search image, class, status">
    </div>
    <div id="imageList"></div>
  </aside>
  <main>
    <div class="top">
      <button id="prevBtn">Prev</button>
      <button id="nextBtn">Next</button>
{draw_controls_html}
      <button id="deleteBtn">Delete selected</button>
      <button id="reviewBtn">Mark image reviewed</button>
      <button id="undoBtn" disabled>Undo</button>
      <button id="saveBtn">Save</button>
      <button id="exportBtn">Export CSV</button>
      <button id="exportReviewedBtn">Export Reviewed CSV</button>
      <button id="exportAuditBtn">Export Audit CSV</button>
      <button id="toggleRefBtn">Reference</button>
      <select id="refSelect"></select>
      <span id="saveStatus" class="save-status">Not saved</span>
      <span id="imageTitle" class="muted"></span>
    </div>
    <div class="canvas-wrap" id="canvasWrap">
      <div class="viewer">
        <div class="viewer-title">RGB + boxes</div>
        <canvas id="canvas" width="256" height="256"></canvas>
      </div>
      <div class="viewer ref-viewer" id="refViewer">
        <div class="viewer-title" id="refTitle">Reference</div>
        <canvas id="refCanvas" width="256" height="256"></canvas>
      </div>
    </div>
  </main>
  <section class="panel">
    <h3>Selected Box</h3>
    <div class="field"><label>Class</label><select id="classSelect"></select></div>
    <div class="field"><label>Status</label><select id="statusSelect"><option>needs_review</option><option>reviewed</option><option>reject</option></select></div>
    <div class="field"><label>Notes</label><textarea id="notesInput" rows="4"></textarea></div>
    <h3>Image Boxes</h3>
    <div id="labelList"></div>
  </section>
</div>
<script>
const DATA = {payload};
const colors = {{ single_cell: '#00ffff', mother_bud_pair: '#ffd200', early_bud_pair: '#ff50dc' }};
const EDITOR_VARIANT = {json.dumps(editor_variant)};
const STORAGE_SCHEMA = 1;
const MAX_UNDO = 80;
function buildDatasetSignature(manifest, labels) {{
  const imageText = manifest.map(m => m.image_id).join('|');
  const labelText = labels.map(l => `${{l.image_id}}:${{l.label_id}}:${{l.class_name}}:${{Math.round(Number(l.x1))}}:${{Math.round(Number(l.y1))}}:${{Math.round(Number(l.x2))}}:${{Math.round(Number(l.y2))}}`).join('|');
  const text = `${{imageText}}||${{labelText}}`;
  let hash = 2166136261;
  for (let i = 0; i < text.length; i += 1) {{
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }}
  return (hash >>> 0).toString(36);
}}
const DATASET_SIGNATURE = buildDatasetSignature(DATA.manifest, DATA.labels);
const STORAGE_KEY = `v6-label-review-progress:${{EDITOR_VARIANT}}:${{DATASET_SIGNATURE}}`;
let current = 0;
let selectedId = null;
let drawMode = false;
let dragStart = null;
let preview = null;
let showReference = true;
let imageQuery = '';
let newBoxClass = DATA.classes[0] || '';
let dirty = false;
let lastSavedAt = null;
let undoStack = [];
let notesUndoLabelId = null;
let notesUndoCaptured = false;
const imageCache = new Map();
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const refCanvas = document.getElementById('refCanvas');
const refCtx = refCanvas.getContext('2d');

function cloneLabels(labels) {{
  return labels.map(l => ({{...l}}));
}}
function pushUndo() {{
  undoStack.push(cloneLabels(DATA.labels));
  if (undoStack.length > MAX_UNDO) undoStack.shift();
  updateUndoButton();
}}
function updateUndoButton() {{
  const button = document.getElementById('undoBtn');
  if (button) button.disabled = undoStack.length === 0;
}}
function formatSavedAt(value) {{
  if (!value) return '';
  return new Date(value).toLocaleTimeString([], {{hour: '2-digit', minute: '2-digit'}});
}}
function updateSaveStatus(message) {{
  const el = document.getElementById('saveStatus');
  if (!el) return;
  if (message) {{
    el.textContent = message;
    return;
  }}
  if (dirty) el.textContent = 'Unsaved changes';
  else if (lastSavedAt) el.textContent = `Saved ${{formatSavedAt(lastSavedAt)}}`;
  else el.textContent = 'Not saved';
}}
function markDirty() {{
  dirty = true;
  updateSaveStatus();
}}
function saveProgress(showAlert = false) {{
  const savedAt = new Date().toISOString();
  const payload = {{
    schema: STORAGE_SCHEMA,
    signature: DATASET_SIGNATURE,
    savedAt,
    current,
    selectedId,
    showReference,
    labels: cloneLabels(DATA.labels),
  }};
  try {{
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
    dirty = false;
    lastSavedAt = savedAt;
    updateSaveStatus(`Saved ${{formatSavedAt(savedAt)}}`);
    if (showAlert) alert('Progress saved in this browser. Use Export Audit CSV when the review is finished.');
    return true;
  }} catch (err) {{
    updateSaveStatus('Save failed');
    if (showAlert) alert(`Could not save progress: ${{err.message || err}}`);
    return false;
  }}
}}
function restoreSavedProgress() {{
  let raw = null;
  try {{
    raw = localStorage.getItem(STORAGE_KEY);
  }} catch (err) {{
    return;
  }}
  if (!raw) return;
  try {{
    const payload = JSON.parse(raw);
    if (!payload || payload.signature !== DATASET_SIGNATURE || !Array.isArray(payload.labels)) return;
    const savedText = payload.savedAt ? ` from ${{new Date(payload.savedAt).toLocaleString()}}` : '';
    if (!window.confirm(`Saved review progress${{savedText}} was found. Restore it?`)) return;
    DATA.labels = cloneLabels(payload.labels);
    current = Math.max(0, Math.min(DATA.manifest.length - 1, Number(payload.current) || 0));
    selectedId = DATA.labels.some(l => l.label_id === payload.selectedId) ? payload.selectedId : null;
    showReference = payload.showReference !== false;
    lastSavedAt = payload.savedAt || null;
    dirty = false;
  }} catch (err) {{
    console.warn('Could not restore saved progress', err);
  }}
}}
function undoLastChange() {{
  if (!undoStack.length) return;
  DATA.labels = undoStack.pop();
  if (!DATA.labels.some(l => l.label_id === selectedId)) selectedId = null;
  notesUndoLabelId = null;
  notesUndoCaptured = false;
  dirty = true;
  updateUndoButton();
  render();
  updateSaveStatus('Undid last change');
}}
function changeWithUndo(fn) {{
  pushUndo();
  fn();
  markDirty();
  render();
}}

function labelsForImage(imageId) {{ return DATA.labels.filter(l => l.image_id === imageId && l.review_status !== 'reject'); }}
function currentImage() {{ return DATA.manifest[current]; }}
function selectedLabel() {{ return DATA.labels.find(l => l.label_id === selectedId); }}
function cacheKey(meta) {{ return meta.image_id || meta.reference_id || meta.reference_path || meta.data_uri; }}
function basenameNoExt(path) {{
  const normalized = String(path || '').replaceAll('\\\\', '/');
  const base = normalized.split('/').pop() || '';
  return base.replace(/\\.[^.]+$/, '');
}}
function referenceTitle(ref) {{
  return basenameNoExt(ref.source_tif || ref.reference_path) || ref.name || 'Reference';
}}
function loadImage(meta) {{
  const key = cacheKey(meta);
  if (imageCache.has(key)) return Promise.resolve(imageCache.get(key));
  return new Promise(resolve => {{
    const img = new Image();
    img.onload = () => {{ imageCache.set(key, img); resolve(img); }};
    img.src = meta.data_uri;
  }});
}}
function scalePoint(evt) {{
  const rect = canvas.getBoundingClientRect();
  return {{ x: (evt.clientX - rect.left) * canvas.width / rect.width, y: (evt.clientY - rect.top) * canvas.height / rect.height }};
}}
async function render() {{
  const meta = currentImage();
  const img = await loadImage(meta);
  renderReferenceControls(meta);
  canvas.width = meta.width; canvas.height = meta.height;
  ctx.drawImage(img, 0, 0);
  for (const label of labelsForImage(meta.image_id)) {{
    drawBox(label, label.label_id === selectedId);
  }}
  if (preview) drawRect(preview.x1, preview.y1, preview.x2, preview.y2, '#ffffff', true);
  await renderReference(meta);
  document.getElementById('imageTitle').textContent = `${{meta.image_id}} (${{current + 1}}/${{DATA.manifest.length}})`;
  renderImageList(); renderLabelList(); renderEditor(); renderCounts();
  updateUndoButton(); updateSaveStatus();
}}
function drawBox(label, selected) {{
  drawRectOn(ctx, label.x1, label.y1, label.x2, label.y2, colors[label.class_name] || '#fff', selected);
}}
function drawRect(x1, y1, x2, y2, color, selected) {{
  drawRectOn(ctx, x1, y1, x2, y2, color, selected);
}}
function drawRectOn(targetCtx, x1, y1, x2, y2, color, selected) {{
  targetCtx.save();
  targetCtx.strokeStyle = color; targetCtx.lineWidth = selected ? 3 : 2; targetCtx.setLineDash(selected ? [5, 3] : []);
  targetCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  targetCtx.restore();
}}
function currentReference(meta) {{
  const refs = meta.references || [];
  if (!refs.length) return null;
  const selected = document.getElementById('refSelect').value;
  return refs.find(ref => ref.reference_id === selected) || refs[0];
}}
function renderReferenceControls(meta) {{
  const refs = meta.references || [];
  const select = document.getElementById('refSelect');
  const button = document.getElementById('toggleRefBtn');
  const prior = select.value;
  if (!refs.length) {{
    select.innerHTML = '<option>No TIF reference</option>';
    select.disabled = true;
    button.disabled = true;
    button.classList.remove('active');
    return;
  }}
  select.disabled = false;
  button.disabled = false;
  select.innerHTML = refs.map(ref => `<option value="${{ref.reference_id}}">${{ref.name}}</option>`).join('');
  select.value = refs.some(ref => ref.reference_id === prior) ? prior : refs[0].reference_id;
  button.classList.toggle('active', showReference);
}}
async function renderReference(meta) {{
  const viewer = document.getElementById('refViewer');
  const ref = currentReference(meta);
  if (!showReference || !ref) {{
    viewer.classList.add('hidden');
    return;
  }}
  viewer.classList.remove('hidden');
  const img = await loadImage(ref);
  refCanvas.width = ref.width || meta.width;
  refCanvas.height = ref.height || meta.height;
  refCtx.drawImage(img, 0, 0, refCanvas.width, refCanvas.height);
  const sx = refCanvas.width / meta.width;
  const sy = refCanvas.height / meta.height;
  for (const label of labelsForImage(meta.image_id)) {{
    drawRectOn(refCtx, label.x1 * sx, label.y1 * sy, label.x2 * sx, label.y2 * sy, colors[label.class_name] || '#fff', label.label_id === selectedId);
  }}
  document.getElementById('refTitle').textContent = referenceTitle(ref);
}}
function imageSearchText(meta) {{
  const labels = labelsForImage(meta.image_id);
  const refs = (meta.references || []).map(referenceTitle);
  return [meta.image_id, meta.split, ...labels.map(l => `${{l.class_name}} ${{l.review_status}} ${{l.notes || ''}}`), ...refs].join(' ').toLowerCase();
}}
function filteredImageIndices() {{
  const query = imageQuery.trim().toLowerCase();
  if (!query) return DATA.manifest.map((_, i) => i);
  return DATA.manifest.map((meta, i) => [meta, i]).filter(([meta]) => imageSearchText(meta).includes(query)).map(([, i]) => i);
}}
function renderImageList() {{
  const countsByImage = Object.fromEntries(DATA.manifest.map(m => [m.image_id, labelsForImage(m.image_id).length]));
  const indices = filteredImageIndices();
  if (!indices.length) {{
    document.getElementById('imageList').innerHTML = '<div class="thumb"><div>No matching images</div></div>';
    return;
  }}
  document.getElementById('imageList').innerHTML = indices.map(i => {{
    const m = DATA.manifest[i];
    return (
    `<div class="thumb ${{i === current ? 'active' : ''}}" onclick="go(${{i}})"><div>${{m.image_id}}<br><small>${{m.split}}</small></div><b>${{countsByImage[m.image_id] || 0}}</b></div>`
    );
  }}).join('');
}}
function renderLabelList() {{
  const rows = labelsForImage(currentImage().image_id);
  document.getElementById('labelList').innerHTML = rows.map(l =>
    `<div class="label-row ${{l.label_id === selectedId ? 'selected' : ''}}" onclick="selectLabel(${{l.label_id}})"><b>${{l.class_name}}</b><br><span class="muted">${{Math.round(l.x1)}},${{Math.round(l.y1)}} - ${{Math.round(l.x2)}},${{Math.round(l.y2)}} | ${{l.review_status}}</span></div>`
  ).join('');
}}
function renderEditor() {{
  const cls = document.getElementById('classSelect');
  const newCls = document.getElementById('newClassSelect');
  cls.innerHTML = DATA.classes.map(c => `<option>${{c}}</option>`).join('');
  if (newCls) {{
    newCls.innerHTML = DATA.classes.map(c => `<option>${{c}}</option>`).join('');
    newCls.value = DATA.classes.includes(newBoxClass) ? newBoxClass : (DATA.classes[0] || '');
    newBoxClass = newCls.value;
  }}
  renderDrawControls();
  const label = selectedLabel();
  for (const el of [cls, document.getElementById('statusSelect'), document.getElementById('notesInput')]) el.disabled = !label;
  if (!label) {{ document.getElementById('notesInput').value = ''; return; }}
  cls.value = label.class_name;
  document.getElementById('statusSelect').value = label.review_status || 'needs_review';
  document.getElementById('notesInput').value = label.notes || '';
}}
function renderCounts() {{
  const kept = DATA.labels.filter(l => l.review_status !== 'reject');
  const reviewed = kept.filter(l => l.review_status === 'reviewed').length;
  const visibleImages = filteredImageIndices().length;
  document.getElementById('counts').textContent = `${{DATA.manifest.length}} images | ${{visibleImages}} shown | ${{kept.length}} boxes | ${{reviewed}} reviewed`;
}}
function renderDrawControls() {{
  const drawBtn = document.getElementById('drawBtn');
  if (drawBtn) drawBtn.classList.toggle('active', drawMode);
  document.querySelectorAll('.draw-class-btn').forEach(button => {{
    const active = drawMode && button.dataset.class === newBoxClass;
    button.classList.toggle('active', active);
  }});
}}
function toggleGenericDrawMode() {{
  drawMode = !drawMode;
  renderDrawControls();
}}
function toggleClassDrawMode(className) {{
  if (drawMode && newBoxClass === className) {{
    drawMode = false;
  }} else {{
    newBoxClass = className;
    drawMode = true;
  }}
  renderDrawControls();
}}
function go(i) {{ current = Math.max(0, Math.min(DATA.manifest.length - 1, i)); selectedId = null; preview = null; notesUndoLabelId = null; notesUndoCaptured = false; render(); }}
function selectLabel(id) {{ selectedId = id; notesUndoLabelId = null; notesUndoCaptured = false; render(); }}
document.getElementById('prevBtn').onclick = () => go(current - 1);
document.getElementById('nextBtn').onclick = () => go(current + 1);
document.getElementById('imageSearch').oninput = e => {{ imageQuery = e.target.value; renderImageList(); renderCounts(); }};
const drawBtn = document.getElementById('drawBtn');
if (drawBtn) drawBtn.onclick = toggleGenericDrawMode;
const newClassSelect = document.getElementById('newClassSelect');
if (newClassSelect) newClassSelect.onchange = e => {{ newBoxClass = e.target.value; renderDrawControls(); }};
document.querySelectorAll('.draw-class-btn').forEach(button => {{
  button.onclick = () => toggleClassDrawMode(button.dataset.class);
}});
document.getElementById('toggleRefBtn').onclick = () => {{ showReference = !showReference; render(); }};
document.getElementById('refSelect').onchange = () => render();
document.getElementById('undoBtn').onclick = undoLastChange;
document.getElementById('saveBtn').onclick = () => saveProgress(true);
document.getElementById('deleteBtn').onclick = () => {{ const l = selectedLabel(); if (l) changeWithUndo(() => {{ l.review_status = 'reject'; selectedId = null; }}); }};
document.getElementById('reviewBtn').onclick = () => {{ const rows = labelsForImage(currentImage().image_id); if (rows.length) changeWithUndo(() => {{ for (const l of rows) l.review_status = 'reviewed'; }}); }};
document.getElementById('classSelect').onchange = e => {{ const l = selectedLabel(); if (l && l.class_name !== e.target.value) changeWithUndo(() => {{ l.class_name = e.target.value; l.review_status = 'reviewed'; }}); }};
document.getElementById('statusSelect').onchange = e => {{ const l = selectedLabel(); if (l && l.review_status !== e.target.value) changeWithUndo(() => {{ l.review_status = e.target.value; }}); }};
document.getElementById('notesInput').onfocus = () => {{ notesUndoLabelId = selectedId; notesUndoCaptured = false; }};
document.getElementById('notesInput').oninput = e => {{
  const l = selectedLabel();
  if (!l) return;
  if (notesUndoLabelId !== l.label_id) {{
    notesUndoLabelId = l.label_id;
    notesUndoCaptured = false;
  }}
  if (!notesUndoCaptured) {{
    pushUndo();
    notesUndoCaptured = true;
  }}
  l.notes = e.target.value;
  markDirty();
}};
document.getElementById('notesInput').onblur = () => {{ notesUndoLabelId = null; notesUndoCaptured = false; }};
canvas.onmousedown = e => {{ if (!drawMode) return; dragStart = scalePoint(e); preview = {{x1: dragStart.x, y1: dragStart.y, x2: dragStart.x, y2: dragStart.y}}; }};
canvas.onmousemove = e => {{ if (!dragStart) return; const p = scalePoint(e); preview = {{x1: Math.min(dragStart.x, p.x), y1: Math.min(dragStart.y, p.y), x2: Math.max(dragStart.x, p.x), y2: Math.max(dragStart.y, p.y)}}; render(); }};
canvas.onmouseup = e => {{
  if (!dragStart || !preview) return;
  if (preview.x2 - preview.x1 > 3 && preview.y2 - preview.y1 > 3) {{
    pushUndo();
    const id = Math.max(0, ...DATA.labels.map(l => l.label_id)) + 1;
    DATA.labels.push({{label_id: id, image_id: currentImage().image_id, class_name: newBoxClass || DATA.classes[0], x1: preview.x1, y1: preview.y1, x2: preview.x2, y2: preview.y2, source: 'manual_review_app', review_status: 'reviewed', notes: ''}});
    selectedId = id;
    markDirty();
  }}
  dragStart = null; preview = null; render();
}};
canvas.onclick = e => {{
  if (drawMode) return;
  const p = scalePoint(e);
  const hit = labelsForImage(currentImage().image_id).slice().reverse().find(l => p.x >= l.x1 && p.x <= l.x2 && p.y >= l.y1 && p.y <= l.y2);
  if (hit) selectLabel(hit.label_id);
}};
function csvEscape(value) {{
  const text = String(value ?? '');
  return /[",\\n]/.test(text) ? '"' + text.replaceAll('"', '""') + '"' : text;
}}
function downloadCsv(filename, rows, columns) {{
  const csvRows = rows.map(l => columns.map(c => csvEscape((c.startsWith('x') || c.startsWith('y')) ? Number(l[c]).toFixed(2) : l[c])).join(','));
  const csv = columns.join(',') + '\\n' + csvRows.join('\\n') + '\\n';
  const blob = new Blob([csv], {{type: 'text/csv'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}}
document.getElementById('exportBtn').onclick = () => {{
  const columns = {json.dumps(LABEL_COLUMNS)};
  downloadCsv('labels_reviewed_export.csv', DATA.labels.filter(l => l.review_status !== 'reject'), columns);
}};
document.getElementById('exportReviewedBtn').onclick = () => {{
  const columns = {json.dumps(LABEL_COLUMNS)};
  downloadCsv('labels_reviewed_only_export.csv', DATA.labels.filter(l => l.review_status === 'reviewed'), columns);
}};
document.getElementById('exportAuditBtn').onclick = () => {{
  const columns = ['label_id'].concat({json.dumps(LABEL_COLUMNS)});
  downloadCsv('labels_audit_export.csv', DATA.labels, columns);
}};
window.addEventListener('beforeunload', e => {{
  if (!dirty) return;
  saveProgress(false);
  e.preventDefault();
  e.returnValue = '';
}});
window.addEventListener('pagehide', () => {{
  if (dirty) saveProgress(false);
}});
restoreSavedProgress();
render();
</script>
</body>
</html>
"""


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--class-draw-buttons", action="store_true", help="Use one draw button per class instead of a class dropdown.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    manifest = read_manifest(args.manifest)
    attach_references(manifest, read_references(args.references))
    labels = read_labels(args.labels)
    classes = read_classes(args.classes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        html_document(manifest, labels, classes, class_draw_buttons=args.class_draw_buttons),
        encoding="utf-8",
    )
    print(f"Wrote interactive label review app to {args.output}")
    print(f"Embedded {len(manifest)} images and {len(labels)} labels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
