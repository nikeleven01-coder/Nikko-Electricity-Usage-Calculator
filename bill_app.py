import streamlit as st
import streamlit.components.v1 as components

# ----------------------------------------
# Embed static app files as Python strings
# ----------------------------------------
INDEX_HTML = """<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Smart Folder</title>
    <link rel=\"stylesheet\" href=\"./styles.css\" />
    <!-- Lucide icons CDN -->
    <script src=\"https://unpkg.com/lucide@latest\"></script>
  </head>
  <body>
    <header class=\"app-header\">
      <h1>Smart Folder</h1>
    </header>

    <section class=\"settings\">
      <div class=\"settings-row\">
        <label class=\"setting-label\" for=\"detectMode\">Detection</label>
        <select id=\"detectMode\" class=\"setting-input\">
          <option value=\"built-in\">Built-in (fast)</option>
          <option value=\"external\">External API (max accuracy)</option>
        </select>
      </div>
      <div id=\"externalConfig\" class=\"settings-row hidden\">
        <label class=\"setting-label\" for=\"apiUrl\">API URL</label>
        <input id=\"apiUrl\" class=\"setting-input\" type=\"text\" placeholder=\"https://api.example.com/detect\" />
        <label class=\"setting-label\" for=\"apiKey\">API Key</label>
        <input id=\"apiKey\" class=\"setting-input\" type=\"password\" placeholder=\"Your API key\" />
      </div>
    </section>

    <section class=\"upload-section\">
      <div id=\"dropzone\" class=\"dropzone\" aria-label=\"Drag and drop files here\">
        <div class=\"dz-inner\">
          <i data-lucide=\"upload\" class=\"icon\"></i>
          <p>Drag & drop files or</p>
          <label for=\"fileInput\" class=\"btn\">Upload</label>
          <input id=\"fileInput\" type=\"file\" multiple aria-label=\"Upload files\" />
        </div>
      </div>
    </section>

    <main class=\"content\">
      <div id=\"folderGrid\" class=\"grid\" aria-live=\"polite\"></div>
    </main>

    <!-- Floating Action Button -->
    <div class=\"fab\" aria-label=\"Actions\">
      <button id=\"fabMain\" class=\"fab-main\" aria-label=\"Main actions\">
        <span class=\"glow\"></span>
        <i data-lucide=\"sparkles\" class=\"icon\"></i>
      </button>
      <div class=\"fab-actions\">
        <button id=\"actionAdd\" class=\"fab-btn\" aria-label=\"Add Folder\">
          <i data-lucide=\"folder-plus\" class=\"icon\"></i>
        </button>
        <button id=\"actionEdit\" class=\"fab-btn\" aria-label=\"Edit Folder Name\">
          <i data-lucide=\"pencil\" class=\"icon\"></i>
        </button>
        <button id=\"actionDelete\" class=\"fab-btn\" aria-label=\"Delete Folder\">
          <i data-lucide=\"trash\" class=\"icon\"></i>
        </button>
        <button id=\"actionRandom\" class=\"fab-btn\" aria-label=\"Generate Random ID\">
          <i data-lucide=\"dice-6\" class=\"icon\"></i>
        </button>
        <button id=\"actionSwap\" class=\"fab-btn\" aria-label=\"Swap Selected Folders\">
          <i data-lucide=\"swap-horizontal\" class=\"icon\"></i>
        </button>
        <button id=\"actionExport\" class=\"fab-btn\" aria-label=\"Export Images\">
          <i data-lucide=\"download\" class=\"icon\"></i>
        </button>
        <div id=\"exportMenu\" class=\"export-menu\" aria-hidden=\"true\" role=\"menu\">
          <button id=\"exportWith\" class=\"menu-btn\" role=\"menuitem\">With Folders</button>
          <button id=\"exportFlat\" class=\"menu-btn\" role=\"menuitem\">Flattened</button>
        </div>
      </div>
    </div>

    <script src=\"./script.js\"></script>
    <script>
      // Initialize icons
      document.addEventListener('DOMContentLoaded', () => {
        if (window.lucide) {
          window.lucide.createIcons();
        }
      });
    </script>
  </body>
  </html>
"""

STYLES_CSS = """
:root {
  --bg: #0f1216;
  --card: #161a20;
  --text: #e6e9ef;
  --muted: #98a2b3;
  --accent: #5b9dff;
  --accent-2: #7c5cff;
  --danger: #ff6b6b;
  --success: #46d19a;
  --grid-gap: 16px;
}

* { box-sizing: border-box; }
html, body { height: 100svh; min-height: 100svh; overflow-x: hidden; overscroll-behavior-x: none; touch-action: pan-y; width: 100%; max-width: 100vw; }
body {
  margin: 0;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto,
    Helvetica, Arial, \"Apple Color Emoji\", \"Segoe UI Emoji\";
  background: radial-gradient(60% 80% at 70% 20%, #12161c 0%, #0c0f13 100%);
  color: var(--text);
  overflow-x: hidden;
  max-width: 100vw;
}

.app-header {
  position: sticky;
  top: 0;
  backdrop-filter: saturate(1.2) blur(8px);
  background: linear-gradient(to bottom, rgba(22, 26, 32, 0.9), rgba(22, 26, 32, 0.3));
  border-bottom: 1px solid rgba(255,255,255,0.06);
  z-index: 20;
}
.app-header h1 {
  margin: 0;
  padding: 18px 24px;
  letter-spacing: 0.3px;
}

.upload-section { padding: 20px 24px; }
.settings { padding: 12px 24px 0; display: grid; gap: 8px; }
.settings-row { display: grid; grid-template-columns: 120px 1fr 120px 1fr; gap: 10px; align-items: center; }
.setting-label { color: var(--muted); }
.setting-input { padding: 8px 10px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.08); background: #161a20; color: var(--text); }
.hidden { display: none; }
.dropzone {
  border: 1.5px dashed rgba(255,255,255,0.18);
  border-radius: 14px;
  min-height: 140px;
  display: grid;
  place-items: center;
  background: rgba(255,255,255,0.02);
  transition: border-color 180ms ease, box-shadow 180ms ease, transform 180ms ease;
}
.dropzone.dragover {
  border-color: var(--accent);
  box-shadow: 0 12px 40px rgba(91, 157, 255, 0.18), inset 0 0 120px rgba(91, 157, 255, 0.06);
  transform: translateY(-2px);
}
.dz-inner { text-align: center; padding: 24px; }
.dz-inner .icon { width: 28px; height: 28px; color: var(--muted); }
.dz-inner p { margin: 10px 0; color: var(--muted); }
.dz-inner .btn {
  display: inline-block;
  padding: 10px 16px;
  border-radius: 10px;
  background: linear-gradient(180deg, #1c2230, #151a24);
  color: var(--text);
  border: 1px solid rgba(255,255,255,0.08);
  box-shadow: 0 10px 20px rgba(0,0,0,0.35), 0 0 0 0 rgba(91,157,255,0.0);
  cursor: pointer;
}
#fileInput { display: none; }

.content { padding: 8px 24px 96px; }
.grid {
  display: flex;
  gap: var(--grid-gap);
  align-items: flex-start;
  flex-wrap: wrap;
  width: 100%;
  overflow-x: hidden;
}
.column {
  display: flex;
  flex-direction: column;
  gap: var(--grid-gap);
  flex: 1 1 0;
  min-width: 0; /* allow columns to shrink */
}

.folder {
  background: linear-gradient(180deg, #1a2029, #141820);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 18px;
  box-shadow: 0 24px 42px rgba(0,0,0,0.38), 0 2px 1px rgba(255,255,255,0.02);
  overflow: hidden;
}
.folder-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 20px;
  cursor: pointer;
  user-select: none;
}
  .folder-header:hover { background: rgba(255,255,255,0.03); }
  .folder-title { font-weight: 600; font-size: 1.15rem; }
  .folder-title-input {
    font-weight: 600;
    font-size: 1.15rem;
    color: var(--text);
    background: transparent;
    border: 1px dashed rgba(255,255,255,0.15);
    border-radius: 8px;
    padding: 2px 6px;
    flex: 1;
    min-width: 0;
  }
.item-count {
  color: var(--muted);
  background: rgba(124,92,255,0.08);
  border: 1px solid rgba(124,92,255,0.16);
  border-radius: 999px;
  padding: 2px 8px;
  font-weight: 500;
  line-height: 1;
  font-size: 0.78rem;
}
@media (min-width: 1024px) { .item-count { font-size: 0.88rem; } }
.folder.selected { outline: 2px solid rgba(123, 92, 255, 0.65); outline-offset: -2px; }

.folder-content {
  max-height: 0;
  overflow: hidden;
  transition: max-height 240ms ease;
  border-top: 1px solid rgba(255,255,255,0.06);
}
.folder-content.expanded {
  max-height: 500px;
  overflow-y: auto;
}
.content-inner { padding: 18px; display: grid; grid-template-columns: 1fr; gap: 14px; }
/* Responsive: mobile 1 per row, desktop 2 per row */
.file-cell { position: relative; transition: transform 240ms ease; will-change: transform; }
@media (min-width: 1024px) {
  .content-inner { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
.file-cell .thumb {
  width: 100%;
  aspect-ratio: 1 / 1;
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  overflow: hidden;
  background: #f9fafb;
}
.file-cell .thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  object-position: center;
  display: block;
}
/* Inline file name and color picker under each thumbnail */
.file-meta {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  margin-top: 6px;
}
.file-meta .name {
  flex: 1;
  font-size: 12px;
  color: #555;
  white-space: normal;
  word-break: break-word;
  overflow: visible;
  text-overflow: unset;
  line-height: 1.2;
}
.color-picker {
  width: 24px;
  height: 24px;
  padding: 0;
  border: none;
  background: transparent;
}
.color-picker::-webkit-color-swatch {
  border: 1px solid #ccc;
  border-radius: 4px;
}
.color-picker::-moz-color-swatch {
  border: 1px solid #ccc;
  border-radius: 4px;
}

.thumb {
  position: relative;
  background: #0f1216;
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 10px;
  overflow: hidden;
  display: block;
}
.thumb img { width: 100%; height: auto; object-fit: cover; display: block; }
.thumb .name { font-size: 0.78rem; color: var(--muted); padding: 6px; text-align: center; word-break: break-word; border-top: 1px solid rgba(255,255,255,0.06); }

/* Palette under thumbnail */
.palette { display: grid; grid-template-columns: repeat(6, 1fr); gap: 4px; padding: 8px; background: #0e1216; border-top: 1px solid rgba(255,255,255,0.06); }
.swatch { height: 16px; border-radius: 4px; border: 1px solid rgba(255,255,255,0.08); box-shadow: inset 0 0 10px rgba(0,0,0,0.25); }
.swatch.main { outline: 1px solid rgba(124,92,255,0.45); }

/* Selected thumbnail highlight */
.thumb.selected {
  outline: 2px solid rgba(91,157,255,0.55);
  outline-offset: -2px;
}

/* Indicate potential drop target on hover */
.thumb.drop-target {
  outline: 2px dashed rgba(91,157,255,0.6);
  outline-offset: -2px;
}

/* Dragging thumbnail visual cue */
.thumb.dragging {
  opacity: 0.6;
  transform: scale(0.98);
}

/* Floating Action Button */
.fab {
  position: fixed !important;
  right: calc(24px + env(safe-area-inset-right));
  bottom: calc(24px + env(safe-area-inset-bottom));
  z-index: 10000; /* ensure above all app content */
  pointer-events: none; /* let clicks pass except on children */
  isolation: isolate; /* create its own stacking context */
}
.fab-main {
  width: 64px; height: 64px; border-radius: 50%;
  border: 1px solid rgba(255,255,255,0.08);
  background: radial-gradient(100% 100% at 30% 30%, #1c2230 0%, #151a24 100%);
  color: var(--text);
  display: grid; place-items: center; cursor: pointer; position: relative;
  z-index: 10001;
  pointer-events: auto; /* clickable even though parent ignores events */
  box-shadow:
    0 22px 42px rgba(0,0,0,0.5),
    0 0 0 2px rgba(124,92,255,0.15),
    0 0 26px rgba(124,92,255,0.35);
  transition: transform 180ms ease, box-shadow 240ms ease;
}
.fab-main:hover { transform: translateY(-2px); box-shadow: 0 28px 58px rgba(0,0,0,0.55), 0 0 0 2px rgba(124,92,255,0.22), 0 0 36px rgba(124,92,255,0.45); }
.fab-main .icon { width: 28px; height: 28px; color: var(--accent-2); }
.fab-actions { position: absolute; right: 0; bottom: 76px; display: none; gap: 10px; pointer-events: auto; }
.fab.open .fab-actions { display: grid; }
.fab-btn {
  width: 44px; height: 44px; border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.08);
  background: linear-gradient(180deg, #1c2230, #151a24);
  color: var(--text);
  display: grid; place-items: center; cursor: pointer;
  box-shadow: 0 10px 20px rgba(0,0,0,0.35), 0 0 16px rgba(91,157,255,0.2);
  transition: transform 160ms ease, box-shadow 200ms ease;
  pointer-events: auto;
}
.fab-btn:hover { transform: translateY(-1px); box-shadow: 0 16px 28px rgba(0,0,0,0.45), 0 0 22px rgba(91,157,255,0.26); }
.fab-btn .icon { width: 22px; height: 22px; color: var(--text); }

/* FAB responsive sizing */
@media (max-width: 640px) {
  .fab { right: calc(16px + env(safe-area-inset-right)); bottom: calc(16px + env(safe-area-inset-bottom)); z-index: 1000; }
  .fab-main { width: 56px; height: 56px; }
  .fab-actions { bottom: 68px; gap: 8px; }
  .fab-btn { width: 52px; height: 52px; }
}

/* Export menu */
.export-menu {
  position: absolute;
  right: 0; /* align to actions right */
  bottom: calc(76px + 220px); /* above the actions stack */
  display: none;
  background: linear-gradient(180deg, #1c2230, #151a24);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 12px;
  box-shadow: 0 16px 28px rgba(0,0,0,0.45), 0 0 24px rgba(91,157,255,0.26);
  padding: 8px;
  gap: 8px;
  pointer-events: auto;
}
.export-menu.show { display: grid; }
.menu-btn {
  padding: 8px 12px;
  border-radius: 10px;
  border: 1px solid rgba(255,255,255,0.08);
  background: #161a20;
  color: var(--text);
  cursor: pointer;
}
.menu-btn:hover { background: #1b2129; }

/* Utility */
.sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0; }
"""

SCRIPT_JS = """
// Smart Folder script: upload, auto-analyze, auto-rename, folders grid

// DOM elements
const dom = {
  detectMode: document.getElementById('detectMode'),
  externalConfig: document.getElementById('externalConfig'),
  apiUrl: document.getElementById('apiUrl'),
  apiKey: document.getElementById('apiKey'),
  dropzone: document.getElementById('dropzone'),
  fileInput: document.getElementById('fileInput'),
  folderGrid: document.getElementById('folderGrid'),
  fabMain: document.getElementById('fabMain'),
  fab: document.querySelector('.fab'),
  actionAdd: document.getElementById('actionAdd'),
  actionEdit: document.getElementById('actionEdit'),
  actionDelete: document.getElementById('actionDelete'),
  actionRandom: document.getElementById('actionRandom'),
  actionSwap: document.getElementById('actionSwap'),
  actionExport: document.getElementById('actionExport'),
  exportMenu: document.getElementById('exportMenu'),
  exportWith: document.getElementById('exportWith'),
  exportFlat: document.getElementById('exportFlat'),
};

// State
/** @typedef {{id:string,name:string,images:string[],expanded?:boolean}} Folder */
/** @typedef {{id:string,dataUrl:string,type:string,clothType?:string,colorName?:string,colorCode?:string,twFamily?:string,twShade?:number,uniqueId?:string,setId?:string,setIndex?:number,setTotal?:number,fileName?:string}} Image */

const state = {
  folders /** @type {Folder[]} */: [],
  images /** @type {Record<string, Image>} */: {},
  selectedFolders: new Set(),
  selectedImages: new Set(),
  // Preserve horizontal scroll position of each folder's image strip
  scrollLeftByFolder: {},
  // Preserve vertical scroll position within each folder panel
  scrollTopByFolder: {},
};

// Touch drag support for mobile
let touchDragId = null;
let touchHoverId = null;

// Utils
function genId(){return Math.random().toString(36).slice(2,10)}
function sanitizeFilename(name){return name.replace(/[^a-z0-9_,\-\.\s]/gi,'').trim()}
function fileToDataURL(file){return new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(r.result);r.onerror=reject;r.readAsDataURL(file);})}
function hexToRgb(hex){const h=hex.replace('#','');const i=parseInt(h,16);return{r:(i>>16)&255,g:(i>>8)&255,b:i&255}}
function rgbToHex(rgb){const toHex=n=>n.toString(16).padStart(2,'0');return `#${toHex(rgb.r)}${toHex(rgb.g)}${toHex(rgb.b)}`}
function srgbToLinear(c){c/=255;return c<=0.04045?c/12.92:Math.pow((c+0.055)/1.055,2.4)}
function rgbToLab({r,g,b}){
  const R=srgbToLinear(r),G=srgbToLinear(g),B=srgbToLinear(b);
  const X=R*0.4124+G*0.3576+B*0.1805;
  const Y=R*0.2126+G*0.7152+B*0.0722;
  const Z=R*0.0193+G*0.1192+B*0.9505;
  const Xn=0.95047,Yn=1.00000,Zn=1.08883;
  const fx = X/Xn>0.008856?Math.cbrt(X/Xn):(7.787*(X/Xn)+16/116);
  const fy = Y/Yn>0.008856?Math.cbrt(Y/Yn):(7.787*(Y/Yn)+16/116);
  const fz = Z/Zn>0.008856?Math.cbrt(Z/Zn):(7.787*(Z/Zn)+16/116);
  return {L:116*fy-16,a:500*(fx-fy),b:200*(fy-fz)};
}
function labDistance(a,b){const dL=a.L-b.L,da=a.a-b.a,db=a.b-b.b;return Math.sqrt(dL*dL+da*da+db*db)}
function rgbDistance(a,b){const dr=a.r-b.r,dg=a.g-b.g,db=a.b-b.b;return dr*dr+dg*dg+db*db}
function loadImage(src){return new Promise((res,rej)=>{const img=new Image();img.onload=()=>res(img);img.onerror=rej;img.src=src;})}
function createUniqueId(clothType){const first=(clothType||'x').trim().charAt(0).toUpperCase()||'X';const num=Math.floor(100000+Math.random()*900000);return `${first}${num}`}

// Tailwind palette subset
const tailwindPalette={
  slate:{50:'#f8fafc',100:'#f1f5f9',200:'#e2e8f0',300:'#cbd5e1',400:'#94a3b8',500:'#64748b',600:'#475569',700:'#334155',800:'#1e293b',900:'#0f172a'},
  gray:{50:'#f9fafb',100:'#f3f4f6',200:'#e5e7eb',300:'#d1d5db',400:'#9ca3af',500:'#6b7280',600:'#4b5563',700:'#374151',800:'#1f2937',900:'#111827'},
  zinc:{50:'#fafafa',100:'#f4f4f5',200:'#e4e4e7',300:'#d4d4d8',400:'#a1a1aa',500:'#71717a',600:'#52525b',700:'#3f3f46',800:'#27272a',900:'#18181b'},
  neutral:{50:'#fafafa',100:'#f5f5f5',200:'#e5e5e5',300:'#d4d4d4',400:'#a3a3a3',500:'#737373',600:'#525252',700:'#404040',800:'#262626',900:'#171717'},
  stone:{50:'#fafaf9',100:'#f5f5f4',200:'#e7e5e4',300:'#d6d3d1',400:'#a8a29e',500:'#78716c',600:'#57534e',700:'#44403c',800:'#292524',900:'#1c1917'},
  red:{50:'#fef2f2',100:'#fee2e2',200:'#fecaca',300:'#fca5a5',400:'#f87171',500:'#ef4444',600:'#dc2626',700:'#b91c1c',800:'#991b1b',900:'#7f1d1d'},
  orange:{50:'#fff7ed',100:'#ffedd5',200:'#fed7aa',300:'#fdba74',400:'#fb923c',500:'#f97316',600:'#ea580c',700:'#c2410c',800:'#9a3412',900:'#7c2d12'},
  amber:{50:'#fffbeb',100:'#fef3c7',200:'#fde68a',300:'#fcd34d',400:'#fbbf24',500:'#f59e0b',600:'#d97706',700:'#b45309',800:'#92400e',900:'#78350f'},
  yellow:{50:'#fefce8',100:'#fef9c3',200:'#fef08a',300:'#fde047',400:'#facc15',500:'#eab308',600:'#ca8a04',700:'#a16207',800:'#854d0e',900:'#713f12'},
  lime:{50:'#f7fee7',100:'#ecfccb',200:'#d9f99d',300:'#bef264',400:'#a3e635',500:'#84cc16',600:'#65a30d',700:'#4d7c0f',800:'#3f6212',900:'#365314'},
  green:{50:'#f0fdf4',100:'#dcfce7',200:'#bbf7d0',300:'#86efac',400:'#4ade80',500:'#22c55e',600:'#16a34a',700:'#15803d',800:'#166534',900:'#14532d'},
  emerald:{50:'#ecfdf5',100:'#d1fae5',200:'#a7f3d0',300:'#6ee7b7',400:'#34d399',500:'#10b981',600:'#059669',700:'#047857',800:'#065f46',900:'#064e3b'},
  teal:{50:'#f0fdfa',100:'#ccfbf1',200:'#99f6e4',300:'#5eead4',400:'#2dd4bf',500:'#14b8a6',600:'#0d9488',700:'#0f766e',800:'#115e59',900:'#134e4a'},
  cyan:{50:'#ecfeff',100:'#cffafe',200:'#a5f3fc',300:'#67e8f9',400:'#22d3ee',500:'#06b6d4',600:'#0891b2',700:'#0e7490',800:'#155e75',900:'#164e63'},
  sky:{50:'#f0f9ff',100:'#e0f2fe',200:'#bae6fd',300:'#7dd3fc',400:'#38bdf8',500:'#0ea5e9',600:'#0284c7',700:'#0369a1',800:'#075985',900:'#0c4a6e'},
  blue:{50:'#eff6ff',100:'#dbeafe',200:'#bfdbfe',300:'#93c5fd',400:'#60a5fa',500:'#3b82f6',600:'#2563eb',700:'#1d4ed8',800:'#1e40af',900:'#1e3a8a'},
  indigo:{50:'#eef2ff',100:'#e0e7ff',200:'#c7d2fe',300:'#a5b4fc',400:'#818cf8',500:'#6366f1',600:'#4f46e5',700:'#4338ca',800:'#3730a3',900:'#312e81'},
  violet:{50:'#f5f3ff',100:'#ede9fe',200:'#ddd6fe',300:'#c4b5fd',400:'#a78bfa',500:'#8b5cf6',600:'#7c3aed',700:'#6d28d9',800:'#5b21b6',900:'#4c1d95'},
  purple:{50:'#faf5ff',100:'#f3e8ff',200:'#e9d5ff',300:'#d8b4fe',400:'#c084fc',500:'#a855f7',600:'#9333ea',700:'#7e22ce',800:'#6b21a8',900:'#581c87'},
  fuchsia:{50:'#fdf4ff',100:'#fae8ff',200:'#f5d0fe',300:'#f0abfc',400:'#e879f9',500:'#d946ef',600:'#c026d3',700:'#a21caf',800:'#86198f',900:'#701a75'},
  pink:{50:'#fdf2f8',100:'#fce7f3',200:'#fbcfe8',300:'#f9a8d4',400:'#f472b6',500:'#ec4899',600:'#db2777',700:'#be185d',800:'#9d174d',900:'#831843'},
  rose:{50:'#fff1f2',100:'#ffe4e6',200:'#fecdd3',300:'#fda4af',400:'#fb7185',500:'#f43f5e',600:'#e11d48',700:'#be123c',800:'#9f1239',900:'#881337'},
};
const shadeOrder=[50,100,200,300,400,500,600,700,800,900];
// Requested descriptive shade names mapping
const shadeNameMap={
  50:'lightest',
  100:'lighter',
  200:'light',
  300:'soft',
  400:'base',
  500:'primary',
  600:'semi-dark',
  700:'dark',
  800:'darker',
  900:'deepest'
};
function friendlyShadeName(family,shade){
  const shadeName=shadeNameMap[shade]||'primary';
  const fam=String(family).toLowerCase();
  return `${fam} ${shadeName}`;
}
function generateAllShadeCombos(family){
  const fam=String(family).toLowerCase();
  const combos=shadeOrder.map(s=>`${fam} ${shadeNameMap[s]}`);
  return combos.join(', ');
}
function nearestTailwindColor(rgb){
  const sampleLab=rgbToLab(rgb);
  let best={dist:Infinity,family:'',shade:500,hex:'#000000',name:''};
  for(const [family,shades] of Object.entries(tailwindPalette)){
    for(const shade of shadeOrder){
      const hex=shades[shade];
      const c=hexToRgb(hex);
      const lab=rgbToLab(c);
      const d=labDistance(sampleLab,lab);
      if(d<best.dist){best={dist:d,family,shade,hex,name:friendlyShadeName(family,shade)};}
    }
  }
  return best;
}
async function getDominantColor(dataUrl){
  const img=await loadImage(dataUrl);
  const canvas=document.createElement('canvas');
  const ctx=canvas.getContext('2d');
  const size=128;canvas.width=size;canvas.height=size;ctx.drawImage(img,0,0,size,size);
  const {data}=ctx.getImageData(0,0,size,size);
  const start=Math.floor(size*0.2),end=Math.floor(size*0.8);
  const samples=[];
  function rgbToHsv(r,g,b){r/=255;g/=255;b/=255;const max=Math.max(r,g,b),min=Math.min(r,g,b);let h=0,s=0,v=max;const d=max-min;s=max===0?0:d/max;if(max===min)h=0;else{switch(max){case r:h=(g-b)/d+(g<b?6:0);break;case g:h=(b-r)/d+2;break;case b:h=(r-g)/d+4;break;}h/=6;}return{h,s,v};}
  for(let y=start;y<end;y++){
    for(let x=start;x<end;x++){
      const i=(y*size+x)*4;const a=data[i+3];if(a<80)continue;const r=data[i],g=data[i+1],b=data[i+2];
      const {s,v}=rgbToHsv(r,g,b); if(v>0.97||v<0.08) continue; if(s<0.15) continue; // skip near white/black and very desaturated
      samples.push({r,g,b});
    }
  }
  if(samples.length<50){
    let r=0,g=0,b=0,count=0;
    for(let y=start;y<end;y++){
      for(let x=start;x<end;x++){
        const i=(y*size+x)*4;const a=data[i+3];if(a<10)continue;r+=data[i];g+=data[i+1];b+=data[i+2];count++;
      }
    }
    if(!count) return {r:127,g:127,b:127};
    return {r:Math.round(r/count),g:Math.round(g/count),b:Math.round(b/count)};
  }
  // k-means (K=3)
  const K=3; const centers=[];
  for(let k=0;k<K;k++){centers.push({...samples[(Math.random()*samples.length)|0]});}
  for(let iter=0;iter<6;iter++){
    const groups=Array.from({length:K},()=>({sumR:0,sumG:0,sumB:0,count:0}));
    for(const p of samples){
      let bi=0,bd=Infinity;for(let k=0;k<K;k++){const d=rgbDistance(p,centers[k]);if(d<bd){bd=d;bi=k;}}
      const g=groups[bi];g.sumR+=p.r;g.sumG+=p.g;g.sumB+=p.b;g.count++;
    }
    for(let k=0;k<K;k++){const g=groups[k];if(g.count>0){centers[k]={r:Math.round(g.sumR/g.count),g:Math.round(g.sumG/g.count),b:Math.round(g.sumB/g.count)};}}
  }
  const counts=centers.map(()=>0);
  for(const p of samples){let bi=0,bd=Infinity;for(let k=0;k<K;k++){const d=rgbDistance(p,centers[k]);if(d<bd){bd=d;bi=k;}}counts[bi]++;}
  let maxI=0;for(let k=1;k<K;k++){if(counts[k]>counts[maxI]) maxI=k;}
  return centers[maxI];
}

// MobileNet clothing suggestion
let mobilenetModelPromise=null;
function loadScript(src){return new Promise((resolve,reject)=>{const s=document.createElement('script');s.src=src;s.async=true;s.onload=resolve;s.onerror=reject;document.head.appendChild(s);})}
async function getMobileNetModel(){if(!mobilenetModelPromise){mobilenetModelPromise=(async()=>{await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js');await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@2.1.0'); // @ts-ignore
const model=await mobilenet.load({version:2,alpha:1.0});return model;})()}return mobilenetModelPromise}
function mapLabelToClothType(label){const l=label.toLowerCase();
  if(/(t[- ]?shirt|tee|shirt)/.test(l))return 't_shirt';
  if(/(hoodie|sweatshirt)/.test(l))return 'hoodie';
  if(/(jeans|denim)/.test(l))return 'jeans';
  if(/(jacket|coat|parka|trench)/.test(l))return 'jacket';
  if(/(dress|gown)/.test(l)){if(/gown/.test(l))return 'long_gown';if(/cocktail/.test(l))return 'cocktail_dress';return 'dress';}
  if(/(suit|tuxedo|blazer)/.test(l))return 'tuxedo';
  if(/(skirt)/.test(l))return 'skirt';
  if(/(shorts)/.test(l))return 'shorts';
  if(/(sweater|pullover|cardigan)/.test(l))return 'sweater';
  if(/(hat|cap|beanie)/.test(l))return 'hat';
  return 'clothing';
}
async function predictClothType(model,dataUrl){const img=await loadImage(dataUrl); // @ts-ignore
  const preds=await model.classify(img);const mapped=preds.map(p=>({type:mapLabelToClothType(p.className),prob:p.probability})).filter(p=>p.type).sort((a,b)=>b.prob-a.prob);return mapped.length?mapped[0].type:'clothing'}

// Upload handlers
function attachUploadHandlers(){
  dom.fileInput.addEventListener('change', async()=>{
    const files=Array.from(dom.fileInput.files||[]).filter(f=>f.type.startsWith('image/'));
    dom.fileInput.value='';
    if(!files.length) return;
    await ingestFiles(files);
  });
  ['dragenter','dragover'].forEach(evt=>dom.dropzone.addEventListener(evt,e=>{e.preventDefault();dom.dropzone.classList.add('dragover');}));
  ['dragleave','drop'].forEach(evt=>dom.dropzone.addEventListener(evt,e=>{e.preventDefault();dom.dropzone.classList.remove('dragover');}));
  dom.dropzone.addEventListener('drop', async(e)=>{
    const dt=e.dataTransfer;const files=Array.from(dt?.files||[]).filter(f=>f.type.startsWith('image/'));
    if(!files.length) return;
    await ingestFiles(files);
  });
}

async function ingestFiles(files){
  const newImages=[];
  for(const file of files){
    const dataUrl=await fileToDataURL(file);
    const id=genId();
    state.images[id]={id,dataUrl,type:file.type};
    newImages.push(state.images[id]);
  }
  const model=await getMobileNetModel();
  for(const img of newImages){
    const avg=await getDominantColor(img.dataUrl);
    const detectedHex=rgbToHex(avg);
    const nearest=nearestTailwindColor(avg);
    img.colorCode=detectedHex; // apply detected dominant color to picker
    img.colorName=nearest.name; img.twFamily=nearest.family; img.twShade=nearest.shade;
    img.colorShadesList=generateAllShadeCombos(nearest.family);
    img.clothType=await predictClothType(model,img.dataUrl);
  }
  // Group by clothType + colorName to assign Unique ID and numbering
  const groups=new Map();
  for(const img of newImages){
    const key=`${img.clothType}__${img.colorName}`;
    if(!groups.has(key)) groups.set(key,[]);
    groups.get(key).push(img);
  }
  for(const [,arr] of groups){
    const uid=createUniqueId(arr[0].clothType);
    arr.forEach((img,idx)=>{img.uniqueId=uid; img.setId=uid; img.setIndex=idx+1; img.setTotal=arr.length;});
  }
  // Compose file names and place images into category folders based on clothType
  function folderNameForType(t){
    const tops=new Set(['t_shirt','hoodie','sweater','jacket','coat','blazer','tuxedo']);
    const bottoms=new Set(['jeans','shorts','skirt','pants','trousers']);
    const dresses=new Set(['dress','cocktail_dress','long_gown','gown']);
    const hats=new Set(['hat','cap','beanie']);
    if (tops.has(t)) return 'Tops';
    if (bottoms.has(t)) return 'Bottoms';
    if (dresses.has(t)) return 'Dresses';
    if (hats.has(t)) return 'Hats';
    return 'Unsorted';
  }
  const folderByName = new Map(state.folders.map(f => [f.name, f]));
  for(const img of newImages){
    const typePart=img.clothType; const shadePart=(img.colorName||'').toLowerCase().replace(/\s+/g,'_');
    const suffix=` ${img.setIndex} of ${img.setTotal}`;
    img.fileName=`${img.uniqueId}, ${typePart}_${shadePart}${suffix}`;
    const fname = folderNameForType(typePart);
    let f = folderByName.get(fname);
    if (!f) { f = {id:genId(), name:fname, images:[]}; state.folders.push(f); folderByName.set(fname, f); }
    f.images.push(img.id);
    // Ensure ingested image Unique_id prefix matches destination folder's first letter
    updateImageUniquePrefixByFolder(img.id, f.id);
  }
  render();
}

// Rendering with row-first placement across responsive columns
function getColumnsCount(){
  const w=window.innerWidth;
  if (w >= 1024) return 4; // Desktop bigger folders
  if (w >= 640) return 2;  // Tablet bigger folders
  return 2;                // Mobile unchanged
}

function render(){
  dom.folderGrid.innerHTML='';
  const colsCount=getColumnsCount();
  const columns=[];
  for(let i=0;i<colsCount;i++){const col=document.createElement('div');col.className='column';columns.push(col);dom.folderGrid.appendChild(col);}  
  state.folders.forEach((folder, idx)=>{
    const colIdx = idx % colsCount;
    const card=document.createElement('div');card.className='folder'+(state.selectedFolders.has(folder.id)?' selected':'');
    const header=document.createElement('div');header.className='folder-header';
    const title=document.createElement('div');title.className='folder-title';title.textContent=folder.name;
    // Inline edit on title click
    title.addEventListener('click',(e)=>{
      e.preventDefault();
      e.stopPropagation();
      beginFolderTitleInlineEdit(folder.id, title);
    });
    // Allow dragging folder only when dragging the title
    title.draggable = true;
    title.addEventListener('dragstart',(e)=>{
      if(e.dataTransfer){ e.dataTransfer.setData('text/folder', folder.id); e.dataTransfer.effectAllowed='move'; }
      card.classList.add('dragging-folder');
    });
    title.addEventListener('dragend',()=>{ card.classList.remove('dragging-folder'); });
    const count=document.createElement('div');count.className='item-count';count.textContent=`${folder.images.length} items`;
    header.appendChild(title);header.appendChild(count);
    const content=document.createElement('div');content.className='folder-content';
    content.dataset.folderId = folder.id;
    if (folder.expanded) content.classList.add('expanded');
    const inner=document.createElement('div');inner.className='content-inner';
    inner.dataset.folderId = folder.id;
    // Track scroll positions as user navigates within the folder
    content.addEventListener('scroll', ()=>{ state.scrollTopByFolder[folder.id] = content.scrollTop; });
    inner.addEventListener('scroll', ()=>{ state.scrollLeftByFolder[folder.id] = inner.scrollLeft; });
    // allow dropping onto folder content to insert at precise position
    inner.addEventListener('dragover',(e)=>{e.preventDefault(); e.stopPropagation(); if(e.dataTransfer) e.dataTransfer.dropEffect='move';});
    inner.addEventListener('drop',(e)=>{e.preventDefault(); e.stopPropagation(); const draggedId=e.dataTransfer?.getData('text/plain'); if(!draggedId) return; const idx=findInsertIndexFromPoint(inner, folder.id, e.clientY||0); insertImageAtIndex(draggedId, folder.id, idx);});
    // mobile touch: drop when touch ends inside this folder
    inner.addEventListener('touchend',()=>{ if(touchDragId){ if(touchHoverId && touchHoverId!==touchDragId){ swapOrInsertWithinFolder(touchDragId, touchHoverId); } else { moveImageToFolder(touchDragId, folder.id); } touchDragId=null; touchHoverId=null; } });
    // also accept drop on full card area with positional insert
    card.addEventListener('dragover',(e)=>{
      e.preventDefault(); e.stopPropagation();
      if(e.dataTransfer){ e.dataTransfer.dropEffect='move'; }
    });
    card.addEventListener('drop',(e)=>{
      e.preventDefault(); e.stopPropagation();
      const draggedFolderId = e.dataTransfer?.getData('text/folder');
      if(draggedFolderId){
        // Swap folders (positions) when a folder is dropped onto another folder
        if(draggedFolderId!==folder.id){ swapFoldersById(draggedFolderId, folder.id); }
        return;
      }
      // Fallback: handle image drop into this folder with positional insert
      const draggedImageId=e.dataTransfer?.getData('text/plain');
      if(!draggedImageId) return;
      const innerEl=card.querySelector('.content-inner')||inner;
      const idx=findInsertIndexFromPoint(innerEl, folder.id, e.clientY||0);
      insertImageAtIndex(draggedImageId, folder.id, idx);
    });
    folder.images.forEach(imgId=>{
      const img=state.images[imgId];
      const cell=document.createElement('div');
      cell.className='file-cell';
      cell.dataset.imgId = img.id;
      const thumb=document.createElement('div');thumb.className='thumb'+(state.selectedImages.has(img.id)?' selected':'');
      thumb.dataset.imgId = img.id;
      thumb.dataset.folderId = folder.id;
      thumb.draggable=true;
      thumb.addEventListener('dragstart',(e)=>{ if(e.dataTransfer){ e.dataTransfer.setData('text/plain', img.id); e.dataTransfer.effectAllowed='move'; } thumb.classList.add('dragging'); });
      thumb.addEventListener('dragend',()=>{thumb.classList.remove('dragging');});
      // allow drop onto a thumbnail to reorder (move-before target)
      thumb.addEventListener('dragover',(e)=>{e.preventDefault(); e.stopPropagation(); if(e.dataTransfer) e.dataTransfer.dropEffect='move'; thumb.classList.add('drop-target');});
      thumb.addEventListener('dragleave',()=>{thumb.classList.remove('drop-target');});
      thumb.addEventListener('drop',(e)=>{e.preventDefault(); e.stopPropagation(); const draggedId=e.dataTransfer?.getData('text/plain'); thumb.classList.remove('drop-target'); if(!draggedId) return; moveImageBeforeTarget(draggedId, img.id);});
      // mobile touch drag
      thumb.addEventListener('touchstart',()=>{touchDragId=img.id;thumb.classList.add('dragging');});
      thumb.addEventListener('touchend',()=>{
        thumb.classList.remove('dragging');
        if(touchDragId && touchHoverId && touchHoverId!==touchDragId){
          moveImageBeforeTarget(touchDragId, touchHoverId);
        }
        document.querySelectorAll('.thumb.drop-target').forEach(th=>th.classList.remove('drop-target'));
        touchDragId=null; touchHoverId=null;
      });
      const image=document.createElement('img');image.src=img.dataUrl;thumb.appendChild(image);
      const meta=document.createElement('div');meta.className='file-meta';
      const name=document.createElement('div');name.className='name';name.textContent=`${img.fileName}`;
      const picker=document.createElement('input');picker.type='color';picker.className='color-picker';picker.value=img.colorCode||'#000000';
      picker.addEventListener('input',(e)=>{
        const hex=picker.value;
        img.colorCode=hex;
        const nearest=nearestTailwindColor(hexToRgb(hex));
        img.twFamily=nearest.family; img.twShade=nearest.shade; img.colorName=nearest.name;
        img.colorShadesList=generateAllShadeCombos(nearest.family);
        // update filename and recompute grouping
        recomputeSetIndicesAndNames();
        render();
      });
      meta.appendChild(name);meta.appendChild(picker);
      cell.appendChild(thumb);cell.appendChild(meta);
      // Toggle image selection and deselect any selected folders
      thumb.addEventListener('click',(e)=>{
        e.preventDefault();
        e.stopPropagation();
        const multi = e.ctrlKey || e.metaKey;
        // Always deselect folders when clicking an image
        state.selectedFolders.clear();
        document.querySelectorAll('.folder.selected').forEach(el=>el.classList.remove('selected'));
        if (!multi) {
          // Clear previous image selections visually and in state
          state.selectedImages.clear();
          document.querySelectorAll('.thumb.selected').forEach(el=>el.classList.remove('selected'));
        }
        if (state.selectedImages.has(img.id)) {
          state.selectedImages.delete(img.id);
          thumb.classList.remove('selected');
        } else {
          state.selectedImages.add(img.id);
          thumb.classList.add('selected');
        }
        // No render() call here to avoid resetting scroll
      });
      inner.appendChild(cell);
    });
    content.appendChild(inner);
    // Restore saved scroll positions after DOM is attached
    requestAnimationFrame(()=>{
      const savedTop = state.scrollTopByFolder && state.scrollTopByFolder[folder.id];
      if (savedTop != null) content.scrollTop = savedTop;
      const savedLeft = state.scrollLeftByFolder && state.scrollLeftByFolder[folder.id];
      if (savedLeft != null) inner.scrollLeft = savedLeft;
    });
    header.addEventListener('click',(e)=>{
      const multi=e.ctrlKey||e.metaKey;
      if(!multi) state.selectedFolders.clear();
      if(state.selectedFolders.has(folder.id)) state.selectedFolders.delete(folder.id); else state.selectedFolders.add(folder.id);
      folder.expanded = !folder.expanded;
      render();
    });
    card.appendChild(header);card.appendChild(content);
    columns[colIdx].appendChild(card);
  });
}

// Actions
// Ensure folder name is unique by appending incrementing suffix " (n)"
function ensureUniqueFolderName(baseName, skipFolderId){
  let name=(baseName||'').trim();
  if(!name) name='Untitled';
  const existing=new Set(state.folders.filter(f=>!skipFolderId || f.id!==skipFolderId).map(f=>f.name));
  if(!existing.has(name)) return name;
  let n=2;
  let candidate;
  do { candidate = `${name} (${n})`; n++; } while(existing.has(candidate));
  return candidate;
}

dom.actionAdd.addEventListener('click',(e)=>{
  e.stopPropagation();
  let raw=null;
  try { if(typeof window.prompt==='function') raw=window.prompt('New folder name'); } catch(_) {}
  const name=ensureUniqueFolderName(raw||`Untitled ${state.folders.length+1}`);
  state.folders.push({id:genId(),name,images:[]});
  render();
});
dom.actionEdit.addEventListener('click',()=>{
  if(state.selectedFolders.size!==1){alert('Select exactly one folder to rename.');return;}
  const fid=[...state.selectedFolders][0];
  const folder=state.folders.find(f=>f.id===fid);
  const raw=prompt('Folder name',folder?.name||'');
  if(raw==null||!folder) return;
  const name=ensureUniqueFolderName(raw, folder.id);
  if(!name) return;
  folder.name=name;
  updateUniqueIdPrefixForFolder(folder.id);
  recomputeSetIndicesAndNames();
  render();
});
dom.actionDelete.addEventListener('click',(e)=>{
  e.stopPropagation();
  // If images are selected, delete images after confirmation
  if(state.selectedImages && state.selectedImages.size > 0){
    const count = state.selectedImages.size;
    let ok = true;
    try { ok = window.confirm(`Delete ${count} selected image(s)?`); } catch(_) {}
    if(!ok) return;
    // Remove each selected image from its folder and from state.images
    const ids = [...state.selectedImages];
    for(const id of ids){
      // Remove from folder arrays
      for(const f of state.folders){
        const idx = f.images.indexOf(id);
        if(idx >= 0){ f.images.splice(idx,1); break; }
      }
      // Remove from images map
      if(state.images && state.images[id]){ delete state.images[id]; }
    }
    state.selectedImages.clear();
    render();
    return;
  }
  // Otherwise, handle folder deletion (with warning if they contain images)
  if(state.selectedFolders.size<1){ console.log('Select image(s) or folder(s) to delete.'); return; }
  const selected=[...state.selectedFolders];
  const foldersWithImages=selected.map(id=>state.folders.find(f=>f.id===id)).filter(f=>f && (f.images||[]).length>0);
  if(foldersWithImages.length>0){
    const totalImages=foldersWithImages.reduce((sum,f)=>sum + (f.images?.length||0), 0);
    let ok=true;
    try {
      ok = window.confirm(`Delete ${foldersWithImages.length} folder(s) containing ${totalImages} image(s)?`);
    } catch(_) {}
    if(!ok) return;
  }
  const ids=new Set(selected);
  state.folders=state.folders.filter(f=>!ids.has(f.id));
  state.selectedFolders.clear();
  render();
});
dom.actionRandom.addEventListener('click', async()=>{
  if(state.selectedImages.size<1){alert('Select image(s) to assign new Unique ID.');return;}
  for(const id of state.selectedImages){
    const img=state.images[id]; if(!img) continue;
    const folder=state.folders.find(f=>f.images.includes(id));
    const letter=getFolderInitial(folder);
    const digits=generateUnusedDigitsForFolder(folder?.id);
    img.uniqueId=`${letter}${digits}`;
    // Cloth type = Folder name
    if(folder){ img.clothType = folder.name; }
    // Detect garment dominant color (ignore background) and update picker
    const hex=await getDominantGarmentColor(img.dataUrl);
    img.colorCode=hex;
    const nearest=nearestTailwindColor(hexToRgb(hex));
    img.twFamily=nearest.family; img.twShade=nearest.shade; img.colorName=nearest.name;
    img.colorShadesList=generateAllShadeCombos(nearest.family);
    // Compose filename
    const typePart=img.clothType; const shadePart=(img.colorName||'').toLowerCase().replace(/\s+/g,'_');
    const suffix=` ${img.setIndex||1} of ${img.setTotal||1}`;
    img.fileName=`${img.uniqueId}, ${typePart}_${shadePart}${suffix}`;
    // Real-time in-place UI update without full re-render
    const cell=document.querySelector(`.file-cell[data-img-id="${id}"]`) || document.querySelector(`.thumb[data-img-id="${id}"]`)?.closest('.file-cell');
    if(cell){
      const nameEl=cell.querySelector('.file-meta .name'); if(nameEl) nameEl.textContent=img.fileName;
      const picker=cell.querySelector('.color-picker'); if(picker) picker.value=hex;
    }
  }
  // Avoid full render to keep scroll/selection stable
});
dom.actionSwap.addEventListener('click',()=>{if(state.selectedFolders.size!==2){alert('Select exactly two folders to swap.');return;}const [aId,bId]=[...state.selectedFolders];const ai=state.folders.findIndex(f=>f.id===aId);const bi=state.folders.findIndex(f=>f.id===bId);if(ai<0||bi<0)return;const tmp=state.folders[ai];state.folders[ai]=state.folders[bi];state.folders[bi]=tmp;render();});
dom.actionExport.addEventListener('click',()=>{dom.exportMenu.classList.toggle('show');});
dom.exportWith.addEventListener('click',()=>{downloadImages(true)});
dom.exportFlat.addEventListener('click',()=>{downloadImages(false)});

// FAB open/close toggle and outside click to close
dom.fabMain.addEventListener('click', (e)=>{
  e.stopPropagation();
  const fabContainer = dom.fabMain?.closest('.fab') || document.querySelector('.fab');
  if (fabContainer) fabContainer.classList.toggle('open');
});
document.addEventListener('click', (e)=>{
  const fabContainer = document.querySelector('.fab');
  if (fabContainer && !fabContainer.contains(e.target)) fabContainer.classList.remove('open');
});

// Mobile: track touch hover to improve drop targeting
document.addEventListener('touchmove',(e)=>{
  if(!touchDragId) return;
  const t=e.touches && e.touches[0]; if(!t) return;
  const el=document.elementFromPoint(t.clientX, t.clientY);
  document.querySelectorAll('.thumb.drop-target').forEach(th=>th.classList.remove('drop-target'));
  const thumbEl=el && el.closest ? el.closest('.thumb') : null;
  if(thumbEl){ thumbEl.classList.add('drop-target'); touchHoverId=thumbEl.dataset.imgId||null; } else { touchHoverId=null; }
}, {passive:false});

function downloadImages(withFolders){
  const imgs=Object.values(state.images);
  for(const img of imgs){
    const folderName=(state.folders.find(f=>f.images.includes(img.id))?.name)||'';
    const a=document.createElement('a'); a.href=img.dataUrl; a.download= withFolders && folderName ? `${folderName}/${sanitizeFilename(img.fileName||'image')}.png` : `${sanitizeFilename(img.fileName||'image')}.png`; a.click();
  }
}

function moveImageToFolder(imgId,targetFolderId){
  const last=capturePositions();
  captureFolderScrollTops();
  const folder=state.folders.find(f=>f.id===targetFolderId); if(!folder) return;
  for(const f of state.folders){const idx=f.images.indexOf(imgId); if(idx>=0){f.images.splice(idx,1); break;}}
  folder.images.push(imgId);
  // Ensure the moved image's Unique ID prefix matches the folder's first letter
  updateImageUniquePrefixByFolder(imgId, targetFolderId);
  render();
  applyFLIP(last);
}
// Capture positions for FLIP animation
function capturePositions(){
  const map=new Map();
  document.querySelectorAll('.file-cell').forEach(cell=>{
    const id=cell.dataset.imgId || cell.querySelector('.thumb')?.dataset.imgId;
    if(!id) return; map.set(id, cell.getBoundingClientRect());
  });
  return map;
}
// Apply FLIP animation based on previous positions
function applyFLIP(prev){
  const cells=document.querySelectorAll('.file-cell');
  cells.forEach(cell=>{
    const id=cell.dataset.imgId || cell.querySelector('.thumb')?.dataset.imgId;
    const before=prev.get(id); if(!before) return;
    const after=cell.getBoundingClientRect();
    const dx=before.left - after.left; const dy=before.top - after.top;
    if(Math.abs(dx)>0.5 || Math.abs(dy)>0.5){
      try {
        cell.animate([
          { transform: `translate(${dx}px, ${dy}px)` },
          { transform: 'translate(0, 0)' }
        ], { duration: 240, easing: 'ease' });
      } catch (_) {
        // Fallback in case WAAPI not available
        cell.style.transition='none';
        cell.style.transform=`translate(${dx}px, ${dy}px)`;
        void cell.getBoundingClientRect();
        cell.style.transition='transform 240ms ease';
        cell.style.transform='translate(0,0)';
        setTimeout(()=>{cell.style.transition=''; cell.style.transform='';},260);
      }
    }
  });
}
// Insert at specific position within a folder
function insertImageAtIndex(imgId,targetFolderId,insertIndex){
  const last=capturePositions();
  captureFolderScrollTops();
  const folder=state.folders.find(f=>f.id===targetFolderId); if(!folder) return;
  for(const f of state.folders){const idx=f.images.indexOf(imgId); if(idx>=0){f.images.splice(idx,1); break;}}
  const clamped=Math.max(0, Math.min(insertIndex, folder.images.length));
  folder.images.splice(clamped, 0, imgId);
  // Ensure the moved image's Unique ID prefix matches the folder's first letter
  updateImageUniquePrefixByFolder(imgId, targetFolderId);
  render();
  applyFLIP(last);
}
// Determine insert index from pointer Y relative to existing cells
function findInsertIndexFromPoint(containerEl, folderId, clientY){
  const cells=Array.from(containerEl.querySelectorAll('.file-cell'));
  if(!cells.length){
    const folder=state.folders.find(f=>f.id===folderId);
    return folder ? folder.images.length : 0;
  }
  for(let i=0;i<cells.length;i++){
    const rect=cells[i].getBoundingClientRect();
    const mid=rect.top + rect.height/2;
    if(clientY < mid) return i;
  }
  return cells.length;
}

// Move dragged image before the target image (reorder-only, no swap)
function moveImageBeforeTarget(dragId,targetId){
  if(dragId===targetId) return;
  const last=capturePositions();
  captureFolderScrollTops();
  const srcFolder=state.folders.find(f=>f.images.includes(dragId));
  const destFolder=state.folders.find(f=>f.images.includes(targetId));
  if(!destFolder) return;
  let targetIndex=destFolder.images.indexOf(targetId);
  if(srcFolder && srcFolder.id===destFolder.id){
    const srcIndex=srcFolder.images.indexOf(dragId);
    if(srcIndex<0 || targetIndex<0) return;
    // remove drag first
    srcFolder.images.splice(srcIndex,1);
    // adjust target index if removal was before target
    if(srcIndex < targetIndex) targetIndex -= 1;
    destFolder.images.splice(targetIndex,0,dragId);
  } else {
    if(srcFolder){const si=srcFolder.images.indexOf(dragId); if(si>=0) srcFolder.images.splice(si,1);}    
    destFolder.images.splice(targetIndex,0,dragId);
  }
  // Ensure the moved image's Unique ID prefix matches the destination folder's first letter
  updateImageUniquePrefixByFolder(dragId, destFolder.id);
  render();
  applyFLIP(last);
}
// Capture current vertical scroll positions for all folders before a re-render
function captureFolderScrollTops(){
  document.querySelectorAll('.folder-content').forEach(el=>{
    const fid = el.dataset.folderId;
    if (fid) state.scrollTopByFolder[fid] = el.scrollTop;
  });
}
// Helper: get folder initial letter (uppercase)
function getFolderInitial(folder){
  return (folder?.name||'').trim().charAt(0).toUpperCase()||'X';
}
// Update a single image's Unique ID prefix to match its folder's first letter
function updateImageUniquePrefixByFolder(imgId, folderId){
  const folder=state.folders.find(f=>f.id===folderId); if(!folder) return;
  const img=state.images[imgId]; if(!img) return;
  const letter=getFolderInitial(folder);
  const old = img.uniqueId || '';
  const digits = old && old.length>1 ? old.slice(1) : String(Math.floor(100000+Math.random()*900000));
  img.uniqueId = `${letter}${digits}`;
  const typePart=img.clothType; const shadePart=(img.colorName||'').toLowerCase().replace(/\s+/g,'_'); const suffix=` ${img.setIndex||1} of ${img.setTotal||1}`;
  img.fileName=`${img.uniqueId}, ${typePart}_${shadePart}${suffix}`;
}
  // Update all images in a folder to have Unique ID prefix = folder's first letter
  function updateUniqueIdPrefixForFolder(folderId){
    const folder=state.folders.find(f=>f.id===folderId); if(!folder) return;
    const letter=getFolderInitial(folder);
    folder.images.forEach(imgId=>{
      const img=state.images[imgId]; if(!img) return;
      const old = img.uniqueId || '';
      const digits = old && old.length>1 ? old.slice(1) : String(Math.floor(100000+Math.random()*900000));
      img.uniqueId = `${letter}${digits}`;
      const typePart=img.clothType; const shadePart=(img.colorName||'').toLowerCase().replace(/\s+/g,'_'); const suffix=` ${img.setIndex||1} of ${img.setTotal||1}`;
      img.fileName=`${img.uniqueId}, ${typePart}_${shadePart}${suffix}`;
    });
  }

  // Swap two folders by id: swap their positions in the folders array
  function swapFoldersById(aId,bId){
    if(!aId || !bId || aId===bId) return;
    const ai=state.folders.findIndex(f=>f.id===aId);
    const bi=state.folders.findIndex(f=>f.id===bId);
    if(ai<0 || bi<0) return;
    // Preserve scroll positions before re-render
    captureFolderScrollTops();
    // Swap positions (the folder objects carry their items and name)
    const tmp=state.folders[ai];
    state.folders[ai]=state.folders[bi];
    state.folders[bi]=tmp;
    render();
  }

  // Begin inline editing for a folder's title
  function beginFolderTitleInlineEdit(folderId, titleEl){
    const folder=state.folders.find(f=>f.id===folderId); if(!folder) return;
    const input=document.createElement('input');
    input.type='text';
    input.className='folder-title-input';
    input.value=folder.name||'';
    input.setAttribute('aria-label','Edit folder name');
    // Prevent header click toggles while editing
    ['click','mousedown','mouseup'].forEach(ev=>{
      input.addEventListener(ev,(e)=>{ e.stopPropagation(); });
    });
    titleEl.replaceWith(input);
    input.focus();
    input.select();
    const finish=(commit)=>{
      input.removeEventListener('blur', onBlur);
      input.removeEventListener('keydown', onKey);
      if(commit){
        const newName=(input.value||'').trim();
        if(newName && newName!==folder.name){
          // Preserve scroll positions before re-render
          captureFolderScrollTops();
          const uniqueName = ensureUniqueFolderName(newName, folder.id);
          folder.name=uniqueName;
          // Update Unique ID prefixes to match new folder initial
          updateUniqueIdPrefixForFolder(folder.id);
          // Recompute names and indices, then re-render
          recomputeSetIndicesAndNames();
          render();
          return; // render replaced DOM
        }
      }
      // Cancel or no change: restore title element in place
      const t=document.createElement('div');
      t.className='folder-title';
      t.textContent=folder.name||'';
      t.addEventListener('click',(e)=>{ e.preventDefault(); e.stopPropagation(); beginFolderTitleInlineEdit(folder.id, t); });
      // Keep dragging restricted to title
      t.draggable = true;
      // We need card element to toggle class on drag; find nearest card
      const cardEl = input.closest('.folder');
      t.addEventListener('dragstart',(e)=>{
        if(e.dataTransfer){ e.dataTransfer.setData('text/folder', folder.id); e.dataTransfer.effectAllowed='move'; }
        if(cardEl) cardEl.classList.add('dragging-folder');
      });
      t.addEventListener('dragend',()=>{ if(cardEl) cardEl.classList.remove('dragging-folder'); });
      input.replaceWith(t);
    };
    const onBlur=()=>finish(true);
    const onKey=(e)=>{
      if(e.key==='Enter'){ e.preventDefault(); finish(true); }
      else if(e.key==='Escape'){ e.preventDefault(); finish(false); }
    };
    input.addEventListener('blur', onBlur);
    input.addEventListener('keydown', onKey);
  }

  // Rule: For images in a folder that are "1 of 1" and whose cloth_type
  // does NOT equal the folder name, change Unique_id to a new value
  // composed of the folder's first letter + unused 6 digits within that folder.
  function applySingletonMismatchRuleToFolder(folderId){
    const folder=state.folders.find(f=>f.id===folderId); if(!folder) return;
    const letter=getFolderInitial(folder);
    folder.images.forEach(imgId=>{
      const img=state.images[imgId]; if(!img) return;
      const isSingleton=(img.setTotal||1)===1;
      const clothMismatch=(img.clothType||'').trim() !== (folder.name||'').trim();
      if(isSingleton && clothMismatch){
        const digits=generateUnusedDigitsForFolder(folderId);
        img.uniqueId=`${letter}${digits}`;
        img.setId=img.uniqueId; img.setIndex=1; img.setTotal=1;
        const typePart=img.clothType; const shadePart=(img.colorName||'').toLowerCase().replace(/\s+/g,'_');
        const suffix=` ${img.setIndex} of ${img.setTotal}`;
        img.fileName=`${img.uniqueId}, ${typePart}_${shadePart}${suffix}`;
      }
    });
  }

// Generate a 6-digit number not used by any image in the folder
function generateUnusedDigitsForFolder(folderId){
  const folder=state.folders.find(f=>f.id===folderId);
  const used=new Set();
  if(folder){
    folder.images.forEach(imgId=>{
      const u=state.images[imgId]?.uniqueId||'';
      const d = u && u.length>1 ? u.slice(1) : null;
      if(d) used.add(d);
    });
  }
  let digits;
  do { digits = String(Math.floor(100000+Math.random()*900000)); } while(used.has(digits));
  return digits;
}

// Compute dominant garment color by sampling central region and filtering near-white backgrounds
async function getDominantGarmentColor(dataUrl){
  try{
    const imgEl=await (new Promise((resolve,reject)=>{ const im=new Image(); im.onload=()=>resolve(im); im.onerror=reject; im.src=dataUrl; }));
    const w=imgEl.naturalWidth||imgEl.width; const h=imgEl.naturalHeight||imgEl.height;
    const canvas=document.createElement('canvas'); canvas.width=w; canvas.height=h;
    const ctx=canvas.getContext('2d',{willReadFrequently:true});
    if(!ctx){ return '#000000'; }
    ctx.drawImage(imgEl,0,0,w,h);
    const cx=Math.floor(w*0.2), cy=Math.floor(h*0.2), cw=Math.floor(w*0.6), ch=Math.floor(h*0.6);
    const imgData=ctx.getImageData(cx,cy,cw,ch).data;
    let r=0,g=0,b=0,count=0;
    for(let i=0;i<imgData.length;i+=4){
      const rr=imgData[i], gg=imgData[i+1], bb=imgData[i+2], aa=imgData[i+3];
      if(aa===0) continue;
      const max=Math.max(rr,gg,bb), min=Math.min(rr,gg,bb);
      const sat=max-min;
      // Filter out near-white and very dark pixels likely background/noise
      if(max<20) continue;
      if(sat<10 && max>220) continue;
      // Filter likely skin tones via HSV hue range and moderate saturation
      const r1=rr/255, g1=gg/255, b1=bb/255;
      const cmax=Math.max(r1,g1,b1), cmin=Math.min(r1,g1,b1);
      const delta=cmax-cmin;
      let hue=0;
      if(delta>0){
        if(cmax===r1){ hue=((g1-b1)/delta)%6; }
        else if(cmax===g1){ hue=(b1-r1)/delta+2; }
        else { hue=(r1-g1)/delta+4; }
        hue*=60; if(hue<0) hue+=360;
      }
      const satN = cmax===0?0:delta/cmax;
      const valN = cmax;
      // Exclude hues commonly associated with skin (approx 10°–50°), moderate saturation
      if(hue>=10 && hue<=50 && satN>=0.15 && satN<=0.75 && valN>0.2) continue;
      r+=rr; g+=gg; b+=bb; count++;
    }
    if(count===0){ return rgbToHex({r,g,b}); }
    r=Math.round(r/count); g=Math.round(g/count); b=Math.round(b/count);
    return rgbToHex({r,g,b});
  } catch(_){
    return '#000000';
  }
}

function recomputeSetIndicesAndNames(){
  const groups=new Map();
  const imgs=Object.values(state.images);
  for(const img of imgs){const key=`${img.clothType}__${img.colorName}`; if(!groups.has(key)) groups.set(key,[]); groups.get(key).push(img);}  
  for(const [,arr] of groups){
    const uid=createUniqueId(arr[0].clothType);
    arr.forEach((img,idx)=>{img.uniqueId=img.uniqueId||uid; img.setId=uid; img.setIndex=idx+1; img.setTotal=arr.length;});
  }
  // Enforce Unique_id first-letter to match folder name for all images
  state.folders.forEach(f=>updateUniqueIdPrefixForFolder(f.id));
  // Apply rule: if cloth_type != folder name and image is 1 of 1, change Unique_id
  state.folders.forEach(f=>applySingletonMismatchRuleToFolder(f.id));
  for(const img of imgs){const typePart=img.clothType; const shadePart=(img.colorName||'').toLowerCase().replace(/\s+/g,'_'); const suffix=` ${img.setIndex||1} of ${img.setTotal||1}`; img.fileName=`${img.uniqueId}, ${typePart}_${shadePart}${suffix}`;}
}

attachUploadHandlers();
render();
window.addEventListener('resize',()=>{render()});

// Ensure FAB floats relative to the viewport at all times
document.addEventListener('DOMContentLoaded', () => {
  const originalFab = document.querySelector('.fab');
  if (!originalFab) return;
  // Create a portal container fixed to the viewport bottom-right
  const portal = document.createElement('div');
  portal.className = 'fab';
  // Move all children into the portal to avoid any transformed ancestors
  while (originalFab.firstChild) {
    portal.appendChild(originalFab.firstChild);
  }
  // Replace original with portal at body level
  document.body.appendChild(portal);
  // Remove the original container and rebind references
  try { originalFab.remove(); } catch(_) {}
  if (window.dom) { window.dom.fab = portal; }

  // Fallback: if position: fixed doesn't stick in some mobile engines,
  // detect movement on scroll and emulate fixed via absolute positioning.
  let useAbsoluteFallback = false;
  let lastRect = portal.getBoundingClientRect();
  function updateFabPosition(){
    const computed = window.getComputedStyle(portal);
    const right = parseInt(computed.right) || 24;
    const bottom = parseInt(computed.bottom) || 24;
    if (useAbsoluteFallback){
      portal.style.position = 'absolute';
      portal.style.top = (window.scrollY + window.innerHeight - bottom - portal.offsetHeight) + 'px';
      portal.style.left = (window.scrollX + window.innerWidth - right - portal.offsetWidth) + 'px';
    } else {
      portal.style.position = 'fixed';
    }
  }
  window.addEventListener('scroll', ()=>{
    const r = portal.getBoundingClientRect();
    if (lastRect && Math.abs(r.top - lastRect.top) > 0.5) {
      useAbsoluteFallback = true;
    }
    lastRect = r;
    updateFabPosition();
  }, { passive: true });
  window.addEventListener('resize', updateFabPosition);
  updateFabPosition();
});
"""

def build_embedded_html():
    """Inline CSS and JS into index.html for rendering in Streamlit."""
    html = INDEX_HTML.replace(
        '<link rel="stylesheet" href="./styles.css" />',
        f'<style>\n{STYLES_CSS}\n</style>'
    ).replace(
        '<script src="./script.js"></script>',
        f'<script>\n{SCRIPT_JS}\n</script>'
    )
    return html


# ----------------------------------------
# Render embedded web app full-screen
# ----------------------------------------
st.set_page_config(page_title="Smart Folder", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    html, body, div[data-testid="stAppViewContainer"], .main, .block-container {
      height: 100svh; min-height: 100svh; overflow: hidden; padding: 0; margin: 0;
    }
    div[data-testid="stAppViewContainer"] { padding: 0 !important; }
    section[data-testid="stSidebar"] { display: none; }
    /* Make the embedded iframe full-screen and prevent outer scroll */
    div[data-testid="stAppViewContainer"] iframe { height: 100svh !important; width: 100vw !important; }
    </style>
    """,
    unsafe_allow_html=True,
)
components.html(build_embedded_html(), height=1200, scrolling=False)
