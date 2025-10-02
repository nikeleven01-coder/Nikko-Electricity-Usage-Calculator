import streamlit as st
import pdfplumber
import re
import pandas as pd

# ----------------------------------------
# APP CONFIG
# ----------------------------------------
st.set_page_config(page_title="Utility Dashboard", page_icon="⚡", layout="centered")

# ----------------------------------------
# DASHBOARD TITLE
# ----------------------------------------
st.title("⚡ Utility Tools Dashboard")

st.write("Select a tool below to get started:")

# ----------------------------------------
# 3x2 TILE GRID
# ----------------------------------------
col1, col2, col3 = st.columns(3)

# --- Tile 1 (Active App) ---
with col1:
    if st.button("💡 Electric Bill Calculator", use_container_width=True):
        st.session_state["page"] = "bill_app"

# --- Coming Soon Tiles ---
with col2:
    st.button("🧮 Coming Soon", use_container_width=True)

with col3:
    st.button("📊 Coming Soon", use_container_width=True)

col4, col5, col6 = st.columns(3)
with col4:
    st.button("🔋 Coming Soon", use_container_width=True)

with col5:
    st.button("🌤️ Coming Soon", use_container_width=True)

with col6:
    st.button("⚙️ Coming Soon", use_container_width=True)

# ----------------------------------------
# BILL APP PAGE (ONLY SHOWS WHEN CLICKED)
# ----------------------------------------
if "page" not in st.session_state:
    st.session_state["page"] = None

if st.session_state["page"] == "bill_app":
    st.markdown("---")
    st.header("💡 Electric Bill Calculator")

    # Tabs
    tab1, tab2 = st.tabs(["📄 Upload Bill", "✍️ Manual Input"])

    # ===========================================
    # TAB 1: UPLOAD BILL
    # ===========================================
    with tab1:
        uploaded_file = st.file_uploader("📂 Upload your electric bill (PDF)", type="pdf")

        total_kwh = None
        rate_per_kwh = None
        df_sections = None

        if uploaded_file:
            with pdfplumber.open(uploaded_file) as pdf:
                text = ""
                for page in pdf.pages:
                    if page.extract_text():
                        text += page.extract_text() + "\n"

                # --- Extract TOTAL AMOUNT DUE ---
                match_due = re.search(r"TOTAL AMOUNT DUE\s+([\d,]+\.\d{2})", text, re.IGNORECASE)
                total_due = float(match_due.group(1).replace(",", "")) if match_due else None

                # --- Extract Total kWh ---
                match_kwh = re.search(r"Billed\s*:\s*(\d+)", text, re.IGNORECASE)
                total_kwh = float(match_kwh.group(1)) if match_kwh else None

                # --- Compute Rate per kWh ---
                if total_due and total_kwh and total_kwh > 0:
                    rate_per_kwh = total_due / total_kwh

                # --- Extract CURRENT CHARGES + Subtotals ---
                charges_match = re.search(r"CURRENT CHARGES(.*?)CURRENT BILL", text, re.S | re.IGNORECASE)
                if charges_match:
                    charges_block = charges_match.group(1)
                    pattern = r"(?P<section>Generation & Transmission|Distribution Charges|Others|Government Charges).*?Sub-Total\s+([\d,]+\.\d{2})"
                    matches = re.finditer(pattern, charges_block, re.S | re.IGNORECASE)

                    rows = []
                    for m in matches:
                        section = m.group("section").strip()
                        subtotal = float(m.group(2).replace(",", ""))
                        est_kwh = subtotal / rate_per_kwh if rate_per_kwh else None
                        rows.append([section, subtotal, est_kwh])

                    if rows:
                        df_sections = pd.DataFrame(rows, columns=["Section", "Sub-Total (₱)", "Est. kWh"])

            # --- Display Results ---
            st.subheader("📊 Bill Summary")
            if total_kwh:
                st.write(f"📏 **Total kWh (from bill):** {total_kwh:,.0f}")
            if rate_per_kwh:
                st.write(f"⚡ **Rate per kWh:** ₱{rate_per_kwh:,.2f}")

            if df_sections is not None:
                with st.expander("📑 Current Charges Breakdown"):
                    df_display = df_sections.copy()
                    df_display["Sub-Total (₱)"] = df_display["Sub-Total (₱)"].map(lambda x: f"₱{x:,.2f}")
                    df_display["Est. kWh"] = df_display["Est. kWh"].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
                    st.dataframe(df_display, use_container_width=True)

            # --- Calculator (based on uploaded bill) ---
            st.subheader("🔢 Enter Your Own kWh Usage")
            manual_kwh = st.number_input("Enter your kWh usage", value=0.0, step=1.0)
            if st.button("💡 Compute My Bill"):
                if rate_per_kwh:
                    computed = manual_kwh * rate_per_kwh
                    st.success(f"💰 Your Computed Bill: ₱{computed:,.2f}")
                else:
                    st.error("⚠️ Please upload a valid bill first.")

    # ===========================================
    # TAB 2: MANUAL INPUT
    # ===========================================
    with tab2:
        st.subheader("🧮 Manual Entry Mode")
        st.write("Manually input your **Diff Rdg (kWh)** and **Total Amount Due** below:")

        manual_diff_rdg = st.number_input("📏 Diff Rdg (kWh)", value=0.0, step=1.0)
        manual_total_due = st.number_input("💰 Total Amount Due (₱)", value=0.0, step=0.01)

        if manual_diff_rdg > 0 and manual_total_due > 0:
            manual_rate_per_kwh = manual_total_due / manual_diff_rdg
            st.info(f"⚡ Computed Rate per kWh: ₱{manual_rate_per_kwh:,.2f}")

            st.subheader("🔢 Try New Usage")
            new_kwh = st.number_input("Enter new kWh usage", value=0.0, step=1.0, key="manual_input")
            if st.button("💡 Compute My Manual Bill"):
                computed_manual = new_kwh * manual_rate_per_kwh
                st.success(f"💰 Your Computed Bill: ₱{computed_manual:,.2f}")
        else:
            st.warning("Please enter both Diff Rdg and Total Due to calculate rate.")
            
            
            # smart_cloth_renamer_grid_folders_fixed.py

import streamlit as st
import os, zipfile, tempfile, re, json, base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import streamlit.components.v1 as components
from collections import defaultdict

# ---------------- Constants ---------------- #
FASHION_COLORS = {
    "Pine Cone": "#556a5f", "Viridian Green": "#009698", "Navy Blue": "#000080",
    "Burgundy": "#800020", "Rose Gold": "#b76e79", "Neon Green": "#39ff14",
    "Champagne": "#f7e7ce", "Emerald Green": "#50c878", "Sky Blue": "#87ceeb",
    "Charcoal": "#36454f", "Ivory": "#fffff0", "Orange": "#ffa500",
    "Pink": "#ffc0cb", "Gold": "#ffd700", "Beige": "#f5f5dc", "Brown": "#a52a2a",
}

STYLE_MAP = {
    "person": "", "shirt": "Shirt", "tshirt": "T-Shirt", "pants": "Pants",
    "jeans": "Jeans", "shorts": "Shorts", "skirt": "Skirt", "dress": "Gown",
    "suit": "Tuxedo", "jacket": "Jacket", "coat": "Coat", "hat": "Hat", "shoe": "Shoes",
    "tuxedo": "Tuxedo", "hoodie": "Hoodie", "long_gown": "Long_Gown",
    "ball_gown": "Ball_Gown", "cocktail_dress": "Cocktail_Dress", "blazer": "Blazer",
}

# ---------------- Color Utilities ---------------- #
def hex_to_rgb_tuple(hex_str):
    s = hex_str.lstrip("#")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))

def rgb_to_lab(rgb):
    arr = np.uint8([[[rgb[0], rgb[1], rgb[2]]]])
    return cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[0, 0].astype(float)

def nearest_palette_name(rgb):
    target = rgb_to_lab(rgb)
    best_name, best_d = None, None
    for name, lab in PALETTE_LAB.items():
        d = np.linalg.norm(target - lab)
        if best_d is None or d < best_d:
            best_d = d
            best_name = name
    return best_name

# Pre-compute color palettes
PALETTE_RGB = {n: hex_to_rgb_tuple(hx) for n, hx in FASHION_COLORS.items()}
PALETTE_LAB = {n: rgb_to_lab(rgb) for n, rgb in PALETTE_RGB.items()}


def dominant_color_kmeans(region_rgb: np.ndarray, k: int = 3) -> tuple:
    # region_rgb: HxWx3 in RGB
    try:
        data = region_rgb.reshape((-1, 3)).astype(np.float32)
        # k-means criteria and attempts
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        attempts = 3
        ret, labels, centers = cv2.kmeans(data, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
        labels = labels.flatten()
        counts = np.bincount(labels)
        dominant_idx = int(np.argmax(counts))
        dominant_rgb = tuple(int(c) for c in centers[dominant_idx])
        return dominant_rgb
    except Exception:
        # Fallback to mean
        return tuple(int(region_rgb[..., i].mean()) for i in range(3))


def mask_center_ellipse(region_rgb: np.ndarray) -> np.ndarray:
    h, w = region_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (max(1, int(w * 0.35)), max(1, int(h * 0.45)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return region_rgb[mask == 255]


def grabcut_refine(arr_bgr: np.ndarray, rect: tuple) -> np.ndarray:
    try:
        mask = np.zeros(arr_bgr.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        x, y, w, h = rect
        cv2.grabCut(arr_bgr, mask, (x, y, w, h), bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        result = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB) * mask2[..., None]
        return result[result.sum(axis=2) > 0]
    except Exception:
        return cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3)


def sanitize_filename(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9\-_\.]", "", s)
    return s


def image_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------- YOLO ---------------- #
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

model = load_yolo()

# ---------------- CLIP ---------------- #
@st.cache_resource
def load_clip():
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return model, processor
    except Exception:
        return None, None

def clip_zero_shot_classify(pil_image, candidate_labels=None):
    if candidate_labels is None:
        candidate_labels = ["Tuxedo", "Hoodie", "Long_Gown", "Gown", "Dress", "Ball_Gown", "Cocktail_Dress",
                           "Suit", "Blazer", "Jacket", "Shirt", "T-Shirt", "Pants", "Jeans", "Shorts", "Skirt"]
    try:
        import torch
        model, processor = load_clip()
        if model is None or processor is None:
            return None
        prompts = [f"a photo of a {lbl.replace('_', ' ')}" for lbl in candidate_labels]
        inputs = processor(text=prompts, images=pil_image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        best_idx = int(probs.argmax())
        return candidate_labels[best_idx]
    except Exception:
        return None


def detect_and_get_color(file_like):
    # Optimize: Resize image for faster processing
    pil = Image.open(file_like).convert("RGB")
    max_size = 800
    if max(pil.size) > max_size:
        pil.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    arr_rgb = np.array(pil)
    arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
    results = model.predict(arr_bgr, verbose=False, imgsz=640)
    if not results or len(results[0].boxes) == 0:
        clip_label = clip_zero_shot_classify(pil)
        # Robust color: mask center ellipse on full image
        dense = mask_center_ellipse(arr_rgb)
        if dense.size == 0:
            dense = arr_rgb.reshape(-1, 3)
        mean_rgb = dominant_color_kmeans(dense.reshape(-1, 3))
        palette_name = nearest_palette_name(mean_rgb)
        return mean_rgb, palette_name, pil, (clip_label or None)

    box = results[0].boxes[0]
    cls_id = int(box.cls[0].item())
    cloth_style = model.names[cls_id]

    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    H, W = arr_rgb.shape[:2]
    x1, x2 = max(0, x1), min(W, x2)
    y1, y2 = max(0, y1), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        clip_label = clip_zero_shot_classify(pil)
        dense = mask_center_ellipse(arr_rgb)
        if dense.size == 0:
            dense = arr_rgb.reshape(-1, 3)
        mean_rgb = dominant_color_kmeans(dense.reshape(-1, 3))
        palette_name = nearest_palette_name(mean_rgb)
        return mean_rgb, palette_name, pil, (clip_label or cloth_style)

    region = arr_rgb[y1:y2, x1:x2]
    if region.size == 0:
        clip_label = clip_zero_shot_classify(pil)
        dense = mask_center_ellipse(arr_rgb)
        if dense.size == 0:
            dense = arr_rgb.reshape(-1, 3)
        mean_rgb = dominant_color_kmeans(dense.reshape(-1, 3))
        palette_name = nearest_palette_name(mean_rgb)
        return mean_rgb, palette_name, pil, (clip_label or cloth_style)

    # Try GrabCut to suppress background around bbox
    rect = (x1, y1, x2 - x1, y2 - y1)
    grab = grabcut_refine(arr_bgr, rect)
    if grab.size > 0:
        region_dense = grab
    else:
        # Fall back to center ellipse mask
        region_dense = mask_center_ellipse(region)
        if region_dense.size == 0:
            region_dense = region.reshape(-1, 3)

    mean_rgb = dominant_color_kmeans(region_dense.reshape(-1, 3))
    palette_name = nearest_palette_name(mean_rgb)

    try:
        region_pil = Image.fromarray(region)
    except Exception:
        region_pil = pil
    clip_label = clip_zero_shot_classify(region_pil)
    if clip_label:
        cloth_style = clip_label

    return mean_rgb, palette_name, pil, cloth_style


# ---------------- Export Functions ---------------- #
def create_export_zip(order_data, items):
    """Create ZIP file from order data and items."""
    try:
        if not items:
            st.warning("No items to export")
            return None
            
        with tempfile.TemporaryDirectory() as td:
            folders = order_data.get("folders", [])
            main_items = order_data.get("mainItems", [])
            
            # Create id->item lookup for fast access
            item_lookup = {str(item["id"]): item for item in items}
            
            files_added = 0
            
            # Save main items (no folder) in root
            for it in main_items:
                original = item_lookup.get(str(it["id"]))
                if original and "pil" in original:
                    out_path = os.path.join(td, it["filename"])
                    original["pil"].save(out_path)
                    files_added += 1
            
            # Save foldered items
            for fo in folders:
                folder_name = fo.get("title") or "Folder"
                folder_dir = os.path.join(td, folder_name)
                os.makedirs(folder_dir, exist_ok=True)
                for it in fo.get("items", []):
                    original = item_lookup.get(str(it["id"]))
                    if original and "pil" in original:
                        out_path = os.path.join(folder_dir, it["filename"])
                        original["pil"].save(out_path)
                        files_added += 1
            
            if files_added == 0:
                st.warning("No valid images found to export")
                return None
            
            # Zip
            buf = BytesIO()
            with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(td):
                    for f in files:
                        fp = os.path.join(root, f)
                        arc = os.path.relpath(fp, td)
                        zf.write(fp, arcname=arc)
            
            st.success(f"Export created successfully with {files_added} files")
            return buf.getvalue()
    except Exception as e:
        st.error(f"Export failed: {e}")
        return None


# ---------------- Assign Filenames ---------------- #
def assign_filenames(items):
    grouped = defaultdict(list)
    for it in items:
        unique_code = f"C{1000 + it['id']}"
        raw_style = it["cloth_style"].lower()
        cloth_style = STYLE_MAP.get(raw_style, raw_style.capitalize())
        color_name = it["color"].replace(" ", "_")
        base_name = f"{unique_code}_{cloth_style}_{color_name}"
        grouped[base_name].append(it)
    for base, group in grouped.items():
        total = len(group)
        for idx, it in enumerate(group, start=1):
            it["filename"] = f"{base}_{idx} of {total}.jpg"
    return items


# ---------------- UI ---------------- #
st.set_page_config(layout="wide")
st.title("Cloth Renamer & Organizer")

uploaded_files = st.file_uploader(
    "Upload images or a ZIP",
    type=["jpg", "jpeg", "png", "zip"],
    accept_multiple_files=True
)

default_folder_name = st.session_state.get("folder_name")
if not default_folder_name:
    if uploaded_files and len(uploaded_files) == 1 and uploaded_files[0].name.lower().endswith(".zip"):
        default_folder_name = os.path.splitext(uploaded_files[0].name)[0]
    else:
        default_folder_name = "New Folder"
folder_name = st.text_input("Folder name", value=default_folder_name, key="folder_name")

if "items" not in st.session_state:
    st.session_state["items"] = []

if uploaded_files:
    st.session_state["items"] = []
    idx_counter = 0
    for f in uploaded_files:
        if f.name.lower().endswith(".zip"):
            with tempfile.TemporaryDirectory() as td:
                zip_path = os.path.join(td, f.name)
                with open(zip_path, "wb") as fh:
                    fh.write(f.read())
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(td)
                for fname in sorted(os.listdir(td)):
                    if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                        fp = os.path.join(td, fname)
                        with open(fp, "rb") as fh:
                            mean_rgb, pname, pil, cloth_style = detect_and_get_color(fh)
                            if mean_rgb:
                                small_b64 = image_to_base64(pil.resize((80, 80)))
                                st.session_state["items"].append({
                                    "id": idx_counter,
                                    "color": pname,
                                    "rgb": mean_rgb,
                                    "image_b64": small_b64,
                                    "pil": pil.copy(),
                                    "cloth_style": cloth_style,
                                })
                                idx_counter += 1
        else:
            mean_rgb, pname, pil, cloth_style = detect_and_get_color(f)
            if mean_rgb:
                small_b64 = image_to_base64(pil.resize((80, 80)))
                st.session_state["items"].append({
                    "id": idx_counter,
                    "color": pname,
                    "rgb": mean_rgb,
                    "image_b64": small_b64,
                    "pil": pil.copy(),
                    "cloth_style": cloth_style,
                })
                idx_counter += 1

items = assign_filenames(st.session_state["items"])
if not items:
    st.stop()

# ---------------- HTML + JS ---------------- #
blocks_data = [{
    "id": it["id"],
    "unique_id": f"C{1000 + it['id']}",
    "cloth_type": STYLE_MAP.get((it["cloth_style"] or "").lower(), (it["cloth_style"] or "").replace(" ", "_").capitalize()),
    "color_raw": it["color"],
    "color_norm": it["color"].replace(" ", "_").lower(),
    "rgb": it["rgb"],
    "img": it["image_b64"],
    "filename": it["filename"],
} for it in items]

html_blocks = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  /* Hide Streamlit emotion cache classes and file uploader preview */
  .st-emotion-cache-fis6aj,
  .e16n7gab7,
  .stFileUploader > div > div > div,
  .stFileUploader [data-testid="stFileUploaderDropzoneInstructions"],
  .stFileUploader [data-testid="stFileUploaderDropzoneInstructions"] + div {{
    display: none !important;
  }}
  
  :root {{ --zoom: 1; }}
  body {{
    margin:0; padding:0; font-family:sans-serif; background:#f7f7f7;
  }}
  .toolbar {{
    position:fixed; top:0; left:0; right:0;
    background:#ffffff; padding:10px 20px;
    box-shadow:0 2px 6px rgba(0,0,0,0.1);
    z-index:200; display:flex; gap:10px; align-items:center;
  }}
  .btn, .add-folder-btn, .edit-btn {{
    background:#007bff; color:white; border:none; border-radius:6px;
    padding:8px 14px; cursor:pointer; font-size:14px;
  }}
  .folder-grid {{
    display:grid;
    grid-template-columns:repeat(auto-fit, minmax(420px, 1fr));
    gap:16px;
    margin-top:70px;
    padding:10px 20px;
    align-items:start;
  }}
  .folder {{
    background:white; border-radius:10px; padding:0;
    box-shadow:0 2px 10px rgba(0,0,0,0.06);
    display:flex; flex-direction:column; min-height:220px; border:1px solid #e9ecef;
  }}
  .folder.collapsed {{ min-height:80px; }}
  .folder-header {{
    display:flex; align-items:center; gap:8px; padding:8px 10px; border-bottom:1px solid #f1f3f5; cursor:grab;
  }}
  .caret {{ cursor:pointer; user-select:none; font-size:16px; padding:0 6px; }}
  .folder.collapsed .list {{ display:none; }}
  .folder-select {{ width:18px; height:18px; }}
  .folder-title {{
    flex:1; font-weight:bold; font-size:15px; border:none; outline:none;
  }}
  .badge {{ background:#eef2ff; color:#334; font-size:11px; padding:2px 6px; border-radius:10px; }}
  .content {{ margin-top:20px; padding:10px 20px; }}
  .list {{
    overflow-y:auto; max-height:620px; padding-right:6px;
    --zoom: 1;
  }}
  .block {{
    width:100%; height:calc(80px * var(--zoom)); display:flex; align-items:center; gap:12px;
    border-radius:8px; padding:6px; margin:8px 0; border:1px solid #ddd;
    box-shadow:0 1px 2px rgba(0,0,0,0.05); background:white;
  }}
  .block.selected {{ border:2px solid #007bff; background:#eef5ff; }}
  .thumb {{ width:calc(68px * var(--zoom)); height:calc(68px * var(--zoom)); border-radius:6px; object-fit:cover; }}
  .handle {{ cursor:grab; font-size:24px; margin-right:8px; width:44px; text-align:center; user-select:none; }}
  input[type=\"color\"].color-picker {{ width:28px; height:28px; padding:0; border:none; background:transparent; }}
  input[type=\"text\"].filename {{ flex:1; font-size:14px; padding:4px; }}
  .sel {{ width:18px; height:18px; }}
</style>

<script src=\"https://cdnjs.cloudflare.com/ajax/libs/Sortable/1.15.0/Sortable.min.js\"></script>
</head>
<body>
  <div class=\"toolbar\">
    <button class=\"add-folder-btn\" onclick=\"addFolder()\">+ Add Folder</button>
    <button class=\"edit-btn\" id=\"editBtn\" onclick=\"toggleEdit()\">Edit</button>
    <button class=\"btn\" id=\"deleteBtn\" onclick=\"onDelete()\">Delete</button>
    <button class=\"btn\" id=\"randomBtn\" onclick=\"randomizeIds()\">Random_ID</button>
    <button class=\"btn\" id=\"exportBtn\" onclick=\"exportZip()\">Export</button>
  </div>

  <div class=\"folder-grid\" id=\"folderGrid\"></div>

  <div class=\"content\" style=\"display:none\">
    <div id=\"mainList\"></div>
  </div>

  <div class=\"modal-backdrop\" id=\"modalBackdrop\" style=\"position:fixed; inset:0; background:rgba(0,0,0,0.4); display:none; align-items:center; justify-content:center; z-index:500;\">
    <div class=\"modal\" style=\"background:#fff; border-radius:8px; padding:16px; width:420px; box-shadow:0 10px 30px rgba(0,0,0,0.2);\">
      <h3 id=\"modalTitle\" style=\"margin:0 0 10px 0;\">Confirm Batch Edit</h3>
      <div class=\"row\" style=\"margin:8px 0; display:flex; gap:8px; align-items:center;\"><label style=\"width:110px; font-size:13px;\">File name</label><input id=\"md_file\" disabled style=\"flex:1; padding:6px 8px; font-size:14px;\"></div>
      <div class=\"row\" style=\"margin:8px 0; display:flex; gap:8px; align-items:center;\"><label style=\"width:110px; font-size:13px;\">Unique ID</label><input id=\"md_uid\" style=\"flex:1; padding:6px 8px; font-size:14px;\"></div>
      <div class=\"row\" style=\"margin:8px 0; display:flex; gap:8px; align-items:center;\"><label style=\"width:110px; font-size:13px;\">Cloth type</label><input id=\"md_type\" style=\"flex:1; padding:6px 8px; font-size:14px;\"></div>
      <div class=\"row\" style=\"margin:8px 0; display:flex; gap:8px; align-items:center;\"><label style=\"width:110px; font-size:13px;\">Color name</label><input id=\"md_color\" style=\"flex:1; padding:6px 8px; font-size:14px;\"></div>
      <div class=\"actions\" style=\"display:flex; gap:8px; justify-content:flex-end; margin-top:12px;\">
        <button class=\"btn\" style=\"background:#6c757d;\" onclick=\"closeBatchModal(true)\">Cancel</button>
        <button class=\"btn\" onclick=\"prefillFromFirst()\">Edit</button>
        <button class=\"btn\" style=\"background:#007bff; color:#fff;\" onclick=\"applyBatch()\">OK</button>
      </div>
    </div>
  </div>

<script>
const data = {json.dumps(blocks_data)};
const PALETTE = {json.dumps({k:v for k,v in FASHION_COLORS.items()})};
const mainList = document.getElementById('mainList');
const folderGrid = document.getElementById('folderGrid');
const modalBackdrop = document.getElementById('modalBackdrop');
const editBtn = document.getElementById('editBtn');
let folderCount = 0;
const folderKeyToInfo = new Map();
const selectedIds = new Set();
const selectedFolders = new Set();
let lastSelectedId = null;

function typeKey(name) {{
  return (name || '').toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
}}

function ensureFolder(name) {{
  const key = typeKey(name);
  if (folderKeyToInfo.has(key)) {{ return folderKeyToInfo.get(key); }}
  const info = createFolder(name);
  info.folder.dataset.typeKey = key;
  folderKeyToInfo.set(key, info);
  return info;
}}

function clearSelection() {{
  const blocks = document.querySelectorAll('.block.selected');
  blocks.forEach(b => {{
    b.classList.remove('selected');
    const cb = b.querySelector('input.sel');
    if (cb) cb.checked = false;
  }});
  selectedIds.clear();
  lastSelectedId = null;
}}

// Color utils: RGB->Lab and CIEDE2000
function rgb2xyz([r,g,b]) {{
  r/=255; g/=255; b/=255;
  r = r>0.04045 ? Math.pow((r+0.055)/1.055,2.4): r/12.92;
  g = g>0.04045 ? Math.pow((g+0.055)/1.055,2.4): g/12.92;
  b = b>0.04045 ? Math.pow((b+0.055)/1.055,2.4): b/12.92;
  let x = (r*0.4124 + g*0.3576 + b*0.1805)/0.95047;
  let y = (r*0.2126 + g*0.7152 + b*0.0722)/1.00000;
  let z = (r*0.0193 + g*0.1192 + b*0.9505)/1.08883;
  const f = t=> t>0.008856? Math.cbrt(t): (7.787*t + 16/116);
  return [116*f(y)-16, 500*(f(x)-f(y)), 200*(f(y)-f(z))];
}}
function hex2rgb(hex) {{ return [parseInt(hex.substr(1,2),16), parseInt(hex.substr(3,2),16), parseInt(hex.substr(5,2),16)]; }}
function deg2rad(d) {{ return d*Math.PI/180; }}
function rad2deg(r) {{ return r*180/Math.PI; }}
function ciede2000(Lab1, Lab2) {{
  const [L1, a1, b1] = Lab1, [L2, a2, b2] = Lab2;
  const kL=1, kC=1, kH=1;
  const C1 = Math.hypot(a1,b1), C2=Math.hypot(a2,b2);
  const Cbar = (C1+C2)/2;
  const G = 0.5*(1-Math.sqrt(Math.pow(Cbar,7)/(Math.pow(Cbar,7)+Math.pow(25,7))));
  const a1p=(1+G)*a1, a2p=(1+G)*a2;
  const C1p=Math.hypot(a1p,b1), C2p=Math.hypot(a2p,b2);
  const h1p = (Math.atan2(b1,a1p)+2*Math.PI)%(2*Math.PI);
  const h2p = (Math.atan2(b2,a2p)+2*Math.PI)%(2*Math.PI);
  const dLp = L2-L1;
  const dCp = C2p-C1p;
  let dhp = 0;
  if (C1p*C2p!==0) {{
    if (Math.abs(h2p-h1p) <= Math.PI) dhp = h2p - h1p;
    else dhp = h2p<=h1p? h2p - h1p + 2*Math.PI: h2p - h1p - 2*Math.PI;
  }}
  const dHp = 2*Math.sqrt(C1p*C2p)*Math.sin(dhp/2);
  const Lbarp = (L1+L2)/2;
  const Cbarp = (C1p+C2p)/2;
  let hbarp = 0;
  if (C1p*C2p===0) hbarp = h1p + h2p;
  else if (Math.abs(h1p-h2p)<=Math.PI) hbarp = (h1p+h2p)/2;
  else hbarp = (h1p+h2p+2*Math.PI)/2;
  const T = 1 - 0.17*Math.cos(hbarp - deg2rad(30)) + 0.24*Math.cos(2*hbarp) + 0.32*Math.cos(3*hbarp + deg2rad(6)) - 0.20*Math.cos(4*hbarp - deg2rad(63));
  const Sl = 1 + (0.015*Math.pow(Lbarp-50,2))/Math.sqrt(20 + Math.pow(Lbarp-50,2));
  const Sc = 1 + 0.045*Cbarp;
  const Sh = 1 + 0.015*Cbarp*T;
  const dTheta = deg2rad(30)*Math.exp(-Math.pow(rad2deg(hbarp)-275,2)/ (25*25));
  const Rc = 2*Math.sqrt(Math.pow(Cbarp,7)/(Math.pow(Cbarp,7)+Math.pow(25,7)));
  const Rt = -Math.sin(2*dTheta)*Rc;
  return Math.sqrt(Math.pow(dLp/(kL*Sl),2) + Math.pow(dCp/(kC*Sc),2) + Math.pow(dHp/(kH*Sh),2) + Rt*(dCp/(kC*Sc))*(dHp/(kH*Sh)));
}}

function nearestPaletteNameFromHex(hex) {{
  const lab = rgb2xyz(hex2rgb(hex));
  let bestName = 'Unknown'; let bestD = Infinity;
  for (const [name, hx] of Object.entries(PALETTE)) {{
    const lab2 = rgb2xyz(hex2rgb(hx));
    const d = ciede2000(lab, lab2);
    if (d < bestD) {{ bestD = d; bestName = name; }}
  }}
  return bestName;
}}

function getColorHexFromName(colorName) {{
  // Convert color name to hex by looking up in PALETTE
  for (const [name, hex] of Object.entries(PALETTE)) {{
    if (name.toLowerCase().replace(/\s+/g, '_') === colorName.toLowerCase()) {{
      return hex;
    }}
  }}
  // If not found, return null
  return null;
}}

document.addEventListener('contextmenu', (e) => {{ e.preventDefault(); clearSelection(); }});

function toggleEdit() {{
  if (modalBackdrop.style.display === 'flex') {{
    closeBatchModal(true);
    clearSelection();
    return;
  }}
  const ids = Array.from(selectedIds);
  if (ids.length < 2) return;
  openBatchModal();
}}

function createBlock(item) {{
  const wrapper = document.createElement('div');
  wrapper.className = 'block';
  wrapper.dataset.id = item.id;
  wrapper.dataset.uniqueId = item.unique_id;
  wrapper.dataset.clothType = item.cloth_type;
  wrapper.dataset.colorName = item.color_norm;
  wrapper.setAttribute('draggable','true');

  const checkbox = document.createElement('input');
  checkbox.type = 'checkbox';
  checkbox.className = 'sel';
  checkbox.addEventListener('change', () => {{
    if (checkbox.checked) {{ selectedIds.add(item.id.toString()); wrapper.classList.add('selected'); lastSelectedId = item.id.toString(); }}
    else {{ selectedIds.delete(item.id.toString()); wrapper.classList.remove('selected'); }}
  }});
  // Right-click selection
  wrapper.addEventListener('contextmenu', (e) => {{
    e.preventDefault();
    if (!wrapper.classList.contains('selected')) {{
      selectedIds.clear();
      wrapper.classList.add('selected');
      selectedIds.add(item.id.toString());
      lastSelectedId = item.id.toString();
      checkbox.checked = true;
    }}
  }});
  
  // Improved drag functionality
  wrapper.addEventListener('dragstart', (e) => {{
    if (!wrapper.classList.contains('selected')) {{
      selectedIds.clear();
      wrapper.classList.add('selected');
      selectedIds.add(item.id.toString());
      lastSelectedId = item.id.toString();
      checkbox.checked = true;
    }}
    e.dataTransfer.effectAllowed = 'move';
    // Store selected IDs for drag operation
    e.dataTransfer.setData('text/selectedIds', JSON.stringify(Array.from(selectedIds)));
  }});
  
  wrapper.appendChild(checkbox);

  const handle = document.createElement('div');
  handle.className = 'handle';
  handle.innerHTML = '≡';
  wrapper.appendChild(handle);

  const img = document.createElement('img');
  img.className = 'thumb';
  img.src = 'data:image/png;base64,' + item.img;
  wrapper.appendChild(img);

  const picker = document.createElement('input');
  picker.type = 'color';
  picker.className = 'color-picker';
  const rgb = item.rgb;
  picker.value = '#' + [rgb[0], rgb[1], rgb[2]].map(x => x.toString(16).padStart(2,'0')).join('');
  picker.addEventListener('input', () => {{
    const name = nearestPaletteNameFromHex(picker.value);
    const norm = name.replace(/\s+/g,'_').toLowerCase();
    wrapper.dataset.colorName = norm;
    const list = wrapper.parentElement;
    const folder = list.closest('.folder');
    const ct = folder ? folder.querySelector('.folder-title').value.trim() : wrapper.dataset.clothType;
    renumber(list, ct, false);
    sendOrder();
  }});
  wrapper.appendChild(picker);

  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'filename';
  input.value = item.filename;
  input.addEventListener('input', sendOrder);
  wrapper.appendChild(input);

  return wrapper;
}}

// Allow dropping onto folder headers to move images
function setupHeaderDrop(header, folder, list) {{
  header.setAttribute('draggable','true');
  header.addEventListener('dragstart', (e) => {{
    e.dataTransfer.setData('text/folderId', folder.dataset.folderId);
  }});
  header.addEventListener('dragover', (e) => {{ e.preventDefault(); header.style.background = '#f1f5ff'; }});
  header.addEventListener('dragleave', () => {{ header.style.background = ''; }});
  header.addEventListener('drop', (e) => {{
    e.preventDefault(); header.style.background = '';
    const srcId = e.dataTransfer.getData('text/folderId');
    if (srcId) {{
      const srcFolder = Array.from(folderGrid.children).find(f => String(f.dataset.folderId) === String(srcId));
      if (srcFolder && srcFolder !== folder) {{ swapNodes(srcFolder, folder); return; }}
    }}
    const ids = Array.from(selectedIds);
    for (const id of ids) {{
      const el = document.querySelector(`.block[data-id=\\"${{id}}\\"]`);
      if (el && el.parentElement !== list) list.appendChild(el);
    }}
    const tName = folder.querySelector('.folder-title').value.trim();
    renumber(list, tName, false);
    updateBadge(folder);
    sendOrder();
  }});
}}

function createFolder(name = "New Folder") {{
  const folder = document.createElement('div');
  folder.className = 'folder';
  folder.dataset.folderId = String(folderCount++);

  const header = document.createElement('div');
  header.className = 'folder-header';

  const caret = document.createElement('span');
  caret.className = 'caret';
  caret.textContent = '▾';
  caret.addEventListener('click', () => {{
    folder.classList.toggle('collapsed');
    caret.textContent = folder.classList.contains('collapsed') ? '▸' : '▾';
  }});

  const folderSel = document.createElement('input');
  folderSel.type = 'checkbox';
  folderSel.className = 'folder-select';
  folderSel.addEventListener('change', () => {{
    const fid = String(folder.dataset.folderId);
    if (folderSel.checked) {{ selectedFolders.add(fid); }} else {{ selectedFolders.delete(fid); }}
  }});

  const title = document.createElement('input');
  title.className = 'folder-title';
  title.value = name;
  title.addEventListener('input', () => {{
    const list = folder.querySelector('.list');
    const newType = title.value.trim();
    const oldKey = folder.dataset.typeKey || '';
    const newKey = typeKey(newType);
    if (oldKey && oldKey !== newKey) {{ folderKeyToInfo.delete(oldKey); }}
    folder.dataset.typeKey = newKey;
    folderKeyToInfo.set(newKey, {{ folder, list, title }});
    const blocks = Array.from(list.querySelectorAll('.block'));
    for (const b of blocks) {{ b.dataset.clothType = newType; }}
    renumber(list, newType, false);
    updateBadge(folder);
    sendOrder();
  }});

  const badge = document.createElement('span');
  badge.className = 'badge';
  badge.textContent = '0 items';

  header.appendChild(caret);
  header.appendChild(folderSel);
  header.appendChild(title);
  header.appendChild(badge);
  folder.appendChild(header);

  const list = document.createElement('div');
  list.className = 'list';
  list.style.setProperty('--zoom', 1);
  list.addEventListener('wheel', (e) => {{ if (!e.ctrlKey) return; e.preventDefault(); const current = parseFloat(getComputedStyle(list).getPropertyValue('--zoom')) || 1; const delta = e.deltaY < 0 ? 0.1 : -0.1; let next = current + delta; if (next < 1) next = 1; list.style.setProperty('--zoom', next.toFixed(2)); }}, {{ passive: false }});

  folder.appendChild(list);
  folderGrid.appendChild(folder);

  Sortable.create(list, {{ group: 'shared', animation: 200, handle: '.handle, .thumb', onEnd: onDragEnd }});
  setupHeaderDrop(header, folder, list);

  updateBadge(folder);
  return {{ folder, list, title }};
}}

function updateBadge(folder) {{
  const count = folder.querySelectorAll('.list .block').length;
  const badge = folder.querySelector('.badge');
  if (badge) badge.textContent = `${{count}} item${{count===1?'':'s'}}`;
}}

function addFolder() {{
  const {{ list, title }} = createFolder();
  folderKeyToInfo.set(typeKey(title.value), {{ folder: list.closest('.folder'), list, title }});
  sendOrder();
}}

function renumber(list, clothType, forceNewUid=false) {{
  const blocks = Array.from(list.querySelectorAll('.block'));
  const groups = new Map();
  for (const b of blocks) {{
    const ct = clothType || b.dataset.clothType || '';
    const color = b.dataset.colorName || '';
    const uid = b.dataset.uniqueId || '';
    // Group by Unique_ID + Cloth_style + Color_name for true duplicates
    const key = `${{uid}}||${{ct}}||${{color}}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(b);
  }}
  for (const [key, arr] of groups) {{
    const [uid, ct, color] = key.split('||');
    // Only generate new Unique_ID if forceNewUid is true (Random_ID button)
    if (forceNewUid) {{
      arr.forEach(b => {{
        b.dataset.uniqueId = generateUniqueId(ct);
      }});
    }}
    const N = arr.length;
    arr.forEach((b, i) => {{
      const input = b.querySelector('.filename');
      const ctUnderscore = (ct || '').replace(/\s+/g, '_');
      const colorUnderscore = (color || '').replace(/\s+/g, '_');
      // Only add numbering if there are true duplicates (same Unique_ID + Cloth_style + Color_name)
      const numbering = N === 1 ? '1 of 1' : `${{i+1}} of ${{N}}`;
      input.value = `${{b.dataset.uniqueId}}_${{ctUnderscore}}_${{colorUnderscore}} ${{numbering}}.jpg`;
    }});
  }}
}}

function generateUniqueId(ct) {{
  const firstChar = (ct || '').replace(/[^A-Za-z]/g, '').charAt(0).toUpperCase() || 'X';
  let digits = '';
  for (let i = 0; i < 6; i++) digits += Math.floor(Math.random() * 10).toString();
  return firstChar + digits;
}}

// Build folders by cloth_type
const types = Array.from(new Set(data.map(d => d.cloth_type)));
for (const t of types) {{
  const info = ensureFolder(t);
  info.title.value = t;
}}

// Place blocks into their cloth_type folders (create if missing)
for (const it of data) {{
  const info = ensureFolder(it.cloth_type);
  const block = createBlock(it);
  info.list.appendChild(block);
  updateBadge(info.folder);
}}
// Initial renumber per folder
for (const folder of folderGrid.children) {{
  const title = folder.querySelector('.folder-title').value.trim();
  const list = folder.querySelector('.list');
  renumber(list, title);
}}

Sortable.create(mainList, {{
  group: 'shared',
  animation: 200,
  handle: '.handle, .thumb',
  onEnd: onDragEnd
}});

// Make folders sortable by header drag and support swap-on-drop
new Sortable(document.getElementById('folderGrid'), {{
  draggable: '.folder',
  handle: '.folder-header',
  animation: 200
}});

function swapNodes(a, b) {{
  const aNext = a.nextSibling;
  const parent = a.parentNode;
  if (b === aNext) {{ parent.insertBefore(b, a); return; }}
  parent.insertBefore(a, b);
  parent.insertBefore(b, aNext);
}}

function openBatchModal() {{
  const ids = Array.from(selectedIds);
  if (ids.length < 2) return;
  const sourceId = lastSelectedId || ids[ids.length-1];
  const first = document.querySelector(`.block[data-id=\\"${{sourceId}}\\"]`);
  const fname = first.querySelector('.filename').value;
  document.getElementById('modalTitle').textContent = fname;
  document.getElementById('md_file').value = fname;
  document.getElementById('md_uid').value = first.dataset.uniqueId;
  document.getElementById('md_type').value = first.dataset.clothType;
  document.getElementById('md_color').value = first.dataset.colorName;
  modalBackdrop.style.display = 'flex';
}}
function closeBatchModal(clearing=false) {{ modalBackdrop.style.display = 'none'; if (clearing) clearSelection(); }}
function prefillFromFirst() {{ /* inputs already prefilled; keep editable */ }}

function applyBatch() {{
  const ids = Array.from(selectedIds);
  if (ids.length < 2) {{ closeBatchModal(true); return; }}
  const sourceId = lastSelectedId || ids[ids.length-1];
  const src = document.querySelector(`.block[data-id=\\"${{sourceId}}\\"]`);
  const baseUidInput = document.getElementById('md_uid').value.trim();
  const baseType = document.getElementById('md_type').value.trim();
  const baseColor = document.getElementById('md_color').value.trim();
  const baseUid = baseUidInput || generateUniqueId(baseType);

  for (const id of ids) {{
    const block = document.querySelector(`.block[data-id=\\"${{id}}\\"]`);
    if (!block) continue;
    block.dataset.uniqueId = baseUid;
    block.dataset.clothType = baseType;
    block.dataset.colorName = baseColor;
    
    // Update color picker for selected images
    const colorPicker = block.querySelector('.color-picker');
    if (colorPicker) {{
      // Convert color name to hex and update picker
      const colorHex = getColorHexFromName(baseColor);
      if (colorHex) {{
        colorPicker.value = colorHex;
      }}
    }}
  }}
  const affectedLists = new Set();
  for (const id of ids) {{ const b = document.querySelector(`.block[data-id=\\"${{id}}\\"]`); if (b) affectedLists.add(b.parentElement); }}
  affectedLists.forEach(list => {{ const tFolder = list.closest('.folder'); const tName = tFolder ? tFolder.querySelector('.folder-title').value.trim() : ''; renumber(list, tName, false); if (tFolder) updateBadge(tFolder); }});
  closeBatchModal(true);
  sendOrder();
}}

function randomizeIds() {{
  if (selectedIds.size === 0) {{
    alert('Please select at least one image first');
    return;
  }}
  const ids = Array.from(selectedIds);
  for (const id of ids) {{
    const block = document.querySelector(`.block[data-id=\\"${{id}}\\"]`);
    if (!block) continue;
    const ct = block.dataset.clothType;
    block.dataset.uniqueId = generateUniqueId(ct);
  }}
  const affected = new Set();
  for (const id of ids) {{ 
    const b = document.querySelector(`.block[data-id=\\"${{id}}\\"]`); 
    if (b) affected.add(b.parentElement); 
  }}
  affected.forEach(list => {{ 
    const f = list.closest('.folder'); 
    const t = f ? f.querySelector('.folder-title').value.trim() : ''; 
    renumber(list, t, false); // Don't force new UID for unselected images
    if (f) updateBadge(f); 
  }});
  sendOrder();
}}

function onDragEnd(evt) {{
  const dragged = evt.item;
  const toList = evt.to;
  const fromList = evt.from;

  // Handle multiple selected images drag
  if (dragged.classList.contains('selected') && selectedIds.size > 1) {{
    const ids = Array.from(selectedIds);
    for (const id of ids) {{
      if (id === dragged.dataset.id) continue;
      const other = document.querySelector(`.block[data-id="${{id}}"]`);
      if (other && other.parentElement !== toList) {{ 
        // Remove from current parent first
        if (other.parentElement) {{
          other.parentElement.removeChild(other);
        }}
        // Add to new parent
        toList.appendChild(other); 
        // Update cloth type for all selected images
        other.dataset.clothType = dragged.dataset.clothType;
      }}
    }}
  }}

  const targetFolder = toList.closest('.folder');
  if (targetFolder) {{
    const tName = targetFolder.querySelector('.folder-title').value.trim();
    const moved = Array.from(toList.querySelectorAll('.block'));
    for (const b of moved) {{
      if (b === dragged || b.classList.contains('selected')) {{
        b.dataset.clothType = tName;
      }}
    }}
    renumber(toList, tName, false);
    updateBadge(targetFolder);
  }}
  const sourceFolder = fromList.closest('.folder');
  if (sourceFolder) {{
    const sName = sourceFolder.querySelector('.folder-title').value.trim();
    renumber(fromList, sName, false);
    updateBadge(sourceFolder);
  }}

  sendOrder();
}}

function onDelete() {{
  if (selectedFolders.size > 0) {{
    if (confirm('Delete selected folder(s) and all contents?')) {{
      for (const fid of Array.from(selectedFolders)) {{
        const folder = Array.from(folderGrid.children).find(f => String(f.dataset.folderId) === String(fid));
        if (folder) folder.remove();
      }}
      selectedFolders.clear();
      sendOrder();
    }}
    return;
  }}
  if (selectedIds.size > 0) {{
    if (confirm('Delete selected image(s)?')) {{
      const affected = new Set();
      for (const id of Array.from(selectedIds)) {{
        const block = document.querySelector(`.block[data-id="${{id}}"]`);
        if (block) {{ affected.add(block.parentElement); block.remove(); }}
      }}
      selectedIds.clear();
      affected.forEach(list => {{
        const tFolder = list.closest('.folder');
        const tName = tFolder ? tFolder.querySelector('.folder-title').value.trim() : '';
        renumber(list, tName, false);
        if (tFolder) updateBadge(tFolder);
      }});
      sendOrder();
    }}
  }}
}}

function sendOrder() {{
  const folders = Array.from(folderGrid.children).map(folder => {{
    const folderId = folder.dataset.folderId;
    const title = folder.querySelector('.folder-title').value;
    const items = Array.from(folder.querySelectorAll('.block')).map(b => {{
      const id = b.dataset.id;
      const input = b.querySelector('input.filename');
      return {{ id, filename: input.value, unique_id: b.dataset.uniqueId, cloth_type: b.dataset.clothType, color_name: b.dataset.colorName }};
    }});
    return {{ folderId, title, items }};
  }});

  const mainItems = Array.from(mainList.querySelectorAll('.block')).map(b => {{
    const id = b.dataset.id;
    const input = b.querySelector('input.filename');
    return {{ id, filename: input.value, unique_id: b.dataset.uniqueId, cloth_type: b.dataset.clothType, color_name: b.dataset.colorName }};
  }});

  window.parent.postMessage({{
    isStreamlitMessage: true,
    type: 'streamlit:setComponentValue',
    value: {{ mainItems, folders }}
  }}, '*');
}}

function exportZip() {{
  const folders = Array.from(folderGrid.children).map(folder => {{
    const folderId = folder.dataset.folderId;
    const title = folder.querySelector('.folder-title').value;
    const items = Array.from(folder.querySelectorAll('.block')).map(b => {{
      const id = b.dataset.id;
      const input = b.querySelector('input.filename');
      return {{ id, filename: input.value, unique_id: b.dataset.uniqueId, cloth_type: b.dataset.clothType, color_name: b.dataset.colorName }};
    }});
    return {{ folderId, title, items }};
  }});

  const mainItems = Array.from(mainList.querySelectorAll('.block')).map(b => {{
    const id = b.dataset.id;
    const input = b.querySelector('input.filename');
    return {{ id, filename: input.value, unique_id: b.dataset.uniqueId, cloth_type: b.dataset.clothType, color_name: b.dataset.colorName }};
  }});

  window.parent.postMessage({{
    isStreamlitMessage: true,
    type: 'streamlit:setComponentValue',
    value: {{ mainItems, folders, export: true }}
  }}, '*');
}}

sendOrder();
</script>
</body>
</html>
"""

order = components.html(html_blocks, height=1200, scrolling=True)

# Export button
if st.button("📦 Export All Folders", type="primary"):
    if isinstance(order, dict):
        zip_data = create_export_zip(order, items)
        if zip_data:
            st.download_button(
                "Download Export", 
                data=zip_data, 
                file_name="cloth_export.zip",
                mime="application/zip"
            )
        else:
            st.error("Failed to create export file")
    else:
        st.warning("No data to export. Please upload some images first.")

# Handle export request from HTML component
if isinstance(order, dict) and order.get("export"):
    zip_data = create_export_zip(order, items)
    if zip_data:
        st.download_button(
            "Download Export", 
            data=zip_data, 
            file_name="cloth_export.zip",
            mime="application/zip"
        )

