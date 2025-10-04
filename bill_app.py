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
import pdfplumber
import pandas as pd

# ---------------- Constants ---------------- #
FASHION_COLORS = {
    # Core existing colors
    "Pine Cone": "#556a5f", "Viridian Green": "#009698", "Navy Blue": "#000080",
    "Burgundy": "#800020", "Rose Gold": "#b76e79", "Neon Green": "#39ff14",
    "Champagne": "#f7e7ce", "Emerald Green": "#50c878", "Sky Blue": "#87ceeb",
    "Charcoal": "#36454f", "Ivory": "#fffff0", "Orange": "#ffa500",
    "Pink": "#ffc0cb", "Gold": "#ffd700", "Beige": "#f5f5dc", "Brown": "#a52a2a",

    # Added common tones and shades
    "Black": "#000000", "White": "#ffffff",
    "Gray": "#808080", "Light Gray": "#d3d3d3", "Dark Gray": "#404040",
    "Red": "#ff0000", "Crimson": "#dc143c",
    "Purple": "#800080", "Lavender": "#e6e6fa", "Indigo": "#4b0082", "Violet": "#8f00ff",
    "Blue": "#0000ff", "Royal Blue": "#4169e1", "Cobalt": "#0047ab",
    "Teal": "#008080", "Turquoise": "#40e0d0", "Aquamarine": "#7fffd4",
    "Green": "#008000", "Forest Green": "#228b22", "Olive": "#808000", "Mint": "#98ff98", "Lime": "#00ff00",
    "Yellow": "#ffff00", "Mustard": "#ffdb58",
    "Peach": "#ffe5b4", "Coral": "#ff7f50", "Salmon": "#fa8072",
    "Tan": "#d2b48c", "Khaki": "#f0e68c",
    "Silver": "#c0c0c0", "Bronze": "#cd7f32", "Copper": "#b87333",
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
    """Robust conversion from an RGB tuple/list/array to LAB.
    Handles None, malformed lengths, numpy arrays, and out-of-range values.
    """
    if rgb is None:
        r = g = b = 0
    else:
        try:
            # Support numpy arrays and sequences
            r, g, b = (int(np.clip(rgb[0], 0, 255)),
                       int(np.clip(rgb[1], 0, 255)),
                       int(np.clip(rgb[2], 0, 255)))
        except Exception:
            # Fallback if rgb is scalar or shorter than 3
            val = int(np.clip(rgb, 0, 255)) if np.isscalar(rgb) else 0
            r = g = b = val
    arr = np.uint8([[[r, g, b]]])
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
def create_export_zip(order_data, items, flatten=False):
    """Create ZIP file from order data and items.
    If flatten is True, all images are stored at zip root without folder directories.
    """
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
                    # Avoid overwrite collisions when flattening or duplicate names
                    base, ext = os.path.splitext(out_path)
                    n = 2
                    while os.path.exists(out_path):
                        out_path = f"{base} ({n}){ext}"
                        n += 1
                    original["pil"].save(out_path)
                    files_added += 1
            
            # Save foldered items
            for fo in folders:
                folder_name = fo.get("title") or "Folder"
                folder_dir = os.path.join(td, folder_name)
                if not flatten:
                    os.makedirs(folder_dir, exist_ok=True)
                for it in fo.get("items", []):
                    original = item_lookup.get(str(it["id"]))
                    if original and "pil" in original:
                        out_path = os.path.join(td if flatten else folder_dir, it["filename"])
                        # Avoid overwrite collisions when flattening or duplicate names
                        base, ext = os.path.splitext(out_path)
                        n = 2
                        while os.path.exists(out_path):
                            out_path = f"{base} ({n}){ext}"
                            n += 1
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
        raw_style = (it.get("cloth_style") or "").lower()
        base_style = (it.get("cloth_style") or "")
        cloth_style = STYLE_MAP.get(raw_style, base_style.replace(" ", "_").capitalize())
        color_name = it["color"].replace(" ", "_")
        base_name = f"{unique_code}_{cloth_style}_{color_name}"
        grouped[base_name].append(it)
    for base, group in grouped.items():
        total = len(group)
        for idx, it in enumerate(group, start=1):
            ext = it.get("ext", ".jpg")
            it["filename"] = f"{base} {idx} of {total}{ext}"
    return items


#############################
# -------- Dashboard --------
#############################
# Use a single page config for the whole app
st.set_page_config(page_title="Utility Dashboard", page_icon="⚡", layout="wide")

# Initialize routing state
if "page" not in st.session_state:
    st.session_state["page"] = None

# Landing dashboard
st.title("⚡ Utility Tools Dashboard")
st.write("Select a tool below to get started:")
st.markdown(
    """
    <style>
    @media (max-width: 768px) {
      [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
      [data-testid="stHorizontalBlock"] > div { width: 100% !important; }
      .block-container { padding-left: 10px !important; padding-right: 10px !important; }
    }
    @media (max-width: 480px) {
      [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
      [data-testid="stHorizontalBlock"] > div { width: 100% !important; }
      .block-container { padding-left: 8px !important; padding-right: 8px !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("💡 Electric Bill Calculator", use_container_width=True):
        st.session_state["page"] = "bill_app"
with col2:
    if st.button("👗 Cloth Renamer & Organizer", use_container_width=True):
        st.session_state["page"] = "renamer_app"
with col3:
    st.button("📊 Coming Soon", use_container_width=True)

col4, col5, col6 = st.columns(3)
with col4:
    st.button("🔋 Coming Soon", use_container_width=True)
with col5:
    st.button("🌤️ Coming Soon", use_container_width=True)
with col6:
    st.button("⚙️ Coming Soon", use_container_width=True)

# ---------------- Bill App Page ---------------- #
if st.session_state["page"] == "bill_app":
    st.markdown("---")
    st.header("💡 Electric Bill Calculator")

    tab1, tab2 = st.tabs(["📄 Upload Bill", "✍️ Manual Input"]) 

    # TAB 1: UPLOAD BILL
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

                # Extract TOTAL AMOUNT DUE
                match_due = re.search(r"TOTAL AMOUNT DUE\s+([\d,]+\.\d{2})", text, re.IGNORECASE)
                total_due = float(match_due.group(1).replace(",", "")) if match_due else None

                # Extract Total kWh
                match_kwh = re.search(r"Billed\s*:\s*(\d+)", text, re.IGNORECASE)
                total_kwh = float(match_kwh.group(1)) if match_kwh else None

                # Compute Rate per kWh
                if total_due and total_kwh and total_kwh > 0:
                    rate_per_kwh = total_due / total_kwh

                # Extract CURRENT CHARGES + Subtotals
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

        # Display Results
        st.subheader("📊 Bill Summary")
        if total_kwh:
            st.write(f"📏 **Total kWh (from bill):** {total_kwh:,.0f}")
        if rate_per_kwh:
            st.write(f"⚡ **Rate per kWh:** ₱{rate_per_kwh:,.2f}")

        if df_sections is not None:
            with st.expander("📑 Current Charges Breakdown"):
                df_display = df_sections.copy()
                df_display["Sub-Total (₱)"] = df_display["Sub-Total (₱)"] .map(lambda x: f"₱{x:,.2f}")
                df_display["Est. kWh"] = df_display["Est. kWh"].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
                st.dataframe(df_display, use_container_width=True)

        # Calculator (based on uploaded bill)
        st.subheader("🔢 Enter Your Own kWh Usage")
        manual_kwh = st.number_input("Enter your kWh usage", value=0.0, step=1.0)
        if st.button("💡 Compute My Bill"):
            if rate_per_kwh:
                computed = manual_kwh * rate_per_kwh
                st.success(f"💰 Your Computed Bill: ₱{computed:,.2f}")
            else:
                st.error("⚠️ Please upload a valid bill first.")

    # TAB 2: MANUAL INPUT
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

# If not on the renamer app, stop before running its UI
if st.session_state.get("page") != "renamer_app":
    st.stop()

# ---------------- UI ---------------- #
# Renamer app title
st.title("Cloth Renamer & Organizer")

# Hide uploaded file list and pagination near the drag-and-drop uploader
st.markdown(
    """
    <style>
    /* Hide uploaded file list items and paginator under the FileUploader */
    [data-testid="stFileUploader"] [data-testid*="uploaded"] { display: none !important; }
    [data-testid="stFileUploader"] [data-testid*="Uploaded"] { display: none !important; }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderUploadedFiles"] { display: none !important; }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] { display: none !important; }
    [data-testid="stFileUploader"] [data-testid*="Paginator"] { display: none !important; }
    [data-testid="stFileUploader"] [data-baseweb="pagination"] { display: none !important; }
    [data-testid="stFileUploader"] div[aria-live="polite"] { display: none !important; }
    /* Hide any extra children beyond the first dropzone container */
    [data-testid="stFileUploader"] > div > div:not(:first-child) { display: none !important; }
    [data-testid="stFileUploader"] > div > :not(:first-child) { display: none !important; }
    [data-testid="stFileUploader"] > div > div > :not(:first-child) { display: none !important; }
    /* Hide lists and status regions */
    [data-testid="stFileUploader"] [role="list"],
    [data-testid="stFileUploader"] [role="status"],
    [data-testid="stFileUploader"] [aria-live] { display: none !important; }
    /* Hide any list container adjacent to dropzone instructions */
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] + div { display: none !important; }
    /* Keep only the dropzone visible; hide everything else inside the uploader */
    [data-testid="stFileUploader"] * { display: none !important; }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] { display: block !important; }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] * { display: initial !important; }
    /* Also hide any global BaseWeb pagination components that might appear */
    [data-baseweb="pagination"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

uploaded_files = st.file_uploader(
    "Upload images or a ZIP",
    type=["jpg", "jpeg", "png", "zip"],
    accept_multiple_files=True
)



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
                                ext = os.path.splitext(fname)[1].lower() or ".jpg"
                                st.session_state["items"].append({
                                    "id": idx_counter,
                                    "color": pname,
                                    "rgb": mean_rgb,
                                    "image_b64": small_b64,
                                    "pil": pil.copy(),
                                    "cloth_style": cloth_style,
                                    "ext": ext,
                                })
                                idx_counter += 1
        else:
            mean_rgb, pname, pil, cloth_style = detect_and_get_color(f)
            if mean_rgb:
                small_b64 = image_to_base64(pil.resize((80, 80)))
                ext = os.path.splitext(f.name)[1].lower() or ".jpg"
                st.session_state["items"].append({
                    "id": idx_counter,
                    "color": pname,
                    "rgb": mean_rgb,
                    "image_b64": small_b64,
                    "pil": pil.copy(),
                    "cloth_style": cloth_style,
                    "ext": ext,
                })
                idx_counter += 1

items = assign_filenames(st.session_state["items"])
if not items:
    st.stop()

# ---------------- HTML + JS ---------------- #
blocks_data = [{
    "id": it["id"],
    "unique_id": f"C{1000 + it['id']}",
    "cloth_type": (
        STYLE_MAP.get(
            (it.get("cloth_style") or "").lower(),
            (it.get("cloth_style") or "").replace(" ", "_").capitalize()
        ) or "New Folder"
    ),
    "color_raw": it["color"],
    "color_norm": it["color"].replace(" ", "_").lower(),
    "rgb": it["rgb"],
    "img": it["image_b64"],
    "filename": it["filename"],
    "ext": it.get("ext", ".jpg"),
} for it in items]

html_blocks = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
  /* Hide Streamlit emotion cache classes and file uploader preview */
  .st-emotion-cache-fis6aj,
  .e16n7gab7,
  .stFileUploader > div > div > div,
  .stFileUploader [data-testid="stFileUploaderDropzoneInstructions"],
  .stFileUploader [data-testid="stFileUploaderDropzoneInstructions"] + div {{
    display: none !important;
  }}
  /* Hide uploaded file list items near the dropzone */
  [data-testid="stFileUploader"] [data-testid="uploadedFile"] {{ display: none !important; }}
  [data-testid="stFileUploader"] [data-testid="uploadedFiles"] {{ display: none !important; }}
  .stFileUploader .uploadedFile {{ display: none !important; }}
  .stFileUploader .uploadedFiles {{ display: none !important; }}
  /* Responsive overrides */
  @media (max-width: 768px) {{
    :root {{ --zoom: 1; --card-w: 180px; }}
    .toolbar {{ flex-wrap: wrap; gap: 10px; padding: 8px 12px; }}
    .folder-grid {{ grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }}
    .folder-grid.grid-fixed {{ grid-template-columns: repeat(2, 1fr); }}
    .btn, .add-folder-btn, .edit-btn {{ font-size: 13px; padding: 8px 12px; }}
  }}
  @media (max-width: 480px) {{
    :root {{ --zoom: 0.9; --card-w: 150px; }}
    .toolbar {{ flex-wrap: wrap; gap: 8px; padding: 8px 10px; }}
    .folder-grid {{ grid-template-columns: 1fr; }}
    .folder-grid.grid-fixed {{ grid-template-columns: 1fr; }}
    .btn, .add-folder-btn, .edit-btn {{ font-size: 12px; padding: 8px 10px; }}
  }}
  
  :root {{ --zoom: 1; --card-w: 240px; }}
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
  .toolbar select.toolbar-select {{
    background:#ffffff; color:#333; border:1px solid #ced4da; border-radius:6px;
    padding:8px 10px; font-size:14px;
  }}
  .toolbar input.toolbar-input {{
    background:#ffffff; color:#333; border:1px solid #ced4da; border-radius:6px;
    padding:8px 10px; font-size:14px;
  }}
  .folder-grid {{
    display:grid;
    gap:16px;
    margin-top:70px;
    padding:10px 20px;
    align-items:start;
  }}
  .folder-grid.grid-auto {{
    grid-template-columns:repeat(auto-fit, minmax(calc(var(--card-w) + 80px), 1fr));
  }}
  .folder-grid.grid-fixed {{
    grid-template-columns:repeat(5, minmax(calc(var(--card-w) + 80px), 1fr));
  }}
  .folder-grid.grid-2 {{
    grid-template-columns:repeat(2, minmax(calc(var(--card-w) + 80px), 1fr));
  }}
  .folder-grid.grid-3 {{
    grid-template-columns:repeat(3, minmax(calc(var(--card-w) + 80px), 1fr));
  }}
  /* Row-first layout using explicit column wrappers */
  .folder-grid.rowfirst {{
    display: flex;
    flex-direction: column;
    gap: 16px;
    margin-top: 70px;
    padding: 10px 20px;
    align-items: stretch;
  }}
  .colgroup {{
    display: grid;
    grid-template-columns: repeat(var(--cols, 3), minmax(calc(var(--card-w) + 80px), 1fr));
    gap: 16px;
    align-items: start;
  }}
  .col {{
    display: flex;
    flex-direction: column;
    gap: 16px;
    align-items: flex-start;
  }}
  /* Independent column layout using CSS multi-columns */
  .folder-grid.columns-2, .folder-grid.columns-3 {{
    display:block; column-gap:16px; column-fill:auto; margin-top:70px; padding:10px 20px;
  }}
  .folder-grid.columns-2 {{ column-count:2; }}
  .folder-grid.columns-3 {{ column-count:3; }}
  /* Desktop: cap 5 rows per column */
  @media (min-width: 769px) {{
    .folder-grid.columns-2 .folder:nth-child(5n),
    .folder-grid.columns-3 .folder:nth-child(5n) {{
      break-after: column;
    }}
  }}
  .folder-grid.columns-2 .folder, .folder-grid.columns-3 .folder {{
    break-inside: avoid; width: auto;
  }}
  .folder {{
    background:white; border-radius:10px; padding:0;
    box-shadow:0 2px 10px rgba(0,0,0,0.06);
    display:flex; flex-direction:column; min-height:220px; border:1px solid #e9ecef;
    width: calc(var(--card-w) + 80px);
  }}
  .folder.collapsed {{ min-height:80px; }}
  .folder-header {{
    display:flex; align-items:center; gap:8px; padding:8px 10px; border-bottom:1px solid #f1f3f5; cursor:grab;
  }}
  .caret {{ cursor:pointer; user-select:none; font-size:16px; padding:0 6px; }}
  /* Smooth expand/collapse using max-height and opacity */
  .list {{
    overflow-y:auto; max-height:500px; padding-right:6px;
    -webkit-overflow-scrolling: touch;
    overscroll-behavior: contain;
    touch-action: pan-y;
    --zoom: 1;
    transition: max-height 0.25s ease, opacity 0.25s ease;
  }}
  .folder.collapsed .list {{
    max-height: 0; opacity: 0;
    /* Allow dropping onto collapsed folders */
    pointer-events: auto;
    min-height: 40px;
  }}
  .folder-select {{ width:18px; height:18px; display:none; }}
  .folder-title {{
    flex:1; font-weight:bold; font-size:15px; border:none; outline:none;
  }}
  .badge {{ background:#eef2ff; color:#334; font-size:11px; padding:2px 6px; border-radius:10px; }}
  .content {{ margin-top:20px; padding:10px 20px; }}
  /* (moved above with animation) */
  .block {{
    width:100%; display:flex; flex-direction:column; align-items:center; gap:8px;
    border-radius:8px; padding:10px; margin:8px 0; border:1px solid #ddd;
    box-shadow:0 1px 2px rgba(0,0,0,0.05); background:white; cursor:pointer;
  }}
  .block.selected {{ border:2px solid #007bff; background:#eef5ff; }}
  .thumb {{ width:var(--card-w); max-width:100%; height:auto; border-radius:6px; object-fit:contain; }}
  .handle {{ display:none; }}
  input[type=\"color\"].color-picker {{ width:32px; height:32px; padding:0; border:none; background:transparent; }}
  input[type=\"text\"].filename {{ width:100%; font-size:14px; padding:6px 8px; text-align:center; white-space:nowrap; overflow:hidden; }}
  .name-row {{ display:flex; gap:8px; align-items:center; width:100%; max-width:var(--card-w); margin:0 auto; }}
  .name-row .filename {{ flex:1; }}
  .folder-header.selected {{ outline:2px solid #007bff; background:#eef5ff; }}

  /* Responsive adjustments for mobile/tablet */
  @media (max-width: 768px) {{
    :root {{ --card-w: 180px; }}
    .folder {{ width: calc(var(--card-w) + 60px); }}
    .folder-header {{ padding: 10px 12px; }}
    .toolbar {{ flex-wrap: wrap; gap: 8px; }}
    .list {{ max-height: calc(70vh); -webkit-overflow-scrolling: touch; overscroll-behavior: contain; }}
    .handle {{ display: inline-flex; align-items: center; justify-content: center; width: 28px; height: 28px; background: #f1f3f5; color: #555; border-radius: 6px; margin-top: 6px; cursor: grab; }}
    .block {{ cursor: default; }}
    /* Reduce columns for responsive layout */
    .folder-grid.columns-3 {{ column-count: 2; }}
    /* Tablet: cap 3 rows per column */
    .folder-grid.columns-2 .folder:nth-child(3n),
    .folder-grid.columns-3 .folder:nth-child(3n) {{ break-after: column; }}
  }}
  @media (max-width: 480px) {{
    :root {{ --card-w: 160px; }}
    .folder {{ width: calc(var(--card-w) + 40px); }}
    .thumb {{ border-radius: 4px; }}
    .toolbar select.toolbar-select {{ padding: 6px 8px; font-size: 13px; }}
    .btn, .add-folder-btn, .edit-btn {{ padding: 6px 10px; font-size: 13px; }}
    /* Mobile: use 2 columns with 2 rows per column */
    .folder-grid.columns-2, .folder-grid.columns-3 {{ column-count: 2; }}
    .folder-grid.columns-2 .folder:nth-child(2n),
    .folder-grid.columns-3 .folder:nth-child(2n) {{ break-after: column; }}
    /* Slightly smaller content cap for mobile */
    .list {{ max-height: calc(65vh); -webkit-overflow-scrolling: touch; overscroll-behavior: contain; }}
  }}
  /* Floating Action Button (bottom-right) */
  .fab-container {{ position: fixed; bottom: 24px; right: 24px; z-index: 1000; pointer-events: none; }}
  .fab-main {{ pointer-events: auto; width: 64px; height: 64px; border-radius: 50%; border: none; cursor: pointer; display: grid; place-items: center; 
    background: #14b8a6; /* clean flat color */ box-shadow: 0 14px 28px rgba(0,0,0,0.22), 0 6px 12px rgba(0,0,0,0.12), 0 0 0 3px rgba(20, 184, 166, 0.25);
    position: relative; overflow: hidden; transition: transform 220ms cubic-bezier(.22,1,.36,1), box-shadow 220ms cubic-bezier(.22,1,.36,1); }}
  .fab-main::before {{ content: ""; position: absolute; inset: 0; border-radius: 50%; background: radial-gradient(120% 120% at 30% 15%, rgba(255,255,255,0.35) 0%, rgba(255,255,255,0.0) 50%); pointer-events: none; }}
  .fab-main:hover {{ transform: translateY(-2px); box-shadow: 0 18px 32px rgba(0,0,0,0.25), 0 8px 16px rgba(0,0,0,0.14), 0 0 0 4px rgba(20, 184, 166, 0.35); }}
  .fab-main:focus {{ outline: 2px solid rgba(20, 184, 166, 0.45); }}
  .fab-icon {{ width: 28px; height: 28px; }}
  .fab-menu {{ position: absolute; bottom: 72px; right: 0; display: grid; gap: 12px; opacity: 0; transform: translateY(10px); pointer-events: none; transition: all 280ms cubic-bezier(.22,1,.36,1); }}
  .fab-container.open .fab-menu {{ opacity: 1; transform: translateY(0); pointer-events: auto; }}
  .fab-container.open .fab-mini {{ animation: bounceIn 300ms cubic-bezier(.28,.84,.42,1); }}
  @keyframes bounceIn {{ 0% {{ transform: translateY(8px) scale(0.96); opacity: 0.6; }} 60% {{ transform: translateY(-2px) scale(1.03); opacity: 1; }} 100% {{ transform: translateY(0) scale(1); }} }}
  .fab-mini {{ width: 48px; height: 48px; border-radius: 12px; backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border: 2px solid rgba(255,255,255,0.10); display: grid; place-items: center; cursor: pointer; position: relative; box-shadow: 0 6px 12px rgba(0,0,0,0.14); }}
  .fab-mini:hover {{ box-shadow: 0 8px 16px rgba(0,0,0,0.18); }}
  .fab-mini:active {{ transform: translateY(-1px) scale(0.99); }}
  .fab-label {{ position: absolute; bottom: -22px; left: 50%; transform: translateX(-50%); font-size: 14px; font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; color: #222; text-shadow: 0 1px 1px rgba(255,255,255,0.25); }}
  /* Mini button theme colors */
  .fab-add {{ background: rgba(255, 241, 150, 0.65); }}
  .fab-edit {{ background: rgba(173, 216, 255, 0.65); }}
  .fab-del {{ background: rgba(255, 190, 190, 0.65); }}
  .fab-rand {{ background: rgba(200, 180, 255, 0.65); }}

  /* Tooltips */
  [data-tooltip] {{ position: relative; }}
  [data-tooltip]::after {{ content: attr(data-tooltip); position: absolute; bottom: 105%; left: 50%; transform: translateX(-50%); background: rgba(30,30,30,0.9); color: #fff; padding: 6px 8px; border-radius: 6px; font-size: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.30); white-space: nowrap; opacity: 0; pointer-events: none; transition: opacity 160ms ease; }}
  [data-tooltip]:hover::after {{ opacity: 1; }}

  /* Ripple effect */
  .ripple {{ position: absolute; border-radius: 50%; transform: scale(0); animation: ripple 600ms ease-out; background: rgba(255,255,255,0.4); }}
  @keyframes ripple {{ to {{ transform: scale(4); opacity: 0; }} }}

  /* Bottom export bar */
  .bottom-export {{ position: fixed; bottom: 16px; left: 50%; transform: translateX(-50%); display: flex; gap: 12px; padding: 10px 14px; border-radius: 12px; 
    background: rgba(255,255,255,0.6); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border: 2px solid rgba(0,0,0,0.10); box-shadow: 0 8px 16px rgba(0,0,0,0.12); z-index: 900; }}
  .bottom-export .export-btn {{ padding: 8px 12px; font-size: 14px; border-radius: 10px; border: none; cursor: pointer; background: #0f766e; color: #fff; box-shadow: 0 4px 8px rgba(0,0,0,0.18); }}
  .bottom-export .export-btn.secondary {{ background: #475569; }}

  /* Dark mode */
  @media (prefers-color-scheme: dark) {{
    body {{ background: linear-gradient(180deg, #0b1020, #121a2a); color: #e7e9ee; }}
    .fab-label {{ color: #e7e9ee; text-shadow: none; }}
    .bottom-export {{ background: rgba(18,24,38,0.55); border-color: rgba(255,255,255,0.08); }}
    .bottom-export .export-btn {{ background: #14b8a6; color: #071318; }}
    .bottom-export .export-btn.secondary {{ background: #64748b; color: #0b1220; }}
  }}
  </style>

<script src=\"https://cdnjs.cloudflare.com/ajax/libs/Sortable/1.15.0/Sortable.min.js\"></script>
</head>
<body>
  <div class=\"toolbar\">
    <input type=\"text\" id=\"newFolderName\" class=\"toolbar-input\" placeholder=\"New folder name\" style=\"width:180px;\" />
    <button class=\"add-folder-btn\" onclick=\"addFolder()\">+ Add Folder</button>
    <button class=\"edit-btn\" id=\"editBtn\" onclick=\"toggleEdit()\">Edit</button>
    <button class=\"btn\" id=\"deleteBtn\" onclick=\"onDelete()\">Delete</button>
    <button class=\"btn\" id=\"randomBtn\" onclick=\"randomizeIds()\">Random_ID</button>
    <span style=\"margin-left:auto; display:flex; align-items:center; gap:8px;\">
      <label style=\"font-size:13px;\" for=\"sortSelect\">Sort</label>
      <select id=\"sortSelect\" class=\"toolbar-select\">
        <option value=\"none\">None</option>
        <option value=\"name\">Name (A→Z)</option>
      </select>
      <label style=\"font-size:13px;\" for=\"layoutSelect\">Layout</label>
      <select id=\"layoutSelect\" class=\"toolbar-select\">
        <option value=\"auto\">Auto</option>
        <option value=\"cols2\">2 columns</option>
        <option value=\"cols3\">3 columns</option>
      </select>
    </span>
  </div>

  <div class=\"folder-grid\" id=\"folderGrid\"></div>

  <div class=\"content\" style=\"display:none\">
    <div id=\"mainList\"></div>
  </div>

  <!-- Floating Action Button -->
  <div class=\"fab-container\" aria-label=\"Actions\">
    <button class=\"fab-main\" id=\"fabMain\" data-tooltip=\"Actions\">
      <svg class=\"fab-icon\" viewBox=\"0 0 24 24\" fill=\"none\" xmlns=\"http://www.w3.org/2000/svg\"> 
        <path d=\"M12 5v14M5 12h14\" stroke=\"#ffffff\" stroke-width=\"2\" stroke-linecap=\"round\"/> 
      </svg>
    </button>
    <div class=\"fab-menu\" id=\"fabMenu\" aria-hidden=\"true\">
      <div class=\"fab-mini fab-add\" id=\"fabAdd\" data-tooltip=\"Add Folder\" onclick=\"addFolder()\">
        <svg width=\"22\" height=\"22\" viewBox=\"0 0 24 24\" fill=\"none\" xmlns=\"http://www.w3.org/2000/svg\"><path d=\"M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v7a2 2 0 01-2 2H5a2 2 0 01-2-2V7z\" stroke=\"#664c00\" stroke-width=\"1.6\"/><path d=\"M12 11v6M9 14h6\" stroke=\"#664c00\" stroke-width=\"1.6\" stroke-linecap=\"round\"/></svg>
        <span class=\"fab-label\">Add Folder</span>
      </div>
      <div class=\"fab-mini fab-edit\" id=\"fabEdit\" data-tooltip=\"Edit\" onclick=\"toggleEdit()\">
        <svg width=\"22\" height=\"22\" viewBox=\"0 0 24 24\" fill=\"none\" xmlns=\"http://www.w3.org/2000/svg\"><path d=\"M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25z\" stroke=\"#1e3a8a\" stroke-width=\"1.6\"/><path d=\"M14.06 6.19l3.75 3.75\" stroke=\"#1e3a8a\" stroke-width=\"1.6\"/></svg>
        <span class=\"fab-label\">Edit</span>
      </div>
      <div class=\"fab-mini fab-del\" id=\"fabDel\" data-tooltip=\"Delete\" onclick=\"onDelete()\">
        <svg width=\"22\" height=\"22\" viewBox=\"0 0 24 24\" fill=\"none\" xmlns=\"http://www.w3.org/2000/svg\"><path d=\"M6 7h12M9 7V5h6v2M10 10v8M14 10v8\" stroke=\"#7f1d1d\" stroke-width=\"1.6\" stroke-linecap=\"round\"/></svg>
        <span class=\"fab-label\">Delete</span>
      </div>
      <div class=\"fab-mini fab-rand\" id=\"fabRand\" data-tooltip=\"Random_ID\" onclick=\"randomizeIds()\">
        <svg width=\"22\" height=\"22\" viewBox=\"0 0 24 24\" fill=\"none\" xmlns=\"http://www.w3.org/2000/svg\"><rect x=\"3\" y=\"3\" width=\"7\" height=\"7\" rx=\"1.5\" stroke=\"#4c1d95\" stroke-width=\"1.6\"/><rect x=\"14\" y=\"14\" width=\"7\" height=\"7\" rx=\"1.5\" stroke=\"#4c1d95\" stroke-width=\"1.6\"/><circle cx=\"7\" cy=\"7\" r=\"1.2\" fill=\"#4c1d95\"/><circle cx=\"17.5\" cy=\"17.5\" r=\"1.2\" fill=\"#4c1d95\"/></svg>
        <span class=\"fab-label\">Random_ID</span>
      </div>
    </div>
  </div>

  <!-- Bottom Export Bar -->
  <div class=\"bottom-export\">
    <button class=\"export-btn\" onclick=\"exportZip(false)\">Export (Folders)</button>
    <button class=\"export-btn secondary\" onclick=\"exportZip(true)\">Export (Flatten)</button>
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
// FAB init and ripple
document.addEventListener('DOMContentLoaded', () => {{
  const fabContainer = document.querySelector('.fab-container');
  const fabMain = document.getElementById('fabMain');
  if (fabMain && fabContainer) {{
    fabMain.addEventListener('click', (e) => {{ addRipple(e.currentTarget, e); fabContainer.classList.toggle('open'); }});
  }}
  document.querySelectorAll('.fab-mini').forEach(btn => {{
    btn.addEventListener('click', (e) => addRipple(e.currentTarget, e));
  }});
  function addRipple(target, evt) {{
    const rect = target.getBoundingClientRect();
    const ripple = document.createElement('span');
    const size = Math.max(rect.width, rect.height);
    ripple.className = 'ripple';
    ripple.style.width = ripple.style.height = size + 'px';
    const x = evt.clientX - rect.left - size/2;
    const y = evt.clientY - rect.top - size/2;
    ripple.style.left = x + 'px';
    ripple.style.top = y + 'px';
    target.appendChild(ripple);
    setTimeout(() => ripple.remove(), 600);
  }}
}});
const data = {json.dumps(blocks_data)};
const PALETTE = {json.dumps({k:v for k,v in FASHION_COLORS.items()})};
const mainList = document.getElementById('mainList');
const folderGrid = document.getElementById('folderGrid');
const modalBackdrop = document.getElementById('modalBackdrop');
const editBtn = document.getElementById('editBtn');
const sortSelect = document.getElementById('sortSelect');
const layoutSelect = document.getElementById('layoutSelect');
let folderCount = 0;
const folderKeyToInfo = new Map();
const selectedIds = new Set();
const selectedFolders = new Set();
let lastSelectedId = null;

// Initialize layout mode
if (layoutSelect) {{
  layoutSelect.addEventListener('change', () => applyLayout(layoutSelect.value));
  // Default to 3-column responsive layout
  layoutSelect.value = 'cols3';
  applyLayout('cols3');
}} else {{
  // Fallback to 3 columns when selector is missing
  applyLayout('cols3');
}}
if (sortSelect) {{
  sortSelect.addEventListener('change', () => applySort(sortSelect.value));
}}

function applyLayout(mode) {{
  if (!folderGrid) return;
  const isAuto = mode === 'auto' || !mode;
  // Reset classes
  folderGrid.classList.remove('columns-2', 'columns-3', 'grid-fixed', 'grid-2', 'grid-3', 'rowfirst');
  folderGrid.classList.toggle('grid-auto', isAuto);
  if (isAuto) {{
    // Flatten back to a simple auto grid
    flattenToGrid();
    setupFolderSortables();
    return;
  }}
  const cols = getColsFromMode(mode);
  folderGrid.classList.add('rowfirst');
  const rowsPerCol = getRowsPerCol();
  redistributeRowFirst(cols, rowsPerCol);
}}

function getColsFromMode(mode) {{
  if (mode === 'cols2') return 2;
  if (mode === 'cols3') return 3;
  return 3; // default
}}

function getRowsPerCol() {{
  // Desktop: 5, Tablet (<=768px): 3, Mobile (<=480px): 2
  if (window.matchMedia('(max-width: 480px)').matches) return 2;
  if (window.matchMedia('(max-width: 768px)').matches) return 3;
  return 5;
}}

function flattenToGrid() {{
  const folders = Array.from(document.querySelectorAll('#folderGrid .folder'));
  folderGrid.innerHTML = '';
  folders.forEach(f => folderGrid.appendChild(f));
}}

function ensureColgroup(cols) {{
  const group = document.createElement('div');
  group.className = 'colgroup';
  group.style.setProperty('--cols', cols);
  for (let c = 0; c < cols; c++) {{
    const col = document.createElement('div');
    col.className = 'col';
    group.appendChild(col);
  }}
  return group;
}}

function redistributeRowFirst(cols, rowsPerCol) {{
  const folders = Array.from(document.querySelectorAll('#folderGrid .folder'));
  folderGrid.innerHTML = '';
  let currentGroupIndex = -1;
  let group = null;
  let columns = [];
  for (let i = 0; i < folders.length; i++) {{
    const gi = Math.floor(i / (cols * rowsPerCol));
    const ci = i % cols;
    if (gi !== currentGroupIndex) {{
      group = ensureColgroup(cols);
      columns = Array.from(group.querySelectorAll('.col'));
      folderGrid.appendChild(group);
      currentGroupIndex = gi;
    }}
    columns[ci].appendChild(folders[i]);
  }}
  setupFolderSortables();
}}

function appendFolderRowFirst(folder) {{
  const mode = layoutSelect ? layoutSelect.value : 'cols3';
  const cols = getColsFromMode(mode);
  const rowsPerCol = getRowsPerCol();
  const count = document.querySelectorAll('#folderGrid .folder').length;
  const gi = Math.floor(count / (cols * rowsPerCol));
  const ci = count % cols;
  const groups = Array.from(folderGrid.querySelectorAll('.colgroup'));
  let group = groups[gi];
  if (!group) {{
    group = ensureColgroup(cols);
    folderGrid.appendChild(group);
  }}
  const columns = Array.from(group.querySelectorAll('.col'));
  columns[ci].appendChild(folder);
  setupFolderSortables();
}}

function setupFolderSortables() {{
  // Enable sortable folders within and across columns
  Array.from(folderGrid.querySelectorAll('.col')).forEach(col => {{
    if (col._sortable) return;
    col._sortable = new Sortable(col, {{
      draggable: '.folder',
      handle: '.folder',
      animation: 0,
      group: 'folders',
      swap: true,
      swapClass: 'swap-target',
      swapThreshold: 0.5,
      onMove: (evt) => {{
        const related = evt.related;
        const dragged = evt.dragged;
        // Only allow when hovering another folder element (strict swap-only)
        return !!(related && related !== dragged);
      }},
      onStart: (evt) => {{
        const item = evt.item;
        // Record original position for swap restoration
        item._origParent = item.parentElement;
        item._origNext = item.nextSibling;
      }},
      onEnd: (evt) => {{
        const dragged = evt.item;
        const origParent = dragged._origParent;
        const origNext = dragged._origNext;
        // Determine pointer position for accurate drop target detection
        let target = (evt.originalEvent && evt.originalEvent.target) ? evt.originalEvent.target : null;
        if (!target && evt.originalEvent) {{
          const oe = evt.originalEvent;
          let x = 0, y = 0;
          if (oe.touches && oe.touches.length) {{
            x = oe.touches[0].clientX; y = oe.touches[0].clientY;
          }} else if (oe.changedTouches && oe.changedTouches.length) {{
            x = oe.changedTouches[0].clientX; y = oe.changedTouches[0].clientY;
          }} else if (oe.clientX != null && oe.clientY != null) {{
            x = oe.clientX; y = oe.clientY;
          }}
          if (x || y) {{ target = document.elementFromPoint(x, y); }}
        }}
        let dropFolder = target ? target.closest('.folder') : null;
        // If pointer is over the dragged folder itself, treat as no target
        if (dropFolder === dragged) {{ dropFolder = null; }}
        // If not directly over a folder, try overlap-based detection
        if (!dropFolder && evt.to) {{
          const siblings = Array.from(evt.to.querySelectorAll('.folder')).filter(el => el !== dragged);
          const dRect = dragged.getBoundingClientRect();
          let best = null, bestArea = 0;
          siblings.forEach(el => {{
            const r = el.getBoundingClientRect();
            const ix = Math.max(0, Math.min(dRect.right, r.right) - Math.max(dRect.left, r.left));
            const iy = Math.max(0, Math.min(dRect.bottom, r.bottom) - Math.max(dRect.top, r.top));
            const area = ix * iy;
            if (area > bestArea) {{ bestArea = area; best = el; }}
          }});
          if (best) {{ dropFolder = best; }}
        }}

        // Strict swap-only: require drop directly on another folder

        // If still not on a folder, restore to original position (no reorder)
        if (!dropFolder && origParent) {{
          if (origNext) {{
            origParent.insertBefore(dragged, origNext);
          }} else {{
            origParent.appendChild(dragged);
          }}
          return;
        }}

        if (dropFolder && dropFolder !== dragged && origParent) {{
          // Capture dropFolder's current position
          const dropParent = dropFolder.parentElement;
          const dropNext = dropFolder.nextSibling;

          // Swap positions: place dropFolder where dragged was, and dragged where dropFolder was
          if (origNext) {{
            origParent.insertBefore(dropFolder, origNext);
          }} else {{
            origParent.appendChild(dropFolder);
          }}
          if (dropNext) {{
            dropParent.insertBefore(dragged, dropNext);
          }} else {{
            dropParent.appendChild(dragged);
          }}
        }}

        // Cleanup and sync order after any move/swap
        dragged._origParent = null;
        dragged._origNext = null;
        sendOrder();
      }}
    }});
  }});

  // Fallback for layouts where folders are direct children (grid/columns layout)
  const hasCols = folderGrid.querySelectorAll('.col').length > 0;
  if (!hasCols) {{
    if (!folderGrid._sortable) {{
      folderGrid._sortable = new Sortable(folderGrid, {{
        draggable: '.folder',
        handle: '.folder',
        animation: 0,
        group: 'folders',
        swap: true,
        swapClass: 'swap-target',
        swapThreshold: 0.5,
        onMove: (evt) => {{
          const related = evt.related;
          const dragged = evt.dragged;
          // Only allow when hovering another folder element (strict swap-only)
          return !!(related && related !== dragged);
        }},
        onStart: (evt) => {{
          const item = evt.item;
          item._origParent = item.parentElement;
          item._origNext = item.nextSibling;
        }},
        onEnd: (evt) => {{
          const dragged = evt.item;
          const origParent = dragged._origParent;
          const origNext = dragged._origNext;
          // Determine pointer position for accurate drop target detection
          let target = (evt.originalEvent && evt.originalEvent.target) ? evt.originalEvent.target : null;
          if (!target && evt.originalEvent) {{
            const oe = evt.originalEvent;
            let x = 0, y = 0;
            if (oe.touches && oe.touches.length) {{
              x = oe.touches[0].clientX; y = oe.touches[0].clientY;
            }} else if (oe.changedTouches && oe.changedTouches.length) {{
              x = oe.changedTouches[0].clientX; y = oe.changedTouches[0].clientY;
            }} else if (oe.clientX != null && oe.clientY != null) {{
              x = oe.clientX; y = oe.clientY;
            }}
            if (x || y) {{ target = document.elementFromPoint(x, y); }}
          }}
          let dropFolder = target ? target.closest('.folder') : null;
          // If pointer is over the dragged folder itself, treat as no target
          if (dropFolder === dragged) {{ dropFolder = null; }}
          // If not directly over a folder, try overlap-based detection
          if (!dropFolder && evt.to) {{
            const siblings = Array.from(evt.to.querySelectorAll('.folder')).filter(el => el !== dragged);
            const dRect = dragged.getBoundingClientRect();
            let best = null, bestArea = 0;
            siblings.forEach(el => {{
              const r = el.getBoundingClientRect();
              const ix = Math.max(0, Math.min(dRect.right, r.right) - Math.max(dRect.left, r.left));
              const iy = Math.max(0, Math.min(dRect.bottom, r.bottom) - Math.max(dRect.top, r.top));
              const area = ix * iy;
              if (area > bestArea) {{ bestArea = area; best = el; }}
            }});
            if (best) {{ dropFolder = best; }}
          }}
          
          // Strict swap-only: require drop directly on another folder
          
          // If still not on a folder, restore to original position (no reorder)
          if (!dropFolder && origParent) {{
            if (origNext) {{
              origParent.insertBefore(dragged, origNext);
            }} else {{
              origParent.appendChild(dragged);
            }}
            return;
          }}
          
          if (dropFolder && dropFolder !== dragged && origParent) {{
            const dropParent = dropFolder.parentElement;
            const dropNext = dropFolder.nextSibling;
            if (origNext) {{
              origParent.insertBefore(dropFolder, origNext);
            }} else {{
              origParent.appendChild(dropFolder);
            }}
            if (dropNext) {{
              dropParent.insertBefore(dragged, dropNext);
            }} else {{
              dropParent.appendChild(dragged);
            }}
          }}
          dragged._origParent = null;
          dragged._origNext = null;
          sendOrder();
        }}
      }});
    }}
  }}
}}

function sortBlocksByName(list, asc=true) {{
  const blocks = Array.from(list.querySelectorAll('.block'));
  blocks.sort((a, b) => {{
    const fa = (a.querySelector('.filename')?.value || '').toLowerCase();
    const fb = (b.querySelector('.filename')?.value || '').toLowerCase();
    return asc ? fa.localeCompare(fb) : fb.localeCompare(fa);
  }});
  blocks.forEach(b => list.appendChild(b));
  const folder = list.closest('.folder');
  const tName = folder ? (folder.querySelector('.folder-title')?.value || '').trim() : '';
  renumber(list, tName, false);
  if (folder) updateBadge(folder);
}}

function applySort(mode) {{
  if (mode !== 'name') return; // 'none' leaves current order as-is
  const lists = Array.from(document.querySelectorAll('.list'));
  if (mainList) lists.push(mainList);
  lists.forEach(list => sortBlocksByName(list, true));
  sendOrder();
}}

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

  function clearAllSelections() {{
    // Clear image selections
    clearSelection();
    // Clear folder selections
    document.querySelectorAll('.folder-header.selected').forEach(h => h.classList.remove('selected'));
    document.querySelectorAll('.folder-select').forEach(cb => {{ cb.checked = false; }});
    selectedFolders.clear();
  }}

  // Auto-fit long filename text to the available input width
  function fitFilename(input) {{
    if (!input) return;
    const rect = input.getBoundingClientRect();
    const available = Math.max(0, rect.width - 16);
    const style = getComputedStyle(input);
    const baseSize = parseFloat(style.fontSize) || 14;
    const family = style.fontFamily || 'sans-serif';
    const canvas = fitFilename._canvas || (fitFilename._canvas = document.createElement('canvas'));
    const ctx = canvas.getContext('2d');
    const text = input.value || '';
    // measure at base size then scale
    ctx.font = `${{baseSize}}px ${{family}}`;
    const measured = ctx.measureText(text).width;
    let size = baseSize;
    if (measured > 0 && available > 0) {{
      const scale = Math.min(1, available / measured);
      size = Math.max(9, Math.floor(baseSize * scale));
    }}
    input.style.fontSize = size + 'px';
  }}

  function fitAllFilenames() {{
    document.querySelectorAll('input.filename').forEach(inp => fitFilename(inp));
  }}
  window.addEventListener('resize', fitAllFilenames);

  // Clicking outside folders and toolbar deselects images and folders
  document.addEventListener('click', (e) => {{
    const insideFolder = e.target.closest('.folder');
    const insideModal = e.target.closest('#modalBackdrop');
    const insideToolbar = e.target.closest('.toolbar');
    if (!insideFolder && !insideModal && !insideToolbar) {{
      clearAllSelections();
    }}
  }});

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

// Helper: RGB array to hex string
function rgbToHex([r,g,b]) {{
  const toHex = (x) => Math.max(0, Math.min(255, Math.round(x))).toString(16).padStart(2, '0');
  return '#' + toHex(r) + toHex(g) + toHex(b);
}}

// Scan a central region of the image to estimate the dominant cloth color
function scanDominantColorFromImage(imgEl) {{
  try {{
    const w = imgEl.naturalWidth || imgEl.width;
    const h = imgEl.naturalHeight || imgEl.height;
    if (!w || !h) return null;
    const canvas = document.createElement('canvas');
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imgEl, 0, 0, w, h);
    // Sample a centered rectangle (50% of width/height)
    const rw = Math.max(1, Math.floor(w * 0.5));
    const rh = Math.max(1, Math.floor(h * 0.5));
    const sx = Math.max(0, Math.floor((w - rw) / 2));
    const sy = Math.max(0, Math.floor((h - rh) / 2));
    const data = ctx.getImageData(sx, sy, rw, rh).data;
    let r = 0, g = 0, b = 0, count = 0;
    for (let i = 0; i < data.length; i += 4) {{
      r += data[i]; g += data[i+1]; b += data[i+2]; count++;
    }}
    if (count === 0) return null;
    r = Math.round(r / count); g = Math.round(g / count); b = Math.round(b / count);
    const hex = rgbToHex([r,g,b]);
    const name = nearestPaletteNameFromHex(hex);
    return {{ rgb: [r,g,b], hex, name }};
  }} catch (e) {{
    return null;
  }}
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
  wrapper.dataset.ext = item.ext;
  wrapper.setAttribute('draggable','true');

  // Click: toggle select/deselect on single click (double-click still toggles).
  wrapper.addEventListener('click', (e) => {{
    const idStr = item.id.toString();
    // Delay to distinguish single vs double click
    if (wrapper._clickTimer) {{ clearTimeout(wrapper._clickTimer); }}
    wrapper._clickTimer = setTimeout(() => {{
      if (wrapper.classList.contains('selected')) {{
        wrapper.classList.remove('selected');
        selectedIds.delete(idStr);
        if (lastSelectedId === idStr) {{ lastSelectedId = null; }}
      }} else {{
        wrapper.classList.add('selected');
        selectedIds.add(idStr);
        lastSelectedId = idStr;
      }}
      sendOrder();
      wrapper._clickTimer = null;
    }}, 200);
  }});

  wrapper.addEventListener('dblclick', (e) => {{
    const idStr = item.id.toString();
    if (wrapper._clickTimer) {{ clearTimeout(wrapper._clickTimer); wrapper._clickTimer = null; }}
    if (wrapper.classList.contains('selected')) {{
      wrapper.classList.remove('selected');
      selectedIds.delete(idStr);
      if (lastSelectedId === idStr) {{ lastSelectedId = null; }}
    }} else {{
      wrapper.classList.add('selected');
      selectedIds.add(idStr);
      lastSelectedId = idStr;
    }}
    sendOrder();
  }});

  // Right-click selects only this image
  wrapper.addEventListener('contextmenu', (e) => {{
    e.preventDefault();
    const idStr = item.id.toString();
    clearSelection();
    wrapper.classList.add('selected');
    selectedIds.add(idStr);
    lastSelectedId = idStr;
  }});

  // Improved drag functionality
  wrapper.addEventListener('dragstart', (e) => {{
    const idStr = item.id.toString();
    if (!wrapper.classList.contains('selected')) {{
      clearSelection();
      wrapper.classList.add('selected');
      selectedIds.add(idStr);
      lastSelectedId = idStr;
    }}
    e.dataTransfer.effectAllowed = 'move';
    const payload = JSON.stringify(Array.from(selectedIds));
    // Set both a custom and plain type for cross-browser (Safari) compatibility
    e.dataTransfer.setData('text/selectedIds', payload);
    e.dataTransfer.setData('text/plain', payload);
  }});

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
  const ct = folder ? (folder.querySelector('.folder-title')?.value || '').trim() : wrapper.dataset.clothType;
  renumber(list, ct, false);
  sendOrder();
  }});
  // Place color picker and filename on one line

  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'filename';
  input.value = item.filename;
  fitFilename(input);
  input.addEventListener('input', () => {{ fitFilename(input); sendOrder(); }});
  const nameRow = document.createElement('div');
  nameRow.className = 'name-row';
  nameRow.appendChild(picker);
  nameRow.appendChild(input);
  wrapper.appendChild(nameRow);

  return wrapper;
}}

// Allow dropping onto folder headers only for folder swap (no image drop)
function setupHeaderDrop(header, folder, list) {{
  header.setAttribute('draggable','true');
  header.addEventListener('dragstart', (e) => {{
    e.dataTransfer.setData('text/folderId', folder.dataset.folderId);
  }});
  header.addEventListener('dragover', (e) => {{ e.preventDefault(); header.style.background = '#f1f5ff'; }});
  header.addEventListener('dragleave', () => {{ header.style.background = ''; }});
  header.addEventListener('drop', (e) => {{
    e.preventDefault(); header.style.background = '';
    // If images were dragged, move them into this folder
    let idsJson = e.dataTransfer.getData('text/selectedIds');
    if (!idsJson) idsJson = e.dataTransfer.getData('text/plain');
    if (idsJson) {{
      try {{
        const ids = JSON.parse(idsJson || '[]');
        const destList = folder.querySelector('.list');
        const tName = (folder.querySelector('.folder-title')?.value || '').trim();
        const affected = new Set([destList]);
        for (const id of ids) {{
          const block = document.querySelector(`.block[data-id="${{id}}"]`);
          if (!block) continue;
          if (block.parentElement !== destList) {{
            if (block.parentElement) affected.add(block.parentElement);
            block.parentElement?.removeChild(block);
            destList.appendChild(block);
          }}
          block.dataset.clothType = tName;
          ensureUidPrefixKeepDigits(block, tName, destList);
          updateFilenameKeepNumbering(block);
        }}
        affected.forEach(l => {{
          const f = l.closest('.folder');
          const name = f ? (f.querySelector('.folder-title')?.value || '').trim() : tName;
          renumber(l, name, false);
          if (f) updateBadge(f);
        }});
        updateBadge(folder);
        syncByPrefix();
        sendOrder();
        return;
      }} catch (err) {{ /* ignore parse errors and continue with folder swap */ }}
    }}
    // Folder swap (dragging folder headers)
    const srcId = e.dataTransfer.getData('text/folderId');
    if (srcId) {{
      const srcFolder = Array.from(document.querySelectorAll('#folderGrid .folder')).find(f => String(f.dataset.folderId) === String(srcId));
      if (srcFolder && srcFolder !== folder) {{
        swapNodes(srcFolder, folder);
        sendOrder();
      }}
    }}
  }});
}}

// Disable folder card drop area for images; swap is handled by Sortable onEnd
// (No-op function retained for safety if referenced elsewhere)
function setupFolderDropArea(folder, list) {{
  // Intentionally no behavior: users cannot drop images onto folder cards
}}

function createFolder(name = "New Folder") {{
  const folder = document.createElement('div');
  folder.className = 'folder';
  folder.dataset.folderId = String(folderCount++);
  // Store non-editable type info on the dataset
  folder.dataset.typeName = name;
  folder.dataset.typeKey = typeKey(name);

  const header = document.createElement('div');
  header.className = 'folder-header';
  // Double-click selects all images inside the folder
  header.addEventListener('dblclick', () => {{
    const fid = String(folder.dataset.folderId);
    const list = folder.querySelector('.list');
    if (header.classList.contains('selected')) {{
      // If folder is selected, double-click deselects the folder and clears highlights
      header.classList.remove('selected');
      selectedFolders.delete(fid);
      document.querySelectorAll('.block.selected').forEach(b => b.classList.remove('selected'));
      selectedIds.clear();
      lastSelectedId = null;
    }} else {{
      // If folder is not selected, double-click selects all images inside
      header.classList.add('selected');
      selectedFolders.add(fid);
      document.querySelectorAll('.block.selected').forEach(b => b.classList.remove('selected'));
      selectedIds.clear();
      Array.from(list.querySelectorAll('.block')).forEach(b => {{
        b.classList.add('selected');
        selectedIds.add(b.dataset.id);
      }});
      lastSelectedId = null;
    }}
  }});

  const caret = document.createElement('span');
  caret.className = 'caret';
  caret.textContent = '▾';
  // Toggle collapse via caret click without affecting other columns
  caret.addEventListener('click', (e) => {{
    e.stopPropagation();
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

  // Restore editable folder title input
  const titleInput = document.createElement('input');
  titleInput.type = 'text';
  titleInput.className = 'folder-title';
  titleInput.value = name;
  titleInput.addEventListener('input', () => {{
    const newName = (titleInput.value || '').trim() || 'New Folder';
    folder.dataset.typeName = newName;
    folder.dataset.typeKey = typeKey(newName);
    const list = folder.querySelector('.list');
    if (list) {{
      const blocks = Array.from(list.querySelectorAll('.block'));
      blocks.forEach(b => {{
        b.dataset.clothType = newName;
        ensureUidPrefixKeepDigits(b, newName, list);
        updateFilenameKeepNumbering(b);
      }});
      renumber(list, newName, false);
    }}
    // Align images to folders by Unique_ID prefix after title changes
    syncByPrefix();
    sendOrder();
  }});

  // Single-click on header toggles collapse (independent accordion per folder)
  header.addEventListener('click', (e) => {{
    // Ignore clicks on controls
    if (e.target === folderSel || e.target === caret || e.target === titleInput) return;
    folder.classList.toggle('collapsed');
    caret.textContent = folder.classList.contains('collapsed') ? '▸' : '▾';
  }});

  const badge = document.createElement('span');
  badge.className = 'badge';
  badge.textContent = '0 items';

  header.appendChild(caret);
  header.appendChild(folderSel);
  header.appendChild(titleInput);
  header.appendChild(badge);
  folder.appendChild(header);

  const list = document.createElement('div');
  list.className = 'list';
  list.style.setProperty('--zoom', 1);
  list.addEventListener('wheel', (e) => {{ if (!e.ctrlKey) return; e.preventDefault(); const current = parseFloat(getComputedStyle(list).getPropertyValue('--zoom')) || 1; const delta = e.deltaY < 0 ? 0.1 : -0.1; let next = current + delta; if (next < 1) next = 1; list.style.setProperty('--zoom', next.toFixed(2)); }}, {{ passive: false }});

  folder.appendChild(list);
  // Append with row-first distribution
  appendFolderRowFirst(folder);

  const isTouch = ('ontouchstart' in window) || window.matchMedia('(max-width: 768px)').matches;
  Sortable.create(list, {{
    group: {{ name: 'shared', pull: true, put: true }},
    animation: 150,
    draggable: '.block',
    handle: isTouch ? '.handle' : '.handle, .thumb',
    forceFallback: true,
    fallbackOnBody: true,
    fallbackTolerance: 3,
    touchStartThreshold: 4,
    dragoverBubble: true,
    onAdd: onItemAdded,
    onEnd: onDragEnd
  }});
  setupHeaderDrop(header, folder, list);
  // Swap-only: no image drop area on folder card

  updateBadge(folder);
  return {{ folder, list }};
}}

function updateBadge(folder) {{
  const count = folder.querySelectorAll('.list .block').length;
  const badge = folder.querySelector('.badge');
  if (badge) badge.textContent = `${{count}} item${{count===1?'':'s'}}`;
}}

function addFolder() {{
  const nameBox = document.getElementById('newFolderName');
  const desired = nameBox ? (nameBox.value || '').trim() : '';
  const {{ folder, list }} = createFolder(desired || 'New Folder');
  const key = folder.dataset.typeKey || typeKey(folder.dataset.typeName || '');
  folderKeyToInfo.set(key, {{ folder, list }});
  if (nameBox) nameBox.value = '';
  sendOrder();
}}

function renumber(list, clothType, forceNewUid=false) {{
  const blocks = Array.from(list.querySelectorAll('.block'));
  // Ensure each block's Unique_ID first letter matches the folder type's first letter
  const expectedPrefix = firstLetterFromType(clothType);
  if (expectedPrefix) {{
    for (const b of blocks) {{
      const uid = b.dataset.uniqueId || '';
      const curPrefix = uid.charAt(0).toUpperCase();
      if (curPrefix !== expectedPrefix) {{
        let digits = getUidDigits(uid);
        if (!digits || digits.length !== 6) {{
          digits = Array.from({{length:6}}, () => Math.floor(Math.random()*10)).join('');
        }}
        b.dataset.uniqueId = expectedPrefix + digits;
      }}
    }}
  }}
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
      const ext = b.dataset.ext || '.jpg';
      input.value = `${{b.dataset.uniqueId}}_${{ctUnderscore}}_${{colorUnderscore}} ${{numbering}}${{ext}}`;
      fitFilename(input);
    }});
  }}
}}

function isUniqueId(uid) {{
  return Array.from(document.querySelectorAll('.block')).every(b => b.dataset.uniqueId !== uid);
}}

function generateUniqueId(ct) {{
  const firstChar = (ct || '').replace(/[^A-Za-z]/g, '').charAt(0).toUpperCase() || 'X';
  let uid = '';
  do {{
    let digits = '';
    for (let i = 0; i < 6; i++) digits += Math.floor(Math.random() * 10).toString();
    uid = firstChar + digits;
  }} while (!isUniqueId(uid));
  return uid;
}}

// Helpers to update Unique_ID when moving items: change prefix only, keep digits unless collision within folder
function firstLetterFromType(ct) {{
  return (ct || '').replace(/[^A-Za-z]/g, '').charAt(0).toUpperCase() || 'X';
}}
function getUidDigits(uid) {{
  const s = String(uid || '');
  const m = s.match(/(\d{6})$/);
  if (m) return m[1];
  // Fallback: remove leading letter and return remainder
  return s.replace(/^[A-Za-z]/,'');
}}
function ensureUidPrefixKeepDigits(block, newType, list) {{
  const prefix = firstLetterFromType(newType);
  let digits = getUidDigits(block.dataset.uniqueId);
  if (!digits || digits.length !== 6) {{
    // Normalize to 6 digits if absent
    digits = Array.from({{length:6}}, () => Math.floor(Math.random()*10)).join('');
  }}
  block.dataset.uniqueId = prefix + digits;
}}

// Update filename to reflect new Unique_ID prefix while preserving existing numbering (e.g., "2 of 2")
function updateFilenameKeepNumbering(block) {{
  const input = block.querySelector('.filename');
  if (!input) return;
  const ext = block.dataset.ext || '.jpg';
  const ctUnderscore = (block.dataset.clothType || '').replace(/\s+/g, '_');
  const colorUnderscore = (block.dataset.colorName || '').replace(/\s+/g, '_');
  const m = String(input.value).match(/\s(\d+\s+of\s+\d+)\.[A-Za-z0-9]+$/);
  const numbering = m ? m[1] : '1 of 1';
  input.value = `${{block.dataset.uniqueId}}_${{ctUnderscore}}_${{colorUnderscore}} ${{numbering}}${{ext}}`;
  fitFilename(input);
}}

// Auto-sync images into folders by Unique_ID first letter matching folder title first letter
function syncByPrefix() {{
  const folders = Array.from(document.querySelectorAll('#folderGrid .folder'));
  const folderByPrefix = new Map();
  folders.forEach(f => {{
    const name = (f.querySelector('.folder-title')?.value || '').trim();
    const pfx = firstLetterFromType(name);
    if (pfx) folderByPrefix.set(pfx, f);
  }});

  const allBlocks = Array.from(document.querySelectorAll('.block'));
  const affectedLists = new Set();
  allBlocks.forEach(b => {{
    const uid = b.dataset.uniqueId || '';
    const pfx = uid.charAt(0).toUpperCase();
    const folder = folderByPrefix.get(pfx);
    if (!folder) return;
    const list = folder.querySelector('.list');
    if (b.parentElement !== list) {{
      if (b.parentElement) affectedLists.add(b.parentElement);
      affectedLists.add(list);
      b.parentElement?.removeChild(b);
      list.appendChild(b);
      const tName = (folder.querySelector('.folder-title')?.value || '').trim();
      b.dataset.clothType = tName;
      ensureUidPrefixKeepDigits(b, tName, list);
      updateFilenameKeepNumbering(b);
    }}
  }});

  affectedLists.forEach(list => {{
    const f = list.closest('.folder');
    const tName = f ? (f.querySelector('.folder-title')?.value || '').trim() : '';
    renumber(list, tName, false);
    if (f) updateBadge(f);
  }});
}}

// Build folders by cloth_type
const types = Array.from(new Set(data.map(d => d.cloth_type)));
for (const t of types) {{
  // ensureFolder will create folder with dataset.typeName/typeKey
  ensureFolder(t);
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
  const title = (folder.dataset.typeName || '').trim();
  const list = folder.querySelector('.list');
  renumber(list, title);
}}

const isTouchMain = ('ontouchstart' in window) || window.matchMedia('(max-width: 768px)').matches;
Sortable.create(mainList, {{
  group: {{ name: 'shared', pull: true, put: true }},
  animation: 200,
  draggable: '.block',
  handle: isTouchMain ? '.handle' : '.handle, .thumb',
  forceFallback: true,
  fallbackOnBody: true,
  fallbackTolerance: 3,
  touchStartThreshold: 4,
  dragoverBubble: true,
  onAdd: onItemAdded,
  onEnd: onDragEnd
}});

// Make folders sortable by header drag and support swap-on-drop
setupFolderSortables();

function swapNodes(a, b) {{
  const aParent = a.parentNode;
  const bParent = b.parentNode;
  const aNext = a.nextSibling;
  const bNext = b.nextSibling;
  if (aParent === bParent) {{
    if (b === aNext) {{ aParent.insertBefore(b, a); return; }}
    aParent.insertBefore(a, b);
    aParent.insertBefore(b, aNext);
  }} else {{
    // Move b into a's parent where a was
    aParent.insertBefore(b, aNext);
    // Move a into b's parent where b was
    bParent.insertBefore(a, bNext);
  }}
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
  // Align images to folders by Unique_ID prefix after batch apply
  syncByPrefix();
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
    // Scan dominant color from image and update color picker and dataset
    const imgEl = block.querySelector('img.thumb');
    const res = imgEl ? scanDominantColorFromImage(imgEl) : null;
    if (res && res.hex && res.name) {{
      const norm = res.name.replace(/\s+/g, '_').toLowerCase();
      block.dataset.colorName = norm;
      const picker = block.querySelector('.color-picker');
      if (picker) picker.value = res.hex;
    }}
    block.dataset.uniqueId = generateUniqueId(ct);
  }}
  const affected = new Set();
  for (const id of ids) {{ 
    const b = document.querySelector(`.block[data-id=\\"${{id}}\\"]`); 
    if (b) affected.add(b.parentElement); 
  }}
  affected.forEach(list => {{
    const f = list.closest('.folder');
    const t = f ? (f.querySelector('.folder-title')?.value || '').trim() : '';
    renumber(list, t, false); // Don't force new UID for unselected images
    if (f) updateBadge(f);
  }});
  // Align images to folders by Unique_ID prefix after randomizing IDs
  syncByPrefix();
  sendOrder();
}}

function groupSelected() {{
  const ids = Array.from(selectedIds);
  if (ids.length < 2) {{ alert('Select two or more images to group'); return; }}
  const sourceId = lastSelectedId || ids[ids.length-1];
  const src = document.querySelector(`.block[data-id=\"${{sourceId}}\"]`);
  if (!src) return;
  const srcType = src.dataset.clothType;
  const srcColor = src.dataset.colorName;
  const destList = src.parentElement;
  const colorHex = getColorHexFromName(srcColor);

  const affectedLists = new Set([destList]);
  for (const id of ids) {{
    const block = document.querySelector(`.block[data-id=\"${{id}}\"]`);
    if (!block || block === src) continue;
    if (block.parentElement !== destList) {{
      affectedLists.add(block.parentElement);
      block.parentElement.removeChild(block);
      destList.appendChild(block);
    }}
    block.dataset.clothType = srcType;
    block.dataset.colorName = srcColor;
    block.dataset.uniqueId = generateUniqueId(srcType);
    const picker = block.querySelector('.color-picker');
    if (picker && colorHex) picker.value = colorHex;
  }}
  affectedLists.forEach(list => {{
    const folder = list.closest('.folder');
    const tName = folder ? (folder.querySelector('.folder-title')?.value || '').trim() : srcType;
    renumber(list, tName, false);
    if (folder) updateBadge(folder);
  }});
  // Align images to folders by Unique_ID prefix
  syncByPrefix();
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
        // Preserve cloth type for selected images; only destination affects UID prefix later
      }}
    }}
  }}

  const targetFolder = toList.closest('.folder');
  if (targetFolder) {{
    const tName = (targetFolder.querySelector('.folder-title')?.value || '').trim();
    const moved = Array.from(toList.querySelectorAll('.block'));
    for (const b of moved) {{
      if (b === dragged || b.classList.contains('selected')) {{
        // Update cloth type and Unique ID prefix to reflect new folder type
        b.dataset.clothType = tName;
        ensureUidPrefixKeepDigits(b, tName, toList);
        updateFilenameKeepNumbering(b);
      }}
    }}
    // Renumber destination list to apply exact-match numbering
    renumber(toList, tName, false);
    updateBadge(targetFolder);
  }}
  const sourceFolder = fromList.closest('.folder');
  if (sourceFolder) {{
    // No renumbering when items leave; numbering stays as-is on remaining items
    updateBadge(sourceFolder);
  }}

  // Align images to folders by Unique_ID prefix
  syncByPrefix();
  sendOrder();
}}

// Fire on cross-list add to ensure destination properties and numbering update instantly
function onItemAdded(evt) {{
  const dragged = evt.item;
  const toList = evt.to;
  const fromList = evt.from;

  // If dragging one of multiple selected, bring the rest along
  if (dragged.classList.contains('selected') && selectedIds.size > 1) {{
    const ids = Array.from(selectedIds);
    for (const id of ids) {{
      if (id === dragged.dataset.id) continue;
      const other = document.querySelector(`.block[data-id="${{id}}"]`);
      if (other && other.parentElement !== toList) {{
        other.parentElement?.removeChild(other);
        toList.appendChild(other);
      }}
    }}
  }}

  const targetFolder = toList.closest('.folder');
  if (targetFolder) {{
    const tName = (targetFolder.querySelector('.folder-title')?.value || '').trim();
    const moved = Array.from(toList.querySelectorAll('.block'));
    for (const b of moved) {{
      if (b === dragged || b.classList.contains('selected')) {{
        b.dataset.clothType = tName;
        ensureUidPrefixKeepDigits(b, tName, toList);
        updateFilenameKeepNumbering(b);
      }}
    }}
    renumber(toList, tName, false);
    updateBadge(targetFolder);
  }}
  const sourceFolder = fromList ? fromList.closest('.folder') : null;
  if (sourceFolder) updateBadge(sourceFolder);
  syncByPrefix();
  sendOrder();
}}

function onDelete() {{
  // Folder delete flow with KEEP/DELETE/CANCEL
  if (selectedFolders.size > 0) {{
    for (const fid of Array.from(selectedFolders)) {{
      const folder = Array.from(document.querySelectorAll('#folderGrid .folder')).find(f => String(f.dataset.folderId) === String(fid));
      if (!folder) continue;
      const list = folder.querySelector('.list');
      const count = list ? list.querySelectorAll('.block').length : 0;
      if (count > 0) {{
        const choice = (prompt('This folder contains images. Type KEEP to delete folder and move images to Draft, DELETE to remove folder and images, or CANCEL to abort.') || '').toUpperCase();
        if (choice === 'CANCEL') {{ continue; }}
        if (choice === 'KEEP') {{
          const draftInfo = ensureFolder('Draft');
          const items = Array.from(list.querySelectorAll('.block'));
          for (const b of items) {{
            list.removeChild(b);
            draftInfo.list.appendChild(b);
            b.dataset.clothType = 'Draft';
            b.dataset.uniqueId = generateUniqueId('Draft');
          }}
          renumber(draftInfo.list, 'Draft', false);
          updateBadge(draftInfo.folder);
          folder.remove();
        }} else if (choice === 'DELETE') {{
          // Delete folder and images
          folder.remove();
        }} else {{
          // Unknown input -> cancel
          continue;
        }}
      }} else {{
        folder.remove();
      }}
    }}
    selectedFolders.clear();
    sendOrder();
    return;
  }}

  // Image delete
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
    const tName = tFolder ? (tFolder.querySelector('.folder-title')?.value || '').trim() : '';
    renumber(list, tName, false);
    if (tFolder) updateBadge(tFolder);
  }});
      sendOrder();
    }}
  }}
}}

function sendOrder() {{
  const folders = Array.from(document.querySelectorAll('#folderGrid .folder')).map(folder => {{
    const folderId = folder.dataset.folderId;
    const title = (folder.querySelector('.folder-title')?.value || '').trim();
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

function exportZip(flatten=false) {{
  const folders = Array.from(document.querySelectorAll('#folderGrid .folder')).map(folder => {{
    const folderId = folder.dataset.folderId;
    const title = (folder.querySelector('.folder-title')?.value || '').trim();
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
    value: {{ mainItems, folders, export: true, flatten }}
  }}, '*');
}}

sendOrder();
</script>
</body>
</html>
"""

order = components.html(html_blocks, height=1200, scrolling=True)

# Handle export request from HTML component
if isinstance(order, dict) and order.get("export"):
    flatten = bool(order.get("flatten", False))
    zip_data = create_export_zip(order, items, flatten=flatten)
    if zip_data:
        st.download_button(
            "Download Export" if not flatten else "Download Export (Flatten)", 
            data=zip_data, 
            file_name="cloth_export.zip" if not flatten else "cloth_export_flat.zip",
            mime="application/zip"
        )
