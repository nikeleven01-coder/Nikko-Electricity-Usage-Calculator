import streamlit as st
import pdfplumber
import re

st.set_page_config(page_title="Electric Bill Calculator", page_icon="⚡", layout="centered")
st.title("⚡ Electric Bill Calculator")

uploaded_file = st.file_uploader("📂 Upload your electric bill (PDF)", type="pdf")

diff_rdg = None
billed = None
rate_per_kwh = None
charges = {}

def to_float(val):
    try:
        return float(val.replace(",", ""))
    except:
        return None

if uploaded_file:
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"

        # --- Extract Total kWh (Diff Rdg) ---
        match_diff = re.search(r"Diff\s*Rdg\s*:\s*(\d+)", text, re.IGNORECASE)
        diff_rdg = int(match_diff.group(1)) if match_diff else None

        match_billed = re.search(r"Billed\s*:\s*(\d+)", text, re.IGNORECASE)
        billed = int(match_billed.group(1)) if match_billed else None

        # --- Extract Total Amount Due ---
        match_due = re.search(r"TOTAL AMOUNT DUE\s+([\d,]+\.\d{2})", text, re.IGNORECASE)
        total_due = to_float(match_due.group(1)) if match_due else None

        if total_due and billed:
            rate_per_kwh = total_due / billed

        # --- Parse CURRENT CHARGES dynamically ---
        lines = text.splitlines()
        current_section = None
        section_headers = ["GENERATION & TRANSMISSION", "DISTRIBUTION CHARGES", "OTHERS", "GOVERNMENT CHARGES", "UNIVERSAL CHARGE"]
        for line in lines:
            upper = line.upper()
            if any(header in upper for header in section_headers):
                current_section = line.strip()
                charges[current_section] = []
                continue

            if current_section:
                # Match charge with optional rate/unit
                match = re.match(r"([A-Za-z\s\-\*\/&]+)\s+([-\d\.]+)?(?:/(kWh|month))?\s+([-\d,]+\.\d{2})", line)
                if match:
                    name = match.group(1).strip()
                    rate = match.group(2) if match.group(2) else ""
                    unit = match.group(3) if match.group(3) else ""
                    amount = to_float(match.group(4))
                    charges[current_section].append({
                        "name": name,
                        "rate": rate,
                        "unit": unit,
                        "amount": amount
                    })
                else:
                    # Capture Sub-Total lines
                    sub_total_match = re.match(r"Sub-Total\s+([-\d,]+\.\d{2})", line, re.IGNORECASE)
                    if sub_total_match:
                        amount = to_float(sub_total_match.group(1))
                        charges[current_section].append({
                            "name": "Sub-Total",
                            "rate": "",
                            "unit": "",
                            "amount": amount
                        })

# --- Show Bill Summary ---
st.markdown("### 📊 Bill Summary")
cols = st.columns(2)
if billed:
    with cols[0]:
        st.metric(label="🔌 Total kWh", value=f"{billed} kWh")
if rate_per_kwh:
    with cols[1]:
        st.metric(label="⚡ Rate per kWh", value=f"{rate_per_kwh:.4f}")

# --- Dynamic dropdown for Current Charges ---
if charges:
    st.markdown("### 📑 Current Charges")
    section_selected = st.selectbox("Select Section to View", options=list(charges.keys()))
    for item in charges[section_selected]:
        rate_display = f" @ {item['rate']}/{item['unit']}" if item['unit'] else ""
        amount_color = "#2c7be5" if item['amount'] >= 0 else "#d62828"
        st.markdown(
            f"""
            <div style='
                padding:8px;
                border-radius:8px;
                background:#f8f9fa;
                margin-bottom:4px;
                display:flex;
                justify-content:space-between;
                font-size:14px;
            '>
                <span><b>{item['name']}{rate_display}</b></span>
                <span style='color:{amount_color}; font-weight:bold;'>₱{item['amount']:,.2f}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

# --- Manual Calculator ---
st.markdown("### 🔢 Compute Your Own Usage")
manual_kwh = st.number_input("Enter your kWh usage", value=0.0)

if st.button("💡 Compute My Bill"):
    if rate_per_kwh:
        computed = manual_kwh * rate_per_kwh
        st.success(f"✅ Your Computed Bill: ₱{computed:,.2f}")
    else:
        st.error("⚠️ Upload a valid bill first.")
