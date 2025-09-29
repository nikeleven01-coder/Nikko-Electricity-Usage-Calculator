import streamlit as st
import pdfplumber
import re
import pandas as pd

st.set_page_config(page_title="Electric Bill Calculator", page_icon="⚡", layout="centered")
st.title("⚡ Electric Bill Calculator")

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

        # --- Extract Total kWh (from Billed/Diff Rdg) ---
        match_kwh = re.search(r"Billed\s*:\s*(\d+)", text, re.IGNORECASE)
        total_kwh = float(match_kwh.group(1)) if match_kwh else None

        # --- Compute Rate per kWh ---
        if total_due and total_kwh and total_kwh > 0:
            rate_per_kwh = total_due / total_kwh

        # --- Extract CURRENT CHARGES sections + subtotals ---
        charges_match = re.search(r"CURRENT CHARGES(.*?)CURRENT BILL", text, re.S | re.IGNORECASE)
        if charges_match:
            charges_block = charges_match.group(1)

            # Capture section headers + subtotals
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

    # --- Show Results ---
    st.subheader("📊 Bill Summary")
    if total_kwh:
        st.write(f"📊 **Total kWh (from bill):** {total_kwh:,.0f}")
    if rate_per_kwh:
        st.write(f"⚡ **Rate per kWh:** ₱{rate_per_kwh:,.2f}")

    # --- Show Sections with Subtotals + Est kWh ---
    if df_sections is not None:
        with st.expander("📑 Current Charges Breakdown"):
            df_display = df_sections.copy()
            df_display["Sub-Total (₱)"] = df_display["Sub-Total (₱)"].map(lambda x: f"₱{x:,.2f}")
            df_display["Est. kWh"] = df_display["Est. kWh"].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
            st.dataframe(df_display, use_container_width=True)

# --- Manual Calculator ---
st.subheader("🔢 Enter your kWh")
manual_kwh = st.number_input("Enter your kWh usage", value=0.0, step=1.0)

if st.button("💡 Compute My Bill"):
    if rate_per_kwh:
        computed = manual_kwh * rate_per_kwh
        st.success(f"Your Computed Bill: ₱{computed:,.2f}")
    else:
        st.error("⚠️ Please upload a valid bill first.")
