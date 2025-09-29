import streamlit as st
import pdfplumber
import re

st.set_page_config(page_title="Electric Bill Calculator", page_icon="⚡", layout="centered")
st.title("⚡ Electric Bill Calculator")

uploaded_file = st.file_uploader("📂 Upload your electric bill (PDF)", type="pdf")

total_due = None
total_kwh = None
rate_per_kwh = None

if uploaded_file:
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"

        # --- Extract TOTAL AMOUNT DUE ---
        match_due = re.search(r"TOTAL AMOUNT DUE\s+([\d,]+\.\d{2})", text, re.IGNORECASE)
        if match_due:
            total_due = float(match_due.group(1).replace(",", ""))

        # --- Extract Total kWh (from Billed/Diff Rdg) ---
        match_kwh = re.search(r"Billed\s*:\s*(\d+)", text, re.IGNORECASE)
        if match_kwh:
            total_kwh = float(match_kwh.group(1))

        # --- Compute Rate per kWh ---
        if total_due and total_kwh and total_kwh > 0:
            rate_per_kwh = total_due / total_kwh

        # --- Show Results ---
        st.subheader("📊 Bill Summary")
        if total_due:
            st.write(f"✅ **TOTAL AMOUNT DUE:** ₱{total_due:,.2f}")
        if total_kwh:
            st.write(f"📊 **Total kWh (from bill):** {total_kwh:,.0f}")
        if rate_per_kwh:
            st.write(f"⚡ **Rate per kWh:** {rate_per_kwh}")

# --- Manual Calculator ---
st.subheader("🔢 Enter your Kwh")
manual_kwh = st.number_input("Enter your kWh usage", value=0.0)

if st.button("💡 Compute My Bill"):
    if rate_per_kwh:
        computed = manual_kwh * rate_per_kwh
        st.success(f"Your Computed Bill: ₱{computed:,.2f}")
    else:
        st.error("⚠️ Upload a valid bill first.")