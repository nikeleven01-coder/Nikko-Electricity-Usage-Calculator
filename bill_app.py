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
    if st.button("🧮 Coming Soon", use_container_width=True):
        st.markdown(
            '<script>window.location.href="http://localhost:5500/";</script>',
            unsafe_allow_html=True,
        )

with col3:
    if st.button("📊 Coming Soon", use_container_width=True):
        st.markdown(
            '<script>window.location.href="http://localhost:5500/";</script>',
            unsafe_allow_html=True,
        )

col4, col5, col6 = st.columns(3)
with col4:
    if st.button("🔋 Coming Soon", use_container_width=True):
        st.markdown(
            '<script>window.location.href="http://localhost:5500/";</script>',
            unsafe_allow_html=True,
        )

with col5:
    if st.button("🌤️ Coming Soon", use_container_width=True):
        st.markdown(
            '<script>window.location.href="http://localhost:5500/";</script>',
            unsafe_allow_html=True,
        )

with col6:
    if st.button("⚙️ Coming Soon", use_container_width=True):
        st.markdown(
            '<script>window.location.href="http://localhost:5500/";</script>',
            unsafe_allow_html=True,
        )

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
