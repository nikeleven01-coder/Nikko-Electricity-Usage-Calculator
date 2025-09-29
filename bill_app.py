import streamlit as st
import pdfplumber
import re

# --- Page Setup ---
st.set_page_config(page_title="Electric Bill Calculator", page_icon="⚡", layout="centered")
st.title("⚡ Electric Bill Calculator")

uploaded_file = st.file_uploader("📂 Upload your electric bill (PDF)", type="pdf")

diff_rdg = None
billed = None
rate_per_kwh = None

# --- Define the structure for all current charges ---
charges_structure = {
    "Generation & Transmission": [
        {"name": "Generation Charge", "rate": "6.5027", "unit": "kWh", "amount": 3439.93},
        {"name": "Power Act Reduction*", "rate": "-0.0047", "unit": "kWh", "amount": -2.08},
        {"name": "Transmission Charge", "rate": "0.8999", "unit": "kWh", "amount": 476.05},
        {"name": "System Loss Charge", "rate": "0.5898", "unit": "kWh", "amount": 312.00},
        {"name": "REC Rate", "rate": "0.00", "unit": "kWh", "amount": 0.00},
        {"name": "Sub-Total", "rate": "", "unit": "", "amount": 4225.90}
    ],
    "Distribution Charges": [
        {"name": "Distribution Charge", "rate": "1.7506", "unit": "kWh", "amount": 926.07},
        {"name": "Supply Charge", "rate": "0.4118", "unit": "kWh", "amount": 217.84},
        {"name": "Metering Charge", "rate": "0.6989", "unit": "kWh", "amount": 369.72},
        {"name": "5.00/month", "rate": "5.00", "unit": "month", "amount": 5.00},
        {"name": "Regulatory Reset Fees Reduction", "rate": "-0.0011", "unit": "kWh", "amount": -0.58},
        {"name": "Sub-Total", "rate": "", "unit": "", "amount": 1518.05}
    ],
    "Others": [
        {"name": "Subsidy on Lifeline Charge", "rate": "0.0005", "unit": "kWh", "amount": 0.26},
        {"name": "Senior Citizen Subsidy Charge", "rate": "0.000187", "unit": "kWh", "amount": 0.10},
        {"name": "Surcharge", "rate": "0.02 of 5,864.00", "unit": "", "amount": 117.28},
        {"name": "Sub-Total", "rate": "", "unit": "", "amount": 117.64}
    ],
    "Government Charges": [
        {"name": "Franchise Tax - Local 0.57% of 5,861.59", "rate": "", "unit": "", "amount": 33.41},
        {"name": "Value Added Tax - Generation", "rate": "", "unit": "", "amount": 351.77},
        {"name": "Value Added Tax - Transmission", "rate": "", "unit": "", "amount": 57.13},
        {"name": "Value Added Tax - System Loss", "rate": "", "unit": "", "amount": 32.54},
        {"name": "Value Added Tax - Distribution", "rate": "", "unit": "", "amount": 182.17},
        {"name": "Value Added Tax - Others", "rate": "", "unit": "", "amount": 14.12},
        {"name": "Missionary Electrification NPC-SPUG", "rate": "0.1949", "unit": "kWh", "amount": 103.10},
        {"name": "Missionary Electrification RE Developer", "rate": "0.0044", "unit": "kWh", "amount": 2.33},
        {"name": "NPC Stranded Contract Costs", "rate": "0.00", "unit": "kWh", "amount": 0.00},
        {"name": "NPC Stranded Debts", "rate": "0.0428", "unit": "kWh", "amount": 22.64},
        {"name": "Feed In Tariff Allowance - FIT-ALL", "rate": "0.1189", "unit": "kWh", "amount": 62.90},
        {"name": "Real Property Tax (RPT)", "rate": "0.0008", "unit": "kWh", "amount": 0.42},
        {"name": "Sub-Total", "rate": "", "unit": "", "amount": 862.53}
    ]
}

# --- Extract Diff Rdg & Billed kWh from PDF ---
if uploaded_file:
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"

        match_diff = re.search(r"Diff\s*Rdg\s*:\s*(\d+)", text, re.IGNORECASE)
        diff_rdg = int(match_diff.group(1)) if match_diff else None

        match_billed = re.search(r"Billed\s*:\s*(\d+)", text, re.IGNORECASE)
        billed = int(match_billed.group(1)) if match_billed else None

        match_due = re.search(r"TOTAL AMOUNT DUE\s+([\d,]+\.\d{2})", text, re.IGNORECASE)
        total_due = float(match_due.group(1).replace(",", "")) if match_due else None

        if total_due and billed:
            rate_per_kwh = total_due / billed

# --- Show Bill Summary ---
st.markdown("### 📊 Bill Summary")
cols = st.columns(2)
if billed:
    with cols[0]:
        st.metric(label="🔌 Total kWh", value=f"{billed} kWh")
if rate_per_kwh:
    with cols[1]:
        st.metric(label="⚡ Rate per kWh", value=f"{rate_per_kwh:.4f}")

# --- Show Current Charges with dropdown ---
st.markdown("### 📑 Current Charges")
section_selected = st.selectbox("Select Section to View", options=list(charges_structure.keys()))

for item in charges_structure[section_selected]:
    rate_display = f" @ {item['rate']}/{item['unit']}" if item['unit'] else ""
    amount_color = "#2c7be5" if item['amount'] >= 0 else "#d62828"  # red for negative
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
