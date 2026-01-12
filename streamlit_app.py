import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Thai Financial Planner", layout="wide")
st.title("Post Retirement Financial Planner")

# --- SESSION STATE SETUP (For Navigation) ---
if 'current_step' not in st.session_state:
    st.session_state['current_step'] = 0

def update_nav():
    st.session_state['nav_radio'] = steps[st.session_state['current_step']]
def next_step():
    if st.session_state['current_step'] < len(steps) - 1:
        st.session_state['current_step'] += 1
        update_nav() # Force radio button to update

def prev_step():
    if st.session_state['current_step'] > 0:
        st.session_state['current_step'] -= 1
        update_nav() # Force radio button to update

def jump_to_step():
    # This runs when the user manually clicks the Radio Button
    selected_step_name = st.session_state['nav_radio']
    st.session_state['current_step'] = steps.index(selected_step_name)

# --- HELPER FUNCTIONS ---
def money_input(label, default_value, key_suffix):
    user_text = st.text_input(
        label, 
        value=f"{default_value:,.0f}", 
        key=f"money_{key_suffix}"
    )
    try:
        clean_value = float(user_text.replace(",", ""))
    except ValueError:
        clean_value = 0.0
    return clean_value

def pct_input(label, key_suffix):
    """Helper for percentage inputs"""
    return st.number_input(label, min_value=0.0, max_value=100.0, value=0.0, step=5.0, key=f"pct_{key_suffix}")

# --- NAVIGATION ---
steps = ["👤 1. User Infomation", "🧩 2. Risk Profile", "📊 3. Portfolio Allocation Preference", "💸 4. Withdrawal Strategy"]
# We ensure the key 'nav_radio' is initialized
if 'nav_radio' not in st.session_state:
    st.session_state['nav_radio'] = steps[0]

st.radio(
    "Go to step:", 
    steps, 
    key="nav_radio", # Linked to session state
    horizontal=True,
    label_visibility="collapsed",
    on_change=jump_to_step # Triggers when user clicks the dots
)
st.progress((st.session_state['current_step'] + 1) / len(steps))
st.markdown("---")

# ==========================================
# TAB 1: USER INFORMATION
# ==========================================
if st.session_state['current_step'] == 0:
    st.header("👤 1. Personal Information (ข้อมูลส่วนตัว)")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        name = st.text_input("Full Name (ชื่อ-นามสกุล)", value="User")
    with col2:
        current_age = st.number_input("Current Age (อายุปัจจุบัน)", 20, 100, 30)
    with col3:
        retire_age = st.number_input("Retirement Age (อายุเกษียณ)", current_age + 1, 100, 60)
    with col4:
        life_expectancy = st.number_input("Expectation Age (อายุขัย)", retire_age + 1, 120, 85)

    st.markdown("---")
    st.header("2. Asset Information (ทรัพย์สิน)")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Liquid Assets (สินทรัพย์สภาพคล่อง)")
        portfolio_val = money_input("Current Portfolio Value (หุ้น/กองทุน)", 0, "port")
        bank_cash = money_input("Total Cash in Bank (เงินสด/เงินฝาก)", 0, "cash")
        total_liquid_assets = portfolio_val + bank_cash
        st.metric("💰 Total Investable Assets", f"{total_liquid_assets:,.2f} THB")
    
    with c2:
        st.subheader("Fixed Assets (สินทรัพย์ถาวร)")
        invest_property = money_input("Investment Property (อสังหาฯ)", 0, "prop")
        other_assets = money_input("Other Assets (ทองคำ/รถยนต์)", 0, "other")
        total_fixed = invest_property + other_assets
        st.metric("🏠 Total Fixed Assets", f"{total_fixed:,.2f} THB")
        
    st.success(f"🏆 **Total Net Worth: {total_liquid_assets + total_fixed:,.2f} THB**")

    st.markdown("---")
    # --- 3. INCOME INFORMATION (Post-Retirement Estimation) ---
    st.header("3. Expected Income (Post-Retirement)")
    st.caption("Enter the monthly income you expect to receive *after* you retire.")
    
    ci1, ci2, ci3 = st.columns(3)
    with ci1:
        gov_benefit = st.number_input("Government Benefit (Social Security/Pension) (THB/Month)", value=0, step=500)
    with ci2:
        fixed_income = st.number_input("Fixed Income (Annuities/Dividends/Rent) (THB/Month)", value=0, step=1000)
    with ci3:
        other_income = st.number_input("Other Post-Retirement Income (THB/Month)", value=0, step=1000)

    total_monthly_income = gov_benefit + fixed_income + other_income
    
    st.info(f"💵 **Guaranteed Monthly Income after Retirement:** {total_monthly_income:,.2f} THB")

    st.markdown("---")

    # --- 4. EXPENSE INFORMATION (The "Burn Rate") ---
    st.header("4. Expense Information (Current)")
    st.caption("This helps estimate your lifestyle cost. We assume this adjusts for inflation later.")

    ce1, ce2 = st.columns(2)
    with ce1:
        insurance = st.number_input("Insurance Premiums (THB/Yearly)", value=0)
        installments = st.number_input("Installments (Car/House) (THB/Month)", value=0)
        debt_obligation = st.number_input("Other Debt Obligations (THB/Month)", value=0)
    
    with ce2:
        nursing = st.number_input("Nursing Home / Caretaker (Estimated Future Need) (THB/Month)", value=0)
        subscription = st.number_input("Subscriptions (Netflix/Gym/Internet) (THB/Month)", value=0)
        other_expense = st.number_input("General Living (Food/Transport/Utilities) (THB/Month)", value=0)

    total_monthly_expense = insurance/12 + installments + debt_obligation + nursing + subscription + other_expense
    
    # Financial Health Snapshot
    st.error(f"💸 **Total Monthly Expenses:** {total_monthly_expense:,.2f} THB")
    
    st.markdown("---")

    st.header("5. Planning Assumptions")
    pc1, pc2 = st.columns(2)
    with pc1:
        current_savings = money_input("Monthly Savings (ออมเพิ่ม)", 0, "save")
        inflation_rate = st.slider("Inflation Rate (%)", 0.0, 10.0, 2.0, 0.1) / 100
    with pc2:
        replacement_ratio = st.slider("Expense Replacement Ratio (%): Estimate how much money you will need to spend in retirement compared to what you spend today", 50, 120, 70)
        
        # --- BUTTONS FOR PAGE 1 ---
    st.markdown("###")
    col_nav1, col_nav2 = st.columns([8, 1])
    with col_nav2:
        st.button("Next Step ➡", on_click=next_step, type="primary", use_container_width=True)
# ==========================================
# TAB 2: RISK ASSESSMENT (Scoring Only)
# ==========================================
elif st.session_state['current_step'] == 1:
    st.header("🧩 แบบประเมินความเสี่ยง (Risk Assessment)")
    st.caption("กรุณาเลือกคำตอบให้ครบทั้ง 10 ข้อ")

    # --- 1. DATA STRUCTURE (Mapping your Q1-Q10 exactly) ---
    questions_data = [
        {
            "q": "Q1: ปัจจุบันคุณกำลังอยู่ในช่วงชีวิตใด",
            "choices": [
                {"label": "อายุยังไม่เกิน 30 ปี เริ่มต้นทำงาน เก็บเงินเก็บทอง", "score": 3},
                {"label": "อายุเกิน 30 แต่ไม่เกิน 55 ปี อยู่ในวัยทำงาน มีเงินเก็บเงินก้อน", "score": 2},
                {"label": "อายุเกิน 55 ปี ใกล้เกษียณอยากพักผ่อน", "score": 1}
            ]
        },
        {
            "q": "Q2: ในเรื่องการลงทุนเมื่อพูดถึง “ความผันผวน” คุณนึกถึงอะไรเป็นอันดับแรก",
            "choices": [
                {"label": "นี่แหละโอกาสทอง ขึ้นก็ขาย ลงก็ซื้อ ได้กำไรตั้งหลายรอบ", "score": 3},
                {"label": "ที่ไหนมีความผันผวน ที่นั่นมีความไม่แน่นอน", "score": 2},
                {"label": "แย่แล้วถ้าราคาตก ก็ขาดทุนสิ!!", "score": 1}
            ]
        },
        {
            "q": "Q3: สไตล์การลงทุนที่ผ่านมาของคุณเป็นแบบไหน",
            "choices": [
                {"label": "กล้าได้กล้าเสีย ถึงเวลาต้องยอมตัดขาดทุน แล้วไปลุยใหม่ สร้างกำไรสูงๆ", "score": 3},
                {"label": "ช้าแต่ชัวร์ ได้น้อยดีกว่าไม่ได้ แต่ไม่อยากขาดทุน", "score": 1},
                {"label": "แล้วแต่จังหวะ แล้วแต่โอกาส บางทีก็เสี่ยงบ้าง มีกำไรพอประมาณ", "score": 2}
            ]
        },
        {
            "q": "Q4: หากลงทุนแล้วขาดทุน อะไรคือสาเหตุในความคิดของคุณ",
            "choices": [
                {"label": "การตัดสินใจที่ผิดพลาดของตัวเรา", "score": 3},
                {"label": "เป็นเพราะความไม่แน่นอนของตลาดและภาวะการลงทุน", "score": 1},
                {"label": "ก็ทั้งตัวเราแล้วก็ภาวะการลงทุนนั่นแหละ", "score": 2}
            ]
        },
        {
            "q": "Q5: ลองหลับตาแล้วมองไปข้างหน้าในอีก 1 ปี คุณอยากเห็นอะไรจากเงินลงทุน",
            "choices": [
                {"label": "ผลตอบแทนแน่นอน 5%", "score": 1},
                {"label": "หวังกำไรถึง 10% แต่ถ้าโชคไม่ดีขาดทุนก็ยอมได้สัก 5%", "score": 2},
                {"label": "หวังกำไรถึง 20% แต่ถ้าโชคไม่ดีขาดทุนก็ยอมได้สัก 10%", "score": 3}
            ]
        },
        {
            "q": "Q6: ถ้าคุณโชคดีถูกล๊อตเตอรี่ได้เงินรางวัล 500,000 บาท คุณจะนำเงินไปลงทุนอะไร",
            "choices": [
                {"label": "ฝากประจำหรือพันธบัตรรัฐบาล เงินต้นอยู่ครบ ผลตอบแทนน้อยหน่อยแต่แน่นอน", "score": 1},
                {"label": "แบ่งครึ่งหนึ่งไปซื้อหุ้นสามัญ อีกครึ่งหนึ่งไปซื้อพันธบัตรรัฐบาล", "score": 2},
                {"label": "โชคดีแบบนี้ไม่ต้องกลัว ซื้อหุ้นไปเลย", "score": 3}
            ]
        },
        {
            "q": "Q7: การได้ไปท่องเที่ยวต่างประเทศแบบหรูหรา... ทว่าโดนเลิกจ้าง...",
            "choices": [
                {"label": "ยกเลิกโปรแกรมท่องเที่ยว จนกว่าจะหางานใหม่ได้", "score": 1},
                {"label": "เปลี่ยนแผนท่องเที่ยว ไปแบบประหยัดแทน", "score": 2},
                {"label": "จองโปรแกรมและไปเที่ยวตามเดิม กลับมาค่อยว่ากัน", "score": 3}
            ]
        },
        {
            "q": "Q8: คุณได้ร่วมรายการเกมโชว์... คุณจะเลือกอย่างไร",
            "choices": [
                {"label": "หยุดเล่นแล้วรับเงินรางวัล 30,000 บาท", "score": 1},
                {"label": "เล่นต่อกับคำถาม 2 ตัวเลือก ตอบถูกรับเงิน 60,000 บาท ตอบผิดไม่ได้อะไรเลย", "score": 2},
                {"label": "เล่นต่อกับคำถาม 4 ตัวเลือก ตอบถูกรับเงิน 120,000 บาท ตอบผิดไม่ได้อะไรเลย", "score": 3}
            ]
        },
        {
            "q": "Q9: เพื่อนชวนลงทุนซื้อที่ดิน... คุณจะร่วมลงทุนเมื่อ...",
            "choices": [
                {"label": "ถึงจะเป็นไปได้น้อย ก็อยากลงทุนด้วย", "score": 3},
                {"label": "ต้องมีความเป็นไปได้ปานกลาง ถึงจะลงทุนด้วย", "score": 2},
                {"label": "ต้องเป็นไปได้มากๆ หน่อย ถึงจะลงทุนด้วย", "score": 1}
            ]
        },
        {
            "q": "Q10: เจ้าของธุรกิจชวนทำงาน... คุณจะเลือกรับผลตอบแทนแบบใด",
            "choices": [
                {"label": "เอารายได้แน่นอนดีกว่า เลือกรับเงินเดือนเป็นหลัก ค่านายหน้านิดหน่อย", "score": 1},
                {"label": "เลือกแบบสมดุล รับเงินเดือนครึ่งหนึ่ง ค่านายหน้าอีกครึ่งหนึ่ง", "score": 2},
                {"label": "เลือกรับรายได้ตามผลงาน เน้นค่านายหน้าเป็นหลัก เงินเดือนเล็กน้อย", "score": 3}
            ]
        }
    ]

    # --- 2. RENDER LOOP ---
    total_score = 0
    all_answered = True
    
    # วนลูปสร้างคำถามอัตโนมัติ
    for i, item in enumerate(questions_data):
        st.subheader(item["q"])
        
        # ใช้ format_func เพื่อแสดงเฉพาะข้อความ (ซ่อนคะแนนไว้ข้างหลัง)
        selected_choice = st.radio(
            f"เลือกคำตอบข้อ {i+1}",
            options=item["choices"],
            format_func=lambda x: x['label'], 
            key=f"q_{i}",
            index=None,  # เริ่มต้นเป็นค่าว่าง
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        if selected_choice is None:
            all_answered = False
        else:
            total_score += selected_choice["score"]

    # --- 3. SCORING LOGIC ---
    if not all_answered:
        st.warning("⚠️ กรุณาตอบให้ครบทุกข้อเพื่อคำนวณผลลัพธ์ (Please answer all questions)")
        profile = "Waiting..."
        mean_return = 0.05
        volatility = 0.10
    else:
        # User Logic Mapping
        if 26 <= total_score <= 30:
            profile = "Aggressive (ความเสี่ยงสูง)"
            mean_return, volatility = 0.09, 0.18
            alloc_text = "Stocks 90% / Bonds 10%"
        elif 21 <= total_score <= 25:
            profile = "Moderate to High (ปานกลางค่อนข้างสูง)"
            mean_return, volatility = 0.07, 0.14
            alloc_text = "Stocks 70% / Bonds 30%"
        elif 16 <= total_score <= 20:
            profile = "Moderate (ปานกลาง)"
            mean_return, volatility = 0.05, 0.10
            alloc_text = "Stocks 50% / Bonds 50%"
        elif 11 <= total_score <= 15:
            profile = "Cautious (ระมัดระวัง)"
            mean_return, volatility = 0.04, 0.06
            alloc_text = "Stocks 30% / Bonds 70%"
        elif total_score == 10:
            profile = "Conservative (ความเสี่ยงต่ำ)"
            mean_return, volatility = 0.03, 0.04
            alloc_text = "Stocks 10% / Bonds 90%"
        else:
            # Fallback (Should not happen given min score is 10)
            profile = "Conservative"
            mean_return, volatility = 0.03, 0.04
            alloc_text = "Stocks 10% / Bonds 90%"

        st.header(f"📊 ผลลัพธ์ของคุณ: {total_score} / 30")
        st.success(f"**Risk Profile:** {profile}")
    # --- BUTTONS FOR PAGE 2 ---
    st.markdown("###")
    col_nav1, col_nav2, col_nav3 = st.columns([1, 8, 1])
    with col_nav1:
        st.button("⬅ Back", on_click=prev_step, use_container_width=True)
    with col_nav3:
        # Button is disabled until all questions are answered
        st.button("Next Step ➡", on_click=next_step, type="primary", use_container_width=True, disabled=not all_answered)    
# ==========================================
# TAB 3: ASSET ALLOCATION 
# ==========================================
elif st.session_state['current_step'] == 2:
    st.header("📊 3. Asset Allocation")
    st.caption("Allocate your portfolio weight (%). Total must be **100%**.")
    
    col_thai, col_us = st.columns(2)
    
    with col_thai:
        st.subheader("Thai Assets (%)")
        w_gov_1y = pct_input("Government Bond 1yr", "gov_1y")
        w_abfth  = pct_input("Bond Fund (ABFTH)", "abfth")
        w_seti   = pct_input("Stock Market (SETI)", "seti")
        w_kblrmf = pct_input("Stock Fund (KBLRMF)", "kblrmf")
        w_gld    = pct_input("Gold ETF (TH-GLD)", "gld")
        w_ktoil  = pct_input("Oil ETF (KTOIL)", "ktoil")
        w_reit   = pct_input("REIT (TH-REIT)", "reit")

    with col_us:
        st.subheader("US Assets (%)")
        w_us_gov = pct_input("US 1yr Bond", "us_gov")
        w_vtblx  = pct_input("US Bond Fund (VTBLX)", "vtblx")
        w_sp500  = pct_input("S&P 500", "sp500")
        w_vti    = pct_input("US Total Stock (VTI)", "vti")
        w_us_gld = pct_input("US Gold (SPDR)", "us_gld")
        w_us_oil = pct_input("US Oil (USO)", "us_oil")
        w_us_reit= pct_input("US REIT (MSCI)", "us_reit")

    # Calculate Total
    total_weight = (
        w_gov_1y + w_abfth + w_seti + w_kblrmf + w_gld + w_ktoil + w_reit +
        w_us_gov + w_vtblx + w_sp500 + w_vti + w_us_gld + w_us_oil + w_us_reit
    )
    
    st.divider()
    col_sum1, col_sum2 = st.columns([2, 2])
    with col_sum2:
        st.markdown("### Total Weight")
        if np.isclose(total_weight, 100.0):
            st.metric("Status", "✅ Perfect", "100%")
        elif total_weight > 100.0:
            st.metric("Status", "❌ Over Limit", f"{total_weight:.1f}%")
            st.error(f"Remove {total_weight-100:.1f}%")
        else:
            st.metric("Status", "⚠️ Under Limit", f"{total_weight:.1f}%")
            st.warning(f"Add {100-total_weight:.1f}%")

    # --- SAVE DATA FUNCTION ---
    def save_and_next():
        # We manually save these inputs to a permanent dictionary
        st.session_state['saved_weights'] = {
            'pct_gov_1y': w_gov_1y, 'pct_abfth': w_abfth,
            'pct_seti': w_seti, 'pct_kblrmf': w_kblrmf,
            'pct_gld': w_gld, 'pct_ktoil': w_ktoil, 'pct_reit': w_reit,
            'pct_us_gov': w_us_gov, 'pct_vtblx': w_vtblx,
            'pct_sp500': w_sp500, 'pct_vti': w_vti,
            'pct_us_gld': w_us_gld, 'pct_us_oil': w_us_oil, 'pct_us_reit': w_us_reit
        }
        next_step()

    # --- NAV BUTTONS ---
    st.markdown("###")
    col_nav1, col_nav2, col_nav3 = st.columns([1, 8, 1])
    with col_nav1:
        st.button("⬅ Back", on_click=prev_step, use_container_width=True)
    with col_nav3:
        # Use the new save_and_next function
        st.button("Next Step ➡", on_click=save_and_next, type="primary", use_container_width=True, disabled=not np.isclose(total_weight, 100.0))
# ==========================================
# PAGE 4: WITHDRAWAL STRATEGY (Fixed Save Logic)
# ==========================================
elif st.session_state['current_step'] == 3:
    st.header("💸 4. Withdrawal Strategy (Monte Carlo)")

    # --- 1. SETUP ASSET DATA ---
    base_asset_map = {
        'pct_gov_1y':  ['TH: Gov Bond 1y',   0.022, 0.015],
        'pct_abfth':   ['TH: Bond Fund',     0.030, 0.040],
        'pct_seti':    ['TH: SET Index',     0.080, 0.160],
        'pct_kblrmf':  ['TH: Stock Fund',    0.085, 0.150],
        'pct_gld':     ['TH: Gold',          0.050, 0.140],
        'pct_ktoil':   ['TH: Oil ETF',       0.060, 0.250],
        'pct_reit':    ['TH: REIT',          0.065, 0.120],
        'pct_us_gov':  ['US: 1y Bond',       0.035, 0.020],
        'pct_vtblx':   ['US: Bond (VTBLX)',  0.040, 0.050],
        'pct_sp500':   ['US: S&P 500',       0.100, 0.180],
        'pct_vti':     ['US: Total Stock',   0.100, 0.185],
        'pct_us_gld':  ['US: Gold (SPDR)',   0.050, 0.140],
        'pct_us_oil':  ['US: Oil (USO)',     0.060, 0.300],
        'pct_us_reit': ['US: REIT (MSCI)',   0.080, 0.170]
    }

    saved_weights = st.session_state.get('saved_weights', {})
    
    rows = []
    for key, (name, mu, sigma) in base_asset_map.items():
        weight = saved_weights.get(key, 0.0) / 100.0
        if weight > 0:
            rows.append({"Asset": name, "Weight": weight, "Mean": mu, "Std Dev": sigma})

    if not rows:
        st.error("⚠️ No assets selected. Please go back to Tab 3.")
    else:
        # Show Assumptions Table
        st.info("👇 **Simulation Assumptions:**")
        df_assumptions = pd.DataFrame(rows)
        # Use Data Editor so user can tweak assumptions live
        edited_df = st.data_editor(
            df_assumptions,
            column_config={
                "Weight": st.column_config.NumberColumn(format="%.2f"),
                "Mean": st.column_config.NumberColumn(format="%.3f"),
                "Std Dev": st.column_config.NumberColumn(format="%.3f")
            },
            disabled=["Asset", "Weight"],
            hide_index=True,
            use_container_width=True
        )

        # Calculate Stats from the EDITED table
        port_mean = 0.0
        port_var = 0.0
        for index, row in edited_df.iterrows():
            port_mean += row['Weight'] * row['Mean']
            port_var += (row['Weight'] * row['Std Dev']) ** 2 
        port_std = np.sqrt(port_var)

        # Inputs
        start_value = st.session_state.get('money_port', 1000000) + st.session_state.get('money_cash', 200000)
        inflation_rate = st.session_state.get('inflation_rate', 0.03)
        target_annual_spending = start_value * 0.04

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Simulated Return", f"{port_mean:.2%}")
        c2.metric("Simulated Volatility", f"{port_std:.2%}")
        c3.metric("Initial Withdrawal", f"{target_annual_spending:,.0f} THB", "4% Rule")

        # --- RUN SIMULATION BUTTON ---
        if st.button("🚀 Run Monte Carlo Simulation", type="primary"):
            
            with st.spinner("Running 10,000 simulations..."):
                years = 30
                sims = 10000 
                random_shock = np.random.normal(port_mean, port_std, (years, sims))
                portfolio_paths = np.zeros((years + 1, sims))
                portfolio_paths[0] = start_value
                inflation_factors = (1 + inflation_rate) ** np.arange(years)
                
                for t in range(1, years + 1):
                    prev_balance = portfolio_paths[t-1]
                    current_withdrawal = target_annual_spending * inflation_factors[t-1]
                    post_withdrawal = np.maximum(prev_balance - current_withdrawal, 0)
                    growth = post_withdrawal * (1 + random_shock[t-1])
                    portfolio_paths[t] = growth

                # Calculate Results
                final_values = portfolio_paths[-1]
                success_rate = (np.sum(final_values > 0) / sims) * 100
                median_result = np.median(final_values)
                
                # --- SAVE RESULTS TO SESSION STATE (Fixes the NameError) ---
                st.session_state['sim_run'] = True
                st.session_state['sim_success_rate'] = success_rate
                st.session_state['sim_median_result'] = median_result
                st.session_state['sim_paths'] = portfolio_paths  # We save the paths for the chart

        # --- DISPLAY RESULTS (Only if Simulation has run) ---
        if st.session_state.get('sim_run', False):
            
            # Retrieve data from session state
            success_rate = st.session_state['sim_success_rate']
            median_result = st.session_state['sim_median_result']
            portfolio_paths = st.session_state['sim_paths']
            years = 30

            # 1. Metric Cards
            if success_rate > 90: color = "green"
            elif success_rate > 75: color = "orange"
            else: color = "red"
            st.write(f"### 🎲 Success Rate: :{color}[{success_rate:.1f}%]")

            # 2. Chart
            p10 = np.percentile(portfolio_paths, 10, axis=1)
            p50 = np.percentile(portfolio_paths, 50, axis=1)
            p90 = np.percentile(portfolio_paths, 90, axis=1)
            x_years = np.arange(years + 1)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.fill_between(x_years, p10, p90, color='blue', alpha=0.15, label="10th-90th Pctl")
            ax.plot(x_years, p50, color='navy', linewidth=2, label="Median")
            ax.axhline(0, color='red', linestyle='--', linewidth=1)
            ax.set_title("30-Year Survival Analysis")
            ax.set_ylabel("Portfolio Value (THB)")
            ax.legend(loc="upper left")
            
            def millions(x, pos): return f'{x/1e6:.1f}M'
            ax.yaxis.set_major_formatter(plt.FuncFormatter(millions))
            st.pyplot(fig)

            # ==========================================
            # 💾 SAVE DATA (MULTI-PAGE PDF w/ MATPLOTLIB)
            # ==========================================
            st.divider()
            st.subheader("💾 Save Your Plan")

            col_dl1, col_dl2 = st.columns(2)

            # --- 1. PREPARE DATA AS PANDAS DATAFRAMES ---
            
            # Data for Page 1 (Health)
            df_health = pd.DataFrame([
                ["Current Age", str(st.session_state.get('current_age', 30))],
                ["Retirement Age", str(st.session_state.get('retire_age', 60))],
                ["Investable Assets", f"{start_value:,.0f}"],
                ["Total Debt", f"{st.session_state.get('money_debt_home',0) + st.session_state.get('money_debt_car',0):,.0f}"],
                ["Monthly Income", f"{st.session_state.get('money_inc_sal',0) + st.session_state.get('money_inc_bonus',0):,.0f}"],
                ["Monthly Savings", f"{st.session_state.get('money_save',0):,.0f}"],
            ], columns=["Item", "Value"])

            # Data for Page 2 (Simulation)
            data_sim = [
                ["Success Rate (30y)", f"{success_rate:.1f}%"],
                ["Median End Value", f"{median_result:,.0f} THB"],
                ["Exp. Annual Return", f"{port_mean*100:.2f}%"],
                ["Volatility (Risk)", f"{port_std*100:.2f}%"],
                ["Initial Withdrawal", f"{target_annual_spending:,.0f} THB"],
            ]
            # Append Asset Allocation to Page 2
            saved_weights = st.session_state.get('saved_weights', {})
            for key, val in saved_weights.items():
                if val > 0:
                    data_sim.append([f"Alloc: {key.replace('pct_', '').upper()}", f"{val:.2f}%"])
            
            df_sim = pd.DataFrame(data_sim, columns=["Metric", "Value"])

            # --- 2. PDF GENERATOR FUNCTION ---
            import io
            from matplotlib.backends.backend_pdf import PdfPages

            def create_multipage_pdf():
                buffer = io.BytesIO()
                with PdfPages(buffer) as pdf:
                    
                    # --- PAGE 1: Health Table ---
                    fig1, ax1 = plt.subplots(figsize=(8, 11))
                    ax1.axis('tight')
                    ax1.axis('off')
                    ax1.set_title("Page 1: Financial Health Profile", fontsize=16, y=0.95)
                    
                    table1 = ax1.table(cellText=df_health.values, colLabels=df_health.columns, loc='center', cellLoc='left')
                    table1.scale(1, 2) # Make rows taller
                    table1.auto_set_font_size(False)
                    table1.set_fontsize(12)
                    
                    # Grey Header
                    for (i, j), cell in table1.get_celld().items():
                        if i == 0: cell.set_facecolor('#e6e6e6')

                    pdf.savefig(fig1, bbox_inches='tight')
                    plt.close(fig1)

                    # --- PAGE 2: Simulation Table ---
                    fig2, ax2 = plt.subplots(figsize=(8, 11))
                    ax2.axis('tight')
                    ax2.axis('off')
                    ax2.set_title("Page 2: Portfolio & Simulation Results", fontsize=16, y=0.95)
                    
                    table2 = ax2.table(cellText=df_sim.values, colLabels=df_sim.columns, loc='center', cellLoc='left')
                    table2.scale(1, 1.5)
                    table2.auto_set_font_size(False)
                    table2.set_fontsize(10)
                    
                    for (i, j), cell in table2.get_celld().items():
                        if i == 0: cell.set_facecolor('#e6e6e6')

                    pdf.savefig(fig2, bbox_inches='tight')
                    plt.close(fig2)

                    # --- PAGE 3: The Chart ---
                    # We redraw the cone chart just for the PDF
                    fig3, ax3 = plt.subplots(figsize=(10, 6))
                    
                    # Re-calc percentiles (quick access)
                    p10 = np.percentile(portfolio_paths, 10, axis=1)
                    p50 = np.percentile(portfolio_paths, 50, axis=1)
                    p90 = np.percentile(portfolio_paths, 90, axis=1)
                    x_years = np.arange(years + 1)

                    ax3.fill_between(x_years, p10, p90, color='blue', alpha=0.15, label="10th-90th Pctl")
                    ax3.plot(x_years, p50, color='navy', linewidth=2, label="Median")
                    ax3.axhline(0, color='red', linestyle='--', linewidth=1)
                    ax3.set_title("Page 3: 30-Year Wealth Projection", fontsize=14)
                    ax3.set_ylabel("Portfolio Value (THB)")
                    ax3.legend(loc="upper left")
                    
                    pdf.savefig(fig3)
                    plt.close(fig3)

                return buffer.getvalue()

            # --- 3. DOWNLOAD BUTTONS ---
            with col_dl1:
                # PDF Export (Multi-page)
                pdf_bytes = create_multipage_pdf()
                st.download_button(
                    label="📕 Download Report (.pdf)",
                    data=pdf_bytes,
                    file_name="financial_report.pdf",
                    mime="application/pdf"
                )

    # --- NAV BUTTONS ---
    st.markdown("###")
    col_nav1, col_nav2 = st.columns([1, 9])
    with col_nav1:
        st.button("⬅ Back", on_click=prev_step, use_container_width=True)
    with col_nav2:
        if st.button("🔄 Reset App"):
            st.session_state['current_step'] = 0
            st.rerun()
