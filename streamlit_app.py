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
    st.header("📊 Portfolio Allocation Preference")
    st.caption("Define your portfolio structure. The total weight must be **100%**.")
    
    col_main1, col_main2 = st.columns([2, 1])
    
    with col_main1:
        # --- 1. CASH ---
        st.subheader("1. Cash & Equivalents")
        w_cash = pct_input("Cash / Money Market (%)", "cash")

        # --- 2. STOCK SECTORS (THAI SET) ---
        st.subheader("2. Stock")
        c1, c2 = st.columns(2)
        with c1:
            w_agro = pct_input("AGRO (Agro & Food Industry) %", "agro")
            w_consump = pct_input("CONSUMP (Consumer Products) %", "consump")
            w_fincial = pct_input("FINCIAL (Financials) %", "fincial")
            w_indus = pct_input("INDUS (Industrials) %", "indus")
        with c2:
            w_propcon = pct_input("PROPCON (Property & Construction) %", "propcon")
            w_resourc = pct_input("RESOURC (Resources) %", "resourc")
            w_service = pct_input("SERVICE (Services) %", "service")
            w_tech = pct_input("TECH (Technology) %", "tech")
        
        total_equity = w_agro + w_consump + w_fincial + w_indus + w_propcon + w_resourc + w_service + w_tech
        st.caption(f"Total Equity Weight: {total_equity}%")

        # --- 3. BONDS ---
        st.subheader("3. Bonds")
        c3, c4 = st.columns(2)
        with c3:
            w_bond_aaa = pct_input("Bond Rating AAA %", "aaa")
            w_bond_aa = pct_input("Bond Rating AA %", "aa")
        with c4:
            w_bond_a = pct_input("Bond Rating A %", "a")
            w_bond_bbb = pct_input("Bond Rating BBB %", "bbb")
        
        total_bond = w_bond_aaa + w_bond_aa + w_bond_a + w_bond_bbb
        st.caption(f"Total Bond Weight: {total_bond}%")

        # --- 4. ALTERNATIVES ---
        st.subheader("4. Derivatives / Alternatives")
        w_gold = pct_input("Gold / Derivatives", "gold")

    # --- CALCULATION & VALIDATION ---
    total_weight = w_cash + total_equity + total_bond + w_gold
    
    # --- WEIGHTED RETURN ESTIMATION (Hidden Logic for Simulation) ---
    # We assign proxy returns to these inputs so Tab 4 can run a simulation
    # (Values are hypothetical annual averages for calculation)
    expected_return = (
        (w_cash * 0.015) + 
        (total_equity * 0.08) +  # Assuming avg equity return 8%
        (w_bond_aaa * 0.025) + (w_bond_aa * 0.030) + (w_bond_a * 0.035) + (w_bond_bbb * 0.045) +
        (w_gold * 0.04)
    ) / 100
    
    estimated_volatility = (
        (w_cash * 0.005) + 
        (total_equity * 0.15) + 
        (total_bond * 0.05) + 
        (w_gold * 0.12)
    ) / 100

    with col_main2:
        st.markdown("### Total Weighting")
        
        if total_weight == 100.0:
            st.metric("Status", "✅ Perfect", f"{total_weight:.0f}%")
            st.success("Allocation Complete!")
            
            st.markdown("---")
            st.markdown("#### Estimated Metrics")
            st.caption("Based on your selection:")
            st.metric("Est. Annual Return", f"{expected_return:.2%}")
            st.metric("Est. Volatility", f"{estimated_volatility:.2%}")
            
        elif total_weight > 100.0:
            st.metric("Status", "❌ Over Limit", f"{total_weight:.0f}%")
            st.error(f"Please remove {total_weight - 100:.0f}%")
        else:
            st.metric("Status", "⚠️ Incomplete", f"{total_weight:.0f}%")
            st.warning(f"Please add {100 - total_weight:.0f}%")

        st.markdown("---")
# --- BUTTONS FOR PAGE 3 ---
    st.markdown("###")
    col_nav1, col_nav2, col_nav3 = st.columns([1, 8, 1])
    with col_nav1:
        st.button("⬅ Back", on_click=prev_step, use_container_width=True)
    with col_nav3:
        # Button is disabled until weight is exactly 100%
        st.button("Next Step ➡", on_click=next_step, type="primary", use_container_width=True, disabled=(total_weight != 100))
# ==========================================
# PAGE 4: WITHDRAWAL STRATEGY (Adapted from your script)
# ==========================================
elif st.session_state['current_step'] == 3:
    st.header("💸 4. Withdrawal Strategy (Monte Carlo)")

    # --- 1. SETUP ASSET DATA (Replaces Excel 'AssetData' Sheet) ---
    # We use the allocation form Page 3 to determine which assets are active
    active_assets = []
    
    # Map Page 3 Inputs to Default Stats (Mean, Std Dev)
    # You can edit these defaults here to match your Excel file
    base_asset_map = {
        'pct_cash':    ['Cash', 0.015, 0.005],
        'pct_agro':    ['Stock: Agro', 0.08, 0.15],
        'pct_consump': ['Stock: Consumer', 0.09, 0.14],
        'pct_fincial': ['Stock: Financial', 0.10, 0.18],
        'pct_indus':   ['Stock: Indus', 0.08, 0.16],
        'pct_propcon': ['Stock: Prop', 0.07, 0.20],
        'pct_resourc': ['Stock: Resource', 0.09, 0.22],
        'pct_service': ['Stock: Service', 0.085, 0.15],
        'pct_tech':    ['Stock: Tech', 0.12, 0.25],
        'pct_aaa':     ['Bond: AAA', 0.025, 0.03],
        'pct_aa':      ['Bond: AA', 0.030, 0.04],
        'pct_a':       ['Bond: A', 0.035, 0.05],
        'pct_bbb':     ['Bond: BBB', 0.045, 0.07],
        'pct_gold':    ['Gold', 0.04, 0.15]
    }

    # Build list of active assets based on user weights from Tab 3
    rows = []
    for key, (name, mu, sigma) in base_asset_map.items():
        weight = st.session_state.get(key, 0.0) / 100.0
        if weight > 0:
            rows.append({"Asset": name, "Weight": weight, "Mean": mu, "Std Dev": sigma})

    if not rows:
        st.error("⚠️ No assets selected in Tab 3. Please go back and allocate your portfolio.")
    else:
        # Show Editable Table (Replaces reading Excel)
        st.info("👇 **Simulation Assumptions:** You can edit the Mean (Return) and Std Dev (Risk) below.")
        df_assumptions = pd.DataFrame(rows)
        
        # Allow user to edit Mean/Std Dev live
        edited_df = st.data_editor(
            df_assumptions, 
            column_config={
                "Weight": st.column_config.NumberColumn(format="%.2f"),
                "Mean": st.column_config.NumberColumn(format="%.3f"),
                "Std Dev": st.column_config.NumberColumn(format="%.3f")
            },
            disabled=["Asset", "Weight"], # Lock weight (edit in Tab 3), allow editing Mean/Std
            hide_index=True,
            use_container_width=True
        )

        # --- 2. CALCULATE PORTFOLIO STATS ---
        # Instead of simulating every asset individually in a slow loop, 
        # we calculate the Weighted Portfolio Mean & Variance.
        # This assumes annual rebalancing (matching your logic).
        
        port_mean = 0.0
        port_var = 0.0
        total_weight = 0.0

        for index, row in edited_df.iterrows():
            w = row['Weight']
            port_mean += w * row['Mean']
            # Simplified Variance (assuming assets are independent for speed)
            # In a full institutional app, we would use a Correlation Matrix here.
            port_var += (w * row['Std Dev']) ** 2  
            total_weight += w

        port_std = np.sqrt(port_var)
        
        # --- 3. SIMULATION SETTINGS ---
        start_value = st.session_state.get('money_port', 1000000) + st.session_state.get('money_cash', 200000)
        inflation_rate = st.session_state.get('inflation_rate', 0.03)
        withdrawal_rate = 0.04 # Strict 4% Rule
        target_annual_spending = start_value * withdrawal_rate

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Simulated Return", f"{port_mean:.2%}")
        c2.metric("Simulated Volatility", f"{port_std:.2%}")
        c3.metric("Initial Withdrawal", f"{target_annual_spending:,.0f} THB", "4% Rule")

        if st.button("🚀 Run Monte Carlo Simulation", type="primary"):
            
            with st.spinner("Running 10,000 simulations..."):
                years = 30
                sims = 10000 
                
                # --- VECTORIZED ENGINE (Fast) ---
                # 1. Generate all random market shocks at once
                random_shock = np.random.normal(port_mean, port_std, (years, sims))
                
                # 2. Initialize Arrays
                portfolio_paths = np.zeros((years + 1, sims))
                portfolio_paths[0] = start_value
                
                # 3. Create Inflation Array [1, 1.03, 1.0609, ...]
                inflation_factors = (1 + inflation_rate) ** np.arange(years)
                
                # 4. Simulation Loop (Year by Year)
                for t in range(1, years + 1):
                    prev_balance = portfolio_paths[t-1]
                    
                    # LOGIC MATCH: 
                    # 1. Calculate Withdrawal amount (Start of Year)
                    current_withdrawal = target_annual_spending * inflation_factors[t-1]
                    
                    # 2. Subtract Withdrawal (Check for Ruin)
                    post_withdrawal = np.maximum(prev_balance - current_withdrawal, 0)
                    
                    # 3. Apply Growth
                    growth = post_withdrawal * (1 + random_shock[t-1])
                    
                    # 4. Store
                    portfolio_paths[t] = growth

                # --- 5. RESULTS & PLOTTING ---
                # Success Rate
                final_values = portfolio_paths[-1]
                success_count = np.sum(final_values > 0)
                success_rate = (success_count / sims) * 100
                
                if success_rate > 90: color = "green"
                elif success_rate > 75: color = "orange"
                else: color = "red"
                
                st.write(f"### 🎲 Success Rate: :{color}[{success_rate:.2f}%]")
                st.caption(f"Portfolio survived {years} years in {success_count:,} out of {sims:,} simulations.")

                # Visualization (Percentile Cone)
                p10 = np.percentile(portfolio_paths, 10, axis=1)
                p50 = np.percentile(portfolio_paths, 50, axis=1)
                p90 = np.percentile(portfolio_paths, 90, axis=1)
                x_years = np.arange(years + 1)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                
                
                # Shaded Cone (10th - 90th)
                ax.fill_between(x_years, p10, p90, color='blue', alpha=0.15, label="10th-90th Percentile")
                # Median Line
                ax.plot(x_years, p50, color='navy', linewidth=2, label="Median (50th)")
                # Zero Line
                ax.axhline(0, color='red', linestyle='--', linewidth=1, label="Depleted")
                
                ax.set_title(f"30-Year Wealth Projection", fontsize=14)
                ax.set_xlabel("Years in Retirement")
                ax.set_ylabel("Portfolio Value (THB)")
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.legend(loc="upper left")
                
                # Millions Formatter
                def millions(x, pos): return f'{x/1e6:.1f}M'
                ax.yaxis.set_major_formatter(plt.FuncFormatter(millions))
                
                st.pyplot(fig)

    # --- NAV BUTTONS ---
    st.markdown("###")
    col_nav1, col_nav2 = st.columns([1, 9])
    with col_nav1:
        st.button("⬅ Back", on_click=prev_step, use_container_width=True)
    with col_nav2:
        if st.button("🔄 Reset App"):
            st.session_state['current_step'] = 0
            st.rerun()
