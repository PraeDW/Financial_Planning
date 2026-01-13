import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as mtick

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Thai Financial Planner", layout="wide")
st.title("Post Retirement Financial Planner")

# ==========================================
# 🧠 CORE SIMULATION ENGINE
# ==========================================
class RetirementSimulator:
    def __init__(self):
        self.life_expectancy = {
            60: 24, 61: 23, 62: 22, 63: 21, 64: 20, 65: 19, 66: 18, 67: 17,
            68: 16, 69: 15, 70: 14, 71: 13, 72: 12, 73: 11, 74: 10, 75: 9,
            76: 8, 77: 7, 78: 6, 79: 5, 80: 4, 81: 3, 82: 2, 83: 1, 84: 1
        }
    
    def get_life_expectancy(self, current_age):
        if current_age in self.life_expectancy:
            return self.life_expectancy[current_age]
        elif current_age > max(self.life_expectancy.keys()):
            return 1
        else:
            return max(self.life_expectancy.values())
    
    def simulate_returns(self, portfolio_allocation, asset_stats, n_simulations, n_years):
        assets_list = list(portfolio_allocation.keys())
        weights = np.array([portfolio_allocation[asset] for asset in assets_list])
        means = np.array([asset_stats[asset]['mean'] for asset in assets_list])
        stds = np.array([asset_stats[asset]['std'] for asset in assets_list])
        
        n_assets = len(assets_list)
        # Dynamic Correlation Matrix
        correlation_matrix = np.eye(n_assets) + 0.4 * (np.ones((n_assets, n_assets)) - np.eye(n_assets))
        cov_matrix = np.outer(stds, stds) * correlation_matrix
        
        portfolio_returns = np.zeros((n_simulations, n_years))
        for sim in range(n_simulations):
            asset_returns = np.random.multivariate_normal(means, cov_matrix, n_years)
            portfolio_returns[sim] = asset_returns @ weights
        return portfolio_returns

    # --- STRATEGIES ---
    def basic_strategy(self, initial_portfolio, withdrawal_rate, inflation_rate, returns, years):
        portfolio_value = initial_portfolio
        withdrawal = initial_portfolio * withdrawal_rate
        balances = [portfolio_value]
        for year in range(years):
            portfolio_value -= withdrawal
            if portfolio_value <= 0:
                balances.extend([0] * (years - year))
                break
            portfolio_value *= (1 + returns[year])
            balances.append(max(0, portfolio_value))
            withdrawal *= (1 + inflation_rate)
        return balances

    def forgoing_inflation_strategy(self, initial_portfolio, withdrawal_rate, inflation_rate, returns, years):
        portfolio_value = initial_portfolio
        withdrawal = initial_portfolio * withdrawal_rate
        balances = [portfolio_value]
        prev_balance = portfolio_value
        for year in range(years):
            portfolio_value -= withdrawal
            if portfolio_value <= 0:
                balances.extend([0] * (years - year))
                break
            portfolio_value *= (1 + returns[year])
            balances.append(max(0, portfolio_value))
            if portfolio_value > prev_balance:
                withdrawal *= (1 + inflation_rate)
            prev_balance = portfolio_value
        return balances

    def rmd_strategy(self, initial_portfolio, starting_age, returns, years):
        portfolio_value = initial_portfolio
        current_age = starting_age
        balances = [portfolio_value]
        for year in range(years):
            life_exp = self.get_life_expectancy(current_age)
            withdrawal = portfolio_value / life_exp if life_exp > 0 else portfolio_value
            portfolio_value -= withdrawal
            if portfolio_value <= 0:
                balances.extend([0] * (years - year))
                break
            portfolio_value *= (1 + returns[year])
            balances.append(max(0, portfolio_value))
            current_age += 1
        return balances

    def guardrails_strategy(self, initial_portfolio, withdrawal_rate, inflation_rate, returns, years):
        portfolio_value = initial_portfolio
        withdrawal = initial_portfolio * withdrawal_rate
        initial_rate = withdrawal_rate
        balances = [portfolio_value]
        for year in range(years):
            portfolio_value -= withdrawal
            if portfolio_value <= 0:
                balances.extend([0] * (years - year))
                break
            portfolio_value *= (1 + returns[year])
            balances.append(max(0, portfolio_value))
            current_rate = withdrawal / portfolio_value if portfolio_value > 0 else 0
            if current_rate < initial_rate * 0.8: withdrawal *= 1.10
            elif current_rate > initial_rate * 1.2: withdrawal *= 0.90
            else: withdrawal *= (1 + inflation_rate)
        return balances

    def run_simulation(self, initial_portfolio, portfolio_allocation, asset_stats, 
                       withdrawal_strategy, withdrawal_rate, n_simulations,
                       years, inflation_rate, starting_age):
        
        returns = self.simulate_returns(portfolio_allocation, asset_stats, n_simulations, years)
        all_balances = []
        
        for sim in range(n_simulations):
            if withdrawal_strategy == "Basic Strategy":
                balances = self.basic_strategy(initial_portfolio, withdrawal_rate, inflation_rate, returns[sim], years)
            elif withdrawal_strategy == "Forgoing Inflation":
                balances = self.forgoing_inflation_strategy(initial_portfolio, withdrawal_rate, inflation_rate, returns[sim], years)
            elif withdrawal_strategy == "RMD Strategy":
                balances = self.rmd_strategy(initial_portfolio, starting_age, returns[sim], years)
            elif withdrawal_strategy == "Guardrails":
                balances = self.guardrails_strategy(initial_portfolio, withdrawal_rate, inflation_rate, returns[sim], years)
            all_balances.append(balances)
        
        all_balances = np.array(all_balances)
        final_values = all_balances[:, -1]
        
        return {
            'survival_rate': np.sum(final_values > 0) / n_simulations,
            'median_balance': np.median(all_balances, axis=0),
            'percentile_10': np.percentile(all_balances, 10, axis=0),
            'percentile_90': np.percentile(all_balances, 90, axis=0),
            'returns_mean': np.mean(returns)
        }

    def recommend_improvements(self, current_survival_rate, portfolio_allocation, withdrawal_rate, min_survival_rate=0.85):
        recommendations = []
        if current_survival_rate >= min_survival_rate:
            return ["✅ Your strategy meets the target survival rate!"]
        if withdrawal_rate > 0.03:
            rec_rate = withdrawal_rate * 0.9
            recommendations.append(f"📉 **Reduce Spending:** Try lowering withdrawal from {withdrawal_rate*100:.1f}% to {rec_rate*100:.1f}%.")
        equity_keys = [k for k in portfolio_allocation.keys() if 'Equity' in k or 'SET' in k or 'S&P' in k or 'Tech' in k]
        equity_weight = sum(portfolio_allocation[k] for k in equity_keys)
        if equity_weight < 0.4:
            recommendations.append(f"📈 **Increase Growth:** Your Equity allocation is low ({equity_weight*100:.0f}%). Consider 50-60%.")
        elif equity_weight > 0.8:
            recommendations.append(f"🛡️ **Reduce Risk:** Your Equity allocation is very high ({equity_weight*100:.0f}%). Consider adding Bonds.")
        recommendations.append("🔄 **Change Strategy:** Try 'Guardrails' or 'Forgoing Inflation' which adapt to market drops.")
        return recommendations

    def find_optimal_withdrawal_rate(self, initial_portfolio, portfolio_allocation, asset_stats,
                                     withdrawal_strategy, years, inflation_rate, starting_age, min_survival_rate=0.85):
        low, high = 0.01, 0.10
        best_rate = 0.01
        for _ in range(10): 
            mid = (low + high) / 2
            res = self.run_simulation(initial_portfolio, portfolio_allocation, asset_stats, 
                                      withdrawal_strategy, mid, 500, years, inflation_rate, starting_age)
            if res['survival_rate'] >= min_survival_rate:
                best_rate = mid
                low = mid
            else:
                high = mid
        return best_rate

# ==========================================
# UI HELPER FUNCTIONS
# ==========================================
if 'current_step' not in st.session_state: st.session_state['current_step'] = 0
steps = ["👤 1. ข้อมูลผู้ใช้", "🧩 2.แบบประเมินความเสี่ยง", "📊 3.จัดสรรพอร์ตโฟลิโอ", "💸 4. กลยุทธ์การถอนเงิน"]

def update_nav(): st.session_state['nav_radio'] = steps[st.session_state['current_step']]
def next_step():
    if st.session_state['current_step'] < len(steps)-1: 
        st.session_state['current_step'] += 1
        update_nav()
def prev_step():
    if st.session_state['current_step'] > 0: 
        st.session_state['current_step'] -= 1
        update_nav()
def jump_step(): st.session_state['current_step'] = steps.index(st.session_state['nav_radio'])

def money_input(label, default, key):
    k = f"m_{key}"
    if k not in st.session_state: st.session_state[k] = f"{default:,.0f}"
    def on_chg():
        try: st.session_state[k] = f"{float(str(st.session_state[k]).replace(',','')):,.0f}"
        except: pass
    st.text_input(label, key=k, on_change=on_chg)
    try: return float(str(st.session_state[k]).replace(',', ''))
    except: return 0.0

def pct_input(label, key):
    return st.number_input(f"{label} (%)", 0.0, 100.0, 0.0, 5.0, key=f"p_{key}", format="%.1f")

# --- NAVIGATION ---
if 'nav_radio' not in st.session_state: st.session_state['nav_radio'] = steps[0]
st.radio("Go to:", steps, key="nav_radio", horizontal=True, label_visibility="collapsed", on_change=jump_step)
st.progress((st.session_state['current_step'] + 1)/len(steps))
st.divider()

# ==========================================
# PAGE 1: FINANCIAL HEALTH CHECK (BLANK INPUTS)
# ==========================================
if st.session_state['current_step'] == 0:
    st.header("👤 1. ข้อมูลผู้ใช้ (Financial Health)")
    
    st.subheader("A. ข้อมูลส่วนตัว")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.text_input("ชื่อ นามสกุล", value="")
    with c2: st.session_state['current_age'] = st.number_input("อายุปัจจุบัน", 0, 100, 0)
    with c3: st.session_state['retire_age'] = st.number_input("อายุเกษียณ", 0, 100, 0)
    with c4: st.number_input("อายุขัย", 0, 120, 0)
    
    st.divider()
    st.subheader("B. ทรัพย์สิน (Assets)")
    ac1, ac2 = st.columns(2)
    with ac1:
        st.markdown("💰 **สินทรัพย์เพื่อการลงทุน**")
        money_cash = money_input("เงินสด/เงินฝาก", 0, "cash_dep")
        money_fund = money_input("กองทุนรวม", 0, "fund")
        money_stock = money_input("หุ้น/พันธบัตร/ทอง", 0, "stock")
        investable_assets = money_cash + money_fund + money_stock
    with ac2:
        st.markdown("🏠 **สินทรัพย์ส่วนตัว**")
        asset_home = money_input("บ้าน/คอนโด", 0, "home")
        asset_car = money_input("รถยนต์", 0, "car")
        personal_assets = asset_home + asset_car + money_input("อื่นๆ", 0, "other")

    st.divider()
    st.subheader("C. หนี้สิน (Debt)")
    lc1, lc2 = st.columns(2)
    with lc1:
        debt_home = money_input("หนี้บ้าน", 0, "debt_home")
        debt_car = money_input("หนี้รถ", 0, "debt_car")
    with lc2:
        debt_cc = money_input("บัตรเครดิต", 0, "debt_cc")
        total_debt = debt_home + debt_car + debt_cc + money_input("หนี้สินอื่น", 0, "debt_other")

    st.divider()
    st.subheader("D. กระแสเงินสด (Cash Flow)")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("📥 **รายได้**")
        income = money_input("เงินเดือน (สุทธิ)", 0, "inc_sal") + money_input("โบนัส/อื่นๆ", 0, "inc_bonus")
    with cc2:
        st.markdown("📤 **รายจ่าย**")
        expense = money_input("รายจ่ายคงที่", 0, "exp_fix") + money_input("รายจ่ายแปรผัน", 0, "exp_var")

    monthly_savings = income - expense

    st.markdown("### 📊 สรุปสถานะการเงิน")
    net_worth = (investable_assets + personal_assets) - total_debt
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("มูลค่าสุทธิ", f"{net_worth:,.0f}")
    m2.metric("เงินลงทุนได้", f"{investable_assets:,.0f}")
    m3.metric("เงินออม/เดือน", f"{monthly_savings:,.0f}")
    m4.metric("หนี้สินรวม", f"{total_debt:,.0f}")

    if monthly_savings < 0:
        st.error(f"⚠️ รายจ่ายมากกว่ารายได้ {abs(monthly_savings):,.0f} บาท")
    
    st.session_state.update({'start_port': investable_assets, 'money_save': monthly_savings, 'money_debt': total_debt})
    st.session_state['inflation'] = st.slider("เงินเฟ้อคาดการณ์ (%)", 0.0, 10.0, 3.0, 0.1) / 100
    
    c_nav1, c_nav2 = st.columns([8, 1])
    with c_nav2: st.button("Next Step ➡", on_click=next_step, type="primary", disabled=(monthly_savings<0))

# ==========================================
# PAGE 2: RISK ASSESSMENT (Thai Questions Preserved)
# ==========================================
elif st.session_state['current_step'] == 1:
    st.header("🧩 2. แบบประเมินความเสี่ยง")
    
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
            "q": "Q7: การได้ไปท่องเที่ยวต่างประเทศแบบหรูหรา เป็นความใฝ่ฝันของคุณที่อุตส่าห์เก็บหอมรอมริบมานานหลายปี ทว่าก่อนจองโปรแกรมท่องเที่ยว คุณโดนเลิกจ้างกะทันหันจากนโยบายลดจำนวนพนักงานของบริษัท คุณจะตัดสินใจอย่างไร",
            "choices": [
                {"label": "ยกเลิกโปรแกรมท่องเที่ยว จนกว่าจะหางานใหม่ได้", "score": 1},
                {"label": "เปลี่ยนแผนท่องเที่ยว ไปแบบประหยัดแทน", "score": 2},
                {"label": "จองโปรแกรมและไปเที่ยวตามเดิม กลับมาค่อยว่ากัน", "score": 3}
            ]
        },
        {
            "q": "Q8: คุณได้ร่วมรายการเกมโชว์ เล่นได้ถึงรอบลึกๆ และมาถึงทางเลือกที่ว่าจะเล่นต่อหรือหยุดเล่น ด้วยเงื่อนไขต่างๆ คุณจะเลือกอย่างไร",
            "choices": [
                {"label": "หยุดเล่นแล้วรับเงินรางวัล 30,000 บาท", "score": 1},
                {"label": "เล่นต่อกับคำถาม 2 ตัวเลือก ตอบถูกรับเงิน 60,000 บาท ตอบผิดไม่ได้อะไรเลย", "score": 2},
                {"label": "เล่นต่อกับคำถาม 4 ตัวเลือก ตอบถูกรับเงิน 120,000 บาท ตอบผิดไม่ได้อะไรเลย", "score": 3}
            ]
        },
        {
            "q": "Q9: เพื่อนของคุณที่เก่งด้านการค้าที่ดิน มาชวนลงทุนซื้อที่ดินด้วยกัน และคาดว่าราคามีโอกาสจะเพิ่มจากตารางวาละ 20,000 บาท เป็น 40,000 บาท ในอีก 1 ปีข้างหน้า แต่ก็มีโอกาสที่ราคาจะไม่เพิ่มขึ้นอยู่เหมือนกัน คุณจะร่วมลงทุนก็ต่อเมื่อโอกาสที่ราคาที่ดินจะเพิ่มขึ้นเป็นแบบใด",
            "choices": [
                {"label": "ถึงจะเป็นไปได้น้อย ก็อยากลงทุนด้วย", "score": 3},
                {"label": "ต้องมีความเป็นไปได้ปานกลาง ถึงจะลงทุนด้วย", "score": 2},
                {"label": "ต้องเป็นไปได้มากๆ หน่อย ถึงจะลงทุนด้วย", "score": 1}
            ]
        },
        {
            "q": "Q10: เจ้าของธุรกิจแห่งหนึ่งชวนคุณไปทำงานด้วย โดยมีเงื่อนไขระหว่าง ให้รับผลตอบแทนเป็นเงินเดือนที่แน่นอน หรือรับเงินเดือนน้อยหน่อยแต่มีค่านายหน้าตามผลงานยอดขายที่ทำได้ คุณจะเลือกรับผลตอบแทนแบบใด",
            "choices": [
                {"label": "เอารายได้แน่นอนดีกว่า เลือกรับเงินเดือนเป็นหลัก ค่านายหน้านิดหน่อย", "score": 1},
                {"label": "เลือกแบบสมดุล รับเงินเดือนครึ่งหนึ่ง ค่านายหน้าอีกครึ่งหนึ่ง", "score": 2},
                {"label": "เลือกรับรายได้ตามผลงาน เน้นค่านายหน้าเป็นหลัก เงินเดือนเล็กน้อย", "score": 3}
            ]
        }
    ]

    total_score = 0
    all_answered = True
    
    for i, item in enumerate(questions_data):
        st.subheader(item["q"])
        choice = st.radio(f"Radio_{i}", item['choices'], format_func=lambda x: x['label'], key=f"q_{i}", index=None, label_visibility="collapsed")
        st.divider()
        if choice is None: all_answered = False
        else: total_score += choice['score']

    if all_answered:
        if total_score >= 26: profile = "Aggressive (เชิงรุก)"
        elif total_score >= 16: profile = "Moderate (ปานกลาง)"
        else: profile = "Conservative (ระมัดระวัง)"
        st.success(f"คะแนน: {total_score} - {profile}")

    c1, c2 = st.columns([1, 8])
    with c1: st.button("⬅ Back", on_click=prev_step)
    with c2: st.button("Next Step ➡", on_click=next_step, type="primary", disabled=not all_answered)

# ==========================================
# PAGE 3: ASSET ALLOCATION
# ==========================================
elif st.session_state['current_step'] == 2:
    st.header("📊 3. จัดสรรพอร์ตโฟลิโอ")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🇹🇭 Thai Assets")
        w1 = pct_input("Gov Bond 1y", "gov")
        w2 = pct_input("Bond Fund", "abf")
        w3 = pct_input("SET Index", "set")
        w4 = pct_input("Stock Fund", "rmf")
        w5 = pct_input("Gold (TH)", "gld")
        w6 = pct_input("Oil ETF", "oil")
        w7 = pct_input("REIT (TH)", "reit")
    with c2:
        st.subheader("🇺🇸 US Assets")
        w8 = pct_input("US Gov 1y", "usgov")
        w9 = pct_input("US Bond Fund", "usbond")
        w10 = pct_input("S&P 500", "sp5")
        w11 = pct_input("US Total Stock", "vti")
        w12 = pct_input("US Gold", "usgld")
        w13 = pct_input("US Oil", "usoil")
        w14 = pct_input("US REIT", "usreit")

    total = w1+w2+w3+w4+w5+w6+w7+w8+w9+w10+w11+w12+w13+w14
    if np.isclose(total, 100.0): st.success(f"Total: {total:.0f}% ✅")
    else: st.error(f"Total: {total:.0f}% (Must be 100%)")

    def save_alloc():
        st.session_state['alloc'] = {
            'pct_gov_1y': w1/100, 'pct_abfth': w2/100, 'pct_seti': w3/100, 'pct_kblrmf': w4/100,
            'pct_gld': w5/100, 'pct_ktoil': w6/100, 'pct_reit': w7/100,
            'pct_us_gov': w8/100, 'pct_vtblx': w9/100, 'pct_sp500': w10/100, 'pct_vti': w11/100,
            'pct_us_gld': w12/100, 'pct_us_oil': w13/100, 'pct_us_reit': w14/100
        }
        next_step()

    c1, c2 = st.columns([1, 8])
    with c1: st.button("⬅ Back", on_click=prev_step)
    with c2: st.button("Next Step ➡", on_click=save_alloc, disabled=not np.isclose(total, 100.0), type="primary")

# ==========================================
# PAGE 4: SIMULATION (FULL FEATURES)
# ==========================================
elif st.session_state['current_step'] == 3:
    st.header("💸 4. Simulation Engine")
    
    # Asset Stats
    stats = {
        'pct_gov_1y': {'mean': 0.022, 'std': 0.015}, 'pct_abfth': {'mean': 0.030, 'std': 0.040},
        'pct_seti': {'mean': 0.080, 'std': 0.160}, 'pct_kblrmf': {'mean': 0.085, 'std': 0.150},
        'pct_gld': {'mean': 0.050, 'std': 0.140}, 'pct_ktoil': {'mean': 0.060, 'std': 0.250},
        'pct_reit': {'mean': 0.065, 'std': 0.120},
        'pct_us_gov': {'mean': 0.035, 'std': 0.020}, 'pct_vtblx': {'mean': 0.040, 'std': 0.050},
        'pct_sp500': {'mean': 0.100, 'std': 0.180}, 'pct_vti': {'mean': 0.100, 'std': 0.185},
        'pct_us_gld': {'mean': 0.050, 'std': 0.140}, 'pct_us_oil': {'mean': 0.060, 'std': 0.300},
        'pct_us_reit': {'mean': 0.080, 'std': 0.170}
    }
    
    alloc = st.session_state.get('alloc', {})
    if not alloc: st.error("No allocation!"); st.stop()
    
    c1, c2 = st.columns(2)
    with c1: strat = st.selectbox("Strategy", ["Basic Strategy", "Forgoing Inflation", "RMD Strategy", "Guardrails"])
    with c2: wd_rate = st.number_input("Withdrawal Rate (%)", 3.0, 10.0, 4.0, 0.1) / 100
    
    if st.button("🚀 Run Simulation", type="primary"):
        sim = RetirementSimulator()
        with st.spinner("Simulating..."):
            res = sim.run_simulation(
                st.session_state['start_port'], alloc, stats, strat, wd_rate, 1000, 30, 
                st.session_state['inflation'], st.session_state['retire_age']
            )
            st.session_state['res'] = res
            st.session_state['strat'] = strat
            st.session_state['wd_rate'] = wd_rate

    # RESULTS
    if 'res' in st.session_state:
        res = st.session_state['res']
        success = res['survival_rate'] * 100
        median_end = res['median_balance'][-1]
        
        st.divider()
        c1, c2 = st.columns(2)
        color = "green" if success > 85 else "red"
        c1.markdown(f"### Success Rate: :{color}[{success:.1f}%]")
        c2.metric("Median End Balance", f"{median_end:,.0f} THB")
        
        # --- PLOT ---
        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(31)
        ax.fill_between(x, res['percentile_10'], res['percentile_90'], alpha=0.2, color='blue', label='10-90th Pctl')
        ax.plot(x, res['median_balance'], color='navy', label='Median')
        ax.axhline(0, color='red', linestyle='--', label='Depleted')
        ax.legend()
        st.pyplot(fig)
        
        # --- RECOMMENDATIONS ---
        if success < 85:
            st.error(f"⚠️ Survival Rate ({success:.1f}%) is below 85% target.")
            sim = RetirementSimulator()
            recs = sim.recommend_improvements(res['survival_rate'], alloc, st.session_state['wd_rate'])
            with st.expander("💡 View Recommendations", expanded=True):
                for r in recs: st.write(r)
        
        # --- OPTIMIZER ---
        st.divider()
        if st.button("🔍 Find Optimal Withdrawal Rate"):
            sim = RetirementSimulator()
            with st.spinner("Optimizing..."):
                opt_rate = sim.find_optimal_withdrawal_rate(
                    st.session_state['start_port'], alloc, stats, st.session_state['strat'], 
                    30, st.session_state['inflation'], st.session_state['retire_age']
                )
            st.success(f"✅ Optimal Safe Withdrawal Rate: **{opt_rate*100:.2f}%**")
            st.caption(f"(Targeting > 85% Success with {st.session_state['strat']})")

        # --- MULTI-PAGE PDF ---
        st.divider()
        st.subheader("💾 Save Your Plan")
        
        col_d1, col_d2 = st.columns(2)
        
        df_health = pd.DataFrame([
            ["Investable Assets", f"{st.session_state['start_port']:,.0f}"],
            ["Total Debt", f"{st.session_state.get('money_debt',0):,.0f}"],
            ["Savings/Mo", f"{st.session_state.get('money_save',0):,.0f}"]
        ], columns=["Metric", "Value"])

        df_sim = pd.DataFrame([
            ["Success Rate", f"{success:.1f}%"],
            ["Median End", f"{median_end:,.0f}"],
            ["Strategy", st.session_state['sim_strat']]
        ], columns=["Metric", "Value"])

        def create_pdf():
            buffer = io.BytesIO()
            with PdfPages(buffer) as pdf:
                # Page 1
                f1, a1 = plt.subplots(figsize=(8,11))
                a1.axis('off')
                a1.set_title("Page 1: Health Profile", fontsize=16)
                t1 = a1.table(cellText=df_health.values, colLabels=df_health.columns, loc='center')
                t1.scale(1, 2)
                pdf.savefig(f1); plt.close(f1)
                # Page 2
                f2, a2 = plt.subplots(figsize=(8,11))
                a2.axis('off')
                a2.set_title("Page 2: Simulation Results", fontsize=16)
                t2 = a2.table(cellText=df_sim.values, colLabels=df_sim.columns, loc='center')
                t2.scale(1, 2)
                pdf.savefig(f2); plt.close(f2)
                # Page 3
                f3, a3 = plt.subplots(figsize=(10,6))
                a3.fill_between(x, res['percentile_10'], res['percentile_90'], alpha=0.2, color='blue')
                a3.plot(x, res['median_balance'], color='navy')
                a3.axhline(0, color='red', linestyle='--')
                a3.set_title("Page 3: Wealth Projection")
                pdf.savefig(f3); plt.close(f3)
            return buffer.getvalue()

        with col_d1:
            csv = df_sim.to_csv().encode('utf-8-sig')
            st.download_button("📄 CSV", csv, "data.csv", "text/csv")
        with col_d2:
            st.download_button("📕 PDF Report", create_pdf(), "report.pdf", "application/pdf")

    st.markdown("###")
    st.button("🔄 Reset App", on_click=lambda: st.session_state.update({'current_step': 0}))
