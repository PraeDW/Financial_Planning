import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from matplotlib.backends.backend_pdf import PdfPages
import csv

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(page_title="Thai Financial Planner", layout="wide")
st.title("Post Retirement Financial Planner")
# =========================================================
# DISCLAIMER
# =========================================================
@st.dialog("⚠️ Disclaimer")
def show_disclaimer():
    st.markdown("""
This website was created by Financial Engineering Students not Financial Planner nor Investment Advisor and we do not have access to any non public information.
We cannot guarantee that the simulation will be 100% correct.
This was created solely for financial planner to use as an assistance for rough estimation and not to be use as a replacement of one.
We are not regulated by any Financial Service Authority.
    """)
    if st.button("I understand"):
        st.rerun()

if "accepted_terms" not in st.session_state:
    show_disclaimer()
    st.session_state["accepted_terms"] = True
# =========================================================
# CORE SIMULATION ENGINE
# =========================================================
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
        assets_list = [k for k in portfolio_allocation.keys() if k in asset_stats]
        if len(assets_list) == 0:
            return np.zeros((n_simulations, n_years))

        weights = np.array([portfolio_allocation[a] for a in assets_list], dtype=float)
        weights = weights / (weights.sum() if weights.sum() != 0 else 1.0)

        means = np.array([asset_stats[a]["mean"] for a in assets_list], dtype=float)
        stds  = np.array([asset_stats[a]["std"]  for a in assets_list], dtype=float)

        n_assets = len(assets_list)
        corr = np.eye(n_assets) + 0.4 * (np.ones((n_assets, n_assets)) - np.eye(n_assets))
        cov = np.outer(stds, stds) * corr

        portfolio_returns = np.zeros((n_simulations, n_years))
        for sim in range(n_simulations):
            asset_returns = np.random.multivariate_normal(means, cov, n_years)
            portfolio_returns[sim] = asset_returns @ weights
        return portfolio_returns

    # -------------------------
    # STRATEGIES (ALL return balances + withdrawals)
    # -------------------------
    def basic_strategy(self, initial_portfolio, withdrawal_rate, inflation_rate, returns, years):
        portfolio_value = initial_portfolio
        withdrawal = initial_portfolio * withdrawal_rate

        balances = [portfolio_value]
        withdrawals = []

        for year in range(years):
            withdrawals.append(max(0.0, withdrawal))
            portfolio_value -= withdrawal

            if portfolio_value <= 0:
                balances.extend([0.0] * (years - year))
                withdrawals.extend([0.0] * (years - 1 - year))
                break

            portfolio_value *= (1 + returns[year])
            balances.append(max(0.0, portfolio_value))
            withdrawal *= (1 + inflation_rate)

        return balances, withdrawals

    def forgoing_inflation_strategy(self, initial_portfolio, withdrawal_rate, inflation_rate, returns, years):
        portfolio_value = initial_portfolio
        withdrawal = initial_portfolio * withdrawal_rate

        balances = [portfolio_value]
        withdrawals = []
        prev_balance = portfolio_value

        for year in range(years):
            withdrawals.append(max(0.0, withdrawal))
            portfolio_value -= withdrawal

            if portfolio_value <= 0:
                balances.extend([0.0] * (years - year))
                withdrawals.extend([0.0] * (years - 1 - year))
                break

            portfolio_value *= (1 + returns[year])
            balances.append(max(0.0, portfolio_value))

            if portfolio_value > prev_balance:
                withdrawal *= (1 + inflation_rate)
            prev_balance = portfolio_value

        return balances, withdrawals

    def rmd_strategy(self, initial_portfolio, starting_age, returns, years):
        portfolio_value = initial_portfolio
        current_age = starting_age

        balances = [portfolio_value]
        withdrawals = []

        for year in range(years):
            life_exp = self.get_life_expectancy(current_age)
            withdrawal = portfolio_value / life_exp if life_exp > 0 else portfolio_value

            withdrawals.append(max(0.0, withdrawal))
            portfolio_value -= withdrawal

            if portfolio_value <= 0:
                balances.extend([0.0] * (years - year))
                withdrawals.extend([0.0] * (years - 1 - year))
                break

            portfolio_value *= (1 + returns[year])
            balances.append(max(0.0, portfolio_value))
            current_age += 1

        return balances, withdrawals

    def guardrails_strategy(self, initial_portfolio, withdrawal_rate, inflation_rate, returns, years):
        portfolio_value = initial_portfolio
        withdrawal = initial_portfolio * withdrawal_rate
        initial_rate = withdrawal_rate

        balances = [portfolio_value]
        withdrawals = []

        for year in range(years):
            withdrawals.append(max(0.0, withdrawal))
            portfolio_value -= withdrawal

            if portfolio_value <= 0:
                balances.extend([0.0] * (years - year))
                withdrawals.extend([0.0] * (years - 1 - year))
                break

            portfolio_value *= (1 + returns[year])
            balances.append(max(0.0, portfolio_value))

            current_rate = withdrawal / portfolio_value if portfolio_value > 0 else 0.0
            if current_rate < initial_rate * 0.8:
                withdrawal *= 1.10
            elif current_rate > initial_rate * 1.2:
                withdrawal *= 0.90
            else:
                withdrawal *= (1 + inflation_rate)

        return balances, withdrawals

    # -------------------------
    # RUN SIMULATION (pad arrays properly)
    # -------------------------
    def run_simulation(
        self,
        initial_portfolio,
        portfolio_allocation,
        asset_stats,
        withdrawal_strategy,
        withdrawal_rate,
        n_simulations,
        years,
        inflation_rate,
        starting_age,
        returns_override=None,
        ):
        returns = returns_override if returns_override is not None else self.simulate_returns(
        portfolio_allocation, asset_stats, n_simulations, years
        )

        strategy_map = {
            "Basic Strategy": self.basic_strategy,
            "Forgoing Inflation": self.forgoing_inflation_strategy,
            "RMD Strategy": self.rmd_strategy,
            "Guardrails": self.guardrails_strategy,
        }

        all_balances = []
        all_withdrawals = []

        for sim in range(n_simulations):
            if withdrawal_strategy == "RMD Strategy":
                balances, wds = strategy_map[withdrawal_strategy](
                initial_portfolio, starting_age, returns[sim], years
                )
            else:
                balances, wds = strategy_map[withdrawal_strategy](
                    initial_portfolio, withdrawal_rate, inflation_rate, returns[sim], years
                )

            # ✅ PAD to fixed lengths
            if len(balances) < years + 1:
                balances = balances + [0.0] * ((years + 1) - len(balances))
            else:
                balances = balances[: years + 1]

            if len(wds) < years:
                wds = wds + [0.0] * (years - len(wds))
            else:
                wds = wds[:years]

            all_balances.append(balances)
            all_withdrawals.append(wds)

        all_balances = np.array(all_balances, dtype=float)       # (sim, years+1)
        all_withdrawals = np.array(all_withdrawals, dtype=float) # (sim, years)

        final_values = all_balances[:, -1]

        return {
            "survival_rate": float(np.mean(final_values > 0)),
            "median_balance": np.median(all_balances, axis=0),
            "percentile_10": np.percentile(all_balances, 10, axis=0),
            "percentile_90": np.percentile(all_balances, 90, axis=0),
            "returns_mean": float(np.mean(returns)),

            "median_withdrawal": np.median(all_withdrawals, axis=0),
            "withdrawal_p10": np.percentile(all_withdrawals, 10, axis=0),
            "withdrawal_p90": np.percentile(all_withdrawals, 90, axis=0),
        }


    # -------------------------
    # RECOMMENDATIONS (fix key mismatch)
    # -------------------------
    def recommend_improvements(self, current_survival_rate, portfolio_allocation, withdrawal_rate, min_survival_rate=0.85):
        recs = []
        if current_survival_rate >= min_survival_rate:
            return ["✅ Your strategy meets the target survival rate!"]

        if withdrawal_rate > 0.03:
            rec_rate = withdrawal_rate * 0.9
            recs.append(f"📉 **Reduce Spending:** Try lowering withdrawal from {withdrawal_rate*100:.1f}% to {rec_rate*100:.1f}%.")

        # equity-ish keys in YOUR alloc
        equity_keys = ["pct_seti", "pct_msci_stock","pct_REITTH","pct_MSCIREITs"]
        equity_weight = sum(float(portfolio_allocation.get(k, 0)) for k in equity_keys)

        if equity_weight < 0.4:
            recs.append(f"📈 **Increase Growth:** Equity allocation seems low ({equity_weight*100:.0f}%). Consider 40–60%.")
        elif equity_weight > 0.8:
            recs.append(f"🛡️ **Reduce Risk:** Equity allocation seems high ({equity_weight*100:.0f}%). Consider adding bonds/cash.")

        recs.append("🔄 **Change Strategy:** Try 'Guardrails' or 'Forgoing Inflation' to adapt during drawdowns.")

        deficit = min_survival_rate - current_survival_rate
        recs.append(f"💰 **Save More:** Consider increasing initial portfolio or reducing spending; gap to target ≈ {(deficit*100):.1f}%.")

        return recs

    # -------------------------
    # OPTIMIZER (wd_rate only)
    # -------------------------
    def find_optimal_withdrawal_rate(
        self,
        initial_portfolio,
        portfolio_allocation,
        asset_stats,
        withdrawal_strategy,
        initial_rate,
        years,
        inflation_rate,
        starting_age,
        min_survival_rate=0.85,
        n_simulations=800,
    ):
        low_rate = 0.01
        high_rate = min(0.12, max(0.06, initial_rate * 2))
        tolerance = 0.001
        best_rate = initial_rate
        max_iterations = 20

        for _ in range(max_iterations):
            if (high_rate - low_rate) <= tolerance:
                break
            test_rate = (low_rate + high_rate) / 2
            results = self.run_simulation(
                initial_portfolio,
                portfolio_allocation,
                asset_stats,
                withdrawal_strategy,
                test_rate,
                n_simulations,
                years,
                inflation_rate,
                starting_age,
            )
            if results["survival_rate"] >= min_survival_rate:
                best_rate = test_rate
                low_rate = test_rate
            else:
                high_rate = test_rate

        return best_rate
    
    def sensitivity_withdrawal_rate(
        self,
        initial_portfolio,
        portfolio_allocation,
        asset_stats,
        withdrawal_strategy,
        wd_grid,
        years,
        inflation_rate,
        starting_age,
        n_simulations=10000,
        returns_override=None,
    ):
        results = []
        for wd in wd_grid:
            res = self.run_simulation(
                initial_portfolio=initial_portfolio,
                portfolio_allocation=portfolio_allocation,
                asset_stats=asset_stats,
                withdrawal_strategy=withdrawal_strategy,
                withdrawal_rate=wd,
                n_simulations=n_simulations,
                years=years,
                inflation_rate=inflation_rate,
                starting_age=starting_age,
                returns_override=returns_override,  # ✅ reuse
            )
            results.append({
                "withdrawal_rate": wd,
                "survival_rate": res["survival_rate"],
                "median_end_balance": float(res["median_balance"][-1]),
            })
        return results
# =========================================================
# UI HELPER FUNCTIONS
# =========================================================
if "current_step" not in st.session_state:
    st.session_state["current_step"] = 0

steps = ["👤 1. ข้อมูลผู้ใช้", "🧩 2.แบบประเมินความเสี่ยง", "📊 3.การจัดสรรสินทรัพย์", "💸 4. กลยุทธ์การถอนเงิน"]

def update_nav():
    st.session_state["nav_radio"] = steps[st.session_state["current_step"]]

def next_step():
    if st.session_state["current_step"] < len(steps) - 1:
        st.session_state["current_step"] += 1
        update_nav()

def prev_step():
    if st.session_state["current_step"] > 0:
        st.session_state["current_step"] -= 1
        update_nav()

def jump_step():
    st.session_state["current_step"] = steps.index(st.session_state["nav_radio"])

def money_input(label, default_val, key_suffix):
    text_key = f"m_{key_suffix}"
    val_key  = f"v_{key_suffix}"

    if text_key not in st.session_state:
        st.session_state[text_key] = f"{default_val:,.0f}"
    if val_key not in st.session_state:
        st.session_state[val_key] = float(default_val)

    def on_change():
        raw = st.session_state.get(text_key, "0")
        try:
            s = str(raw).strip()
            num = float(s.replace(",", "")) if s else 0.0
        except:
            num = 0.0
        st.session_state[val_key] = num
        st.session_state[text_key] = f"{num:,.0f}"

    st.text_input(label, key=text_key, on_change=on_change)

    # keep val_key synced even if no on_change fired
    raw = st.session_state.get(text_key, "0")
    try:
        s = str(raw).strip()
        st.session_state[val_key] = float(s.replace(",", "")) if s else 0.0
    except:
        st.session_state[val_key] = 0.0

    return float(st.session_state.get(val_key, 0.0))

def pct_input(label, key):
    return st.number_input(f"{label} (%)", 0.0, 100.0, 0.0, 5.0, key=f"p_{key}", format="%.1f")

def get_val_num(key_suffix):
    return float(st.session_state.get(f"v_{key_suffix}", 0.0))
def get_num(key_suffix):
    try:
        return float(st.session_state.get(f"v_{key_suffix}", 0.0) or 0.0)
    except:
        return 0.0
    
def build_full_report_csv(export_data, res, alloc, years=30):
    def fnum(x, nd=2, default=0.0):
        try:
            return f"{float(x):,.{nd}f}"
        except:
            return f"{default:,.{nd}f}"

    def fpct(x, nd=2):
        try:
            return f"{float(x)*100:.{nd}f}%"
        except:
            return ""

    def to_int(x, default=None):
        try:
            return int(float(x))
        except:
            return default

    ASSET_LABELS = {
        "pct_deposit": "Fixed Deposit",
        "pct_gov_bond": "Thai Gov Bond 1Y",
        "pct_seti": "SET Index",
        "pct_XAUTHB": "Gold (THB)",
        "pct_REITTH": "Thai REIT",
        "pct_msci_stock": "MSCI World Equity",
        "pct_msci_gov_bond": "MSCI Gov Bond",
        "pct_XAUUSD": "Gold (USD)",
        "pct_MSCIREITs": "Global REIT",
    }

    # --- Pull profile ---
    name = export_data.get("name", "-")
    retire_age = export_data.get("retire_age", "")
    life_exp = export_data.get("life_exp", "")
    inflation = export_data.get("inflation", None)

    retire_age_int = to_int(retire_age, None)

    # --- Cashflow (annual) ---
    total_income = export_data.get("total_income")
    total_expense = export_data.get("total_expense")
    yearly_savings = export_data.get("yearly_savings")

    # --- Assets / debt ---
    cash = export_data.get("cash")
    bond = export_data.get("bond")
    stock_th = export_data.get("stock_th")
    stock_gl = export_data.get("stock_gl")
    other = export_data.get("other")
    investable = float(export_data.get("investable") or 0.0)
    total_debt = float(export_data.get("total_debt") or 0.0)
    net_worth = float(export_data.get("net_worth") or (investable - total_debt))

    # --- Simulation settings ---
    sim_strat = export_data.get("sim_strat", "-")
    wd_rate = export_data.get("wd_rate", None)

    rows = []

    # =========================
    # SECTION A: PROFILE/INPUTS
    # =========================
    rows.append(["SECTION", "FIELD", "VALUE"])
    rows.append(["PROFILE", "Name", name])
    rows.append(["PROFILE", "Retire Age", retire_age])
    rows.append(["PROFILE", "Life Expectancy", life_exp])

    if inflation is not None:
        rows.append(["SETTINGS", "Inflation", fpct(inflation)])

    rows.append(["CASHFLOW (ANNUAL)", "Total Income (THB)", fnum(total_income, 2)])
    rows.append(["CASHFLOW (ANNUAL)", "Total Expense (THB)", fnum(total_expense, 2)])
    rows.append(["CASHFLOW (ANNUAL)", "Yearly Savings (THB)", fnum(yearly_savings, 2)])

    rows.append(["ASSETS", "Cash (THB)", fnum(cash, 2)])
    rows.append(["ASSETS", "Bonds (THB)", fnum(bond, 2)])
    rows.append(["ASSETS", "Thai Stocks (THB)", fnum(stock_th, 2)])
    rows.append(["ASSETS", "Global Stocks (THB)", fnum(stock_gl, 2)])
    rows.append(["ASSETS", "Gold/Other (THB)", fnum(other, 2)])
    rows.append(["ASSETS", "TOTAL Investable (THB)", fnum(investable, 2)])

    rows.append(["DEBT", "TOTAL Debt (THB)", fnum(total_debt, 2)])
    rows.append(["SUMMARY", "Net Worth (THB)", fnum(net_worth, 2)])

    rows.append(["SIMULATION", "Strategy", sim_strat])
    if wd_rate is not None:
        rows.append(["SIMULATION", "Withdrawal Rate", fpct(wd_rate)])

    if res is not None:
        rows.append(["SIMULATION", "Survival Rate", f"{res['survival_rate']*100:.1f}%"])
        rows.append(["SIMULATION", "Median End Balance (Year 30)", fnum(res["median_balance"][-1], 0)])

    # --- Asset Allocation ---
    rows.append([])
    rows.append(["SECTION", "ASSET", "WEIGHT (%)"])
    if alloc:
        for k, v in alloc.items():
            label = ASSET_LABELS.get(k, k)
            rows.append(["ALLOCATION", label, f"{float(v)*100:.2f}"])
    else:
        rows.append(["ALLOCATION", "No allocation found", ""])

    # --- Sensitivity (WD rate) ---
    rows.append([])
    rows.append(["SENSITIVITY (WD RATE)", "", ""])
    rows.append(["Withdrawal Rate", "Survival Rate", "Median End Balance"])

    sens = export_data.get("sensitivity")
    if sens:
        for r in sens:
            rows.append([
                f"{float(r['withdrawal_rate'])*100:.2f}%",
                f"{float(r['survival_rate'])*100:.1f}%",
                f"{float(r['median_end_balance']):,.0f}",
            ])
    else:
        rows.append(["No sensitivity results", "", ""])

    # =========================
    # SECTION B: YEARLY PROJECTION
    # =========================
    rows.append([])
    rows.append(["YEARLY PROJECTION (30Y)"])
    rows.append([
        "Year",
        "Age",
        "Median_Balance",
        "P10_Balance",
        "P90_Balance",
        "Median_Withdrawal",
        "P10_Withdrawal",
        "P90_Withdrawal",
        "P10_Depleted_Flag",
    ])

    if res is not None:
        mb = res.get("median_balance")
        p10b = res.get("percentile_10")
        p90b = res.get("percentile_90")
        mw = res.get("median_withdrawal")
        p10w = res.get("withdrawal_p10")
        p90w = res.get("withdrawal_p90")

        # sanity check
        if mb is None or p10b is None or p90b is None or mw is None or p10w is None or p90w is None:
            rows.append(["ERROR", "Missing arrays in res", "Run simulation again"])
        elif len(mb) < years + 1 or len(p10b) < years + 1 or len(p90b) < years + 1:
            rows.append(["ERROR", "Balance arrays length mismatch", f"len(mb)={len(mb)}"])
        elif len(mw) < years or len(p10w) < years or len(p90w) < years:
            rows.append(["ERROR", "Withdrawal arrays length mismatch", f"len(mw)={len(mw)}"])
        else:
            for y in range(1, years + 1):
                age = (retire_age_int + (y - 1)) if retire_age_int is not None else ""
                median_bal = float(mb[y])
                p10_bal = float(p10b[y])
                p90_bal = float(p90b[y])

                median_wd = float(mw[y - 1])
                p10_wd = float(p10w[y - 1])
                p90_wd = float(p90w[y - 1])

                depleted_flag = 1 if p10_bal <= 0 else 0

                rows.append([
                    y,
                    age,
                    round(median_bal, 2),
                    round(p10_bal, 2),
                    round(p90_bal, 2),
                    round(median_wd, 2),
                    round(p10_wd, 2),
                    round(p90_wd, 2),
                    depleted_flag,
                ])
    else:
        rows.append(["No simulation results found. Please run simulation first."])

    out = io.StringIO()
    csv.writer(out).writerows(rows)
    return out.getvalue().encode("utf-8-sig")

def build_pdf_bytes(data, res):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:

        # Page 1
        import matplotlib.pyplot as plt
        f1, a1 = plt.subplots(figsize=(8, 11))
        a1.axis("off")
        a1.set_title("Financial Health Report", fontsize=18, fontweight="bold", pad=20)

        y = 0.85
        a1.text(0.1, y, f"Name: {data['name']}", fontsize=12, fontweight="bold"); y -= 0.03
        a1.text(0.1, y, f"Retire Age: {data['retire_age']} | Life Exp: {data['life_exp']}", fontsize=11); y -= 0.05

        a1.text(0.1, y, "SUMMARY", fontsize=12, fontweight="bold"); y -= 0.04
        a1.text(0.1, y, f"Investable Assets: {data['investable']:,.0f} THB", fontsize=11); y -= 0.03
        a1.text(0.1, y, f"Total Debt: {data['total_debt']:,.0f} THB", fontsize=11); y -= 0.03
        a1.text(0.1, y, f"Yearly Savings: {data['yearly_savings']:,.0f} THB", fontsize=11); y -= 0.03
        a1.text(0.1, y, f"Net Worth: {data['net_worth']:,.0f} THB", fontsize=11); y -= 0.05

        details = [
            ["Total Income", f"{data['total_income']:,.0f}"],
            ["Total Expense", f"{data['total_expense']:,.0f}"],
        ]
        t1 = a1.table(cellText=details, colLabels=["Item", "THB"], bbox=[0.1, 0.35, 0.8, 0.18])
        t1.auto_set_font_size(False); t1.set_fontsize(10)
        pdf.savefig(f1); plt.close(f1)

        # Page 2
        if res is not None:
            f2, a2 = plt.subplots(figsize=(8, 11))
            a2.axis("off")
            a2.set_title("Simulation Results", fontsize=16, pad=20)

            sim_table = [
                ["Strategy", data.get("sim_strat", "-")],
                ["Withdrawal Rate", f"{data.get('wd_rate', 0)*100:.2f}%"],
                ["Success Rate", f"{res['survival_rate']*100:.1f}%"],
                ["Median End Balance", f"{res['median_balance'][-1]:,.0f} THB"],
            ]
            t2 = a2.table(cellText=sim_table, colLabels=["Metric", "Result"], bbox=[0.1, 0.73, 0.8, 0.20])
            t2.auto_set_font_size(False); t2.set_fontsize(10)

            ax = f2.add_axes([0.1, 0.12, 0.8, 0.50])
            x = range(len(res["median_balance"]))
            ax.fill_between(x, res["percentile_10"], res["percentile_90"], alpha=0.2)
            ax.plot(x, res["median_balance"])
            ax.set_title("Wealth Projection")
            ax.set_xlabel("Year")
            ax.set_ylabel("Portfolio Value (THB)")

            pdf.savefig(f2); plt.close(f2)

    return buf.getvalue()
def next():
    # ดึงค่าจาก widget "user_name" แล้วเก็บลง key ถาวร
    st.session_state["profile_name"] = (st.session_state.get("user_name") or "").strip()
    next_step()
# =========================================================
# NAV BAR
# =========================================================
if "nav_radio" not in st.session_state:
    st.session_state["nav_radio"] = steps[0]

st.radio("Go to:", steps, key="nav_radio", horizontal=True, label_visibility="collapsed", on_change=jump_step)
st.progress((st.session_state["current_step"] + 1) / len(steps))
st.divider()
# ========================================================
# PAGE 1: FINANCIAL HEALTH 
# =========================================================
if st.session_state["current_step"] == 0:
    st.header("👤 1. ข้อมูลผู้ใช้ (Financial Health)")

    st.subheader("A. ข้อมูลส่วนตัว")

    def validate_ages():
        r_age = st.session_state.get("retire_age", 60)
        l_exp = st.session_state.get("life_expectancy", r_age + 25)
        if l_exp < r_age:
            st.session_state["life_expectancy"] = r_age

    c1, c2, c3 = st.columns(3)
    with c1:
        st.text_input("ชื่อ", key="user_name", placeholder="ชื่อของคุณ")
    with c2:
        st.number_input("อายุเกษียณ", min_value=40, max_value=100, value=60, key="retire_age", on_change=validate_ages)
    with c3:
        current_retire = st.session_state.get("retire_age", 60)
        st.number_input("อายุขัย", min_value=current_retire, max_value=120, value=current_retire + 25, key="life_expectancy", on_change=validate_ages)

    # Assets
    st.subheader("B. ทรัพย์สิน (Investable Assets Only)")
    with st.expander("📝 รายละเอียดทรัพย์สิน", expanded=True):
        st.markdown("💰 สินทรัพย์เพื่อการลงทุน")
        i1, i2 = st.columns(2)
        with i1:
            money_cash = money_input("เงินสด/เงินฝาก (Cash)", 0, "cash_dep")
            money_bond = money_input("ตราสารหนี้ (Bond)", 0, "bond")
        with i2:
            money_stock = money_input("หุ้นไทย (Thai Equity)", 0, "stock")
            money_glstock = money_input("หุ้นต่างประเทศ (Global Equity)", 0, "gl_stock")
            other_invest = money_input("ทองคำ/ทรัพย์สินเพื่อการลงทุนอื่นๆ (Gold/Alternative)", 0, "other_invest")
        investable_assets = money_cash + money_bond + money_stock + money_glstock + other_invest

    # Debt
    st.subheader("C. หนี้สิน (Debt)")
    with st.expander("📝 รายละเอียดหนี้สินรวม", expanded=True):
        st.markdown("💳 หนี้สินต่างๆ")
        lc1, lc2 = st.columns(2)
        with lc1:
            debt_home = money_input("หนี้บ้าน", 0, "debt_home")
            debt_car = money_input("หนี้รถ", 0, "debt_car")
        with lc2:
            debt_cc = money_input("บัตรเครดิต", 0, "debt_cc")
            debt_other = money_input("หนี้สินอื่น", 0, "debt_other")
        total_debt = debt_home + debt_car + debt_cc + debt_other
    st.metric("หนี้สินรวมทั้งหมด", f"{total_debt:,.0f}")

    # Cashflow (post-retire)
    st.subheader("D. กระแสเงินสด (Cash Flow) — หลังเกษียณ")
    cc1, cc2 = st.columns(2)
    with cc1:
        with st.expander("📝 รายละเอียดรายได้", expanded=True):
            st.markdown("📥รายได้ต่อปี (หลังเกษียณ)")
            income = money_input("เงินบำนาญ/รายได้ประจำ (Annual)", 0, "inc_sal")
            rental = money_input("ค่าเช่า (Annual)", 0, "inc_rental")
            others = money_input("รายได้อื่นๆ (Annual)", 0, "inc_other")
        total_income = income + rental + others
        st.metric("รวมรายได้ทั้งหมด/ปี", f"{total_income:,.0f}")

    with cc2:
        with st.expander("📝 รายละเอียดรายจ่าย", expanded=True):
            st.markdown("รายจ่ายคงที่ต่อปี (Fixed Expenses)")
            with st.expander("🔻 รายละเอียด Fixed", expanded=False):
                c_f1, c_f2 = st.columns(2)
                with c_f1:
                    e_insurance = money_input("ประกัน (Insurance)", 0, "exp_insurance")
                    e_sub = money_input("ค่าสมาชิก (Sub)", 0, "exp_sub")
                    e_home = money_input("ค่าที่พัก (Home)", 0, "exp_home")
                with c_f2:
                    e_other = money_input("อื่นๆ (Other)", 0, "exp_other")
            total_fixed = e_insurance + e_sub + e_home + e_other
            st.metric("รวมรายจ่ายคงที่/ปี", f"{total_fixed:,.0f}")

            st.markdown("รายจ่ายไม่คงที่ต่อปี (Not Fixed Expenses)")
            with st.expander("🔻 รายละเอียด Not Fixed", expanded=False):
                e_transport = money_input("ค่าเดินทาง (Transport)", 0, "exp_variable")
                e_food = money_input("ค่าอาหาร (Food)", 0, "exp_food")
                e_entertain = money_input("ค่าบันเทิง (Entertainment)", 0, "exp_entertain")
                e_travel = money_input("ค่าเที่ยว (Travel)", 0, "exp_travel")
                e_health = money_input("ค่ารักษาพยาบาล (Health)", 0, "exp_health")
                e_other_var = money_input("ค่าใช้จ่ายเบ็ดเตล็ด (Other)", 0, "exp_var_other")
            total_non_fixed = e_transport + e_food + e_entertain + e_travel + e_health + e_other_var
            st.metric("รวมรายจ่ายไม่คงที่/ปี", f"{total_non_fixed:,.0f}")

            expense = total_fixed + total_non_fixed

        st.metric("รวมรายจ่ายทั้งหมด/ปี", f"{expense:,.0f}")

    yearly_savings = total_income - expense
    net_worth = investable_assets - total_debt

    st.markdown("### 📊 สรุปสถานะการเงิน (หลังเกษียณ)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("มูลค่าสุทธิ (Net Worth)", f"{net_worth:,.0f}")
    m2.metric("เงินลงทุนได้ (Investable)", f"{investable_assets:,.0f}")
    m3.metric("เงินคงเหลือ/ปี (Income-Expense)", f"{yearly_savings:,.0f}")
    m4.metric("หนี้สินรวม", f"{total_debt:,.0f}")

    if yearly_savings < 0:
        st.warning(f"⚠️ รายจ่ายมากกว่ารายได้ {abs(yearly_savings):,.0f} บาท/ปี (ยังสามารถจำลองต่อได้)")

    # store core
    st.session_state["start_port"] = investable_assets
    st.session_state["money_save"] = yearly_savings
    st.session_state["money_debt"] = total_debt

    # inflation (store as 'inflation' ONLY)
    st.session_state["inflation"] = st.slider("เงินเฟ้อคาดการณ์ (%)", 0.0, 10.0, 3.0, 0.1) / 100

    st.subheader("เป้าหมายทางการเงิน (Goal)")
    with st.expander("📝 รายละเอียดเป้าหมายทางการเงิน", expanded=True):
        st.session_state["target_amount"] = money_input("จำนวนเงินที่ต้องการ (THB)", 0, "goal_amount")
        st.session_state["importance_level"] = st.slider("ระดับความสำคัญ (%)", 0, 100, 10)

    c_nav1,c_nav2 = st.columns([10, 3])
    with c_nav2:
        st.button("Next Step ➡", on_click=next, type="primary")
# =========================================================
# PAGE 2: RISK ASSESSMENT
# =========================================================
elif st.session_state["current_step"] == 1:
    st.header("🧩 2. แบบประเมินความเสี่ยง")

    questions_data = [
        {"q": "Q1: ปัจจุบันคุณกำลังอยู่ในช่วงชีวิตใด", "choices": [{"label": "อายุยังไม่เกิน 30 ปี เริ่มต้นทำงาน เก็บเงินเก็บทอง", "score": 3}, {"label": "อายุเกิน 30 แต่ไม่เกิน 55 ปี อยู่ในวัยทำงาน มีเงินเก็บเงินก้อน", "score": 2}, {"label": "อายุเกิน 55 ปี ใกล้เกษียณอยากพักผ่อน", "score": 1}]},
        {"q": "Q2: ในเรื่องการลงทุนเมื่อพูดถึง “ความผันผวน” คุณนึกถึงอะไรเป็นอันดับแรก", "choices": [{"label": "นี่แหละโอกาสทอง ขึ้นก็ขาย ลงก็ซื้อ ได้กำไรตั้งหลายรอบ", "score": 3}, {"label": "ที่ไหนมีความผันผวน ที่นั่นมีความไม่แน่นอน", "score": 2}, {"label": "แย่แล้วถ้าราคาตก ก็ขาดทุนสิ!!", "score": 1}]},
        {"q": "Q3: สไตล์การลงทุนที่ผ่านมาของคุณเป็นแบบไหน", "choices": [{"label": "กล้าได้กล้าเสีย ถึงเวลาต้องยอมตัดขาดทุน แล้วไปลุยใหม่ สร้างกำไรสูงๆ", "score": 3}, {"label": "ช้าแต่ชัวร์ ได้น้อยดีกว่าไม่ได้ แต่ไม่อยากขาดทุน", "score": 1}, {"label": "แล้วแต่จังหวะ แล้วแต่โอกาส บางทีก็เสี่ยงบ้าง มีกำไรพอประมาณ", "score": 2}]},
        {"q": "Q4: หากลงทุนแล้วขาดทุน อะไรคือสาเหตุในความคิดของคุณ", "choices": [{"label": "การตัดสินใจที่ผิดพลาดของตัวเรา", "score": 3}, {"label": "เป็นเพราะความไม่แน่นอนของตลาดและภาวะการลงทุน", "score": 1}, {"label": "ก็ทั้งตัวเราแล้วก็ภาวะการลงทุนนั่นแหละ", "score": 2}]},
        {"q": "Q5: ลองหลับตาแล้วมองไปข้างหน้าในอีก 1 ปี คุณอยากเห็นอะไรจากเงินลงทุน", "choices": [{"label": "ผลตอบแทนแน่นอน 5%", "score": 1}, {"label": "หวังกำไรถึง 10% แต่ถ้าโชคไม่ดีขาดทุนก็ยอมได้สัก 5%", "score": 2}, {"label": "หวังกำไรถึง 20% แต่ถ้าโชคไม่ดีขาดทุนก็ยอมได้สัก 10%", "score": 3}]},
        {"q": "Q6: ถ้าคุณโชคดีถูกล๊อตเตอรี่ได้เงินรางวัล 500,000 บาท คุณจะนำเงินไปลงทุนอะไร", "choices": [{"label": "ฝากประจำหรือพันธบัตรรัฐบาล เงินต้นอยู่ครบ ผลตอบแทนน้อยหน่อยแต่แน่นอน", "score": 1}, {"label": "แบ่งครึ่งหนึ่งไปซื้อหุ้นสามัญ อีกครึ่งหนึ่งไปซื้อพันธบัตรรัฐบาล", "score": 2}, {"label": "โชคดีแบบนี้ไม่ต้องกลัว ซื้อหุ้นไปเลย", "score": 3}]},
        {"q": "Q7: การได้ไปท่องเที่ยวต่างประเทศแบบหรูหรา เป็นความใฝ่ฝันของคุณที่อุตส่าห์เก็บหอมรอมริบมานานหลายปี ทว่าก่อนจองโปรแกรมท่องเที่ยว คุณโดนเลิกจ้างกะทันหันจากนโยบายลดจำนวนพนักงานของบริษัท คุณจะตัดสินใจอย่างไร", "choices": [{"label": "ยกเลิกโปรแกรมท่องเที่ยว จนกว่าจะหางานใหม่ได้", "score": 1}, {"label": "เปลี่ยนแผนท่องเที่ยว ไปแบบประหยัดแทน", "score": 2}, {"label": "จองโปรแกรมและไปเที่ยวตามเดิม กลับมาค่อยว่ากัน", "score": 3}]},
        {"q": "Q8: คุณได้ร่วมรายการเกมโชว์ เล่นได้ถึงรอบลึกๆ และมาถึงทางเลือกที่ว่าจะเล่นต่อหรือหยุดเล่น ด้วยเงื่อนไขต่างๆ คุณจะเลือกอย่างไร", "choices": [{"label": "หยุดเล่นแล้วรับเงินรางวัล 30,000 บาท", "score": 1}, {"label": "เล่นต่อกับคำถาม 2 ตัวเลือก ตอบถูกรับเงิน 60,000 บาท ตอบผิดไม่ได้อะไรเลย", "score": 2}, {"label": "เล่นต่อกับคำถาม 4 ตัวเลือก ตอบถูกรับเงิน 120,000 บาท ตอบผิดไม่ได้อะไรเลย", "score": 3}]},
        {"q": "Q9: เพื่อนของคุณที่เก่งด้านการค้าที่ดิน มาชวนลงทุนซื้อที่ดินด้วยกัน และคาดว่าราคามีโอกาสจะเพิ่มจากตารางวาละ 20,000 บาท เป็น 40,000 บาท ในอีก 1 ปีข้างหน้า แต่ก็มีโอกาสที่ราคาจะไม่เพิ่มขึ้นอยู่เหมือนกัน คุณจะร่วมลงทุนก็ต่อเมื่อโอกาสที่ราคาที่ดินจะเพิ่มขึ้นเป็นแบบใด ", "choices": [{"label": "ถึงจะเป็นไปได้น้อย ก็อยากลงทุนด้วย", "score": 3}, {"label": "ต้องมีความเป็นไปได้ปานกลาง ถึงจะลงทุนด้วย", "score": 2}, {"label": "ต้องเป็นไปได้มากๆ หน่อย ถึงจะลงทุนด้วย", "score": 1}]},
        {"q": "Q10: เจ้าของธุรกิจแห่งหนึ่งชวนคุณไปทำงานด้วย โดยมีเงื่อนไขระหว่าง ให้รับผลตอบแทนเป็นเงินเดือนที่แน่นอน หรือรับเงินเดือนน้อยหน่อยแต่มีค่านายหน้าตามผลงานยอดขายที่ทำได้ คุณจะเลือกรับผลตอบแทนแบบใด", "choices": [{"label": "เอารายได้แน่นอนดีกว่า เลือกรับเงินเดือนเป็นหลัก ค่านายหน้านิดหน่อย", "score": 1}, {"label": "เลือกแบบสมดุล รับเงินเดือนครึ่งหนึ่ง ค่านายหน้าอีกครึ่งหนึ่ง", "score": 2}, {"label": "เลือกรับรายได้ตามผลงาน เน้นค่านายหน้าเป็นหลัก เงินเดือนเล็กน้อย", "score": 3}]}
    ]

    total_score = 0
    all_answered = True
    for i, item in enumerate(questions_data):
        st.subheader(item["q"])
        choice = st.radio(
            f"Radio_{i}",
            item["choices"],
            format_func=lambda x: x["label"],
            key=f"q_{i}",
            index=None,
            label_visibility="collapsed",
        )
        st.divider()
        if choice is None:
            all_answered = False
        else:
            total_score += int(choice["score"])

    if all_answered:
        if total_score >= 26:
            profile = "Aggressive (เชิงรุก)"
        elif total_score >= 16:
            profile = "Moderate (ปานกลาง)"
        else:
            profile = "Conservative (ระมัดระวัง)"
        st.success(f"คะแนน: {total_score} - {profile}")

    c1, c2 = st.columns([1, 8])
    with c1:
        st.button("⬅ Back", on_click=prev_step)
    with c2:
        st.button("Next Step ➡", on_click=next_step, type="primary", disabled=not all_answered)
# =========================================================
# PAGE 3: ASSET ALLOCATION
# =========================================================
elif st.session_state["current_step"] == 2:
    st.header("📊 3. การจัดสรรสินทรัพย์")
    c1, c2 = st.columns(2)
    w1 = pct_input("Fix Deposit", "deposit")
    with c1:
        st.subheader("Thai Equity")
        w2 = pct_input("Government Bond 1 year", "gov_bond")
        w3 = pct_input("SET", "seti")
        w4 = pct_input("Gold", "XAUTHB")
        w5 = pct_input("REITs", "REITTH")
    with c2:
        st.subheader("Global Equity")
        w6 = pct_input("MSCI stock", "msci_stock")
        w7 = pct_input("MSCI government bond", "msci_gov_bond")
        w8 = pct_input("Gold US", "XAUUSD")
        w9 = pct_input("REITs", "MSCIREITs")

    total = w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8 + w9
    if np.isclose(total, 100.0):
        st.success(f"Total: {total:.0f}% ✅")
    else:
        st.error(f"Total: {total:.0f}% (Must be 100%)")

    def save_alloc():
        st.session_state["saved_alloc"] = {
            "pct_deposit": w1 / 100,
            "pct_gov_bond": w2 / 100,
            "pct_seti": w3 / 100,
            "pct_XAUTHB": w4 / 100,
            "pct_REITTH": w5 / 100,
            "pct_msci_stock": w6 / 100,
            "pct_msci_gov_bond": w7 / 100,
            "pct_XAUUSD": w8 / 100,
            "pct_MSCIREITs": w9 / 100,
        }
        next_step()

    c1, c2 = st.columns([1, 8])
    with c1:
        st.button("⬅ Back", on_click=prev_step)
    with c2:
        st.button("Next Step ➡", on_click=save_alloc, disabled=not np.isclose(total, 100.0), type="primary")

# =========================================================
# PAGE 4: SIMULATION + EXPORT (wd_rate only, no cashflow mode)
# =========================================================
elif st.session_state["current_step"] == 3:
    st.header("💸 4.กลยุทธ์การถอนเงิน")

    YEARS = 30
    N_SIM = 10000

    asset_stats = {
        "pct_deposit": {"mean": 0.0505, "std": 0.0572},
        "pct_gov_bond": {"mean": 0.0206, "std": 0.0125},
        "pct_seti": {"mean": 0.1227, "std": 0.3266},
        "pct_XAUTHB": {"mean": 0.065, "std": 0.150},
        "pct_REITTH": {"mean": 0.070, "std": 0.200},

        "pct_msci_stock": {"mean": 0.030, "std": 0.025},
        "pct_msci_gov_bond": {"mean": 0.040, "std": 0.035},
        "pct_XAUUSD": {"mean": 0.060, "std": 0.200},
        "pct_MSCIREITs": {"mean": 0.070, "std": 0.160}
    }

    alloc = st.session_state.get("saved_alloc", {})
    if not alloc:
        st.error("⚠️ No allocation data. Please go to Page 3.")
        if st.button("Go to Page 3"):
            st.session_state["current_step"] = 2
            st.rerun()
        st.stop()

    c1, c2 = st.columns(2)
    strat_options = ["Basic Strategy", "Forgoing Inflation", "RMD Strategy", "Guardrails"]
    with c1:
        strat_selection = st.selectbox("กลยุทธ์", strat_options)
    with c2:
        wd_rate = st.number_input("อัตราการถอน (%)", 3.0, 10.0, 4.0, 0.1) / 100

    start_port = st.session_state.get("start_port", 1_000_000.0)
    inflation = st.session_state.get("inflation", 0.03)
    retire_age = st.session_state.get("retire_age", 60)

    # =========================
    # 1) RUN SIMULATION (generate returns once, reuse later)
    # =========================
    if st.button("🚀 Run Simulation", type="primary"):
        sim = RetirementSimulator()
        with st.spinner("Simulating..."):
            mc_returns = sim.simulate_returns(alloc, asset_stats, N_SIM, YEARS)
            st.session_state["mc_returns"] = mc_returns  # ✅ cache returns

            res = sim.run_simulation(
                initial_portfolio=start_port,
                portfolio_allocation=alloc,
                asset_stats=asset_stats,
                withdrawal_strategy=strat_selection,
                withdrawal_rate=wd_rate,
                n_simulations=N_SIM,
                years=YEARS,
                inflation_rate=inflation,
                starting_age=retire_age,
                returns_override=mc_returns,  # ✅ reuse the same returns
            )

        st.session_state["res"] = res
        st.session_state["sim_strat"] = strat_selection
        st.session_state["wd_rate"] = wd_rate

        # clear export cache
        st.session_state.pop("export_pdf_bytes", None)
        st.session_state.pop("export_csv_bytes", None)

    # =========================
    # 2) RUN SENSITIVITY (reuse cached returns, no new random)
    # =========================
    if st.button("📊 Run Sensitivity Analysis"):
        mc_returns = st.session_state.get("mc_returns", None)
        if mc_returns is None:
            st.error("กรุณากด Run Simulation ก่อน เพื่อสร้าง Monte Carlo returns (จะได้ไม่สุ่มใหม่)")
        else:
            sim = RetirementSimulator()
            wd_grid = [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
            with st.spinner("Running sensitivity..."):
                sens = sim.sensitivity_withdrawal_rate(
                    initial_portfolio=start_port,
                    portfolio_allocation=alloc,
                    asset_stats=asset_stats,
                    withdrawal_strategy=strat_selection,
                    wd_grid=wd_grid,
                    years=YEARS,
                    inflation_rate=inflation,
                    starting_age=retire_age,
                    n_simulations=N_SIM,
                    returns_override=mc_returns,  # ✅ reuse
                )

            st.session_state["sensitivity"] = sens
            st.session_state["sensitivity_settings"] = {
                "strategy": strat_selection,
                "years": YEARS,
                "inflation": inflation,
                "start_port": start_port,
                "retire_age": retire_age,
            }

    # =========================
    # 3) RESULTS
    # =========================
    if "res" in st.session_state:
        res = st.session_state["res"]
        success = res["survival_rate"] * 100
        median_end = res["median_balance"][-1]

        st.divider()
        c1, c2 = st.columns(2)
        color = "green" if success > 85 else "red"
        c1.markdown(f"### Success Rate: :{color}[{success:.1f}%]")
        c2.metric("Median End Balance (Year 30)", f"{median_end:,.0f} THB")

        # ✅ Graph AFTER median_end, BEFORE recommendations (as requested)
        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(res["median_balance"]))
        ax.fill_between(x, res["percentile_10"], res["percentile_90"], alpha=0.2, label="10-90th Pctl")
        ax.plot(x, res["median_balance"], label="Median")
        ax.axhline(0, linestyle="--", label="Depleted")
        ax.legend()
        ax.set_xlabel("Year")
        ax.set_ylabel("Portfolio Value (THB)")
        ax.set_title("Wealth Projection")
        st.pyplot(fig)

        # =========================
        # 4) SENSITIVITY DISPLAY (table only, no graph)
        # =========================
        if "sensitivity" in st.session_state:
            st.subheader("📌 Sensitivity (Withdrawal Rate)")
            df = pd.DataFrame(st.session_state["sensitivity"]).copy()
            df["Withdrawal %"] = df["withdrawal_rate"] * 100
            df["Survival %"] = df["survival_rate"] * 100
            df = df[["Withdrawal %", "Survival %", "median_end_balance"]]
            df = df.rename(columns={"median_end_balance": "Median End Balance (THB)"})
            st.dataframe(df, use_container_width=True)

        # =========================
        # 5) Recommendations
        # =========================
        sim = RetirementSimulator()
        st.subheader("💡 Recommendations")
        if success < 85:
            st.error(f"⚠️ Survival Rate ({success:.1f}%) is below 85% target.")
            recs = sim.recommend_improvements(
                current_survival_rate=res["survival_rate"],
                portfolio_allocation=alloc,
                withdrawal_rate=st.session_state.get("wd_rate", wd_rate),
            )
            with st.expander("👉 View Action Plan", expanded=True):
                for r in recs:
                    st.info(r)
        else:
            st.success("✅ Your plan looks solid! You have a high chance of success.")

        # =========================
        # 6) Optimizer
        # =========================
        st.divider()
        if st.button("🔍 Find Optimal Withdrawal Rate"):
            with st.spinner("Optimizing..."):
                opt_rate = sim.find_optimal_withdrawal_rate(
                    initial_portfolio=start_port,
                    portfolio_allocation=alloc,
                    asset_stats=asset_stats,
                    withdrawal_strategy=st.session_state.get("sim_strat", "Basic Strategy"),
                    initial_rate=st.session_state.get("wd_rate", wd_rate),
                    years=YEARS,
                    inflation_rate=inflation,
                    starting_age=retire_age,
                )

            curr = st.session_state.get("wd_rate", wd_rate)
            diff = opt_rate - curr

            c_opt1, c_opt2 = st.columns(2)
            c_opt1.metric("Current Rate", f"{curr*100:.2f}%")
            c_opt2.metric("Optimal Rate", f"{opt_rate*100:.2f}%", f"{diff*100:.2f}%")

            if diff > 0:
                st.success(f"🎉 You can safely increase your withdrawal by {diff*100:.2f}%!")
            else:
                st.warning(f"⚠️ You should reduce your withdrawal by {abs(diff*100):.2f}% to be safe.")

    # =========================================================
    # EXPORT (Page 4 only)
    # =========================================================
    st.divider()
    st.subheader("💾 Save Your Plan")

    if st.button("✅ Prepare Export Files"):
        res = st.session_state.get("res")
        alloc = st.session_state.get("saved_alloc", {})

        raw_name = st.session_state.get("profile_name") or st.session_state.get("user_name")
        name = (raw_name or "").strip() or "ไม่ระบุชื่อ"

        export_data = {
            "name": name,
            "retire_age": st.session_state.get("retire_age", 60),
            "life_exp": st.session_state.get("life_expectancy", 85),
            "inflation": st.session_state.get("inflation", 0.03),
            "sim_strat": st.session_state.get("sim_strat", "-"),
            "wd_rate": st.session_state.get("wd_rate", 0.0),
            "sensitivity": st.session_state.get("sensitivity", None),
        }

        # numeric v_* keys
        cash = get_num("cash_dep")
        bond = get_num("bond")
        stock_th = get_num("stock")
        stock_gl = get_num("gl_stock")
        other = get_num("other_invest")
        investable = cash + bond + stock_th + stock_gl + other

        debt_home = get_num("debt_home")
        debt_car = get_num("debt_car")
        debt_cc = get_num("debt_cc")
        debt_other = get_num("debt_other")
        total_debt = debt_home + debt_car + debt_cc + debt_other

        inc_sal = get_num("inc_sal")
        inc_rental = get_num("inc_rental")
        inc_other = get_num("inc_other")
        total_income = inc_sal + inc_rental + inc_other

        exp_ins = get_num("exp_insurance")
        exp_sub = get_num("exp_sub")
        exp_home = get_num("exp_home")
        exp_oth_fix = get_num("exp_other")
        total_fixed = exp_ins + exp_sub + exp_home + exp_oth_fix

        exp_trans = get_num("exp_variable")
        exp_food = get_num("exp_food")
        exp_ent = get_num("exp_entertain")
        exp_trav = get_num("exp_travel")
        exp_oth_var = get_num("exp_var_other")
        total_variable = exp_trans + exp_food + exp_ent + exp_trav + exp_oth_var

        total_expense = total_fixed + total_variable
        yearly_savings = total_income - total_expense
        net_worth = investable - total_debt

        export_data.update({
            "cash": cash, "bond": bond, "stock_th": stock_th, "stock_gl": stock_gl, "other": other,
            "investable": investable,
            "total_debt": total_debt,
            "total_income": total_income,
            "total_expense": total_expense,
            "total_fixed": total_fixed,
            "total_variable": total_variable,
            "yearly_savings": yearly_savings,
            "net_worth": net_worth,
        })

        st.session_state["export_data"] = export_data
        st.session_state["export_csv_bytes"] = build_full_report_csv(export_data, res, alloc, years=YEARS)
        st.session_state["export_pdf_bytes"] = build_pdf_bytes(export_data, res)

        st.success("Export files prepared ✅")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "📄 Download Full Report CSV",
            data=st.session_state.get("export_csv_bytes", b""),
            file_name="full_retirement_report.csv",
            mime="text/csv",
            disabled=("export_csv_bytes" not in st.session_state),
        )
    with c2:
        st.download_button(
            "📕 Download PDF",
            data=st.session_state.get("export_pdf_bytes", b""),
            file_name="report.pdf",
            mime="application/pdf",
            disabled=("export_pdf_bytes" not in st.session_state),
        )
