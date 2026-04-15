"""
app.py v2 — LP Centrifugal Compressor VFD Analysis Dashboard
Based on Gas_Compressor_v17.ipynb | Data: Petrobras/ccp (Apache 2.0)

Changes vs v1:
- Inputs moved to dedicated tab; outputs only update on "Run Analysis"
- Gas properties update in real-time in input tab
- Compression ratio max capped at 4.56 (physical design limit)
- Production profile sliders + numeric inputs
- Compressor map: all 20 points visible + efficiency degradation chart
- Overhaul year markers corrected (year AFTER overhaul)
- "Produced Gas Flow" label; total compressed gas added
- VFD speed: line chart; overhaul lines removed from VFD tab
- MM USD notation
- Economics: electricity price on secondary axis
- Reset to defaults button
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d as _interp1d

import model

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LP Compressor — VFD Analysis",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
[data-testid="collapsedControl"] { display: none; }
.kpi-card {
    background:#f8fafc; border:1px solid #dbe3ee;
    border-radius:10px; padding:14px 18px; text-align:center;
}
.kpi-label { font-size:13px; color:#5a6b7d; margin-bottom:4px; }
.kpi-value { font-size:26px; font-weight:700; color:#1f2d3d; }
.kpi-sub   { font-size:11px; color:#8a9ab0; margin-top:2px; }
.section-header {
    font-size:15px; font-weight:700; color:#1f3a5f;
    border-bottom:2px solid #dbe3ee;
    padding-bottom:4px; margin:8px 0 10px 0;
}
.derived-info {
    background:#f0f7ff; border-left:3px solid #3498db;
    padding:8px 12px; border-radius:4px;
    font-size:13px; color:#2c3e50; margin:6px 0;
}
.warn-info {
    background:#fff8f0; border-left:3px solid #e67e22;
    padding:8px 12px; border-radius:4px;
    font-size:13px; margin:6px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
N_FIXED = 9831.0
COLORS  = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c", "#9b59b6"]
CONSTRAINT_COLORS = {
    "antisurge":     "#2ecc71",
    "pressure":      "#e67e22",
    "N_max+recycle": "#e74c3c",
}
R_P_MAX = 4.56   # physical design limit

@st.cache_resource(show_spinner="Loading performance curves from Petrobras/ccp GitHub…")
def load_curves():
    curves  = model.load_performance_curves("lp-sec1-caso-a", eff_cutoff=0.70)
    splines = model.build_splines(curves)
    return splines

splines = load_curves()
SPEEDS  = sorted(splines.keys())
EFF_BEP = splines[N_FIXED]["eff_bep"]

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULTS = dict(
    surge_margin=10, use_rp=True, r_p_required=3.5,
    p_suc_bar=4.08, t_suc_c=34,
    yrs_decline1=5, yrs_plateau2=5, load_plateau2=75,
    decay_rate=1.5, overhaul_interval=5,
    overhaul_recovery=45, degradation_floor=85,
    elec_price_base=0.08, elec_escalation=3.0,
    hours_per_year=8000, discount_rate=8,
    vfd_usd_kw=250, vfd_install_f=1.30,
)

def _init():
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)
    for c, v in model.COMPOSITION_Y1_DEFAULT.items():
        st.session_state.setdefault(f"y1_{c}", v)
    for c, v in model.COMPOSITION_Y20_DEFAULT.items():
        st.session_state.setdefault(f"y20_{c}", v)
    st.session_state.setdefault("results", None)
    st.session_state.setdefault("dirty", False)

def _reset():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    for c, v in model.COMPOSITION_Y1_DEFAULT.items():
        st.session_state[f"y1_{c}"] = v
    for c, v in model.COMPOSITION_Y20_DEFAULT.items():
        st.session_state[f"y20_{c}"] = v
    st.session_state["results"] = None
    st.session_state["dirty"]   = False

def _mark_dirty():
    st.session_state["dirty"] = True

_init()

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("⚙️ LP Centrifugal Compressor — VFD Investment Analysis")
st.markdown(
    "Curve: `lp-sec1-caso-a` (Petrobras/ccp, Apache 2.0) · "
    "N_fixed = 9,831 RPM · PR EOS + ChemSep BIPs · 20-year techno-economic model  \n"
    "📄 [Read the article (LinkedIn)](https://www.linkedin.com/pulse/python-extended-simulation-centrifugal-gas-compressor-lopez-andreu-dzc1f/)  ·  "
    "💻 [Source code (GitHub)](https://github.com/malandreu/compressor-vfd-analysis)"
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_in, tab_map, tab_prod, tab_vfd, tab_econ = st.tabs([
    "⚙️ Inputs",
    "📈 Compressor Map",
    "📊 Production & Recycle",
    "🎛️ VFD Strategy",
    "💰 Economics",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — INPUTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_in:

    # Run / Reset bar
    rc1, rc2, rc3 = st.columns([2, 1, 5])
    is_running = st.session_state.get("running", False)
    run_top    = rc1.button("⏳ Calculating…" if is_running else "▶ Run Analysis",
                             type="primary", use_container_width=True,
                             disabled=is_running)
    reset_btn  = rc2.button("↺ Reset", use_container_width=True, disabled=is_running)
    if reset_btn:
        _reset(); st.rerun()

    if is_running:
        rc3.warning("⏳ **Calculating…** Please wait.")
    elif st.session_state["dirty"] and st.session_state["results"] is not None:
        rc3.warning("⚠️ Inputs changed — press **Run Analysis** to update.")
    elif st.session_state["results"] is not None:
        rc3.success("✅ Results up to date.")
    else:
        rc3.info("ℹ️ Set inputs and press **Run Analysis**.")

    st.divider()

    # ── A ─────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">A · Compressor Settings & Operating Conditions</div>',
                unsafe_allow_html=True)
    st.caption(f"Fixed speed reference: **{N_FIXED:.0f} RPM** (design BEP — not adjustable for this curve)")

    ca1, ca2, ca3, ca4 = st.columns(4)
    surge_margin_pct = ca1.slider(
        "Surge margin [%]", 0, 20, st.session_state["surge_margin"], 1,
        key="surge_margin", on_change=_mark_dirty)

    with ca2:
        use_rp = st.toggle(
            "Discharge pressure constraint",
            value=st.session_state["use_rp"], key="use_rp", on_change=_mark_dirty)
        if use_rp:
            r_p_required = st.slider(
                f"Compression ratio [-]  (max: {R_P_MAX})",
                2.0, R_P_MAX, float(st.session_state["r_p_required"]), 0.1,
                key="r_p_required", on_change=_mark_dirty)
        else:
            r_p_required = None
            st.caption("None → pure antisurge model")

    p_suc_bar = ca3.slider(
        "Suction pressure [bar abs]", 2.5, 5.0,
        float(st.session_state["p_suc_bar"]), 0.01,
        key="p_suc_bar", on_change=_mark_dirty)

    t_suc_c = ca4.slider(
        "Suction temperature [°C]", 15, 50,
        int(st.session_state["t_suc_c"]), 1,
        key="t_suc_c", on_change=_mark_dirty)

    surge_margin = surge_margin_pct / 100.0

    st.divider()

    # ── B ─────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">B · Gas Composition</div>',
                unsafe_allow_html=True)
    st.caption("Adjust mole fractions. Press Normalize to enforce sum = 1. "
               "Gas properties update instantly.")

    def comp_block(label, prefix, defaults):
        st.markdown(f"**{label}**")
        vals = {}
        cols = st.columns(2)
        for idx, (comp, dflt) in enumerate(defaults.items()):
            with cols[idx % 2]:
                v = st.number_input(
                    comp, 0.0, 1.0,
                    float(st.session_state.get(f"{prefix}_{comp}", dflt)),
                    step=0.001, format="%.5f",
                    key=f"{prefix}_{comp}", on_change=_mark_dirty)
                vals[comp] = v
        total = sum(vals.values())
        nb1, nb2 = st.columns([1, 2])
        if nb1.button("Normalize", key=f"norm_{prefix}"):
            if total > 0:
                for c in vals:
                    st.session_state[f"{prefix}_{c}"] = vals[c] / total
                _mark_dirty(); st.rerun()
        diff = abs(total - 1.0)
        if diff > 1e-4:
            nb2.warning(f"Sum = {total:.5f}  (Δ = {diff:.5f})")
        else:
            nb2.success(f"Sum = {total:.5f} ✓")
        return vals

    bca, bcb = st.columns(2)
    with bca:
        comp_y1_raw  = comp_block("Year 1 (design)", "y1",  model.COMPOSITION_Y1_DEFAULT)
    with bcb:
        comp_y20_raw = comp_block("Year 20 (field depletion)", "y20", model.COMPOSITION_Y20_DEFAULT)

    def _norm(d):
        t = sum(d.values())
        return {k: v/t for k, v in d.items()} if t > 0 else d

    comp_y1  = _norm(comp_y1_raw)
    comp_y20 = _norm(comp_y20_raw)

    # Real-time gas properties
    st.markdown("**Gas properties** *(real-time — no model run required)*")
    try:
        props_y1  = model.gas_properties(comp_y1,  p_suc_bar, t_suc_c)
        props_y20 = model.gas_properties(comp_y20, p_suc_bar, t_suc_c)
        gp1, gp2 = st.columns(2)
        gp1.table(pd.DataFrame(props_y1.items(),  columns=["Property","Year 1"]).set_index("Property"))
        gp2.table(pd.DataFrame(props_y20.items(), columns=["Property","Year 20"]).set_index("Property"))
        mw1, mw20 = props_y1["MW [g/mol]"], props_y20["MW [g/mol]"]
        st.markdown(
            f'<div class="derived-info">MW drift: <b>{mw1:.3f} → {mw20:.3f} g/mol</b>'
            f' (+{(mw20/mw1-1)*100:.1f}%)  ·  '
            f'ρ: {props_y1["ρ [kg/m³]"]:.4f} → {props_y20["ρ [kg/m³]"]:.4f} kg/m³</div>',
            unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Gas properties error: {e}")

    st.divider()

    # ── C ─────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">C · Production Profile</div>',
                unsafe_allow_html=True)

    # Use physical load_min for UI preview (fast, no EOS call)
    load_min_frac, Q_as_nmin, Q_design = model.compute_load_min(
        splines, surge_margin, N_FIXED)
    load_min_pct = max(int(np.ceil(load_min_frac * 100)), 1)

    cc1, cc2, cc3, cc4 = st.columns(4)

    with cc1:
        st.markdown("**Decline 1 [years]**")
        d1s = st.slider("d1_sl", 2, 10, int(st.session_state["yrs_decline1"]),
                         1, label_visibility="collapsed",
                         key="d1_sl", on_change=_mark_dirty)
        yrs_decline1 = st.number_input(
            "d1_ni", 2, 10, d1s, 1,
            label_visibility="collapsed", key="yrs_decline1",
            on_change=_mark_dirty)

    with cc2:
        st.markdown("**Plateau 2 [years]**")
        p2s = st.slider("p2_sl", 2, 10, int(st.session_state["yrs_plateau2"]),
                         1, label_visibility="collapsed",
                         key="p2_sl", on_change=_mark_dirty)
        yrs_plateau2 = st.number_input(
            "p2_ni", 2, 10, p2s, 1,
            label_visibility="collapsed", key="yrs_plateau2",
            on_change=_mark_dirty)

    with cc3:
        st.markdown(f"**Plateau 2 load [%]** (min {load_min_pct+1}%)")
        lp2_pct = st.slider(
            "lp2_sl", load_min_pct+1, 95,
            int(max(st.session_state["load_plateau2"], load_min_pct+1)),
            1, label_visibility="collapsed",
            key="load_plateau2", on_change=_mark_dirty)
        st.number_input("lp2_ni", float(load_min_pct+1), 95.0,
                         float(lp2_pct), 1.0,
                         label_visibility="collapsed", disabled=True)
        load_plateau2 = lp2_pct / 100.0

    with cc4:
        yrs_decline2 = 20 - 5 - yrs_decline1 - yrs_plateau2
        st.markdown("**Decline 2 [years]** *(derived)*")
        st.markdown(
            f'<div class="derived-info" style="margin-top:8px">'
            f'<b>{yrs_decline2}</b> years<br>'
            f'(5 + {yrs_decline1} + {yrs_plateau2} + {yrs_decline2} = 20)</div>',
            unsafe_allow_html=True)

    if yrs_decline2 < 1:
        st.error("Decline 2 < 1 year — reduce Decline 1 or Plateau 2.")

    # Preview chart
    _loads_p = model.build_load_profile(
        20, 5, yrs_decline1, load_plateau2, yrs_plateau2,
        max(yrs_decline2, 1), load_min_frac)
    fig_p = go.Figure()
    fig_p.add_scatter(x=np.arange(1,21), y=_loads_p*100,
                      mode="lines+markers", fill="tozeroy",
                      fillcolor="rgba(52,152,219,0.1)",
                      line=dict(color="#3498db", width=2))
    fig_p.add_hline(y=load_min_frac*100,
                    line=dict(color="red", dash="dash"),
                    annotation_text=f"LOAD_MIN {load_min_frac*100:.0f}%")
    fig_p.update_layout(height=200, showlegend=False,
                         margin=dict(t=5,b=30,l=40,r=5),
                         yaxis_title="Load [%]", xaxis_title="Year")
    st.plotly_chart(fig_p, use_container_width=True)

    st.divider()

    # ── D ─────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">D · Performance Degradation Model</div>',
                unsafe_allow_html=True)
    cd1, cd2, cd3, cd4 = st.columns(4)

    decay_rate_pct = cd1.slider(
        "Efficiency decay rate [%/yr]", 0.5, 3.0,
        float(st.session_state["decay_rate"]), 0.1,
        key="decay_rate", on_change=_mark_dirty)
    decay_rate = decay_rate_pct / 100.0

    overhaul_interval = cd2.slider(
        "Overhaul interval [years]", 1, 10,
        int(st.session_state["overhaul_interval"]), 1,
        key="overhaul_interval", on_change=_mark_dirty)

    oh_rec_pct = cd3.slider(
        "Recovery at overhaul [%]", 0, 100,
        int(st.session_state["overhaul_recovery"]), 5,
        key="overhaul_recovery", on_change=_mark_dirty)
    overhaul_recovery = oh_rec_pct / 100.0

    deg_floor_pct = cd4.slider(
        "Min allowable efficiency [% of design]", 50, 90,
        int(st.session_state["degradation_floor"]), 1,
        key="degradation_floor", on_change=_mark_dirty)
    degradation_floor = deg_floor_pct / 100.0

    prev = model.degradation_preview(decay_rate, overhaul_interval,
                                      overhaul_recovery, EFF_BEP)
    st.markdown(
        f'<div class="derived-info">'
        f'→ Poly. efficiency at overhaul: <b>{prev["eta_bop"]:.1f}% → {prev["eta_eop"]:.1f}%</b>'
        f' (−{prev["loss_pct"]:.1f}%)<br>'
        f'→ Recovered: {prev["rec_pct"]:.1f}%  ·  '
        f'Permanent loss/cycle: {prev["perm_pct"]:.1f}%</div>',
        unsafe_allow_html=True)

    st.divider()

    # ── E ─────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">E · Economics</div>',
                unsafe_allow_html=True)
    ce1, ce2, ce3 = st.columns(3)

    with ce1:
        elec_price_base = st.slider(
            "Electricity price [USD/kWh]", 0.03, 0.25,
            float(st.session_state["elec_price_base"]), 0.01,
            key="elec_price_base", on_change=_mark_dirty)
        elec_esc_pct = st.slider(
            "Annual escalation [%/yr] (linear)", 0.0, 10.0,
            float(st.session_state["elec_escalation"]), 0.5,
            key="elec_escalation", on_change=_mark_dirty)
        elec_escalation = elec_esc_pct / 100.0

    with ce2:
        hours_per_year = st.slider(
            "Operating hours [h/yr]", 1000, 8760,
            int(st.session_state["hours_per_year"]), 100,
            key="hours_per_year", on_change=_mark_dirty)
        disc_pct = st.slider(
            "Discount rate [%]", 5, 20,
            int(st.session_state["discount_rate"]), 1,
            key="discount_rate", on_change=_mark_dirty)
        discount_rate = disc_pct / 100.0

    with ce3:
        vfd_usd_kw = st.slider(
            "VFD installed CAPEX [USD/kW]", 150, 400,
            int(st.session_state["vfd_usd_kw"]), 10,
            key="vfd_usd_kw", on_change=_mark_dirty)
        vfd_install_f = st.slider(
            "Installation factor [-]", 1.1, 1.6,
            float(st.session_state["vfd_install_f"]), 0.05,
            key="vfd_install_f", on_change=_mark_dirty)

    st.divider()
    _is_running2 = st.session_state.get("running", False)
    run_bot = st.columns([2, 5])[0].button(
        "⏳ Calculating…" if _is_running2 else "▶ Run Analysis",
        type="primary", use_container_width=True,
        key="run_bot", disabled=_is_running2)

# ══════════════════════════════════════════════════════════════════════════════
# RUN MODEL
# ══════════════════════════════════════════════════════════════════════════════
if run_top or run_bot:
    if yrs_decline2 < 1:
        st.error("Cannot run: Decline 2 < 1 year.")
    else:
        st.session_state["running"] = True
        st.rerun()

if st.session_state.get("running", False):
    model.clear_caches()

    with st.spinner("Setting up thermodynamics…"):
        ctx = model.setup_thermo(comp_y1, p_suc_bar, t_suc_c)

    with st.spinner("Computing physical turndown limit…"):
        lmf_std, Q_std_design, _, _ = model.compute_load_min_std(
            splines, surge_margin, N_FIXED, ctx,
            comp_y1, comp_y20,
            project_life=20,
            yrs_plateau1=5, yrs_decline1=yrs_decline1,
            load_plateau2=load_plateau2,
            yrs_plateau2=yrs_plateau2, yrs_decline2=yrs_decline2)

    load_profile = model.build_load_profile(
        20, 5, yrs_decline1, load_plateau2,
        yrs_plateau2, yrs_decline2, lmf_std)
    comp_profile = model.build_composition_profile(load_profile, comp_y1, comp_y20)
    eta_profile  = model.build_eta_profile(
        20, decay_rate, overhaul_interval,
        overhaul_recovery, degradation_floor)
    elec_prices  = model.build_elec_prices(20, elec_price_base, elec_escalation)

    with st.spinner("Running 20-year model…"):
        fc = model.compute_annual_cashflows(
            splines, ctx, load_profile, comp_profile,
            eta_profile, elec_prices, surge_margin,
            r_p_required, N_FIXED, overhaul_interval,
            hours_per_year, discount_rate, vfd_usd_kw, vfd_install_f,
            Q_std_design=Q_std_design)
    st.session_state["results"] = {
        "fc": fc, "ctx": ctx,
        "load_profile": load_profile,
        "eta_profile": eta_profile,
        "elec_prices": elec_prices,
    }
    st.session_state["dirty"]   = False
    st.session_state["running"] = False
    st.rerun()

# ── Unpack or stop ────────────────────────────────────────────────────────────
results = st.session_state.get("results")
if results is None:
    for _t in [tab_map, tab_prod, tab_vfd, tab_econ]:
        with _t:
            st.info("ℹ️ Go to **⚙️ Inputs** and press **Run Analysis**.")
    st.stop()

fc           = results["fc"]
ctx          = results["ctx"]
load_profile = results["load_profile"]
eta_profile  = results["eta_profile"]
elec_prices  = results["elec_prices"]
df_fc        = fc["df"]
years_arr    = np.arange(1, 21)
oh_years     = [y for y in years_arr
                if ((y-1) % overhaul_interval == 0) and y > 1]

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — COMPRESSOR MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab_map:
    st.subheader("Compressor Performance Map")
    st.caption("All 20 annual operating points shown · ★ = BEP")

    # Surge interpolation
    Q_sp = [splines[N]["Q_surge"] for N in SPEEDS]
    H_sp = [splines[N]["head_surge"] for N in SPEEDS]
    Q_ap = [splines[N]["Q_surge"]*(1+surge_margin) for N in SPEEDS]
    H_ap = [float(splines[N]["cs_head"](splines[N]["Q_surge"]*(1+surge_margin)))
            for N in SPEEDS]
    Nf   = np.linspace(SPEEDS[0], SPEEDS[-1], 200)
    csq  = _interp1d(SPEEDS, Q_sp, kind="cubic")
    csh  = _interp1d(SPEEDS, H_sp, kind="cubic")
    csaq = _interp1d(SPEEDS, Q_ap, kind="cubic")
    csah = _interp1d(SPEEDS, H_ap, kind="cubic")

    fig_map = go.Figure()

    for N, col in zip(SPEEDS, COLORS):
        s = splines[N]
        Qp = np.linspace(s["Q_surge"], s["Q_max"], 200)
        fig_map.add_scatter(x=Qp, y=s["cs_head"](Qp),
                            mode="lines", name=f"{N:.0f} RPM",
                            line=dict(color=col, width=2))
        fig_map.add_scatter(
            x=[s["Q_bep"]], y=[float(s["cs_head"](s["Q_bep"]))],
            mode="markers", showlegend=False,
            marker=dict(color=col, size=10, symbol="star",
                        line=dict(color="white", width=1)))

    fig_map.add_scatter(x=csq(Nf), y=csh(Nf), mode="lines",
                        name="Surge limit",
                        line=dict(color="red", dash="dash", width=2))
    fig_map.add_scatter(x=csaq(Nf), y=csah(Nf), mode="lines",
                        name=f"Anti-surge (+{surge_margin*100:.0f}%)",
                        line=dict(color="orange", dash="dot", width=1.5))

    # Fixed speed — all 20 points
    Qa, Ha, Ya = [], [], []
    for _, row in df_fc.iterrows():
        Qtot = row["Q_process"] + row["Q_recycle_fixed"]
        try:
            h, _ = model._get_head_eff(splines, Qtot, N_FIXED)
            Qa.append(Qtot); Ha.append(h); Ya.append(int(row["year"]))
        except Exception:
            pass
    if Qa:
        fig_map.add_scatter(x=Qa, y=Ha, mode="lines+markers",
                            name="Fixed speed (Q_total)",
                            line=dict(color="#2c3e50", width=2),
                            marker=dict(color="#2c3e50", size=6),
                            text=[f"Yr {y}" for y in Ya],
                            hovertemplate="Fixed Yr%{text}<br>Q=%{x:.0f}<br>H=%{y:.1f}<extra></extra>")

    # VFD — all 20 points coloured by constraint
    for cst, cc in CONSTRAINT_COLORS.items():
        mask = df_fc["constraint"] == cst
        if not mask.any():
            continue
        Qb, Hb, Yb = [], [], []
        for _, row in df_fc[mask].iterrows():
            Qtot = row["Q_process"] + row["Q_recycle_vfd"]
            try:
                h, _ = model._get_head_eff(splines, Qtot, row["N_vfd"])
                Qb.append(Qtot); Hb.append(h); Yb.append(int(row["year"]))
            except Exception:
                pass
        if Qb:
            fig_map.add_scatter(x=Qb, y=Hb, mode="markers",
                                name=f"VFD ({cst})",
                                marker=dict(color=cc, size=9, symbol="circle",
                                            line=dict(color="white", width=1)),
                                text=[f"Yr {y}" for y in Yb],
                                hovertemplate="VFD %{text}<br>Q=%{x:.0f}<br>H=%{y:.1f}<extra></extra>")

    fig_map.update_layout(
        height=430, xaxis_title="Inlet flow [m³/h]",
        yaxis_title="Polytropic head [kJ/kg]",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=20, b=10, l=10, r=10))
    st.plotly_chart(fig_map, use_container_width=True)

    st.divider()
    st.subheader("Polytropic Efficiency over Project Life")

    eta_abs = eta_profile * EFF_BEP * 100
    fig_eta = go.Figure()
    fig_eta.add_scatter(x=years_arr, y=eta_abs, mode="lines+markers",
                        name="Absolute polytropic efficiency",
                        line=dict(color="#9b59b6", width=2),
                        marker=dict(size=6))
    fig_eta.add_scatter(x=years_arr, y=eta_profile*100, mode="lines",
                        name="eta_factor × 100 (relative)",
                        line=dict(color="#3498db", width=1.5, dash="dot"))
    fig_eta.add_hline(y=degradation_floor * EFF_BEP * 100,
                      line=dict(color="red", dash="dash"),
                      annotation_text="Mandatory replacement floor")
    for oh in oh_years:
        fig_eta.add_vline(x=oh, line=dict(color="#ccc", dash="dot", width=1))
        fig_eta.add_annotation(x=oh, y=eta_abs.max()*1.015,
                                text="OH", showarrow=False,
                                font=dict(size=9, color="#999"))
    fig_eta.update_layout(
        height=280, yaxis_title="Polytropic efficiency [%]",
        xaxis_title="Project year",
        xaxis=dict(tickvals=list(range(1,21))),
        legend=dict(orientation="h", y=-0.28),
        margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig_eta, use_container_width=True)
    st.caption("OH marker = year after overhaul (when efficiency is restored).")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PRODUCTION & RECYCLE
# ══════════════════════════════════════════════════════════════════════════════
with tab_prod:
    st.subheader("Production Profile & Gas Compression")

    fig_prod = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=(
            "Production load profile",
            "Produced Gas Flow & Total Compressed Gas [m³/h]",
            "Recycle fraction — Fixed speed vs VFD",
        ),
        vertical_spacing=0.08)

    # P1 — load %
    fig_prod.add_scatter(row=1, col=1, x=years_arr, y=load_profile*100,
                         mode="lines+markers", name="Load %",
                         line=dict(color="#3498db", width=2),
                         fill="tozeroy", fillcolor="rgba(52,152,219,0.1)")
    fig_prod.add_hline(y=fc["load_min"]*100, row=1, col=1,
                       line=dict(color="red", dash="dash"),
                       annotation_text=f"LOAD_MIN {fc['load_min']*100:.0f}%")

    # P2 — flows
    Q_proc  = df_fc["Q_process"]
    Q_tot_A = df_fc["Q_process"] + df_fc["Q_recycle_fixed"]
    Q_tot_B = df_fc["Q_process"] + df_fc["Q_recycle_vfd"]

    fig_prod.add_scatter(row=2, col=1, x=df_fc["year"], y=Q_proc,
                         mode="lines+markers", name="Produced Gas Flow",
                         line=dict(color="#3498db", width=2.5),
                         fill="tozeroy", fillcolor="rgba(52,152,219,0.08)")
    fig_prod.add_scatter(row=2, col=1, x=df_fc["year"], y=Q_tot_A,
                         mode="lines+markers", name="Total compressed (Fixed speed)",
                         line=dict(color="#2c3e50", width=1.5, dash="dot"))
    fig_prod.add_scatter(row=2, col=1, x=df_fc["year"], y=Q_tot_B,
                         mode="lines+markers", name="Total compressed (VFD)",
                         line=dict(color="#e67e22", width=1.5, dash="dash"))
    fig_prod.add_hline(y=fc["Q_as_nmin"], row=2, col=1,
                       line=dict(color="red", dash="dash"),
                       annotation_text=f"Q_antisurge(N_min) {fc['Q_as_nmin']:.0f} m³/h (actual)")

    # P3 — recycle fraction
    frac_A = (df_fc["Q_recycle_fixed"] / Q_tot_A * 100).fillna(0)
    frac_B = (df_fc["Q_recycle_vfd"] / Q_tot_B.clip(lower=1) * 100).fillna(0)

    fig_prod.add_scatter(row=3, col=1, x=df_fc["year"], y=frac_A,
                         mode="lines+markers", name="Fixed speed recycle %",
                         line=dict(color="#2c3e50", width=2))
    for cst, cc in CONSTRAINT_COLORS.items():
        mask = df_fc["constraint"] == cst
        if mask.any():
            fig_prod.add_scatter(row=3, col=1,
                                 x=df_fc.loc[mask, "year"],
                                 y=frac_B[mask], mode="markers",
                                 name=f"VFD ({cst})",
                                 marker=dict(color=cc, size=8))

    fig_prod.update_yaxes(title_text="Load [%]",   row=1, col=1)
    fig_prod.update_yaxes(title_text="Flow [m³/h]", row=2, col=1)
    fig_prod.update_yaxes(title_text="Recycle [%]", row=3, col=1)
    fig_prod.update_xaxes(title_text="Project year", row=3, col=1,
                           tickvals=list(range(1, 21)))
    fig_prod.update_layout(height=720,
                            legend=dict(orientation="h", y=-0.06),
                            margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(fig_prod, use_container_width=True)
    st.caption(
        "**Produced Gas Flow** = reservoir output.  "
        "**Total compressed** = produced + recycle.  "
        "Fixed speed must compress additional gas to avoid surge.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — VFD STRATEGY
# ══════════════════════════════════════════════════════════════════════════════
with tab_vfd:
    rp_lbl = ("None (antisurge)" if r_p_required is None else f"{r_p_required:.2f}")
    pd_lbl = ("N/A" if r_p_required is None
               else f"{r_p_required*p_suc_bar:.2f} bar abs")
    st.subheader("VFD Operating Strategy — Dual Constraint Analysis")
    st.caption(f"Compression ratio = **{rp_lbl}**  ·  P_disch required = **{pd_lbl}**")

    fig_vfd = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=(
            "VFD operating speed by year — binding constraint",
            "Annual energy saving by constraint [MM USD]",
        ),
        vertical_spacing=0.10)

    # P1 — Line chart coloured by constraint
    for cst, cc in CONSTRAINT_COLORS.items():
        mask = df_fc["constraint"] == cst
        if not mask.any():
            continue
        fig_vfd.add_scatter(row=1, col=1,
                            x=df_fc.loc[mask, "year"],
                            y=df_fc.loc[mask, "N_vfd"],
                            mode="markers+lines", name=cst,
                            line=dict(color=cc, width=1),
                            marker=dict(color=cc, size=9))

    fig_vfd.add_hline(y=N_FIXED, row=1, col=1,
                      line=dict(color="#2c3e50", dash="dash", width=1.5),
                      annotation_text=f"Fixed {N_FIXED:.0f} RPM")
    fig_vfd.add_hline(y=SPEEDS[0], row=1, col=1,
                      line=dict(color="gray", dash="dot", width=1),
                      annotation_text=f"N_min {SPEEDS[0]:.0f} RPM")

    # P2 — Saving bars (MM USD)
    for cst, cc in CONSTRAINT_COLORS.items():
        mask = df_fc["constraint"] == cst
        if mask.any():
            fig_vfd.add_bar(row=2, col=1,
                            x=df_fc.loc[mask, "year"],
                            y=df_fc.loc[mask, "saving_USD"]/1e6,
                            name=cst, marker_color=cc,
                            opacity=0.85, showlegend=False)

    fig_vfd.add_vline(x=fc["payback"]+0.5, row=2, col=1,
                      line=dict(color="red", dash="dash"),
                      annotation_text=f"Payback yr {fc['payback']:.1f}")

    fig_vfd.update_yaxes(title_text="VFD speed [RPM]",     row=1, col=1)
    fig_vfd.update_yaxes(title_text="Saving [MM USD/yr]",  row=2, col=1)
    fig_vfd.update_xaxes(title_text="Project year", row=2, col=1,
                          tickvals=list(range(1, 21)))
    fig_vfd.update_layout(height=550, barmode="stack",
                           legend=dict(orientation="h", y=1.05),
                           margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(fig_vfd, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("🟢 Antisurge years", fc["n_antisurge"])
    m2.metric("🟠 Pressure years",  fc["n_pressure"])
    m3.metric("🔴 N_max+recycle",   fc["n_nmax"])

    if r_p_required is not None and fc["n_antisurge"] > 0:
        s_as  = df_fc[df_fc["constraint"]=="antisurge"]["saving_USD"].sum()
        s_tot = df_fc["saving_USD"].sum()
        pct   = (s_as - s_tot) / (s_as + 1e-9) * 100
        st.markdown(
            f'<div class="warn-info">⚠️ Saving lost vs unconstrained model: '
            f'<b>{pct:.1f}%</b></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ECONOMICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_econ:
    st.subheader("Economic Analysis — VFD Investment over 20 Years")

    def kpi(label, value, sub=""):
        return (f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{value}</div>'
                f'<div class="kpi-sub">{sub}</div></div>')

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi("VFD CAPEX",
                    f"{fc['capex_vfd']/1e6:.2f} MM$",
                    f"{fc['kW_peak']:.0f} kW × {vfd_usd_kw} × {vfd_install_f:.2f}"),
                unsafe_allow_html=True)
    k2.markdown(kpi("20-year NPV",
                    f"{fc['npv_total']/1e6:+.2f} MM$",
                    f"Discount rate {disc_pct:.0f}%"),
                unsafe_allow_html=True)
    irr_v = fc["irr"]*100 if np.isfinite(fc["irr"]) else float("nan")
    k3.markdown(kpi("IRR",
                    f"{irr_v:.1f}%" if np.isfinite(irr_v) else "N/A",
                    "Internal rate of return"),
                unsafe_allow_html=True)
    pb = fc["payback"]
    k4.markdown(kpi("Simple payback",
                    f"{pb:.1f} yr" if np.isfinite(pb) else ">20 yr",
                    f"Discounted: {fc['payback_disc']:.1f} yr"
                    if np.isfinite(fc["payback_disc"]) else "Discounted: >20 yr"),
                unsafe_allow_html=True)

    st.divider()

    if not df_fc.empty:
        ap = df_fc["year"]

        # P1 built separately to avoid secondary_y label overlap in subplots
        fig_p1 = make_subplots(specs=[[{"secondary_y": True}]])
        fig_p1.add_scatter(x=ap, y=df_fc["load_pct"],
                           mode="lines", name="Load %", fill="tozeroy",
                           fillcolor="rgba(52,152,219,0.15)",
                           line=dict(color="#3498db", width=2),
                           secondary_y=False)
        fig_p1.add_scatter(x=ap, y=df_fc["elec_price"],
                           mode="lines", name="Electricity price [USD/kWh]",
                           line=dict(color="#e67e22", dash="dash", width=2),
                           secondary_y=True)
        fig_p1.update_yaxes(title_text="Load [%]", secondary_y=False)
        fig_p1.update_yaxes(title_text="Electricity [USD/kWh]",
                            secondary_y=True, showgrid=False)
        fig_p1.update_xaxes(title_text="Project year")
        fig_p1.update_layout(height=260, title_text="Load profile & electricity price",
                              legend=dict(orientation="h", y=-0.3),
                              margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_p1, use_container_width=True)

        # ── Panel 2: Annual saving ───────────────────────────────────────────
        fig_saving = go.Figure()
        for cst, cc in CONSTRAINT_COLORS.items():
            mask = df_fc["constraint"] == cst
            if mask.any():
                fig_saving.add_bar(x=df_fc.loc[mask,"year"],
                                   y=df_fc.loc[mask,"saving_USD"]/1e6,
                                   name=cst, marker_color=cc, opacity=0.85)
        fig_saving.add_vline(x=pb+0.5, line=dict(color="red", dash="dash"),
                             annotation_text=f"Payback yr {pb:.1f}")
        fig_saving.update_layout(
            barmode="stack", height=280,
            title_text="Annual saving [MM USD]",
            yaxis_title="Saving [MM USD/yr]",
            xaxis=dict(title="Project year", tickvals=list(range(1,21))),
            legend=dict(orientation="h", y=-0.3),
            margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig_saving, use_container_width=True)

        # ── Panel 3: Cumulative NPV ───────────────────────────────────────────
        npv_vals = df_fc["npv_cum"]/1e6
        fig_npv = go.Figure()
        fig_npv.add_scatter(x=ap, y=npv_vals,
                            mode="lines", name="Cumulative NPV",
                            line=dict(color="#2c3e50", width=2.5),
                            fill="tozeroy",
                            fillcolor="rgba(46,204,113,0.12)" if float(npv_vals.iloc[-1]) >= 0
                                       else "rgba(231,76,60,0.12)")
        fig_npv.add_hline(y=0, line=dict(color="gray", dash="dash"))
        if np.isfinite(fc["payback_disc"]):
            fig_npv.add_vline(x=fc["payback_disc"]+0.5,
                              line=dict(color="red", dash="dash"),
                              annotation_text=f"Breakeven yr {fc['payback_disc']:.1f}")
        fig_npv.update_layout(
            height=280,
            title_text="Cumulative discounted NPV [MM USD]",
            yaxis_title="NPV [MM USD]",
            xaxis=dict(title="Project year", tickvals=list(range(1,21))),
            margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig_npv, use_container_width=True)

        # ── Panel 4: Energy cost comparison ──────────────────────────────────
        fig_cost = go.Figure()
        fig_cost.add_scatter(x=ap, y=df_fc["cost_fixed"]/1e6,
                             mode="lines", name="Fixed speed",
                             line=dict(color="#e74c3c", width=2),
                             fill="tozeroy", fillcolor="rgba(231,76,60,0.1)")
        fig_cost.add_scatter(x=ap, y=df_fc["cost_vfd"]/1e6,
                             mode="lines", name="VFD",
                             line=dict(color="#2ecc71", width=2),
                             fill="tozeroy", fillcolor="rgba(46,204,113,0.1)")
        fig_cost.update_layout(
            height=280,
            title_text="Annual energy cost comparison [MM USD]",
            yaxis_title="Energy cost [MM USD/yr]",
            xaxis=dict(title="Project year", tickvals=list(range(1,21))),
            legend=dict(orientation="h", y=-0.3),
            margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig_cost, use_container_width=True)

    st.divider()
    with st.expander("📋 Annual results table"):
        if not df_fc.empty:
            df_d = df_fc[["year","load_pct","kW_fixed","kW_vfd",
                           "saving_kW","saving_USD","npv_cum",
                           "N_vfd","constraint","overhaul"]].copy()
            df_d["saving_USD"] = (df_d["saving_USD"]/1e6).round(3)
            df_d["npv_cum"]    = (df_d["npv_cum"]/1e6).round(3)
            df_d = df_d.rename(columns={
                "saving_USD": "saving [MM$]",
                "npv_cum":    "NPV_cum [MM$]"})
            df_d["load_pct"]  = df_d["load_pct"].round(1)
            for c in ["kW_fixed","kW_vfd","saving_kW","N_vfd"]:
                df_d[c] = df_d[c].round(0).astype(int)
            st.dataframe(df_d, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "**Data**: Petrobras/ccp (Apache 2.0) · "
    "**EOS**: Peng-Robinson + ChemSep BIPs via `thermo` (Caleb Bell, MIT) · "
    "**References**: Lapina (1982), Campbell TOTM, Kurz et al. (2020)  \n"
    "📄 [Article (LinkedIn)](https://www.linkedin.com/pulse/python-extended-simulation-centrifugal-gas-compressor-lopez-andreu-dzc1f/) · "
    "💻 [Notebook & code (GitHub)](https://github.com/malandreu/compressor-vfd-analysis)"
)
