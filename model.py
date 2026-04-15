"""
model.py — Centrifugal Compressor Calculation Engine
Aligned with Gas_Compressor_VFD_Analysis.ipynb (final published version)

Key change vs previous version:
  Production profile is defined in standard volumetric flow [Sm³/h] at
  15 °C / 1.01325 bara.  Each annual operating point is converted to
  actual suction conditions via PR-EOS before the compressor calculation.
"""

import numpy as np
import pandas as pd
import urllib.request
import io, contextlib, warnings
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import brentq
from thermo import ChemicalConstantsPackage, PRMIX, CEOSLiquid, CEOSGas, FlashVL
from thermo.interaction_parameters import IPDB

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
_BASE_URL  = "https://raw.githubusercontent.com/petrobras/ccp/main/ccp/tests/data/"
N_FIXED_DEFAULT    = 9831.0
EFF_CUTOFF_DEFAULT = 0.70
IRR_LOWER = 0.001
IRR_UPPER = 10.0
IRR_XTOL  = 1e-6
CROSS_TOL = 1e-12

# Standard conditions (ISO / field convention)
P_STD_BAR_DEFAULT = 1.01325   # bara
T_STD_C_DEFAULT   = 15.0      # °C

COMPONENTS_DEFAULT = [
    "methane", "ethane", "propane", "butane", "isobutane",
    "pentane", "isopentane", "nitrogen", "hydrogen sulfide", "CO2",
]
COMPOSITION_Y1_DEFAULT = {
    "methane": 0.58976, "ethane": 0.03099, "propane": 0.00600,
    "butane": 0.00080, "isobutane": 0.00050, "pentane": 0.00010,
    "isopentane": 0.00010, "nitrogen": 0.00550,
    "hydrogen sulfide": 0.00020, "CO2": 0.36605,
}
COMPOSITION_Y20_DEFAULT = {
    "methane": 0.54000, "ethane": 0.02500, "propane": 0.00800,
    "butane": 0.00150, "isobutane": 0.00100, "pentane": 0.00020,
    "isopentane": 0.00020, "nitrogen": 0.00550,
    "hydrogen sulfide": 0.00020, "CO2": 0.41840,
}

# ── Module-level caches ───────────────────────────────────────────────────────
_ANTISURGE_CACHE:  dict = {}
_OPERATING_CACHE:  dict = {}
_CASHFLOW_CACHE:   dict = {}

def clear_caches():
    _ANTISURGE_CACHE.clear()
    _OPERATING_CACHE.clear()
    _CASHFLOW_CACHE.clear()


# ══════════════════════════════════════════════════════════════════════════════
# 1. PERFORMANCE CURVES
# ══════════════════════════════════════════════════════════════════════════════

def load_performance_curves(curve_name: str, eff_cutoff: float = 0.70) -> dict:
    """Load head and efficiency curves from Petrobras/ccp GitHub."""
    result = {}
    for metric in ("head", "eff"):
        url = _BASE_URL + f"{curve_name}-{metric}.csv"
        with urllib.request.urlopen(url) as r:
            text = r.read().decode("utf-8")
        current_speed, qs, vals = None, [], []
        for line in text.strip().splitlines():
            parts = line.strip().split(",")
            if parts[0] == "x":
                if current_speed is not None:
                    _store_curve(result, current_speed, metric,
                                 np.array(qs), np.array(vals), eff_cutoff)
                current_speed = float(parts[1])
                qs, vals = [], []
            else:
                qs.append(float(parts[0]))
                vals.append(float(parts[1]))
        if current_speed is not None:
            _store_curve(result, current_speed, metric,
                         np.array(qs), np.array(vals), eff_cutoff)
    return result


def _store_curve(result, speed, metric, qs, vals, eff_cutoff):
    if speed not in result:
        result[speed] = {}
    if metric == "eff":
        mask = vals >= eff_cutoff
        qs, vals = qs[mask], vals[mask]
    elif metric == "head":
        keep = [0]
        for i in range(1, len(vals)):
            if vals[i] < vals[keep[-1]]:
                keep.append(i)
        qs, vals = qs[keep], vals[keep]
    result[speed][f"Q_{metric}"] = qs
    result[speed][metric] = vals


def build_splines(curves: dict) -> dict:
    """Fit cubic splines; identify BEP, surge, stonewall for each speed."""
    splines = {}
    for N, data in sorted(curves.items()):
        Q_head = data["Q_head"]
        Q_eff  = data["Q_eff"]
        cs_head = CubicSpline(Q_head, data["head"], extrapolate=False)
        cs_eff  = CubicSpline(Q_eff,  data["eff"],  extrapolate=False)
        idx_bep = np.argmax(data["eff"])
        splines[N] = {
            "cs_head":    cs_head,
            "cs_eff":     cs_eff,
            "Q_surge":    float(Q_head.min()),
            "Q_max":      float(Q_head.max()),
            "Q_bep":      float(Q_eff[idx_bep]),
            "eff_bep":    float(data["eff"][idx_bep]),
            "head_surge": float(cs_head(Q_head.min())),
        }
    return splines


# ══════════════════════════════════════════════════════════════════════════════
# 2. THERMODYNAMIC SETUP
# ══════════════════════════════════════════════════════════════════════════════

def setup_thermo(composition: dict, P_suc_bar: float, T_suc_C: float,
                 P_std_bar: float = P_STD_BAR_DEFAULT,
                 T_std_C:   float = T_STD_C_DEFAULT) -> dict:
    """
    Build thermodynamic context from composition and suction conditions.
    Includes standard-condition state for Sm³/h ↔ m³/h conversion.
    """
    components = list(composition.keys())
    z = list(composition.values())
    P_pa     = P_suc_bar * 1e5
    T_K      = T_suc_C  + 273.15
    P_std_pa = P_std_bar * 1e5
    T_std_K  = T_std_C  + 273.15

    consts, props = ChemicalConstantsPackage.from_IDs(components)
    kijs   = IPDB.get_ip_asymmetric_matrix("ChemSep PR", consts.CASs, "kij")
    eos_kw = dict(Tcs=consts.Tcs, Pcs=consts.Pcs, omegas=consts.omegas, kijs=kijs)
    liq    = CEOSLiquid(PRMIX, HeatCapacityGases=props.HeatCapacityGases, eos_kwargs=eos_kw)
    gas    = CEOSGas(PRMIX,  HeatCapacityGases=props.HeatCapacityGases, eos_kwargs=eos_kw)
    flasher = FlashVL(consts, props, liquid=liq, gas=gas)

    suc = flasher.flash(T=T_K, P=P_pa, zs=z)
    MW  = sum(z[i] * consts.MWs[i] for i in range(len(components)))

    return {
        "flasher":    flasher,
        "consts":     consts,
        "components": components,
        "z_design":   z,
        "MW_design":  MW,
        "Z_design":   suc.gas.Z(),
        "k_design":   suc.gas.Cp() / suc.gas.Cv(),
        "rho_design": suc.gas.rho_mass(),
        "h1_design":  suc.gas.H(),
        "P_suc_pa":   P_pa,
        "P_suc_bar":  P_suc_bar,
        "T_suc_K":    T_K,
        "T_suc_C":    T_suc_C,
        "P_std_pa":   P_std_pa,
        "P_std_bar":  P_std_bar,
        "T_std_K":    T_std_K,
        "T_std_C":    T_std_C,
    }


def _flash_Z(ctx: dict, T_K: float, P_pa: float, z: list) -> float:
    """Compressibility factor at given state."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        state = ctx["flasher"].flash(T=T_K, P=P_pa, zs=z)
    return float(state.gas.Z())


def q_std_from_suction(ctx: dict, Q_suc_m3h: float, z: list) -> float:
    """Convert actual suction flow [m³/h] → standard flow [Sm³/h]."""
    Z_suc = _flash_Z(ctx, ctx["T_suc_K"], ctx["P_suc_pa"], z)
    Z_std = _flash_Z(ctx, ctx["T_std_K"], ctx["P_std_pa"], z)
    return float(Q_suc_m3h
                 * (Z_std / Z_suc)
                 * (ctx["T_std_K"] / ctx["T_suc_K"])
                 * (ctx["P_suc_pa"] / ctx["P_std_pa"]))


def q_suction_from_std(ctx: dict, Q_std_Sm3h: float, z: list) -> float:
    """Convert standard flow [Sm³/h] → actual suction flow [m³/h]."""
    Z_suc = _flash_Z(ctx, ctx["T_suc_K"], ctx["P_suc_pa"], z)
    Z_std = _flash_Z(ctx, ctx["T_std_K"], ctx["P_std_pa"], z)
    return float(Q_std_Sm3h
                 * (Z_suc / Z_std)
                 * (ctx["T_suc_K"] / ctx["T_std_K"])
                 * (ctx["P_std_pa"] / ctx["P_suc_pa"]))


def gas_properties(composition: dict, P_suc_bar: float, T_suc_C: float) -> dict:
    """MW, Z, rho, k, h1 at suction conditions."""
    ctx = setup_thermo(composition, P_suc_bar, T_suc_C)
    return {
        "MW [g/mol]":  round(ctx["MW_design"],  3),
        "Z [-]":       round(ctx["Z_design"],    4),
        "ρ [kg/m³]":  round(ctx["rho_design"],  4),
        "k (Cp/Cv)":   round(ctx["k_design"],    4),
        "h₁ [J/mol]": round(ctx["h1_design"],   2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. CORE COMPUTE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _get_head_eff(splines: dict, Q_m3h: float, N_rpm: float):
    speeds = sorted(splines.keys())
    N_min, N_max = speeds[0], speeds[-1]
    if not (N_min <= N_rpm <= N_max):
        raise ValueError(f"N={N_rpm} out of range")
    if N_rpm in splines:
        s = splines[N_rpm]
        if not (s["Q_surge"] <= Q_m3h <= s["Q_max"]):
            raise ValueError(f"Q={Q_m3h:.0f} out of range at N={N_rpm:.0f}")
        return float(s["cs_head"](Q_m3h)), float(s["cs_eff"](Q_m3h))
    N_lo  = max(n for n in speeds if n <= N_rpm)
    N_hi  = min(n for n in speeds if n >= N_rpm)
    alpha = (N_rpm - N_lo) / (N_hi - N_lo)
    s_lo, s_hi = splines[N_lo], splines[N_hi]
    phi   = Q_m3h / N_rpm
    Q_lo  = np.clip(phi*N_lo, s_lo["Q_surge"], s_lo["Q_max"])
    Q_hi  = np.clip(phi*N_hi, s_hi["Q_surge"], s_hi["Q_max"])
    Q_sg  = (1-alpha)*s_lo["Q_surge"] + alpha*s_hi["Q_surge"]
    Q_mx  = (1-alpha)*s_lo["Q_max"]   + alpha*s_hi["Q_max"]
    if not (Q_sg <= Q_m3h <= Q_mx):
        raise ValueError(f"Q={Q_m3h:.0f} out of interpolated range")
    head = ((1-alpha)*float(s_lo["cs_head"](Q_lo))*(N_rpm/N_lo)**2
            + alpha  *float(s_hi["cs_head"](Q_hi))*(N_rpm/N_hi)**2)
    eta  = ((1-alpha)*float(s_lo["cs_eff"](Q_lo))
            + alpha  *float(s_hi["cs_eff"](Q_hi)))
    return head, eta


def _discharge_state(ctx, W_poly_Jkg, eta_poly,
                     h1_Jmol, MW_kgmol, Z_ref, k_ref, z_gas):
    h2    = h1_Jmol + (W_poly_Jkg / eta_poly) * MW_kgmol
    n_p   = (k_ref*eta_poly) / (k_ref*eta_poly - (k_ref-1))
    r_est = (1 + W_poly_Jkg*(n_p-1) /
             (n_p*Z_ref*8.314/MW_kgmol*ctx["T_suc_K"]))**(n_p/(n_p-1))
    P_est = ctx["P_suc_pa"] * r_est
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            d = ctx["flasher"].flash(H=h2, P=P_est, zs=z_gas)
        return d.T - 273.15, P_est/1e5
    except Exception:
        return ctx["T_suc_K"]*r_est**((n_p-1)/n_p) - 273.15, P_est/1e5


def compute_operating_point(splines, ctx, Q_m3h, N_rpm,
                             eta_factor=1.0, MW_override=None) -> dict:
    """Lapina (1982) simplified — design Z, k, rho."""
    try:
        head, eta = _get_head_eff(splines, Q_m3h, N_rpm)
    except ValueError as e:
        return {"is_valid": False, "reason": str(e)}
    eta *= eta_factor
    MW_use = MW_override if MW_override is not None else ctx["MW_design"]
    if MW_override is not None:
        head *= MW_use / ctx["MW_design"]
    W      = head * 1000.0
    m_dot  = (Q_m3h/3600.0) * ctx["rho_design"]
    kW     = m_dot * W / eta / 1000.0
    T_d, P_d = _discharge_state(ctx, W, eta, ctx["h1_design"],
                                 ctx["MW_design"]/1000.0,
                                 ctx["Z_design"], ctx["k_design"], ctx["z_design"])
    return {"is_valid": True, "Q_m3h": Q_m3h, "N_rpm": N_rpm,
            "head_kJkg": head, "eta_poly": eta,
            "power_kW": kW, "m_dot_kgs": m_dot,
            "P_disch_bar": P_d, "T_disch_C": T_d,
            "r_pressure": P_d/ctx["P_suc_bar"]}


def compute_operating_point_rigorous(splines, ctx, Q_m3h, N_rpm,
                                      z_gas, eta_factor=1.0) -> dict:
    """Rigorous EOS — full flash per call."""
    try:
        head, eta = _get_head_eff(splines, Q_m3h, N_rpm)
    except ValueError as e:
        return {"is_valid": False, "reason": str(e)}
    eta *= eta_factor
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        suc = ctx["flasher"].flash(T=ctx["T_suc_K"], P=ctx["P_suc_pa"], zs=z_gas)
    MW  = sum(z_gas[i]*ctx["consts"].MWs[i] for i in range(len(z_gas)))
    rho = suc.gas.rho_mass()
    h1  = suc.gas.H()
    Z   = suc.gas.Z()
    k   = suc.gas.Cp()/suc.gas.Cv()
    head *= MW / ctx["MW_design"]
    W   = head * 1000.0
    kW  = (Q_m3h/3600.0)*rho*W/eta/1000.0
    T_d, P_d = _discharge_state(ctx, W, eta, h1, MW/1000.0, Z, k, z_gas)
    return {"is_valid": True, "Q_m3h": Q_m3h, "N_rpm": N_rpm,
            "MW_real": MW, "rho_real": rho,
            "head_kJkg": head, "eta_poly": eta,
            "power_kW": kW, "m_dot_kgs": (Q_m3h/3600.0)*rho,
            "P_disch_bar": P_d, "T_disch_C": T_d,
            "r_pressure": P_d/ctx["P_suc_bar"]}


# ══════════════════════════════════════════════════════════════════════════════
# 4. ANTI-SURGE AND STRATEGY COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def _get_surge_flow(splines, N_rpm):
    speeds = sorted(splines.keys())
    if N_rpm in splines:
        return splines[N_rpm]["Q_surge"]
    N_lo  = max(n for n in speeds if n <= N_rpm)
    N_hi  = min(n for n in speeds if n >= N_rpm)
    alpha = (N_rpm-N_lo)/(N_hi-N_lo)
    return (1-alpha)*splines[N_lo]["Q_surge"] + alpha*splines[N_hi]["Q_surge"]


def _apply_antisurge(splines, ctx, Q_process, N_rpm, surge_margin, fn, **kw):
    Qs   = _get_surge_flow(splines, N_rpm) * (1+surge_margin)
    Qrec = max(0.0, Qs - Q_process)
    Qtot = Q_process + Qrec
    pto  = fn(splines, ctx, Qtot, N_rpm, **kw)
    if not pto["is_valid"]:
        return {"is_valid": False, "reason": pto.get("reason")}
    frac = Qrec/Qtot if Qtot > 0 else 0.0
    return {"is_valid": True,
            "Q_process": Q_process, "Q_recycle": Qrec, "Q_total": Qtot,
            "power_kW": pto["power_kW"],
            "power_recycle_kW": pto["power_kW"]*frac,
            "P_disch_bar": pto["P_disch_bar"],
            "T_disch_C": pto["T_disch_C"],
            "eta_poly": pto["eta_poly"]}


def _cached_antisurge(splines, ctx, Q_process, N_rpm, surge_margin,
                       eta_factor, z_gas, cache_tag):
    key = (cache_tag,
           round(Q_process,4), round(N_rpm,4), round(eta_factor,8),
           tuple(round(zi,10) for zi in z_gas) if z_gas is not None else None)
    if key not in _ANTISURGE_CACHE:
        if z_gas is not None:
            r = _apply_antisurge(splines, ctx, Q_process, N_rpm, surge_margin,
                                  compute_operating_point_rigorous,
                                  z_gas=z_gas, eta_factor=eta_factor)
        else:
            r = _apply_antisurge(splines, ctx, Q_process, N_rpm, surge_margin,
                                  compute_operating_point, eta_factor=eta_factor)
        _ANTISURGE_CACHE[key] = dict(r)
    return dict(_ANTISURGE_CACHE[key])


def compare_strategies(splines, ctx, Q_process, surge_margin,
                        r_p_required, N_fixed, eta_factor=1.0,
                        z_gas=None, cache_tag=()) -> dict:
    """Fixed speed + recycle (A) vs VFD (B). Q_process in actual m³/h."""
    speeds = sorted(splines.keys())
    Qsp = np.array([splines[N]["Q_surge"] for N in speeds])
    cs  = interp1d(speeds, Qsp, kind="cubic")
    Qamin = float(cs(speeds[0]))  * (1+surge_margin)
    Qamax = float(cs(speeds[-1])) * (1+surge_margin)

    A = _cached_antisurge(splines, ctx, Q_process, N_fixed, surge_margin,
                           eta_factor, z_gas, cache_tag)
    if not A["is_valid"]:
        return {"is_valid": False}

    # N_antisurge
    if   Q_process <= Qamin: N_as = speeds[0]
    elif Q_process >= Qamax: N_as = speeds[-1]
    else:
        N_as = brentq(lambda N: float(cs(N))*(1+surge_margin)-Q_process,
                      speeds[0], speeds[-1], xtol=1.0)

    # N_pressure
    if r_p_required is None:
        N_vfd, constraint, N_pr = N_as, "antisurge", N_as
    else:
        p_req = ctx["P_suc_bar"] * r_p_required
        _lc   = {}
        def _pd(N):
            k = round(N,4)
            if k not in _lc:
                r = _cached_antisurge(splines, ctx, Q_process, N, surge_margin,
                                       eta_factor, z_gas, cache_tag)
                v = float(r["P_disch_bar"]) if r["is_valid"] and r["P_disch_bar"] else 0.0
                _lc[k] = v if np.isfinite(v) else 0.0
            return _lc[k]

        pN = _pd(N_fixed); pn = _pd(speeds[0])
        if pN < p_req:
            N_pr, constraint = N_fixed, "N_max+recycle"
        elif pn >= p_req:
            N_pr, constraint = speeds[0], "antisurge"
        elif pN <= pn or not np.isfinite(pN-pn):
            N_pr, constraint = N_fixed, "pressure"
        else:
            try:
                N_pr = brentq(lambda N: _pd(N)-p_req, speeds[0], N_fixed, xtol=1.0)
                constraint = "pressure" if N_pr > N_as else "antisurge"
            except Exception:
                N_pr, constraint = N_fixed, "pressure"

        N_vfd = min(max(N_as, N_pr), N_fixed)
        if constraint != "N_max+recycle":
            constraint = "pressure" if N_pr > N_as else "antisurge"

    B = _cached_antisurge(splines, ctx, Q_process, N_vfd, surge_margin,
                           eta_factor, z_gas, cache_tag)
    if not B["is_valid"]:
        return {"is_valid": False}

    dkW = A["power_kW"] - B["power_kW"]
    return {
        "is_valid": True, "Q_process": Q_process,
        "A_N_rpm": N_fixed,   "A_power_kW": A["power_kW"],
        "A_Q_recycle": A["Q_recycle"], "A_Q_total": A["Q_total"],
        "A_P_disch": A["P_disch_bar"], "A_T_disch_C": A["T_disch_C"],
        "A_eta_poly": A["eta_poly"],
        "B_N_rpm": N_vfd,     "B_power_kW": B["power_kW"],
        "B_Q_recycle": B["Q_recycle"], "B_Q_total": B["Q_total"],
        "B_P_disch": B["P_disch_bar"], "B_T_disch_C": B["T_disch_C"],
        "B_eta_poly": B["eta_poly"],
        "saving_kW": dkW,
        "saving_pct": dkW/A["power_kW"]*100 if A["power_kW"]>0 else np.nan,
        "constraint": constraint,
        "N_antisurge": N_as,
        "N_pressure":  N_pr if r_p_required is not None else N_as,
        "P_disch_required": ctx["P_suc_bar"]*r_p_required if r_p_required else None,
    }


def compute_load_min(splines, surge_margin, N_fixed) -> tuple:
    """Physical turndown: (load_min_fraction, Q_antisurge_Nmin, Q_design)."""
    speeds    = sorted(splines.keys())
    Q_as_nmin = splines[speeds[0]]["Q_surge"] * (1+surge_margin)
    Q_design  = splines[N_fixed]["Q_bep"]
    return Q_as_nmin/Q_design, Q_as_nmin, Q_design


# ══════════════════════════════════════════════════════════════════════════════
# 5. PRODUCTION PROFILE, COMPOSITION, DEGRADATION
# ══════════════════════════════════════════════════════════════════════════════

def build_load_profile(project_life, yrs_plateau1, yrs_decline1,
                        load_plateau2, yrs_plateau2, yrs_decline2,
                        load_min) -> np.ndarray:
    """Fractional load profile on a standard-flow basis."""
    def _load(y):
        if y <= yrs_plateau1:
            return 1.0
        elif y <= yrs_plateau1 + yrs_decline1:
            t = y - yrs_plateau1
            k = -np.log(load_plateau2/1.0)/yrs_decline1
            return np.exp(-k*t)
        elif y <= yrs_plateau1 + yrs_decline1 + yrs_plateau2:
            return load_plateau2
        else:
            t = y - (yrs_plateau1+yrs_decline1+yrs_plateau2)
            k = -np.log(load_min/load_plateau2)/yrs_decline2
            return load_plateau2*np.exp(-k*t)
    return np.array([_load(y) for y in range(1, project_life+1)])


def _composition_from_cumulative(year, cum_prod, comp_y1, comp_y20):
    """Composition interpolated by cumulative production fraction."""
    components = list(comp_y1.keys())
    total = float(cum_prod[-1])
    start = float(cum_prod[0])
    if len(cum_prod) == 1 or total <= start:
        alpha = 0.0
    else:
        alpha = min(max((float(cum_prod[year-1])-start)/(total-start), 0.0), 1.0)
    z = [comp_y1[c] + alpha*(comp_y20[c]-comp_y1[c]) for c in components]
    s = sum(z)
    return {c: zi/s for c, zi in zip(components, z)}


def build_composition_profile(load_profile, comp_y1, comp_y20) -> list:
    """List of composition dicts, one per year."""
    cum = np.cumsum(load_profile)
    return [_composition_from_cumulative(y, cum, comp_y1, comp_y20)
            for y in range(1, len(load_profile)+1)]


def compute_load_min_std(splines, surge_margin, N_fixed,
                          ctx, comp_y1, comp_y20,
                          project_life=20,
                          yrs_plateau1=5, yrs_decline1=5,
                          load_plateau2=0.75, yrs_plateau2=5, yrs_decline2=5,
                          Q_std_design=None) -> tuple:
    """
    Compute LOAD_MIN on a standard-flow basis (iterative, as in the notebook).
    Returns (load_min_std, Q_std_design, Q_as_nmin_m3h, Q_design_m3h).
    """
    load_min_phys, Q_as_nmin, Q_design = compute_load_min(splines, surge_margin, N_fixed)

    # Design standard flow
    z_design = list(comp_y1.values())
    if Q_std_design is None:
        Q_std = q_std_from_suction(ctx, Q_design, z_design)
    else:
        Q_std = float(Q_std_design)

    # Iterative conversion of LOAD_MIN to standard-flow basis
    guess = load_min_phys
    for _ in range(6):
        _lp = build_load_profile(project_life, yrs_plateau1, yrs_decline1,
                                  load_plateau2, yrs_plateau2, yrs_decline2, guess)
        cum  = np.cumsum(_lp)
        comp_end = _composition_from_cumulative(project_life, cum, comp_y1, comp_y20)
        z_end    = list(comp_end.values())
        Q_std_min = q_std_from_suction(ctx, Q_as_nmin, z_end)
        new_guess = Q_std_min / Q_std
        if abs(new_guess - guess) < 1e-6:
            guess = new_guess
            break
        guess = new_guess

    return guess, Q_std, Q_as_nmin, Q_design


def build_eta_profile(project_life, decay_rate, overhaul_interval,
                       overhaul_recovery, degradation_floor) -> np.ndarray:
    def _eta(y):
        yi = (y-1) % overhaul_interval
        oh = (y-1) // overhaul_interval
        es = 1.0
        for _ in range(oh):
            ee = es * np.exp(-decay_rate*overhaul_interval)
            es = ee + overhaul_recovery*(es-ee)
        return max(degradation_floor, es*np.exp(-decay_rate*yi))
    return np.array([_eta(y) for y in range(1, project_life+1)])


def build_elec_prices(project_life, elec_price_base, elec_escalation) -> np.ndarray:
    return np.array([elec_price_base*(1+elec_escalation)**(y-1)
                     for y in range(1, project_life+1)])


def degradation_preview(decay_rate, overhaul_interval, overhaul_recovery, eff_bep) -> dict:
    eta_end  = np.exp(-decay_rate*overhaul_interval)
    loss_pct = (1-eta_end)*100
    rec_pct  = overhaul_recovery*(1-eta_end)*100
    perm_pct = (1-eta_end)*(1-overhaul_recovery)*100
    return {"eta_bop":  round(eff_bep*100,2),
            "eta_eop":  round(eff_bep*eta_end*100,2),
            "loss_pct": round(loss_pct,2),
            "rec_pct":  round(rec_pct,2),
            "perm_pct": round(perm_pct,2)}


# ══════════════════════════════════════════════════════════════════════════════
# 6. ANNUAL OPERATING PROFILE  (new — matches notebook Cell 16)
# ══════════════════════════════════════════════════════════════════════════════

def compute_annual_operating_profile(
        splines, ctx, load_profile, comp_profile, eta_profile,
        surge_margin, r_p_required, N_fixed,
        overhaul_interval, Q_std_design,
        cache_tag=()) -> dict:
    """
    Build the annual technical operating profile.

    Inputs are on a STANDARD FLOW basis (Sm³/h).
    Each year is converted to actual suction m³/h via PR-EOS before the
    compressor calculation — matching the notebook implementation exactly.
    """
    scenario_key = (
        cache_tag,
        tuple(round(v,8) for v in load_profile),
        tuple(tuple(round(zi,8) for zi in comp.values()) for comp in comp_profile),
        tuple(round(v,8) for v in eta_profile),
        round(Q_std_design,4),
        int(overhaul_interval),
    )
    if scenario_key in _OPERATING_CACHE:
        cached = _OPERATING_CACHE[scenario_key]
        return {k: v.copy(deep=True) if isinstance(v, pd.DataFrame) else v
                for k, v in cached.items()}

    years = np.arange(1, len(load_profile)+1)
    rows  = []
    for year, load, comp_yr, eta in zip(years, load_profile, comp_profile, eta_profile):
        z_yr    = list(comp_yr.values())
        q_std   = float(load * Q_std_design)
        q_suc   = q_suction_from_std(ctx, q_std, z_yr)
        MW_yr   = sum(z_yr[i]*ctx["consts"].MWs[i] for i in range(len(z_yr)))
        oh      = (((int(year)-1) % overhaul_interval == 0) and int(year) > 1)

        r = compare_strategies(splines, ctx, q_suc, surge_margin,
                                r_p_required, N_fixed,
                                eta_factor=float(eta), z_gas=z_yr,
                                cache_tag=cache_tag)
        if not r["is_valid"]:
            continue

        rows.append({
            "year":              int(year),
            "load_pct":          float(load*100),
            "Q_std_Sm3h":        q_std,
            "Q_process":         q_suc,
            "eta_factor":        float(eta),
            "MW":                float(MW_yr),
            "overhaul":          oh,
            "kW_fixed":          r["A_power_kW"],
            "kW_vfd":            r["B_power_kW"],
            "saving_kW":         r["saving_kW"],
            "N_fixed":           r["A_N_rpm"],
            "N_vfd":             r["B_N_rpm"],
            "N_antisurge":       r["N_antisurge"],
            "N_pressure":        r["N_pressure"],
            "Q_total_fixed":     r["A_Q_total"],
            "Q_total_vfd":       r["B_Q_total"],
            "Q_recycle_fixed":   r["A_Q_recycle"],
            "Q_recycle_vfd":     r["B_Q_recycle"],
            "eta_poly_fixed":    r["A_eta_poly"],
            "eta_poly_vfd":      r["B_eta_poly"],
            "P_disch_fixed_bar": r["A_P_disch"],
            "P_disch_vfd_bar":   r["B_P_disch"],
            "T_disch_fixed_C":   r["A_T_disch_C"],
            "T_disch_vfd_C":     r["B_T_disch_C"],
            "constraint":        r["constraint"],
        })

    df = pd.DataFrame(rows)
    result = {
        "df": df,
        "n_antisurge":  int((df["constraint"]=="antisurge").sum())    if not df.empty else 0,
        "n_pressure":   int((df["constraint"]=="pressure").sum())     if not df.empty else 0,
        "n_nmax":       int((df["constraint"]=="N_max+recycle").sum()) if not df.empty else 0,
    }
    _OPERATING_CACHE[scenario_key] = {
        k: v.copy(deep=True) if isinstance(v, pd.DataFrame) else v
        for k, v in result.items()}
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 7. ECONOMIC ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _cross(x, y, target, y0):
    xv, yv = np.asarray(x,float), np.asarray(y,float)
    idx = np.where(yv >= target)[0]
    if not len(idx): return float("inf")
    i = idx[0]
    x0, y0_ = (0.0,y0) if i==0 else (xv[i-1],yv[i-1])
    x1, y1  = xv[i], yv[i]
    if abs(y1-y0_) < CROSS_TOL or not np.isfinite(y1-y0_): return float(x1)
    frac = min(max((target-y0_)/(y1-y0_),0.0),1.0)
    return float(x0+frac*(x1-x0))


def compute_annual_cashflows(
        splines, ctx,
        load_profile, comp_profile, eta_profile, elec_prices,
        surge_margin, r_p_required, N_fixed, overhaul_interval,
        hours_per_year, discount_rate, vfd_usd_kw, vfd_install_f,
        Q_std_design=None,
) -> dict:
    """
    Full 20-year techno-economic model.
    Production is on a standard-flow basis; conversion to suction flow is
    done inside compute_annual_operating_profile.
    """
    load_min_std, Q_std, Q_as_nmin, Q_design = compute_load_min_std(
        splines, surge_margin, N_fixed, ctx,
        dict(zip(COMPONENTS_DEFAULT, comp_profile[0].values()
                 if hasattr(list(comp_profile[0].values())[0], '__iter__')
                 else comp_profile[0].values())),
        dict(zip(COMPONENTS_DEFAULT, comp_profile[-1].values()
                 if hasattr(list(comp_profile[-1].values())[0], '__iter__')
                 else comp_profile[-1].values())),
        len(load_profile),
        Q_std_design=Q_std_design or None,
    )
    Q_std_use = Q_std_design if Q_std_design else Q_std

    cache_tag = (
        round(surge_margin,6),
        None if r_p_required is None else round(r_p_required,6),
        round(N_fixed,2),
        round(ctx["P_suc_bar"],6),
        round(ctx["T_suc_C"],4),
        tuple(round(zi,8) for zi in ctx["z_design"]),
    )

    op = compute_annual_operating_profile(
        splines, ctx, load_profile, comp_profile, eta_profile,
        surge_margin, r_p_required, N_fixed,
        overhaul_interval, Q_std_use, cache_tag=cache_tag)

    df = op["df"].copy(deep=True)
    if df.empty:
        return {"df": df, "capex_vfd": np.nan, "kW_peak": np.nan,
                "npv_total": np.nan, "saving_total": 0.0,
                "payback": np.inf, "payback_disc": np.inf,
                "irr": np.nan,
                "n_antisurge": 0, "n_pressure": 0, "n_nmax": 0,
                "Q_design": Q_design, "Q_as_nmin": Q_as_nmin,
                "load_min": load_min_std, "Q_std_design": Q_std_use}

    price_map     = {int(y): float(p)
                     for y, p in zip(range(1,len(load_profile)+1), elec_prices)}
    df["elec_price"] = df["year"].map(price_map)
    df["E_fixed_MWh"] = df["kW_fixed"] * hours_per_year / 1000
    df["E_vfd_MWh"]   = df["kW_vfd"]   * hours_per_year / 1000
    df["cost_fixed"]  = df["kW_fixed"] * hours_per_year * df["elec_price"]
    df["cost_vfd"]    = df["kW_vfd"]   * hours_per_year * df["elec_price"]
    df["saving_USD"]  = df["cost_fixed"] - df["cost_vfd"]
    df["disc_factor"] = 1/(1+discount_rate)**df["year"]
    df["saving_pv"]   = df["saving_USD"] * df["disc_factor"]
    df["saving_cum"]  = df["saving_USD"].cumsum()

    kW_peak   = float(df["kW_fixed"].max())
    capex_vfd = kW_peak * vfd_usd_kw * vfd_install_f
    df["npv_cum"] = df["saving_pv"].cumsum() - capex_vfd

    payback      = _cross(df["year"], df["saving_cum"], capex_vfd, 0.0)
    payback_disc = _cross(df["year"], df["npv_cum"],    0.0, -capex_vfd)

    cfs = [-capex_vfd] + list(df["saving_USD"])
    try:
        irr = brentq(lambda r: sum(f/(1+r)**t for t,f in enumerate(cfs)),
                     IRR_LOWER, IRR_UPPER, xtol=IRR_XTOL)
    except Exception:
        irr = np.nan

    return {
        "df": df, "capex_vfd": capex_vfd, "kW_peak": kW_peak,
        "npv_total":    float(df["npv_cum"].iloc[-1]),
        "saving_total": float(df["saving_USD"].sum()),
        "payback": payback, "payback_disc": payback_disc,
        "irr": irr,
        "n_antisurge": op["n_antisurge"],
        "n_pressure":  op["n_pressure"],
        "n_nmax":      op["n_nmax"],
        "Q_design":    Q_design,
        "Q_as_nmin":   Q_as_nmin,
        "load_min":    load_min_std,
        "Q_std_design": Q_std_use,
    }
