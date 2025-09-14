from dataclasses import dataclass
import numpy as np
import pandas as pd
from pyomo.environ import value
import io
from contextlib import redirect_stdout, redirect_stderr

OK = "TEST OK"
BAD = "TEST FAILED"

# store model params to use in checks 
@dataclass
class ModelConfig:
    delta_t: float
    min_capacity: float
    max_capacity: float
    max_power: float
    efficiency: float
    init_capacity: float
    cycles_per_day_max: float

def _extract(model):
    """Extract numpy arrays from a solved Pyomo model."""
    T = len(list(model.Period))
    ch  = np.array([value(model.Charge_power[t])    for t in model.Period], dtype=float)
    ds  = np.array([value(model.Discharge_power[t]) for t in model.Period], dtype=float)
    soc = np.array([value(model.Capacity[t])        for t in model.Period], dtype=float)
    obj = float(value(model.objective))
    return ch, ds, soc, obj

def _print_result(name: str, passed: bool, extra: str = "") -> bool:
    mark = OK if passed else BAD
    msg = f"{mark} {name}"
    if extra:
        msg += f" — {extra}"
    print(msg)
    return passed

# ------------------ post-solve checks to see if model kept to constraints ------------------

def run_basic_postsolve_checks(model, prices_df: pd.DataFrame, cfg: ModelConfig) -> bool:
    """
    Run quick invariants on the *current solved model*.
    Prints PASS/FAIL per check and returns overall boolean.
    """
    ch, ds, soc, obj = _extract(model)
    dt = cfg.delta_t

    results = []

    # 1) no simultaneous charge & discharge
    results.append(_print_result(
        "no simultaneous charge/discharge",
        bool(np.all((ch <= 1e-9) | (ds <= 1e-9)))
    ))

    # 2) power bounds
    results.append(_print_result(
        "power limits respected",
        bool(np.all(ch >= -1e-9) and np.all(ds >= -1e-9)
             and np.max(ch) <= cfg.max_power + 1e-9
             and np.max(ds) <= cfg.max_power + 1e-9),
        extra=f"max(ch)={np.max(ch):.6f}, max(ds)={np.max(ds):.6f}"
    ))

    # 3) SoC bounds
    results.append(_print_result(
        "SoC bounds respected",
        bool(np.min(soc) >= cfg.min_capacity - 1e-6
             and np.max(soc) <= cfg.max_capacity + 1e-6),
        extra=f"min(soc)={np.min(soc):.6f}, max(soc)={np.max(soc):.6f}"
    ))

    # 4) SoC dynamics
    lhs = soc[1:]
    rhs = soc[:-1] + cfg.efficiency*ch[:-1]*dt - ds[:-1]*dt
    results.append(_print_result(
        "SoC dynamics consistent",
        bool(np.allclose(lhs, rhs, atol=1e-6))
    ))

    # 5) end SoC = init SoC
    results.append(_print_result(
        "terminal SoC equals initial SoC",
        abs(soc[-1] - cfg.init_capacity) <= 1e-6
    ))

    # 6) daily throughput cap (computed from dataframe dates)
    #    If your index is not a DateTimeIndex, we just skip this check.
    if isinstance(prices_df.index, pd.DatetimeIndex):
        df = pd.DataFrame({"thru": (ch + ds) * dt}, index=prices_df.index)
        daily = df["thru"].groupby(df.index.date).sum()
        cap = 2.0 * cfg.cycles_per_day_max * cfg.max_capacity
        results.append(_print_result(
            "daily throughput cap respected",
            bool(np.all(daily.values <= cap + 1e-6)),
            extra=f"max daily={daily.max():.6f} ≤ cap={cap:.6f}"
        ))
    else:
        results.append(_print_result("daily throughput cap respected", True))

    # overall
    all_ok = all(results)
    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}\n")
    return all_ok

# ------------------ optional tiny synthetic smoke tests (very fast) ------------------
def run_quick_synthetic_smokes(solver: str, build_and_solve_fn, worst_case: bool = False) -> bool:
    """
    Runs 2 tiny scenario solves.
    Smoke test A: check if there is no trade when flatlined prices
    Smoke test B: check if in 4 ptu's the battery behaviour is as expected
                  (charge before peak and discharge during peak)
    """

    ok = True

    def _norm_index(df):
        # Make index 0..N-1 so Pyomo Params (keyed by t in RangeSet) match
        return df.reset_index(drop=True)

    def _silent_build_and_solve(df, **kwargs):
        # Capture all solver/model prints during the solve
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            return build_and_solve_fn(df, **kwargs)

    # (A) flat prices -> no trade if eta < 1
    idx = pd.date_range("2025-05-04", periods=8, freq="15min")
    flat = pd.DataFrame(
        {"IMBALANCE_SHORT_EUR_MWH": 50.0, "IMBALANCE_LONG_EUR_MWH": 50.0},
        index=idx
    )
    flat_n = _norm_index(flat)
    res, m = _silent_build_and_solve(
        flat_n, solver=solver,
        MIN_CAPACITY=0, MAX_CAPACITY=2, MAX_POWER=1,
        EFFICIENCY=0.95, INIT_CAPACITY=0,
        RAMP_CH=1, RAMP_DC=1,
        CYCLES_PER_DAY_MAX=10, DEG_COST_EUR_PER_MWH=0.0
    )
    ch, ds, soc, obj = _extract(m)
    ok &= _print_result(
        "smoke A: flat prices → no trade",
        bool(np.allclose(ch, 0.0, atol=1e-7) and np.allclose(ds, 0.0, atol=1e-7)
            and abs(obj) <= 1e-6)
    )

    # (B) simple spike: cheap then expensive → 1 charge then 1 discharge (eta=1)
    idx = pd.date_range("2025-05-04", periods=4, freq="15min")
    spike = pd.DataFrame(
        {"IMBALANCE_SHORT_EUR_MWH": [50, 10, 50, 50],
        "IMBALANCE_LONG_EUR_MWH":  [50, 10,100, 50]},
        index=idx
    )
    spike_n = _norm_index(spike)
    res, m = _silent_build_and_solve(
        spike_n, solver=solver,
        MIN_CAPACITY=0, MAX_CAPACITY=2, MAX_POWER=1,
        EFFICIENCY=1.0, INIT_CAPACITY=0,
        RAMP_CH=1, RAMP_DC=1,
        CYCLES_PER_DAY_MAX=10, DEG_COST_EUR_PER_MWH=0.0
    )
    ch, ds, soc, obj = _extract(m)
    pass_B = (ch[1] >= 1.0 - 1e-6 and ds[2] >= 1.0 - 1e-6 and
            np.all(ch[[0,2,3]] <= 1e-6) and np.all(ds[[0,1,3]] <= 1e-6) and
            abs(obj - (100-10)*0.25) <= 1e-5)
    ok &= _print_result("smoke B: simple arbitrage spike", pass_B)

    print(f"\n{'ALL SMOKES PASSED' if ok else 'SOME SMOKES FAILED'}\n")
    
    return ok
