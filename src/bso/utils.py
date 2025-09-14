import pandas as pd
from pyomo.environ import *
from pyomo.environ import value, maximize, minimize
from math import isfinite
import os
from pathlib import Path


def import_imbalance_prices(filename="imbalance_prices.csv"):
    """
    Load the imbalance price CSV from the project `data` folder,
    independent of where the script is executed from.
    """
    # Location of this utils.py file
    here = Path(__file__).resolve()
    # Project root: two levels up from src/bso/utils.py
    root = here.parents[2]
    data_file = root / "data" / filename
    return pd.read_csv(data_file)

def retrieve_days_daylabels(imbalance_prices):
    # to prep for day and timeofday sets that are used to constrain the max cycles per day
    idx = pd.to_datetime(imbalance_prices.index)
    day_labels = [pd.Timestamp(t).strftime("%Y-%m-%d") for t in idx]

    days = sorted(set(day_labels))
    times_by_day = {d: [t for t, dlab in enumerate(day_labels) if dlab == d] for d in days}

    return days, times_by_day

def save_battery_results(m, datetimes,filename="battery_results", Δt=0.25, eps=1e-9):
    """
    Extract results from a solved Pyomo battery model and save to CSV.

    Args:
        m:       Pyomo model (already solved)
        datetimes: list/Series of timestamps aligned with m.Period
        filename: output CSV filename
        Δt:       hours per period (default 0.25 for 15 minutes)
        eps:      numerical tolerance (treat very small values as 0)

    Returns:
        results_df: pandas DataFrame with results
    """

    periods = list(m.Period)

    rows = []
    for k, t in enumerate(periods):
        Price_short = float(m.Price_short[t])
        Price_long  = float(m.Price_long[t])

        p_ch  = m.Charge_power[t].value or 0.0
        p_dis = m.Discharge_power[t].value or 0.0
        soc   = m.Capacity[t].value or 0.0

        # Clean tiny numerical noise
        if abs(p_ch)  < eps: p_ch  = 0.0
        if abs(p_dis) < eps: p_dis = 0.0
        if abs(soc)   < eps: soc   = 0.0

        # Convention: discharge positive, charge negative
        net_power = p_dis - p_ch

        # Imbalance volume (MWh) over PTU
        imbalance_mwh = net_power * Δt

        # Revenue using "worst price" rule
        revenue = Price_long * (p_dis * Δt) - Price_short * (p_ch * Δt)

        rows.append({
            "datetime": datetimes[k],
            "price_short": Price_short,
            "price_long":  Price_long,
            "charge_power_mw": p_ch,
            "discharge_power_mw": p_dis,
            "net_power_mw": net_power,
            "imbalance_mwh": imbalance_mwh,
            "soc_mwh": soc,
            "revenue": revenue,
        })

    results_df = pd.DataFrame(rows).set_index("datetime").sort_index()
    results_df["cumulative_revenue"] = results_df["revenue"].cumsum()

    # Save to CSV
    results_df.to_csv(f"{filename}.csv", index=True)

    print(f"Results saved to {filename}.csv")
    return results_df

def mip_report(model, results, obj=None):
    """
    Extract post-solve KPIs (objective, best bound, mip gap, nodes, time, status)
    from a Pyomo solve() results object across solvers (HiGHS/GLPK/CBC/Gurobi).

    Args:
        model   : your Pyomo model (with a single objective named 'objective' by default)
        results : the return value from opt.solve(model, ...)
        obj     : optional Pyomo objective object; defaults to model.objective

    Returns:
        dict with keys: sense, objective, best_bound, abs_gap, rel_gap,
                        nodes, iterations, time_sec, status, termination, message
    """
    obj = obj or model.objective
    sense = "maximize" if obj.sense == maximize else "minimize"

    # Incumbent objective value
    incumbent = float(value(obj))

    # Try to find the solver's best bound and other stats in a robust way
    s = results.solver
    other = getattr(s, "other", {}) or {}

    # Common places different writers put these
    best_bound = (
        other.get("best_bound") or
        getattr(s, "best_bound", None) or
        other.get("Best objective bound") or
        other.get("bound")
    )
    if best_bound is not None:
        best_bound = float(best_bound)

    nodes = (
        other.get("branch_and_bound_nodes") or
        other.get("nodes") or
        getattr(s, "branch_and_bound_nodes", None)
    )
    iterations = (
        other.get("iter") or
        getattr(s, "iterations", None)
    )
    time_sec = (
        getattr(s, "time", None) or
        other.get("time") or
        other.get("cpu time (seconds)")
    )

    status = getattr(s, "status", None)
    term   = getattr(s, "termination_condition", None)
    msg    = getattr(s, "message", None) or other.get("message")

    # Compute gaps (use solver-provided if available)
    rel_gap = (
        other.get("relative_gap") or
        getattr(s, "gap", None) or
        getattr(s, "mipgap", None)
    )

    abs_gap = None
    if rel_gap is None and best_bound is not None and isfinite(incumbent):
        # Compute our own gap. For maximize, gap = (best_bound - incumbent)/|incumbent|
        # For minimize, gap = (incumbent - best_bound)/|incumbent|
        denom = max(1e-12, abs(incumbent))
        if sense == "maximize":
            rel_gap = (best_bound - incumbent) / denom
            abs_gap = best_bound - incumbent
        else:
            rel_gap = (incumbent - best_bound) / denom
            abs_gap = incumbent - best_bound
    else:
        # If solver gave rel_gap, compute abs_gap when possible
        if rel_gap is not None and best_bound is not None:
            if sense == "maximize":
                abs_gap = best_bound - incumbent
            else:
                abs_gap = incumbent - best_bound

    return {
        "sense": sense,
        "objective": incumbent,
        "best_bound": best_bound,
        "abs_gap": abs_gap,
        "rel_gap": rel_gap,
        "nodes": nodes,
        "iterations": iterations,
        "time_sec": time_sec,
        "status": str(status),
        "termination": str(term),
        "message": msg,
    }

def save_model_results(model, output_path):

    imbalance_prices = import_imbalance_prices('imbalance_prices.csv')

    here = Path(__file__).resolve()
    # Project root: two levels up from src/bso/utils.py
    # next versions don't hardcode this
    root = here.parents[2]
    data_file = root / "results" / output_path

    save_battery_results(
        model,
        datetimes=imbalance_prices["START_DATETIME_UTC"],
        filename=data_file
    )
