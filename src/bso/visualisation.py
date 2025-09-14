import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import seaborn as sns


# seaborn is optional; plot works without it
try:
    import seaborn as sns
    _HAVE_SNS = True
except Exception:
    _HAVE_SNS = False


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _set_theme():
    """Light, readable defaults (works with or without seaborn)."""
    import matplotlib as mpl
    if _HAVE_SNS:
        sns.set_theme(style="whitegrid", context="talk")
    mpl.rcParams.update({
        "axes.edgecolor": "#cccccc",
        "axes.grid": True,
        "grid.alpha": 0.30,
        "grid.linestyle": "-",
        "figure.autolayout": False,
        "axes.titleweight": "bold",
        "axes.labelweight": "regular",
        "axes.titlesize": "large",
        "legend.frameon": True,
    })


def _ensure_dt_index(df, time_col=None):
    """
    Return a copy of df indexed by datetime.
    - If df already has a DatetimeIndex: sort and return.
    - Else, use time_col if provided.
    - Else try to auto-detect a datetime column or common names.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()

    out = df.copy()

    # Explicit column provided
    if time_col is not None:
        dt = pd.to_datetime(out[time_col], errors="coerce", utc=False)
        if dt.isna().all():
            raise ValueError(f"Could not parse any datetimes from column '{time_col}'.")
        out.index = dt
        out = out[~out.index.isna()]
        return out.sort_index()

    # Auto-detect: any datetime-typed column
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out.index = out[c]
            return out.drop(columns=[c]).sort_index()

    # Auto-detect by common names, attempt parsing
    for c in ("timestamp", "time", "datetime", "date", "ptu_start", "ptu_time", "start"):
        if c in out.columns:
            dt = pd.to_datetime(out[c], errors="coerce", utc=False)
            if dt.notna().any():
                out.index = dt
                return out.drop(columns=[c]).sort_index()

    raise ValueError(
        "No datetime index/column found. Pass time_col='your_timestamp_col' or "
        "pre-index your frame: df = df.set_index('your_timestamp_col')."
    )


def _bar_width_days(index, scale=1.35):
    """
    Return bar width in Matplotlib 'date' units (days).
    Uses median spacing; scale>1 => wider bars.
    """
    if len(index) > 1:
        spacing = pd.Series(index).diff().median()
        if pd.isna(spacing):
            spacing = pd.Timedelta(minutes=15)
    else:
        spacing = pd.Timedelta(minutes=15)
    return (spacing / pd.Timedelta(days=1)) * 0.80 * float(scale)


# ------------------------------------------------------------------
# Main plotters
# ------------------------------------------------------------------

def plot_day_like_example_sns(
    df,
    price_col="price_long",
    power_col="net_power_mw",
    soc_col="soc_mwh",
    imbalance_col="imbalance_mwh",

    start=None, end=None, day=None, revenue_col="revenue",

    linewidth_scale=1.6,
    bar_scale=1.35,
    edgewidth=1.2,
    figsize=(18, 8.5),

    hour_tick_interval=8,
    legend_outside=True,
    vline_thinner=0.7,

    title_prefix="",
    time_col=None,  # <-- NEW: specify your timestamp column if not indexed by time
):
    """
    Plot (1) imbalance price with action markers, (2) PTU imbalance bars,
    and (3) SoC line + end-of-PTU bars for a chosen day or time window.
    """
    # --- colors ---
    COLOR_PRICE      = "#1f77b4"
    COLOR_CHARGE     = "#2ca02c"  # charge / import (–)
    COLOR_DISCHARGE  = "#d62728"  # discharge / export (+)

    _set_theme()
    view_all = _ensure_dt_index(df, time_col=time_col)

    # ----- pick range -----
    if start is not None or end is not None:
        view = view_all.loc[start:end]
        chosen_label = f"{pd.to_datetime(start).date() if start else ''} → {pd.to_datetime(end).date() if end else ''}"
    else:
        # choose a 'hero' day by revenue if available; else by |power| throughput
        if day is None:
            by_day = view_all.groupby(view_all.index.date)
            if revenue_col in view_all.columns:
                day_scores = by_day[revenue_col].sum().sort_values(ascending=False)
            else:
                # sum absolute power as a proxy for "interesting" days
                power_abs = by_day[power_col].apply(lambda x: np.abs(pd.to_numeric(x, errors="coerce")).fillna(0).sum())
                day_scores = power_abs.sort_values(ascending=False)
            hero = day_scores.index[0]
            day = pd.to_datetime(str(hero)).date()
        else:
            day = pd.to_datetime(str(day)).date()
        view = view_all.loc[str(day):str(day)]
        chosen_label = str(day)

    if view.empty:
        raise ValueError("No rows in the selected window. Check your start/end/day and time index/column.")

    # safe copy before in-place operations
    view = view.copy()

    # numeric coercion (if present)
    for c in [price_col, power_col, soc_col]:
        if c in view.columns:
            view[c] = pd.to_numeric(view[c], errors="coerce")

    # ----- PTU length (hours) inferred from median spacing -----
    if len(view.index) > 1:
        spacing = pd.Series(view.index).diff().median()
        delta_h = (spacing / pd.Timedelta(hours=1)) if pd.notna(spacing) else 0.25
    else:
        delta_h = 0.25

    # ----- imbalance per PTU (MWh) -----
    if imbalance_col in view.columns:
        imbal = pd.to_numeric(view[imbalance_col], errors="coerce")
    else:
        imbal = pd.to_numeric(view[power_col], errors="coerce") * float(delta_h)

    # ----- SoC end-of-PTU and change (for coloring) -----
    soc_end = view[soc_col].shift(-1)
    dsoc = soc_end - soc_end.shift(1)
    up, dn = dsoc > 0, dsoc < 0

    # action masks for vlines
    is_discharge = view[power_col] > 0   # export (+)
    is_charge    = view[power_col] < 0   # import (–)

    # sizes
    w = _bar_width_days(view.index, scale=bar_scale)
    price_lw  = 2.0 * linewidth_scale
    vline_lw  = price_lw * vline_thinner

    # ----- figure (3 panels) -----
    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(3, 1, height_ratios=[3.0, 1.6, 2.2], hspace=0.18)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)

    # (0) Price + action markers
    if _HAVE_SNS:
        sns.lineplot(ax=ax0, x=view.index, y=view[price_col], color=COLOR_PRICE,
                     linewidth=price_lw, label="Imbalance Price (€/MWh)")
    else:
        ax0.plot(view.index, view[price_col], linewidth=price_lw, label="Imbalance Price (€/MWh)")
    ymin, ymax = ax0.get_ylim()
    # draw charge/discharge markers as vertical lines
    ax0.vlines(view.index[is_charge],    ymin=ymin, ymax=ymax,
               colors=COLOR_CHARGE, linestyles="-", linewidth=vline_lw, label="Charge")
    ax0.vlines(view.index[is_discharge], ymin=ymin, ymax=ymax,
               colors=COLOR_DISCHARGE, linestyles=":", linewidth=vline_lw, label="Discharge")
    ax0.set_ylim(ymin, ymax)
    ax0.set_ylabel("Imbalance Price\n(€/MWh)")
    # ax0.set_title(f"{title_prefix}{chosen_label}")

    # (1) Imbalance per PTU (bars)
    pos = imbal.clip(lower=0)   # export (+) -> discharge -> red
    neg = imbal.clip(upper=0)   # import (–) -> charge -> green
    ax1.bar(view.index, pos, width=w, alpha=0.9, edgecolor="black", linewidth=edgewidth,
            color=COLOR_DISCHARGE, label="Export (+)")
    ax1.bar(view.index, neg, width=w, alpha=0.9, edgecolor="black", linewidth=edgewidth,
            color=COLOR_CHARGE, label="Import (–)")
    ax1.axhline(0, lw=1.2)
    ax1.set_ylabel("Imbalance\n(MWh)")

    # (2) SoC line + end-of-PTU bars (up/down only)
    if _HAVE_SNS:
        sns.lineplot(ax=ax2, x=view.index, y=view[soc_col], linewidth=price_lw*0.9, color=COLOR_PRICE)
    else:
        ax2.plot(view.index, view[soc_col], linewidth=price_lw*0.9)
    ax2.bar(soc_end.index[up], soc_end[up], width=w, alpha=0.9,
            edgecolor="black", linewidth=edgewidth, color=COLOR_CHARGE, label="SoC (end) ↑")
    ax2.bar(soc_end.index[dn], soc_end[dn], width=w, alpha=0.9,
            edgecolor="black", linewidth=edgewidth, color=COLOR_DISCHARGE, label="SoC (end) ↓")
    ax2.set_ylabel("State of Charge\n(MWh)")

    # x-axis ticks
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=hour_tick_interval))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    for ax in (ax0, ax1, ax2):
        ax.grid(True, axis="y", alpha=0.30)

    # Legends
    if legend_outside:
        ax0.legend(loc="upper left", bbox_to_anchor=(0.5, 1.15), frameon=True, ncol=3, borderaxespad=0.0)
        plt.tight_layout(rect=(0.10, 0.0, 1.0, 0.98))
    else:
        ax0.legend(ncol=3, frameon=True, loc="upper left")
        ax1.legend(ncol=2, frameon=True, loc="upper left")
        ax2.legend(ncol=3, frameon=True, loc="upper left")
        plt.tight_layout()

    plt.show()


def plot_cumulative_revenue_same_style(
    df,
    revenue_col="revenue",
    time_col="datetime",

    # window (optional)
    start=None, end=None,

    # start plotting only after N full days from the window start
    start_after_days=3,
    rebase_at_cutoff=True,      # start curve at 0 at the cutoff

    # visuals
    linewidth_scale=1.6,
    figsize=(18, 4.8),
    legend_outside_right=True,
    label="",
):
    COLOR_PRICE = "#1f77b4"  # match your price line color

    _set_theme()
    view = _ensure_dt_index(df, time_col=time_col)
    if start is not None or end is not None:
        view = view.loc[start:end]
    if view.empty:
        raise ValueError("No rows in the selected window for cumulative revenue.")

    # Cut off the first N days (dates only on x-axis)
    window_start_date = view.index.min().normalize()
    cutoff = window_start_date + pd.Timedelta(days=int(start_after_days))
    view = view.loc[view.index >= cutoff]
    if view.empty:
        raise ValueError(f"No data on/after {cutoff.date()} (after {start_after_days} days).")

    # Build cumulative
    rev = pd.to_numeric(view[revenue_col], errors="coerce").fillna(0.0)
    cum = rev.cumsum() if not rebase_at_cutoff else (rev - rev.iloc[0] + rev.iloc[0]).cumsum()
    if rebase_at_cutoff:
        # simpler: just start at zero at cutoff
        cum = rev.cumsum()

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.lineplot(ax=ax, x=cum.index, y=cum.values,
                 color=COLOR_PRICE, linewidth=2.0 * linewidth_scale, label=label)

    ax.axhline(0, lw=1.2, alpha=0.6)
    ax.set_ylabel("Cumulative Revenue (€)")
    ax.set_xlabel("Date")
    

    # DATE ticks (no time) — one tick per day
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    ax.grid(True, axis="y", alpha=0.30)

    # if legend_outside_right:
    #     ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
    #               frameon=True, ncol=1, borderaxespad=0.0)
    #     plt.tight_layout(rect=(0.0, 0.0, 0.86, 1.0))
    # else:
    #     ax.legend(loc="upper left", frameon=True)
    #     plt.tight_layout()
    ax.set_title("Cumulative Revenue - Week")
    plt.show()


def plot_imbalance_prices_same_style(
    df,
    time_col="START_DATETIME_UTC",
    long_col="IMBALANCE_LONG_EUR_MWH",
    short_col="IMBALANCE_SHORT_EUR_MWH",
    which="long",                 # "long" | "short" | "both"

    # optional window
    start=None, end=None,

    # start plotting only after N full days from the window start
    start_after_days=3,

    # visuals (match your style)
    linewidth_scale=1.6,
    figsize=(18, 4.8),
    legend_outside_right=True,
    label_long="Imbalance price (long) (€/MWh)",
    label_short="Imbalance price (short) (€/MWh)",
    title="Imbalance Price - Week",
):
    # same palette as your price line
    COLOR_PRICE = "#1f77b4"   # primary (use for "long")
    COLOR_ALT   = "#ff7f0e"   # secondary (use for "short" when plotting both)

    # use your existing helpers
    _set_theme()
    view = _ensure_dt_index(df, time_col=time_col)
    if start is not None or end is not None:
        view = view.loc[start:end]
    if view.empty:
        raise ValueError("No rows in the selected window for imbalance prices.")

    # keep only needed cols, coerce to numeric
    cols_present = []
    if which in ("long", "both") and long_col in view:
        view[long_col] = pd.to_numeric(view[long_col], errors="coerce")
        cols_present.append(long_col)
    if which in ("short", "both") and short_col in view:
        view[short_col] = pd.to_numeric(view[short_col], errors="coerce")
        cols_present.append(short_col)
    if not cols_present:
        raise ValueError("Requested columns not found. Check 'which', 'long_col', and 'short_col'.")

    # cut off the first N full days
    window_start_date = view.index.min().normalize()
    cutoff = window_start_date + pd.Timedelta(days=int(start_after_days))
    view = view.loc[view.index >= cutoff]
    if view.empty:
        raise ValueError(f"No data on/after {cutoff.date()} (after {start_after_days} days).")

    # figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # plot the lines
    if which in ("long", "both") and long_col in view:
        sns.lineplot(
            ax=ax, x=view.index, y=view[long_col],
            color=COLOR_PRICE, linewidth=2.0 * linewidth_scale, label=label_long
        )
    if which in ("short", "both") and short_col in view:
        sns.lineplot(
            ax=ax, x=view.index, y=view[short_col],
            color=(COLOR_PRICE if which != "both" else COLOR_ALT),
            linewidth=2.0 * linewidth_scale,
            label=(label_short if which == "both" else label_short.replace(" (short)", ""))
        )

    ax.axhline(0, lw=1.2, alpha=0.6)
    ax.set_ylabel("Imbalance Price (€/MWh)")
    ax.set_xlabel("Date")
    ax.set_title(title)

    # DATE ticks only (one per day)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    ax.grid(True, axis="y", alpha=0.30)

    # legend outside on the right
    if legend_outside_right:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  frameon=True, ncol=1, borderaxespad=0.0)
        plt.tight_layout(rect=(0.0, 0.0, 0.86, 1.0))
    else:
        ax.legend(loc="upper left", frameon=True)
        plt.tight_layout()

    plt.show()


# ---------------- greedy baseline ----------------
def greedy_battery_trader(
    prices_df,
    time_col="START_DATETIME_UTC",
    long_col="IMBALANCE_LONG_EUR_MWH",
    short_col="IMBALANCE_SHORT_EUR_MWH",
    # battery / market params
    Delta_t=0.25,
    MAX_CAPACITY=2.0,
    MIN_CAPACITY=0.0,
    MAX_POWER=1.0,
    INIT_CAPACITY=0.0,
    EFFICIENCY=0.9,             # roundtrip
    RAMP_CH=1.0,                # MW/PTU (towards more negative, i.e., charging)
    RAMP_DC=1.0,                # MW/PTU (towards more positive, i.e., discharging)
    CYCLES_PER_DAY_MAX=2.0,     # equivalent full cycles / day
    DEG_COST_EUR_PER_MWH=1.5,
    # greedy thresholds
    low_q=0.30,
    high_q=0.70,
):
    """
    Returns a DataFrame with:
      datetime, price_long, price_short, price_sell, price_buy, p_mw, e_mwh, soc_mwh,
      revenue, cum_revenue, throughput_mwh_day, efc_used_day
    """
    # prepare
    df = prices_df.copy()
    # rename/standardize
    df.rename(columns={time_col: "datetime", long_col: "price_long", short_col: "price_short"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.sort_values("datetime").reset_index(drop=True)
    df["date"] = df["datetime"].dt.date

    # worst-case prices
    df["price_sell"] = df[["price_long", "price_short"]].min(axis=1)  # for export (+)
    df["price_buy"]  = df[["price_long", "price_short"]].max(axis=1)  # for import (–)

    # greedy thresholds on a mid-price
    df["p_mid"] = 0.5*(df["price_long"] + df["price_short"])
    low_thr  = df["p_mid"].quantile(low_q)
    high_thr = df["p_mid"].quantile(high_q)

    # split η round-trip symmetrically
    eta_leg = np.sqrt(float(EFFICIENCY))
    eta_ch, eta_dc = eta_leg, eta_leg

    # daily throughput budget (MWh of |grid energy|) from EFC cap:
    # EFC = throughput / (2*Capacity)  => throughput_limit = 2 * Capacity * EFC_max
    daily_throughput_limit = 2.0 * MAX_CAPACITY * CYCLES_PER_DAY_MAX

    # simulate
    rows = []
    soc = float(INIT_CAPACITY)
    prev_p = 0.0
    used_throughput_today = 0.0
    current_day = df["date"].iloc[0] if not df.empty else None
    cum_rev = 0.0

    for i, r in df.iterrows():
        day = r["date"]
        if day != current_day:
            # reset daily counters at day boundary
            current_day = day
            used_throughput_today = 0.0

        # --- decide action: target power (MW) before constraints ---
        if r["p_mid"] <= low_thr:
            p_tgt = -MAX_POWER  # charge (import)
        elif r["p_mid"] >= high_thr:
            p_tgt = +MAX_POWER  # discharge (export)
        else:
            p_tgt = 0.0

        # --- apply ramp limits (per-direction)
        # allowed range given previous power
        p_min_by_ramp = prev_p - RAMP_CH
        p_max_by_ramp = prev_p + RAMP_DC
        p = float(np.clip(p_tgt, p_min_by_ramp, p_max_by_ramp))

        # --- apply power bounds
        p = float(np.clip(p, -MAX_POWER, +MAX_POWER))

        # --- apply SoC feasibility -> cap p further
        if p >= 0:  # discharge: must have enough energy in SoC to deliver p*Δt
            p_by_soc = (soc * eta_dc) / Delta_t
            p = min(p, p_by_soc)
        else:       # charge: must have enough headroom to accept p*Δt*η_ch
            headroom = MAX_CAPACITY - soc
            p_abs_by_soc = headroom / (eta_ch * Delta_t)
            p = max(p, -p_abs_by_soc)

        # --- apply daily cycle throughput cap
        remaining_throughput = max(0.0, daily_throughput_limit - used_throughput_today)
        if remaining_throughput <= 1e-12:
            p = 0.0
        else:
            e_if_p = abs(p) * Delta_t
            if e_if_p > remaining_throughput:
                # scale p down proportionally
                p = np.sign(p) * (remaining_throughput / Delta_t)

        # --- compute SoC update with efficiencies
        if p >= 0:  # discharge to grid
            e_grid = p * Delta_t                       # MWh delivered to grid
            e_from_soc = e_grid / eta_dc               # MWh drawn from battery
            soc_new = soc - e_from_soc
        else:       # charge from grid
            e_grid = p * Delta_t                       # negative MWh (import)
            e_into_soc = (-e_grid) * eta_ch            # MWh stored
            soc_new = soc + e_into_soc

        # enforce exact SoC bounds (tiny numerical safety)
        soc_new = min(MAX_CAPACITY, max(MIN_CAPACITY, soc_new))

        # --- revenue (worst-case dual pricing)
        price = df.at[i, "price_sell"] if e_grid >= 0 else df.at[i, "price_buy"]
        revenue = e_grid * price - DEG_COST_EUR_PER_MWH * abs(e_grid)
        cum_rev += revenue

        # --- update trackers
        used_throughput_today += abs(e_grid)
        prev_p = p
        soc = soc_new

        rows.append({
            "datetime": r["datetime"],
            "price_long": r["price_long"],
            "price_short": r["price_short"],
            "price_sell": df.at[i, "price_sell"],
            "price_buy": df.at[i, "price_buy"],
            "p_mw": p,
            "e_mwh": e_grid,
            "soc_mwh": soc,
            "revenue": revenue,
            "cum_revenue": cum_rev,
            "throughput_used_today_mwh": used_throughput_today,
            "efc_used_today": used_throughput_today / (2.0 * MAX_CAPACITY),
            "day": str(day)
        })

    res = pd.DataFrame(rows)
    return res



def plot_cumrev_greedy_vs_opt(
    greedy_df,
    opt_df,
    greedy_rev_col="revenue",
    opt_rev_col="revenue",
    greedy_time_col="datetime",
    opt_time_col="datetime",

    # optional window
    start=None, end=None,

    # start plotting only after N full days from the earliest of the two series
    start_after_days=0,
    rebase_at_cutoff=True,          # start both curves at 0 at the cutoff

    # visuals
    linewidth_scale=1.6,
    figsize=(18, 4.8),
    legend_outside_right=True,
    greedy_label="Greedy cumulative (€)",
    opt_label="Optimised cumulative (€)",
    title="Cumulative Revenue — Greedy vs Optimised",
    step=False                      # set True for step-style curves
):
    # colors (match your palette for price as the greedy line)
    COLOR_GREEDY = "#1f77b4"  # blue (same as imbalance price)
    COLOR_OPT    = "#ff7f0e"  # orange

    _set_theme()
    g = _ensure_dt_index(greedy_df, time_col=greedy_time_col).copy()
    o = _ensure_dt_index(opt_df,    time_col=opt_time_col).copy()

    if start is not None or end is not None:
        g = g.loc[start:end]
        o = o.loc[start:end]

    if g.empty or o.empty:
        raise ValueError("No rows in the selected window for one or both series.")

    # Common cutoff (earliest start across both), then +N days if requested
    earliest = min(g.index.min(), o.index.min()).normalize()
    cutoff = earliest + pd.Timedelta(days=int(start_after_days))
    g = g.loc[g.index >= cutoff]
    o = o.loc[o.index >= cutoff]
    if g.empty or o.empty:
        raise ValueError(f"No data on/after {cutoff.date()} for one or both series.")

    # Build cumulative series
    g_rev = pd.to_numeric(g[greedy_rev_col], errors="coerce").fillna(0.0)
    o_rev = pd.to_numeric(o[opt_rev_col],    errors="coerce").fillna(0.0)

    g_cum = g_rev.cumsum()
    o_cum = o_rev.cumsum()
    if rebase_at_cutoff:
        if len(g_cum) > 0: g_cum = g_cum - g_cum.iloc[0]
        if len(o_cum) > 0: o_cum = o_cum - o_cum.iloc[0]

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if step:
        ax.step(g_cum.index, g_cum.values, where="post",
                color=COLOR_GREEDY, linewidth=2.0*linewidth_scale, label=greedy_label)
        ax.step(o_cum.index, o_cum.values, where="post",
                color=COLOR_OPT,    linewidth=2.0*linewidth_scale, label=opt_label)
    else:
        sns.lineplot(ax=ax, x=g_cum.index, y=g_cum.values,
                     color=COLOR_GREEDY, linewidth=2.0*linewidth_scale, label=greedy_label)
        sns.lineplot(ax=ax, x=o_cum.index, y=o_cum.values,
                     color=COLOR_OPT,    linewidth=2.0*linewidth_scale, label=opt_label)

    ax.axhline(0, lw=1.2, alpha=0.6)
    ax.set_ylabel("Cumulative Revenue (€)")
    ax.set_xlabel("Date")
    ax.set_title(title)

    # Show dates (no times)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    ax.grid(True, axis="y", alpha=0.30)

    if legend_outside_right:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  frameon=True, ncol=1, borderaxespad=0.0)
        plt.tight_layout(rect=(0.0, 0.0, 0.86, 1.0))
    else:
        ax.legend(loc="upper left", frameon=True)
        plt.tight_layout()

    plt.show()

def kpis_extended(
    df,
    Delta_t=0.25,
    CAPACITY_MWH=2.0,
    DEG_COST_EUR_PER_MWH=1.5,
    power_cols=("p_mw","net_power_mw"),
    energy_cols=("e_mwh","imbalance_mwh"),
    soc_col="soc_mwh",
    long_col="price_long",
    short_col="price_short",
    time_col=None,  # pass your datetime column name here (e.g., "datetime")
):
    import pandas as pd, numpy as np

    d = df.copy()

    # ---------- time index ----------
    if not isinstance(d.index, pd.DatetimeIndex):
        cand = None
        if time_col and time_col in d.columns:
            cand = time_col
        else:
            common = {"datetime","start_datetime_utc","timestamp","time","date"}
            for c in d.columns:
                if c.lower() in common:
                    cand = c; break
            if cand is None:
                for c in d.columns:
                    lc = c.lower()
                    if ("date" in lc) or ("time" in lc):
                        cand = c; break
        if cand is None:
            raise ValueError("No datetime-like column found. Pass time_col='your_column_name'.")
        d[cand] = pd.to_datetime(d[cand], utc=True, errors="coerce")
        if d[cand].isna().all():
            raise ValueError(f"Could not parse datetime from column '{cand}'.")
        d = d.sort_values(cand).set_index(cand)

    # ---------- energy per PTU ----------
    ecol = next((c for c in energy_cols if c in d.columns), None)
    if ecol:
        d["e_mwh"] = pd.to_numeric(d[ecol], errors="coerce").fillna(0.0)
    else:
        pcol = next((c for c in power_cols if c in d.columns), None)
        if pcol:
            d[pcol] = pd.to_numeric(d[pcol], errors="coerce").fillna(0.0)
            d["e_mwh"] = d[pcol] * float(Delta_t)
        else:
            d["e_mwh"] = 0.0  # keep code running even if absent

    # ---------- SoC (optional) ----------
    if soc_col in d.columns:
        d[soc_col] = pd.to_numeric(d[soc_col], errors="coerce")

    # ---------- prices & dual-pricing mask (optional) ----------
    have_prices = (long_col in d.columns) and (short_col in d.columns)
    if have_prices:
        d[long_col]  = pd.to_numeric(d[long_col], errors="coerce")
        d[short_col] = pd.to_numeric(d[short_col], errors="coerce")
        d["price_sell"] = d[[long_col, short_col]].min(axis=1)
        d["price_buy"]  = d[[long_col, short_col]].max(axis=1)
        dual_mask = d[long_col].ne(d[short_col])
    else:
        d["price_sell"] = np.nan
        d["price_buy"]  = np.nan
        dual_mask = pd.Series(False, index=d.index)

    # ---------- revenue (use existing, else reconstruct) ----------
    if "revenue" not in d.columns:
        if have_prices:
            gross = np.where(d["e_mwh"] >= 0,
                             d["e_mwh"] * d["price_sell"],
                             d["e_mwh"] * d["price_buy"])
            d["revenue"] = gross - DEG_COST_EUR_PER_MWH * d["e_mwh"].abs()
        else:
            d["revenue"] = 0.0

    # ---------- aggregates ----------
    exp_mwh   = d["e_mwh"].clip(lower=0).sum()
    imp_mwh   = (-d["e_mwh"].clip(upper=0)).sum()
    throughput = d["e_mwh"].abs().sum()
    efc_total  = throughput / (2 * CAPACITY_MWH) if CAPACITY_MWH > 0 else np.nan

    exp_eur = (d["e_mwh"].clip(lower=0) * d["price_sell"]).sum(skipna=True)
    imp_eur = (d["e_mwh"].clip(upper=0) * d["price_buy"]).sum(skipna=True)  # negative
    deg_eur = DEG_COST_EUR_PER_MWH * throughput
    net_eur = d["revenue"].sum()

    # weighted prices
    w_buy  = d["e_mwh"].clip(upper=0).abs()
    w_sell = d["e_mwh"].clip(lower=0)
    avg_buy  = (w_buy * d["price_buy"]).sum()  / w_buy.sum()  if w_buy.sum()  > 0 else np.nan
    avg_sell = (w_sell * d["price_sell"]).sum() / w_sell.sum() if w_sell.sum() > 0 else np.nan
    realised_margin_per_mwh = (exp_eur + imp_eur - deg_eur) / throughput if throughput > 0 else np.nan

    # ---------- daily stats ----------
    d["date"] = d.index.date
    daily_rev = d.groupby("date")["revenue"].sum()
    best_day  = (daily_rev.idxmax(), float(daily_rev.max())) if len(daily_rev) else (None, np.nan)
    worst_day = (daily_rev.idxmin(), float(daily_rev.min())) if len(daily_rev) else (None, np.nan)
    vol_day   = float(daily_rev.std()) if len(daily_rev) else np.nan

    # EFC per day (abs before groupby)
    daily_throughput = d["e_mwh"].abs().groupby(d["date"]).sum()
    efc_per_day      = daily_throughput / (2 * CAPACITY_MWH) if CAPACITY_MWH > 0 else pd.Series(dtype=float)
    avg_efc_per_day  = float(efc_per_day.mean()) if not efc_per_day.empty else np.nan
    max_efc_per_day  = float(efc_per_day.max())  if not efc_per_day.empty else np.nan

    # drawdown
    cum    = d["revenue"].cumsum()
    max_dd = float((cum - cum.cummax()).min()) if len(cum) else np.nan

    # utilisation & behaviour
    pcol = next((c for c in power_cols if c in d.columns), None)
    if pcol:
        active = (d[pcol].abs() > 1e-6)
        util_pct = 100 * active.mean()
        charge_pct = 100 * (d[pcol] < -1e-6).mean()
        discharge_pct = 100 * (d[pcol] >  1e-6).mean()
        avg_abs_p_active = float(d.loc[active, pcol].abs().mean()) if active.any() else 0.0
        sign = np.sign(d[pcol].values)
        sign[np.abs(d[pcol].values) <= 1e-6] = 0
        switches = int(np.sum((sign[1:] * sign[:-1]) == -1))
    else:
        util_pct = charge_pct = discharge_pct = avg_abs_p_active = 0.0
        switches = 0

    # SoC stats (optional)
    soc_stats = {}
    if soc_col in d.columns:
        soc_stats = {
            "soc_min": float(d[soc_col].min()),
            "soc_avg": float(d[soc_col].mean()),
            "soc_max": float(d[soc_col].max()),
            "soc_at_min_pct": 100 * (np.isclose(d[soc_col], 0.0, atol=1e-3)).mean(),
            "soc_at_max_pct": 100 * (np.isclose(d[soc_col], CAPACITY_MWH, atol=1e-3)).mean(),
        }

    # dual exposure
    dual_pct = 100 * dual_mask.mean() if len(dual_mask) else np.nan
    dual_eur_share = 100 * d.loc[dual_mask, "revenue"].sum() / net_eur if (net_eur and dual_mask.any()) else np.nan

    return {
        # Economics
        "net_revenue_eur": float(net_eur),
        "export_revenue_eur": float(exp_eur),
        "import_cost_eur": float(imp_eur),            # negative
        "degradation_cost_eur": float(deg_eur),
        "eur_per_mwh_throughput": float(net_eur / throughput) if throughput > 0 else np.nan,
        "eur_per_efc": float(net_eur / efc_total) if efc_total and not np.isnan(efc_total) else np.nan,
        "best_day": best_day,
        "worst_day": worst_day,
        "daily_volatility_eur": vol_day,
        "max_drawdown_eur": max_dd,

        # Operations
        "throughput_mwh": float(throughput),
        "total_efc": float(efc_total) if not np.isnan(efc_total) else np.nan,
        "avg_efc_per_day": avg_efc_per_day,   
        "max_efc_per_day": max_efc_per_day,   
        "utilisation_pct": float(util_pct),
        "charge_pct": float(charge_pct),
        "discharge_pct": float(discharge_pct),
        "avg_abs_power_active_mw": float(avg_abs_p_active),
        "switches": switches,
        **soc_stats,

        # Market interaction
        "dual_windows_pct": float(dual_pct),
        "dual_windows_revenue_share_pct": float(dual_eur_share),
        "avg_buy_price_eur_per_mwh": float(avg_buy),
        "avg_sell_price_eur_per_mwh": float(avg_sell),
        "realised_margin_per_mwh_eur": float(realised_margin_per_mwh),
    }


# ---------------- ai generated greedy baseline, just for the sake of comparison and showing concept of greedy vs optimised ----------------
def greedy_battery_trader(
    prices_df,
    time_col="START_DATETIME_UTC",
    long_col="IMBALANCE_LONG_EUR_MWH",
    short_col="IMBALANCE_SHORT_EUR_MWH",
    # battery / market params
    Delta_t=0.25,
    MAX_CAPACITY=2.0,
    MIN_CAPACITY=0.0,
    MAX_POWER=1.0,
    INIT_CAPACITY=0.0,
    EFFICIENCY=0.9,             # roundtrip
    RAMP_CH=1.0,                # MW/PTU (towards more negative, i.e., charging)
    RAMP_DC=1.0,                # MW/PTU (towards more positive, i.e., discharging)
    CYCLES_PER_DAY_MAX=2.0,     # equivalent full cycles / day
    DEG_COST_EUR_PER_MWH=1.5,
    # greedy thresholds
    low_q=0.30,
    high_q=0.70,
):
    """
    Returns a DataFrame with:
      datetime, price_long, price_short, price_sell, price_buy, p_mw, e_mwh, soc_mwh,
      revenue, cum_revenue, throughput_mwh_day, efc_used_day
    """
    # prepare
    df = prices_df.copy()
    # rename/standardize
    df.rename(columns={time_col: "datetime", long_col: "price_long", short_col: "price_short"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.sort_values("datetime").reset_index(drop=True)
    df["date"] = df["datetime"].dt.date

    # worst-case prices
    df["price_sell"] = df[["price_long", "price_short"]].min(axis=1)  # for export (+)
    df["price_buy"]  = df[["price_long", "price_short"]].max(axis=1)  # for import (–)

    # greedy thresholds on a mid-price
    df["p_mid"] = 0.5*(df["price_long"] + df["price_short"])
    low_thr  = df["p_mid"].quantile(low_q)
    high_thr = df["p_mid"].quantile(high_q)

    # split η round-trip symmetrically
    eta_leg = np.sqrt(float(EFFICIENCY))
    eta_ch, eta_dc = eta_leg, eta_leg

    # daily throughput budget (MWh of |grid energy|) from EFC cap:
    # EFC = throughput / (2*Capacity)  => throughput_limit = 2 * Capacity * EFC_max
    daily_throughput_limit = 2.0 * MAX_CAPACITY * CYCLES_PER_DAY_MAX

    # simulate
    rows = []
    soc = float(INIT_CAPACITY)
    prev_p = 0.0
    used_throughput_today = 0.0
    current_day = df["date"].iloc[0] if not df.empty else None
    cum_rev = 0.0

    for i, r in df.iterrows():
        day = r["date"]
        if day != current_day:
            # reset daily counters at day boundary
            current_day = day
            used_throughput_today = 0.0

        # --- decide action: target power (MW) before constraints ---
        if r["p_mid"] <= low_thr:
            p_tgt = -MAX_POWER  # charge (import)
        elif r["p_mid"] >= high_thr:
            p_tgt = +MAX_POWER  # discharge (export)
        else:
            p_tgt = 0.0

        # --- apply ramp limits (per-direction)
        # allowed range given previous power
        p_min_by_ramp = prev_p - RAMP_CH
        p_max_by_ramp = prev_p + RAMP_DC
        p = float(np.clip(p_tgt, p_min_by_ramp, p_max_by_ramp))

        # --- apply power bounds
        p = float(np.clip(p, -MAX_POWER, +MAX_POWER))

        # --- apply SoC feasibility -> cap p further
        if p >= 0:  # discharge: must have enough energy in SoC to deliver p*Δt
            p_by_soc = (soc * eta_dc) / Delta_t
            p = min(p, p_by_soc)
        else:       # charge: must have enough headroom to accept p*Δt*η_ch
            headroom = MAX_CAPACITY - soc
            p_abs_by_soc = headroom / (eta_ch * Delta_t)
            p = max(p, -p_abs_by_soc)

        # --- apply daily cycle throughput cap
        remaining_throughput = max(0.0, daily_throughput_limit - used_throughput_today)
        if remaining_throughput <= 1e-12:
            p = 0.0
        else:
            e_if_p = abs(p) * Delta_t
            if e_if_p > remaining_throughput:
                # scale p down proportionally
                p = np.sign(p) * (remaining_throughput / Delta_t)

        # --- compute SoC update with efficiencies
        if p >= 0:  # discharge to grid
            e_grid = p * Delta_t                       # MWh delivered to grid
            e_from_soc = e_grid / eta_dc               # MWh drawn from battery
            soc_new = soc - e_from_soc
        else:       # charge from grid
            e_grid = p * Delta_t                       # negative MWh (import)
            e_into_soc = (-e_grid) * eta_ch            # MWh stored
            soc_new = soc + e_into_soc

        # enforce exact SoC bounds (tiny numerical safety)
        soc_new = min(MAX_CAPACITY, max(MIN_CAPACITY, soc_new))

        # --- revenue (worst-case dual pricing)
        price = df.at[i, "price_sell"] if e_grid >= 0 else df.at[i, "price_buy"]
        revenue = e_grid * price - DEG_COST_EUR_PER_MWH * abs(e_grid)
        cum_rev += revenue

        # --- update trackers
        used_throughput_today += abs(e_grid)
        prev_p = p
        soc = soc_new

        rows.append({
            "datetime": r["datetime"],
            "price_long": r["price_long"],
            "price_short": r["price_short"],
            "price_sell": df.at[i, "price_sell"],
            "price_buy": df.at[i, "price_buy"],
            "p_mw": p,
            "e_mwh": e_grid,
            "soc_mwh": soc,
            "revenue": revenue,
            "cum_revenue": cum_rev,
            "throughput_used_today_mwh": used_throughput_today,
            "efc_used_today": used_throughput_today / (2.0 * MAX_CAPACITY),
            "day": str(day)
        })

    res = pd.DataFrame(rows)
    return res
