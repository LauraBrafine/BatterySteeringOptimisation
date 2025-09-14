import os
from pathlib import Path
import sys
# Project root = parent of this scripts/ directory
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bso import model, utils
from bso import test_validation as tv  

###################### MAIN CODE ######################
def main():
    # select solver
    solver = "highs"

    # Set input parameters
    Delta_t              = 0.25    # hours per period, since each t is a PTU, used to convert power to energy
    MAX_CAPACITY         = 2       # MWh
    MIN_CAPACITY         = 0       # MWh
    MAX_POWER            = 1       # MW for charging and discharging
    INIT_CAPACITY        = 0       # MWh
    EFFICIENCY           = 0.9     # roundtrip efficiency
    RAMP_CH              = 1.0     # MW per PTU for discharging
    RAMP_DC              = 1.0     # MW per PTU for discharging
    CYCLES_PER_DAY_MAX   = 2     # <= 1.5 equivalent full cycles per day  
    DEG_COST_EUR_PER_MWH = 1.5     # linear degradation €/MWh throughput

    # input data and results paths
    data_file = ROOT / "data" / "imbalance_prices.csv"
    out_stem  = ROOT / "results" / "battery_week_results"

    # load data
    imbalance_prices = utils.import_imbalance_prices(data_file)

    # run model
    result, model_obj = model.battery_optimisation(
        imbalance_prices, solver, MIN_CAPACITY, MAX_CAPACITY, MAX_POWER,
        EFFICIENCY, INIT_CAPACITY, RAMP_CH, RAMP_DC,
        CYCLES_PER_DAY_MAX, DEG_COST_EUR_PER_MWH
    )

    # build config for running checks and smoke tests
    cfg = tv.ModelConfig(
        delta_t=Delta_t,
        min_capacity=MIN_CAPACITY,
        max_capacity=MAX_CAPACITY,
        max_power=MAX_POWER,
        efficiency=EFFICIENCY,
        init_capacity=INIT_CAPACITY,
        cycles_per_day_max=CYCLES_PER_DAY_MAX,
    )
    # run checks
    def _build_and_solve(prices_df, **kwargs):
        return model.battery_optimisation(
        prices_df, kwargs["solver"],
        kwargs["MIN_CAPACITY"], kwargs["MAX_CAPACITY"], kwargs["MAX_POWER"],
        kwargs["EFFICIENCY"], kwargs["INIT_CAPACITY"],
        kwargs["RAMP_CH"], kwargs["RAMP_DC"],
        kwargs["CYCLES_PER_DAY_MAX"], kwargs["DEG_COST_EUR_PER_MWH"]
        )
    
    # run smoek tests
    ok_smokes = tv.run_quick_synthetic_smokes(solver, _build_and_solve)
    ok_checks = tv.run_basic_postsolve_checks(model_obj, imbalance_prices, cfg)

    # only save the data to results output if all tests are OK
    if ok_checks and ok_smokes:
        print("Model solution validated.")
        utils.save_model_results(model_obj, out_stem)
    else:
        print("Validation failed (see messages above).")


if __name__ == "__main__":
    main()