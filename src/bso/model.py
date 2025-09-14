from pyomo.environ import *
from .utils import import_imbalance_prices, save_model_results, retrieve_days_daylabels

def battery_optimisation(imbalance_prices, solver, MIN_CAPACITY, MAX_CAPACITY, MAX_POWER, EFFICIENCY, 
                         INIT_CAPACITY, RAMP_CH, RAMP_DC, CYCLES_PER_DAY_MAX, DEG_COST_EUR_PER_MWH):
    
    ####################### initiate model #######################
    model = ConcreteModel()
    # set solver, use glpk solver, free and ogood for small MILP
    # tho solver can have impact on solution
    opt   = SolverFactory(solver) 

    ###################### create sets ##########################
    # create set for time range to be optimised
    # CHECK OUT WARNING MESSAGE
    # model.Period = Set(initialize=imbalance_prices.index.tolist())
    model.Period = RangeSet(0, len(imbalance_prices)-1)  # guaranteed ordered ints
    # New day sets/params
    days, times_by_day = retrieve_days_daylabels(imbalance_prices)
    model.Days = Set(initialize=days)
    model.TimesOfDay = Set(model.Days, initialize=times_by_day)  # subset: timesteps in each day

    # set delta-t
    Delta_t = 0.25

    ########## create parameters ##############
    # short price and long price imbalance create parameters
    model.Price_short       = Param(model.Period, initialize=imbalance_prices["IMBALANCE_SHORT_EUR_MWH"].to_dict())
    model.Price_long        = Param(model.Period, initialize=imbalance_prices["IMBALANCE_LONG_EUR_MWH"].to_dict())
    # Ramping & cost params (as Pyomo Params so they show up in LP)
    model.RampChUp   = Param(initialize=RAMP_CH)
    model.RampChDown = Param(initialize=RAMP_CH)
    model.RampDsUp   = Param(initialize=RAMP_DC)
    model.RampDsDown = Param(initialize=RAMP_DC)

    model.CyclesPerDayMax = Param(initialize=CYCLES_PER_DAY_MAX)
    model.DegCost         = Param(initialize=DEG_COST_EUR_PER_MWH)

    ######################### create variables ##################
    # create variables for max charge and discharge power, and for the capacity bounded by min max capacity
    model.Capacity          = Var(model.Period, bounds=(MIN_CAPACITY, MAX_CAPACITY))
    model.Charge_power      = Var(model.Period, bounds=(0, MAX_POWER))
    model.Discharge_power   = Var(model.Period, bounds=(0, MAX_POWER))

    # create two binary decision variables one for charging on for discharging
    # could also use 1 variable and use big M to enforce no charge discharge at the same time
    # but this option more robust if big M too loose
    model.y_c               = Var(model.Period, within = Binary)  # 1 if chargin in t
    model.y_d               = Var(model.Period, within = Binary)  # 1 if discharging in t

    ################ create objective function ###################

    # define objective function maximise revenue
    def objective_rule(model):
        energy_revenue = sum(
            model.Price_long[t]  * model.Discharge_power[t] * Delta_t
        - model.Price_short[t] * model.Charge_power[t]    * Delta_t
        for t in model.Period
        )
        degrade_cost = model.DegCost * sum(
            (model.Charge_power[t] + model.Discharge_power[t]) * Delta_t
            for t in model.Period
        )
        return energy_revenue - degrade_cost

    # add the objective rule to the model
    model.objective = Objective(rule=objective_rule, sense=maximize)

    ################## create constraints ##########################

    # no simultaneous charge and dischare constraint
    def no_simul_cd_rule(model, t):
        return model.y_c[t] + model.y_d[t] <= 1
    
    # charging constraints
    # linking charging to decsision variable
    def charging_rule(model, t):
        return model.Charge_power[t] <= MAX_POWER * model.y_c[t]

    # linking discharging to decsision variable
    def discharging_rule(model, t):
        return model.Discharge_power[t] <= MAX_POWER * model.y_d[t]

    # making sure no over charging
    def over_charge_rule(model, t):
        return model.Charge_power[t] <= (MAX_CAPACITY - model.Capacity[t]) / (Delta_t * EFFICIENCY)

    # making sure no over discharging
    def over_discharge_rule(model, t):
        return model.Discharge_power[t] <= model.Capacity[t] / (Delta_t)
    
    # SOC constraint
    # put effiency in SOC instead of objective function
    def soc_rule(model, t):
        # skip first time step, where init is already set
        if t == model.Period.first():
            return Constraint.Skip
            # since roundtrip efficiency is used ony has to be implemented either for charging or discharging
        return model.Capacity[t] == (model.Capacity[t-1] 
                                    + EFFICIENCY * model.Charge_power[t-1] * Delta_t 
                                    - model.Discharge_power[t-1] * Delta_t)
    
    # define cyclicity, make sure end is same as beginning
    def init_soc_rule(model):
        return model.Capacity[model.Period.first()] == INIT_CAPACITY

    def terminal_soc_rule(model):
        return model.Capacity[model.Period.last()] == INIT_CAPACITY
    
    def ch_ramp_up_rule(m, t):
        if t == m.Period.first(): 
            return Constraint.Skip
        return m.Charge_power[t] - m.Charge_power[t-1] <= m.RampChUp

    def ch_ramp_down_rule(m, t):
        if t == m.Period.first():
            return Constraint.Skip
        return m.Charge_power[t-1] - m.Charge_power[t] <= m.RampChDown

    def ds_ramp_up_rule(m, t):
        if t == m.Period.first():
            return Constraint.Skip
        return m.Discharge_power[t] - m.Discharge_power[t-1] <= m.RampDsUp

    def ds_ramp_down_rule(m, t):
        if t == m.Period.first():
            return Constraint.Skip
        return m.Discharge_power[t-1] - m.Discharge_power[t] <= m.RampDsDown

    def daily_throughput_rule(m, d):
        # energy moved in/out during day d (MWh)
        thru = sum( (m.Charge_power[t] + m.Discharge_power[t]) * Delta_t
                    for t in m.TimesOfDay[d] )
        return thru <= 2.0 * m.CyclesPerDayMax * MAX_CAPACITY
    
    # add constraints to model
    model.no_simul_cd_rule      = Constraint(model.Period, rule = no_simul_cd_rule)
    model.charging_rule         = Constraint(model.Period, rule = charging_rule)
    model.disharging_rule       = Constraint(model.Period, rule = discharging_rule)
    model.over_charge_rule      = Constraint(model.Period, rule = over_charge_rule)
    model.over_discharge_rule   = Constraint(model.Period, rule = over_discharge_rule)
    model.soc_rule              = Constraint(model.Period, rule = soc_rule)
    model.init_soc_rule         = Constraint(rule = init_soc_rule)
    model.terminal_soc_rule     = Constraint(rule = terminal_soc_rule)
    model.ch_ramp_up            = Constraint(model.Period, rule=ch_ramp_up_rule)
    model.ch_ramp_down          = Constraint(model.Period, rule=ch_ramp_down_rule)
    model.ds_ramp_up            = Constraint(model.Period, rule=ds_ramp_up_rule)
    model.ds_ramp_down          = Constraint(model.Period, rule=ds_ramp_down_rule)
    model.daily_throughput      = Constraint(model.Days, rule=daily_throughput_rule)

    #################### Solve Model #######################

    result = opt.solve(model, tee=True)#, keepfiles = True, logfile = 'cbc_run.log', load_solutions = False)
    #  Check solver status
    print(f"solver status: {result.solver.status}, terminal condition: {result.solver.termination_condition}")

    return result, model


if __name__ == "__main__":

    solver = 'highs'

    output_path = 'battery_week_results'

    Delta_t          = 0.25     # hours per period, since each t is a PTU, used to convert power to energy
    MAX_CAPACITY     = 2        # MWh
    MIN_CAPACITY     = 0        # MWh
    MAX_POWER        = 1        # MW for charging and discharging

    INIT_CAPACITY    = 0
    EFFICIENCY       = 0.9 

    RAMP_CH = 1.0
    RAMP_DC = 1.0                      # MW per PTU for discharging
    CYCLES_PER_DAY_MAX = 1.5           # ≤ 1.5 equivalent full cycles per day
    DEG_COST_EUR_PER_MWH = 2.0         # linear degradation €/MWh throughput

    imbalance_prices = import_imbalance_prices('imbalance_prices.csv')

    result, model = battery_optimisation(imbalance_prices, solver, MIN_CAPACITY, MAX_CAPACITY, 
                                              MAX_POWER, EFFICIENCY, INIT_CAPACITY, RAMP_CH, 
                                              RAMP_DC, CYCLES_PER_DAY_MAX, DEG_COST_EUR_PER_MWH)

    print(result)

    save_model_results(model, output_path, '1')