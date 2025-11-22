import pybamm
import numpy as np

class BatterySim:
    def __init__(self, 
                 step_size_minutes=15, 
                 battery_capex=10.0, 
                 eol_soh=0.80):
        """
        A battery simulator that tracks physics and economic degradation.
        step_size_minutes (float): Time step size in minutes
        battery_capex (float): Initial cost of battery ($)
        eol_soh (float): State of health at which we terminate battery (buy new one)
        """
        
        # 1. Economic variables
        self.dt_seconds = step_size_minutes * 60
        self.capex = battery_capex
        self.eol_soh = eol_soh

        # 2. Physics variables
        options = {
            "SEI": "ec reaction limited",
            "SEI porosity change": "true",
            "thermal": "lumped", 
        }
        
        # PROBABLY NEED TO CHANGE TO INDUSTRIAL SCALE
        self.model = pybamm.lithium_ion.SPM(options=options)
        self.param = pybamm.ParameterValues("Mohtat2020")
        self.param.update({"Current function [A]": "[input]"})
        
        # 3. Internal state variables
        self.sim = None
        self.n_li_init = None       # Initial Mols of Lithium (for SOH)
        self.nominal_capacity = None # Initial Capacity in Ah (for SOC)
        self.current_capacity = None # Current Capacity in Ah (for SOC)
        self.prev_lithium_lost = 0.0
        
        # Degradation rate (Exponential Moving Average)
        self.degradation_rate_ema = 0.0
        self.ema_alpha = 0.01 # Smoothing factor (approx last 100 steps)
        
        self.reset()

    def reset(self):
        self.sim = pybamm.Simulation(self.model, parameter_values=self.param)
        
        # makes ministep to initialize variables
        self.sim.solve(t_eval=[0, 1e-6], inputs={"Current function [A]": 0})
        
        # Initialize Trackers (values are now available in self.sim.solution)
        self.n_li_init = self.sim.solution["Total lithium lost [mol]"].entries[-1] + \ #type: ignore
                         self.sim.solution["Total lithium in particles [mol]"].entries[-1] #type: ignore
        
        # Use evaluating parameters directly doesn't require a solve, but good to have
        self.nominal_capacity = self.param.evaluate(self.model.param.Q)
        self.current_capacity = self.nominal_capacity 
        self.prev_lithium_lost = 0.0
        
        # EMA for degradation rate
        self.deg_rate_ema = 0.0
        self.ema_alpha = 0.05 
        
        return self._get_current_state()

    def step(self, current_amps, instant_elec_cost, avg_daily_profit):
        """
        Args:
            current_amps (float): (+) Discharge, (-) Charge
            instant_elec_cost (float): $/kWh
            avg_daily_profit (float): $/day
        """
        
        # 1. Run Physics Solver
        try:
            sol = self.sim.step(dt=self.dt_seconds, inputs={"Current function [A]": current_amps}) #type: ignore
        except pybamm.SolverError:
            return None, True

        # 2. Extract Physical Properties
        lithium_lost_cumulative = sol["Total lithium lost [mol]"].entries[-1]
        voltage = sol["Terminal voltage [V]"].entries[-1]
        temp_k = sol["Cell temperature [K]"].entries[-1]

        ohmic = np.mean(sol["Ohmic heating [W]"].entries)
        irrev = np.mean(sol["Irreversible electrochemical heating [W]"].entries)
        heating_watts = ohmic + irrev

        # 3. Update SoC manually
        ah_exchanged = current_amps * (self.dt_seconds / 3600.0) # Current (A) * Time (h) = Capacity (Ah)
        self.current_capacity -= ah_exchanged # Update current capacity (flip signs)
        soc = np.clip(self.current_capacity / self.nominal_capacity, 0.0, 1.0)

        # 4. Calculate degradation
        delta_li_loss = lithium_lost_cumulative - self.prev_lithium_lost # Mols lost this step
        self.prev_lithium_lost = lithium_lost_cumulative # Update for next step
        
        # Exponential moving average of degradation rate
        if self.degradation_rate_ema == 0.0:
            self.degradation_rate_ema = delta_li_loss
        else:
            self.degradation_rate_ema = (self.ema_alpha * delta_li_loss) + ((1 - self.ema_alpha) * self.degradation_rate_ema)

        # Current State of Health (SOH)
        current_soh = 1.0 - (lithium_lost_cumulative / self.n_li_init)

        # Economic Costs
        costs = self._calculate_costs(delta_li_loss, heating_watts, current_soh, instant_elec_cost, avg_daily_profit)

        # 5. Package Output
        state = {
            "physics": {
                "voltage": voltage,
                "soc": soc, 
                "soh": current_soh,
                "temperature_c": temp_k - 273.15,
                "lithium_lost_step": delta_li_loss
            },
            "costs": costs
        }

        # Check for End of Life
        done = current_soh < self.eol_soh
        
        return state, done

    def _calculate_costs(self, delta_li_loss, heating_watts, current_soh, instant_elec_cost, avg_daily_profit):
        """Internal helper to compute the three cost components."""
        
        # 1. Hardware Wear Cost (Capital Depreciation)
        allowable_li_loss = self.n_li_init * (1.0 - self.eol_soh) # How many mol of lithium can be lost before EOL? # type: ignore
        fraction_life_consumed = delta_li_loss / allowable_li_loss # Frac of total allowable_li_loss, lost in this step?
        cost_wear = fraction_life_consumed * self.capex # Estimate of wear cost this step

        # 2. Inefficiency Heat Cost
        heat_joules = heating_watts * self.dt_seconds  # Energy = Power * Time
        heat_kwh = heat_joules / 3.6e6 # Convert J to kWh
        cost_heat = heat_kwh * instant_elec_cost

        # 3. Revenue Compression (Opportunity Cost of Wear)
        val_per_mol_daily = avg_daily_profit / self.n_li_init # How much does 1 mol of lithium earn us per day? ($/mol/day)
        steps_per_day = 86400 / self.dt_seconds
        mol_lost_per_day = self.degradation_rate_ema * steps_per_day # Avg mol lost per step * steps per day
        pct_lost_per_day = mol_lost_per_day / self.n_li_init # Turn into a rate #type: ignore
        if pct_lost_per_day <= 1e-9:
            days_remaining = 3650 # Default to 10 years if first step or negligible degradation
        else:
            days_remaining = (current_soh - self.eol_soh) / pct_lost_per_day
        cost_opportunity = delta_li_loss * val_per_mol_daily * days_remaining # capacity lost (%) * daily value of a mol ($/day) * expected days remaining (days)

        return {
            "wear": cost_wear,
            "heat": cost_heat,
            "opportunity": cost_opportunity,
            "total": cost_wear + cost_heat + cost_opportunity
        }

    def _get_current_state(self):
        return {
            "physics": {
                "voltage": 0,
                "soc": 0.0, # Start empty
                "soh": 1.0,
                "temperature_c": 25.0,
                "lithium_lost_step": 0.0
            },
            "costs": {"wear": 0.0, "heat": 0.0, "opportunity": 0.0, "total": 0.0}
        }