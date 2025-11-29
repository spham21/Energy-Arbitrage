"""
Manuel Sage, January 2024

Base classes of environments.
"""
import abc
from typing import Optional

import numpy as np
import gymnasium as gym
from battery_env import BatterySim
from datetime import datetime

#import pandas as pd

class DiscreteEnv(gym.Env):

    def __init__(self, modeling_period: int, dataframe: str, precision_level: Optional[str] = "low" ):
        """
        :param env_name: A string that represents the name of the environment.
        :param modeling_period: An integer that represents the modeling period (in whatever increment the provided dataframe is in)
        :param precision_level: precision level of stored floats

        """
        super().__init__()
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=np.array([0.2,-np.inf]), high=np.array([0.8,np.inf])) # No predictions used
        

        self.env_name = "Energy grid"
        self.modeling_period = modeling_period
        self.precision = precision_level
        self.count = None
        self.state = None
        self.data = dataframe.reset_index(drop=True) # Date should be in datetime format!!
        self.step_size_minutes = (self.data.iloc[1].Date-self.data.iloc[0].Date).seconds/60
        self.battery = BatterySim(self.step_size_minutes, plant_capacity=10.0, plant_capex=300000) 
        self.battery_state = None
        self.cum_reward = 0.05 # seed estimate
        self.init_index = None

    @property
    def precision(self):
        """
        A property that returns the precision of the variables as a dictionary with keys 'float' and 'int'.

        :return: A dictionary that represents the precision of the variables.
        """
        return {"float": self._precision_float, "int": self._precision_int}

    @precision.setter
    def precision(self, value):
        """
        A setter method for the precision of the variables.

        :param value: A string that represents the precision level of the variables. Must be either 'low', 'medium', or
        'high'.
        """
        if value == "low":
            self._precision_float = np.float32
            self._precision_int = np.int8
        elif value == "medium":
            self._precision_float = np.float64
            self._precision_int = np.int16
        elif value == "high":
            self._precision_float = np.float128
            self._precision_int = np.int32

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.

        :param seed: An integer that represents the random seed. Default is None.
        :param options: A dictionary of options. Default is None.
        """
        super().reset(seed=seed)  # Defines np's random generator
        # ADD: get state from first line
        self.init_index = round(np.random.uniform(0, len(self.data)-self.modeling_period-1))
        self.count = 0

        self.battery_state = self.battery._get_current_state()
        soc = self.battery_state["physics"]["soc"]

        self.state = [soc, self.data.iloc[self.count+self.init_index].price]

        return self.state, self._get_info()
        

    def step(self, action):
        #self._action_checker(action)
        done = False

        action_list = [-1, 0, 1] # in MW        
        action_power = action_list[action]

        profit_guess = 0 if self.count == 0 else self.cum_reward/((self.step_size_minutes*self.count)*60*24)
        
        self.battery_state, battery_done = self.battery.step(action_power, self.state[1]/1000, profit_guess) # HARDCODED DAILY PROFIT GUESS
        
        if self.battery_state is None: # pybamm solver error 
            return [0,0], 0, True, False, self._get_info()
        
        actual_power = self.battery_state["physics"]["power_actual"] # in kW

        gross_profit = actual_power*1000 * self.step_size_minutes/60 * (self.state[1]/1000)
        reward = gross_profit - self.battery_state["costs"]["total"]
        self.cum_reward += reward
        
        self.count += 1
        price = self.data.iloc[self.count+self.init_index].price
        self.state = [self.battery_state["physics"]["soc"], price]
        print(f"Step {self.count}: Action {action_power} kW, Price {price} $/MWh, Reward {reward:.2f} $, SoC {self.state[0]:.4f}")

        if battery_done or (self.count >= self.modeling_period):
            done = True

        return self.state, reward, done, False, self._get_info()  # Truncated is false


    def _get_obs(self):
        return self.state


    def _action_checker(self, action: np.array):
        """
        Checks if the action is within the bounds of the action space and does not contain NaN values.

        :param action: A numpy array that represents the action to check.
        """
        if np.isnan(action).any():
            raise gym.error.InvalidAction(f'Action must not contain NaN values. Passed action: {action}')
        if not self.action_space.contains(action):
            raise gym.error.InvalidAction(f'Action must be within the action space bounds. Passed action: {action}')

    def _get_info(self):        
        return {}