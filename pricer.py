import numpy as np
from typing import Union, Callable
from enum import Enum
from pde_solver import solve_1d_pde, FiniteDifferencingScheme
import scipy

class OptionType(Enum):
    CALL = 0
    PUT = 1

    def __eq__(self, other):
        return self.__class__ is other.__class__ and other.value == self.value


def price(maturity: float,
          strike: float,
          option_type: OptionType,
          number_of_spatial_levels: int,
          number_of_time_steps: int,
          pde_coefficients: Callable[[float], dict],
          scheme: FiniteDifferencingScheme,
          spot: float,
          volatility: float,
          interest_rate: float,
          number_of_std_deviations: int = 4,
          minimum_time_step_size: float = 1e-5,
          ) -> Union[float, None]:
    """Finds a price of a product using numerical PDE solver.

        number_of_spatial_levels (int):  number of spatial levels, i.e. number of state variable slices.
        number_of_time_steps (int): number of time steps, i.e. number of time variable slices.
    """

    # If step_size (which is equal to T / (M - 1)) is small enough, truncate it up to the lower bound and hence 
    # effective number of time steps is bounded from above. 
    step_size = maturity/np.abs(number_of_time_steps - 1)
    if step_size < minimum_time_step_size:
        number_of_time_steps = int(maturity / minimum_time_step_size)
    
    # x_min and x_max correspond to a left and right boundaries in phase space. 
    x_min, x_max = None, None

    # The solution domain in phase space is [-n_std * std, +n_std * std]. The standard deviation is basically sigma * sqrt(T). 
    x_min = -number_of_std_deviations * volatility * np.sqrt(maturity)
    x_max = number_of_std_deviations * volatility * np.sqrt(maturity)

    # If strike K is too high, that is K > S_0 * e^{x_max + (r - sigma ** 2 / 2) * T}, then
    # if Call, the price is set to 0, 
    # if Put, the price is set to its intrinsic value K * e^{-rT} - S_0. 
    if (strike > spot * np.exp(x_max + (interest_rate - volatility ** 2 / 2) * maturity)):
        if option_type == OptionType.CALL:
            return 0
        return strike * np.exp(-interest_rate * maturity) - spot
                
    # If strike K is too low, that is K < S_0 * e^{x_min + (r - sigma ** 2 / 2) * T}, then
    # if Call, the price is set to its intrinsic value S_0 - K * e^{-rT}. 
    # if Put, the price is set to 0.
    if (strike < spot * np.exp(x_min + (interest_rate - volatility ** 2 / 2) * maturity)):
        if option_type == OptionType.CALL:
            return spot - strike * np.exp(-interest_rate * maturity)
        return 0

    # Define a lower boundary condition, that is for x_min. 
    # If Call, the boundary condition is set to 0.
    # If Put, the boundary condition is set to Put intrinsic value at time t, forwarded to expiration T, that is 
    # e^{r(T-t)} * (K * e^{-r(T-t) - S_0 * e^{x + (r - \sigma ** 2 / 2) * t}}).
    def lower_boundary_condition(x: float, time: float):
        if option_type == OptionType.CALL:
            return 0
        return np.exp(interest_rate * (maturity - time)) * (strike * np.exp(interest_rate * (time - maturity)) - spot * np.exp(x + (interest_rate - volatility ** 2 / 2) * time))

    # Define an upper boundary condition, that is for x_max. 
    # If Put, the boundary condition is set to 0.
    # If Call, the boundary condition is set to Call intrinsic value at time t, forwarded to expiration T, that is 
    # e^{r(T-t)} * (S_0 * e^{x + (r - \sigma ** 2 / 2) * t}} - K * e^{-r(T-t)}).
    def upper_boundary_condition(x: float, time: float):
        if option_type == OptionType.PUT:
            return 0
        return np.exp(interest_rate * (maturity - time)) * (spot * np.exp(x + (interest_rate - volatility ** 2 / 2) * time) - strike * np.exp(interest_rate * (time - maturity)))

    # Define a terminal condition at t=T, which is (S_T(x)) - K)_{+} for Calls and (K - S_T(x)))_{+} for Puts
    # with S_T(x) = S_0 * e^{x + (r - sigma ** 2 / 2) * T}
    def terminal_condition(x): 
        payoff = lambda s: np.maximum(s - strike, 0) if option_type == OptionType.CALL else np.maximum(strike - s, 0)
        return payoff(spot * np.exp(x + (interest_rate - volatility ** 2 / 2) * maturity))

    # Call the PDE solver to retrieve the solution. 
    solution = solve_1d_pde(scheme=scheme,
                           lower_boundary_condition=lower_boundary_condition,
                           upper_boundary_condition=upper_boundary_condition,
                           terminal_condition=terminal_condition,
                           pde_coefficients=pde_coefficients,
                           number_of_spatial_levels=number_of_spatial_levels,
                           number_of_time_steps=number_of_time_steps,
                           x_min=x_min,
                           x_max=x_max,
                           t_start=0,
                           t_end=maturity)  
              
    # If HeatEquatinon, we need to discount the T-forward prices to a value date. 
    solution["FunctionValues"] *= np.exp(-interest_rate * maturity)
            
    # Tranform the spatial grid [x_min, ..., x_max] to [s_min, ..., s_max]
    spots = spot * np.exp(solution["SpatialGrid"])

    # Interpolate the prices given at corresponding spot values [s_min, ..., s_max]
    interp = scipy.interpolate.interp1d(spots, solution["FunctionValues"], kind="cubic")

    # Return the interplated value 
    return interp(spot)