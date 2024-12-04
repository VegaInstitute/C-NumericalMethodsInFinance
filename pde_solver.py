from typing import Callable, Dict
from enum import Enum
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu, spsolve
from copy import deepcopy


class FiniteDifferencingScheme(Enum):
    Explicit = 0
    Implicit = 1

    def __eq__(self, other):
        return self.__class__ is other.__class__ and other.value == self.value


def solve_1d_pde(scheme: FiniteDifferencingScheme, lower_boundary_condition: Callable[[float, float], float], upper_boundary_condition: Callable[[float, float], float],
               terminal_condition: Callable[[np.ndarray], float], pde_coefficients: Callable[[float], dict], number_of_spatial_levels: int, number_of_time_steps: int,
               x_min: float, x_max: float, t_start: float, t_end: float) -> tuple[Dict[str, np.ndarray], np.ndarray]:
    """PDE solver.

    Args:
        scheme (FiniteDifferencingScheme): a numerical scheme used to obtain numerical solution (Explicit or Implicit)
        lower_boundary_condition (Callable[[float, float]): a lower boundary condition at $x=x_{min}$ and arbitrary t; a function from (x,t) to a boundary value.
        upper_boundary_condition (Callable[[float, float]): an upper boundary condition at $x=x_{max}$ and arbitrary t; a function from (x,t) to a boundary value.
        terminal_condition (Callable[[np.ndarray], float]): a terminal condition at $t=t_{end}$ and arbitrary x; a function from an array of x values to a terminal boundary value.
        pde_coefficients (Callable[[float], dict]): a function from state variable x to a mapping from a correposnding term in PDE to a coefficient in front of it.
        number_of_spatial_levels (int):  number of spatial levels, i.e. number of state variable slices.
        number_of_time_steps (int): number of time steps, i.e. number of time variable slices.
        x_min (float): the leftmost value in spatial grid.
        x_max (float): the rightmost value in spatial grid.
        t_start (float): the first value in time grid.
        t_end (float): the last value in time grid.
    """
    # Initialize appropriate time and spatial grids
    time_grid = np.linspace(t_start, t_end, number_of_time_steps)
    spatial_grid = np.linspace(x_min, x_max, number_of_spatial_levels)

    # Estimate the time step dt and spacial step dx
    dt = (t_end - t_start) / (number_of_time_steps - 1)
    dx = (x_max - x_min) / (number_of_spatial_levels - 1)

    # Initialize a workspace slice (which is, effectively, a u^{m} in scheme notations) with terminal condition at spatial grid
    workspace_slice = terminal_condition(spatial_grid)
    # Initialize a result slice (which is, effectively, a u^{m+1} in scheme notations)
    result_slice = np.empty_like(workspace_slice)

    # Estimate the coefficients in front of correposnding derivatives in PDE generator.
    # In case of heat equtation, that is, effectively, a {"V_ss": 0.5 * sigma ** 2, "V_t": 1}
    pde_coefficients_at_grid_values = pde_coefficients(spatial_grid)

    match scheme:
        case FiniteDifferencingScheme.Explicit:
            # Initialize the propagator tridiagonal matrix.
            main_diagonal = 1 + pde_coefficients_at_grid_values["U"] * dt - 2 * dt / (
                dx ** 2) * pde_coefficients_at_grid_values["U_xx"]
            upper_diagonal = pde_coefficients_at_grid_values["U_xx"] * dt / (
                dx ** 2) + pde_coefficients_at_grid_values["U_x"] * dt / (2 * dx)
            lower_diagonal = pde_coefficients_at_grid_values["U_xx"] * dt / \
                dx ** 2 - \
                pde_coefficients_at_grid_values["U_x"] * dt / (2 * dx)

            # Cast the main diagonal element to an array if needed.
            if isinstance(main_diagonal, float):
                main_diagonal = np.full(
                    number_of_spatial_levels, main_diagonal)

            # Cast the upper diagonal element to an array if needed.
            if isinstance(upper_diagonal, float):
                upper_diagonal = np.full(
                    number_of_spatial_levels, upper_diagonal)

            # Cast the lower diagonal element to an array if needed.
            if isinstance(lower_diagonal, float):
                lower_diagonal = np.full(
                    number_of_spatial_levels, lower_diagonal)

            # Initialize the propagator matrix A (that is, such matrix that u^{m+1} = A * u^{m}) in CSR (compressed sparse row) format.
            propagator = sparse.diags(
                [lower_diagonal[1:], main_diagonal, upper_diagonal[:-1]], (-1, 0, 1)).tocsc()

            # Main propagation loop, evlauating u^{m+1} = A * u^{m}.
            for i, t in enumerate(reversed(time_grid[:-1])):
                # YOUR CODE HERE
                # Compute u^{m+1} = A * u^{m}.
                result_slice = propagator @ workspace_slice
                # Handling lower boundary conditon.
                result_slice[0] = lower_boundary_condition(spatial_grid[0], t)
                # Handling upper boundary conditon.
                result_slice[-1] = upper_boundary_condition(
                    spatial_grid[-1], t)
                # Now workspsace slice points to a new slice u^{m+1}.
                workspace_slice = result_slice

        case FiniteDifferencingScheme.Implicit:
            # Initialize the propagator tridiagonal matrix.
            main_diagonal = 1 - pde_coefficients_at_grid_values["U"] * dt + 2 * dt / (
                dx ** 2) * pde_coefficients_at_grid_values["U_xx"]
            lower_diagonal = - pde_coefficients_at_grid_values["U_xx"] * dt / (
                dx ** 2) + pde_coefficients_at_grid_values["U_x"] * dt / (2 * dx)
            upper_diagonal = - pde_coefficients_at_grid_values["U_xx"] * dt / (
                dx ** 2) - pde_coefficients_at_grid_values["U_x"] * dt / (2 * dx)

            # Cast the main diagonal element to an array if needed.
            if isinstance(main_diagonal, float):
                main_diagonal = np.full(
                    number_of_spatial_levels, main_diagonal)

            # Cast the upper diagonal element to an array if needed.
            if isinstance(upper_diagonal, float):
                upper_diagonal = np.full(
                    number_of_spatial_levels, upper_diagonal)

            # Cast the lower diagonal element to an array if needed.
            if isinstance(lower_diagonal, float):
                lower_diagonal = np.full(
                    number_of_spatial_levels, lower_diagonal)

            # Initialize the propagator matrix A (that is, such matrix that u^{m} = A * u^{m+1}) in CSR (compressed sparse row) format.
            propagator = sparse.diags(
                [lower_diagonal[2:-1], main_diagonal[1:-1], upper_diagonal[1:-2]], (-1, 0, 1)).tocsc()
            # Compute LU decomposition of propagator A.
            inv_propagator = splu(propagator)
            # Main propagation loop, evlauating u^{m} = A * u^{m+1}.
            for i, t in enumerate(reversed(time_grid[:-1])):
                # YOUR CODE HERE
                # Handling lower boundary conditon.
                result_slice[0] = lower_boundary_condition(spatial_grid[0], t)
                # Handling upper boundary conditon.
                result_slice[-1] = upper_boundary_condition(
                    spatial_grid[-1], t)
                # Adjusting the second element u^{m}_{1} in work space slice to satisfy boyndary conditions.
                workspace_slice[1] -= lower_diagonal[1] * result_slice[0]
                # Adjusting the penultimate element u^{m}_{N-2} in work space slice to satisfy boyndary conditions.
                workspace_slice[-2] -= upper_diagonal[-2] * result_slice[-1]
                # Solve u^{m} = A * u^{m+1} and retrieve u^{m+1}.
                result_slice[1:-1] = inv_propagator.solve(workspace_slice[1:-1])
                # Now workspsace slice points to a new slice u^{m+1}.
                workspace_slice = result_slice
                
    return {"SpatialGrid": spatial_grid, "FunctionValues": result_slice}