import numpy as np
import pytest

from fenicsxconcrete.finite_element_problem import ConcreteThermoMechanical
from fenicsxconcrete.util import ureg


def test_hydration_function():

    T = ureg.Quantity(35.0, ureg.degC).to_base_units().magnitude
    dt = 60 * 30
    time_list = [40000]
    parameter = {}
    parameter["B1"] = 2.916e-4
    parameter["B2"] = 0.0024229
    parameter["eta"] = 5.554
    parameter["alpha_max"] = 0.875
    parameter["E_act"] = 47002
    parameter["T_ref"] = ureg.Quantity(25.0, ureg.degC).to_base_units().magnitude
    parameter["Q_pot"] = 500e3

    # initiate material problem
    experiment, parameters = ConcreteThermoMechanical.default_parameters()
    material_problem = ConcreteThermoMechanical(experiment=experiment, parameters=parameters)
    # get the respective function
    hydration_fkt = material_problem.get_heat_of_hydration_ftk()

    heat_list, doh_list = hydration_fkt(T, time_list, dt, parameter)
    # print(heat_list)
    # exit()
    assert heat_list == pytest.approx(np.array([239.06484735]))
    assert doh_list == pytest.approx(np.array([0.47812969]))
    # assert heat_list == pytest.approx(np.array([169.36164423]))
    # assert doh_list == pytest.approx(np.array([0.33872329]))
    # problem.experiment.apply_displ_load(displacement)
    # problem.solve()  # solving this
