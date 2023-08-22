import os
from pathlib import Path

import numpy as np
import pytest

from fenicsxconcrete.experimental_setup import SimpleCube
from fenicsxconcrete.finite_element_problem import ConcreteThermoMechanical, LinearElasticity
from fenicsxconcrete.sensor_definition.displacement_sensor import DisplacementSensor
from fenicsxconcrete.sensor_definition.strain_sensor import StrainSensor
from fenicsxconcrete.sensor_definition.stress_sensor import StressSensor
from fenicsxconcrete.util import ureg


@pytest.mark.parametrize("dim", [2, 3])
def test_mechanical_only(dim: int) -> None:
    # defining experiment parameters
    parameters_exp = {}
    parameters_exp["dim"] = dim * ureg("")
    parameters_exp["num_elements_length"] = 2 * ureg("")
    parameters_exp["num_elements_height"] = 2 * ureg("")
    parameters_exp["num_elements_width"] = 2 * ureg("")
    parameters_exp["strain_state"] = "uniaxial" * ureg("")

    displacement = 0.01 * ureg("m")

    parameters = {}
    parameters["rho"] = 7750 * ureg("kg/m^3")
    parameters["E"] = 210e9 * ureg("N/m^2")
    parameters["nu"] = 0.28 * ureg("")

    experiment = SimpleCube(parameters_exp)

    problem_elastic = LinearElasticity(experiment, parameters, pv_name=f"pure_mechanical_test_{dim}", pv_path="")

    _, parameters_thermo = ConcreteThermoMechanical.default_parameters()
    parameters_thermo["nu"] = parameters["nu"].copy()
    parameters_thermo["E_28"] = parameters["E"].copy()
    parameters_thermo["q_degree"] = 4 * ureg("")

    problem_thermo_mechanical = ConcreteThermoMechanical(
        experiment, parameters_thermo, pv_name=f"thermo_mechanical_test_{dim}", pv_path=""
    )

    # apply displacement load and solve
    experiment.apply_displ_load(displacement)
    experiment.apply_body_force()

    problem_elastic.solve()
    problem_elastic.pv_plot()

    # problem_thermo_mechanical.experiment.apply_displ_load(displacement)
    problem_thermo_mechanical.temperature_problem.q_alpha.vector.array[:] = parameters_thermo["alpha_max"].magnitude

    problem_thermo_mechanical.mechanics_solver.solve(problem_thermo_mechanical.fields.displacement)
    problem_thermo_mechanical.pv_plot()

    assert problem_thermo_mechanical.q_fields.youngs_modulus.vector.array[:] == pytest.approx(
        parameters["E"].magnitude
    )

    np.testing.assert_allclose(
        problem_thermo_mechanical.fields.displacement.vector.array,
        problem_elastic.fields.displacement.vector.array,
        rtol=1e-4,
    )
