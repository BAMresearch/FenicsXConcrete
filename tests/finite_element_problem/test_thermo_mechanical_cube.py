import os
from pathlib import Path

import numpy as np
import pytest

from fenicsxconcrete.experimental_setup import MinimalCubeExperiment
from fenicsxconcrete.finite_element_problem import ConcreteThermoMechanical, LinearElasticity
from fenicsxconcrete.sensor_definition import DisplacementSensor, StrainSensor, StressSensor
from fenicsxconcrete.sensor_definition.strain_sensor import StrainSensor
from fenicsxconcrete.sensor_definition.stress_sensor import StressSensor
from fenicsxconcrete.util import ureg


@pytest.mark.parametrize("dim", [2, 3])
def test_mechanical_only(dim: int) -> None:

    # defining experiment parameters
    parameters = {}

    parameters["dim"] = dim * ureg("")
    parameters["num_elements_length"] = 2 * ureg("")
    parameters["num_elements_height"] = 2 * ureg("")
    parameters["num_elements_width"] = 2 * ureg("")

    displacement = 0.01 * ureg("m")

    parameters["rho"] = 7750 * ureg("kg/m^3")
    parameters["E"] = 210e9 * ureg("N/m^2")
    parameters["nu"] = 0.28 * ureg("")

    # setting up the problem
    experiment = MinimalCubeExperiment(parameters)
    problem_elastic = LinearElasticity(experiment, parameters, pv_name="pure_mechanical_test.xdmf", pv_path="")

    parameters_thermo = ConcreteThermoMechanical.default_parameters()
    parameters_thermo["nu"] = parameters["nu"]
    parameters_thermo["E_28"] = parameters["E"]
    problem_thermo_mechanical = ConcreteThermoMechanical(
        experiment, parameters, pv_name="thermo_echanical_test.xdmf", pv_path=""
    )

    if dim == 2:
        sensor_location = [0.5, 0.5, 0.0]
    elif dim == 3:
        sensor_location = [0.5, 0.5, 0.5]

    # add sensors
    problem_elastic.add_sensor(StressSensor(sensor_location))
    problem_thermo_mechanical.add_sensor(StrainSensor(sensor_location))

    # apply displacement load and solve
    problem_elastic.experiment.apply_displ_load(displacement)
    problem_elastic.solve()

    problem_thermo_mechanical.experiment.apply_displ_load(displacement)
    problem_thermo_mechanical.mechanics_problem.q_values.degree_of_hydration.vector.array[:] = 1.0
    problem_thermo_mechanical.solve()
    # problem_elastic.pv_plot()


#     # checks
#     analytic_eps = (displacement.to_base_units() / (1.0 * ureg("m"))).magnitude

#     strain_result = problem.sensors["StrainSensor"].get_last_entry().magnitude
#     stress_result = problem.sensors["StressSensor"].get_last_entry().magnitude
#     if dim == 2:
#         # strain in yy direction
#         assert strain_result[-1] == pytest.approx(analytic_eps)
#         # strain in xx direction
#         assert strain_result[0] == pytest.approx(-problem.parameters["nu"].magnitude * analytic_eps)
#         # strain in xy and yx direction
#         assert strain_result[1] == pytest.approx(strain_result[2])
#         assert strain_result[1] == pytest.approx(0.0)
#         # stress in yy direction
#         assert stress_result[-1] == pytest.approx((analytic_eps * problem.parameters["E"]).magnitude)

#     elif dim == 3:
#         # strain in zz direction
#         assert strain_result[-1] == pytest.approx(analytic_eps)
#         # strain in yy direction
#         assert strain_result[4] == pytest.approx(-problem.parameters["nu"].magnitude * analytic_eps)
#         # strain in xx direction
#         assert strain_result[0] == pytest.approx(-problem.parameters["nu"].magnitude * analytic_eps)
#         # shear strains
#         sum_mixed_strains = (
#             strain_result[1]  # xy
#             - strain_result[3]  # yx
#             - strain_result[2]  # xz
#             - strain_result[6]  # zx
#             - strain_result[5]  # yz
#             - strain_result[7]  # zy
#         )
#         assert sum_mixed_strains == pytest.approx(0.0)

#         # stress in zz direction
#         assert stress_result[-1] == pytest.approx((analytic_eps * problem.parameters["E"].magnitude))


# @pytest.mark.parametrize("dim", [2, 3])
# def test_strain_state_error(dim: int) -> None:
#     setup_parameters = UniaxialCubeExperiment.default_parameters()
#     setup_parameters["dim"] = dim * ureg("")
#     setup_parameters["strain_state"] = "wrong" * ureg("")
#     setup = UniaxialCubeExperiment(setup_parameters)
#     default_setup, fem_parameters = LinearElasticity.default_parameters()
#     with pytest.raises(ValueError):
#         fem_problem = LinearElasticity(setup, fem_parameters)


# @pytest.mark.parametrize("dim", [2, 3])
# def test_multiaxial_strain(dim: int) -> None:
#     setup_parameters = UniaxialCubeExperiment.default_parameters()
#     setup_parameters["dim"] = dim * ureg("")
#     setup_parameters["strain_state"] = "multiaxial" * ureg("")
#     setup = UniaxialCubeExperiment(setup_parameters)
#     default_setup, fem_parameters = LinearElasticity.default_parameters()
#     fem_problem = LinearElasticity(setup, fem_parameters)

#     displ = -0.01
#     fem_problem.experiment.apply_displ_load(displ * ureg("m"))

#     if dim == 2:
#         target = np.array([displ, displ])
#         sensor_location = [fem_problem.p["length"], fem_problem.p["height"], 0.0]
#     elif dim == 3:
#         target = np.array([displ, displ, displ])
#         sensor_location = [fem_problem.p["length"], fem_problem.p["width"], fem_problem.p["height"]]

#     sensor = DisplacementSensor(sensor_location)

#     fem_problem.add_sensor(sensor)
#     fem_problem.solve()
#     result = fem_problem.sensors.DisplacementSensor.get_last_entry().magnitude
#     assert result == pytest.approx(target)
