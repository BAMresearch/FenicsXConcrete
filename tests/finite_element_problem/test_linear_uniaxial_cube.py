import os
from pathlib import Path

import numpy as np
import pint
import pytest

from fenicsxconcrete.experimental_setup.uniaxial_cube import UniaxialCubeExperiment
from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity
from fenicsxconcrete.sensor_definition.strain_sensor import StrainSensor
from fenicsxconcrete.sensor_definition.stress_sensor import StressSensor
from fenicsxconcrete.unit_registry import ureg


@pytest.mark.parametrize("dim", [2, 3])
def test_disp(dim: int):
    """uniaxial tension test for different dimensions (dim)"""

    # setup paths and directories
    data_dir = "data_files"
    data_path = Path(__file__).parent / data_dir

    # define file name and path for paraview output
    file_name = f"test_linear_uniaxial_{dim}d"
    files = [data_path / (file_name + ".xdmf"), data_path / (file_name + ".h5")]
    # delete file if it exisits (only relevant for local tests)
    for file in files:
        if file.is_file():
            os.remove(file)

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

    if dim == 2:
        # change default stress_state
        parameters["stress_state"] = "plane_stress" * ureg("")

    # setting up the problem
    experiment = UniaxialCubeExperiment(parameters)
    problem = LinearElasticity(experiment, parameters, pv_name=file_name, pv_path=data_path)

    # add sensors
    if dim == 2:
        problem.add_sensor(StressSensor([[0.5, 0.5, 0.0]]))
        problem.add_sensor(StrainSensor([[0.5, 0.5, 0.0]]))
    elif dim == 3:
        problem.add_sensor(StressSensor([[0.5, 0.5, 0.5]]))
        problem.add_sensor(StrainSensor([[0.5, 0.5, 0.5]]))

    # apply displacement load and solve
    problem.experiment.apply_displ_load(displacement)
    problem.solve()
    problem.pv_plot()

    # checks
    analytic_eps = displacement.to_base_units() / (1.0 * ureg("m"))
    if dim == 2:
        # strain in yy direction
        assert problem.sensors["StrainSensor"].data[-1][-1] == pytest.approx(analytic_eps)
        # strain in xx direction
        assert problem.sensors["StrainSensor"].data[-1][0] == pytest.approx(-problem.parameters["nu"] * analytic_eps)
        # strain in xy and yx direction
        assert problem.sensors["StrainSensor"].data[-1][1] == pytest.approx(
            problem.sensors["StrainSensor"].data[-1][2]
        )
        assert problem.sensors["StrainSensor"].data[-1][1] == pytest.approx(0.0)
        # stress in yy direction
        assert problem.sensors["StressSensor"].data[-1][-1].magnitude == pytest.approx(
            (analytic_eps * problem.parameters["E"]).magnitude
        )
    elif dim == 3:
        # strain in zz direction
        assert problem.sensors["StrainSensor"].data[-1][-1] == pytest.approx(analytic_eps)
        # strain in yy direction
        assert problem.sensors["StrainSensor"].data[-1][4] == pytest.approx(-problem.parameters["nu"] * analytic_eps)
        # strain in xx direction
        assert problem.sensors["StrainSensor"].data[-1][0] == pytest.approx(-problem.parameters["nu"] * analytic_eps)
        # shear strains
        sum_mixed_strains = (
            problem.sensors["StrainSensor"].data[-1][1].magnitude  # xy
            - problem.sensors["StrainSensor"].data[-1][3].magnitude  # yx
            - problem.sensors["StrainSensor"].data[-1][2].magnitude  # xz
            - problem.sensors["StrainSensor"].data[-1][6].magnitude  # zx
            - problem.sensors["StrainSensor"].data[-1][5].magnitude  # yz
            - problem.sensors["StrainSensor"].data[-1][7].magnitude  # zy
        )
        assert sum_mixed_strains == pytest.approx(0.0)

        # stress in zz direction
        assert problem.sensors["StressSensor"].data[-1][-1].magnitude == pytest.approx(
            (analytic_eps * problem.parameters["E"]).magnitude
        )
