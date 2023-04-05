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


# @pytest.mark.parametrize("dim",[2,3])
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
        parameters["stress_state"] = "plane_stress" * ureg("")

    # setting up the problem
    experiment = UniaxialCubeExperiment(parameters)
    problem = LinearElasticity(experiment, parameters, pv_name=file_name, pv_path=data_path)

    problem.add_sensor(StressSensor([[0.0, 0.0, 0.0]]))
    problem.add_sensor(StrainSensor([[0.0, 0.0, 0.0]]))

    # apply displacement
    problem.experiment.apply_displ_load(displacement)
    problem.solve()
    problem.pv_plot()

    # get stresses and strains over time
    print("Stress sensor", problem.sensors["StressSensor"].data)
    print("strain sensor", problem.sensors["StrainSensor"].data)
    print("time", problem.sensors["StrainSensor"].time)

    if dim == 2:
        # checks
        assert problem.sensors["StrainSensor"].data[-1][-1] == pytest.approx(displacement / (1.0 * ureg("m")))
        assert problem.sensors["StrainSensor"].data[-1][0] == pytest.approx(
            -problem.parameters["nu"] * displacement / (1.0 * ureg("m"))
        )
        assert problem.sensors["StressSensor"].data[-1][-1].magnitude == pytest.approx(
            (displacement / (1.0 * ureg("m")) * problem.parameters["E"]).magnitude
        )
    elif dim == 3:
        print("not yet done")


if __name__ == "__main__":

    test_disp(2)

    # test_disp(3)
