import os
from pathlib import Path

import numpy as np
import pytest

# for know copy material law from fenics-constitutive to tests/finite_element_problem should be a module later
from linear_elasticity_model import LinearElasticityModel
from mises_plasticity_isotropic_hardening import VonMises3D

from fenicsxconcrete.experimental_setup.simple_cube import SimpleCube
from fenicsxconcrete.finite_element_problem.concrete_am_fc import ConcreteAMFC
from fenicsxconcrete.sensor_definition.displacement_sensor import DisplacementSensor
from fenicsxconcrete.sensor_definition.strain_sensor import StrainSensor
from fenicsxconcrete.sensor_definition.stress_sensor import StressSensor
from fenicsxconcrete.util import ureg


@pytest.mark.parametrize("dim", [3])
def test_fc(dim: int) -> None:
    """easy cube test for checking interface fenicsxconcrete - fencis_constitutive
    uniaxial tension test"""

    # setup paths and directories
    data_dir = "data_files"
    data_path = Path(__file__).parent / data_dir

    # define file name and path for paraview output
    file_name = f"test_mat_fc_uniaxial_{dim}d"
    files = [data_path / (file_name + ".xdmf"), data_path / (file_name + ".h5")]
    # delete file if it exists (only relevant for local tests)
    for file in files:
        if file.is_file():
            os.remove(file)

    # defining experiment parameters
    parameters = {}

    parameters["dim"] = dim * ureg("")
    parameters["num_elements_length"] = 2 * ureg("")
    parameters["num_elements_height"] = 2 * ureg("")
    parameters["num_elements_width"] = 2 * ureg("")

    displacement = 0.005 * ureg("m")

    # choose material and set parameters
    # material_law = VonMises3D
    # parameters["p_ka"] = 175000 * ureg("MPa")  # bulk modulus
    # parameters["p_mu"] = 80769 * ureg("MPa")  # shear modulus
    # parameters["p_y0"] = 1200 * ureg("MPa")  # initial yield stress
    # parameters["p_y00"] = 2500 * ureg("MPa")  # final yield stress
    # parameters["p_w"] = 200 * ureg("")  # saturation parameter
    #
    material_law = LinearElasticityModel
    parameters["E"] = 42000 * ureg("Pa")  # young's modulus
    parameters["nu"] = 0.3 * ureg("")  # poisson ratio

    parameters["rho"] = 2000 * ureg("kg/m^3")
    parameters["strain_state"] = "uniaxial" * ureg("")
    parameters["dt"] = 0.1 * ureg("s")

    # setting up the problem
    experiment = SimpleCube(parameters)
    problem = ConcreteAMFC(experiment, parameters, material_law, pv_name=file_name, pv_path=data_path)

    sensor_location = [0.5, 0.5, 0.5]

    # add sensors
    problem.add_sensor(StressSensor(sensor_location))
    problem.add_sensor(StrainSensor(sensor_location))
    problem.add_sensor(DisplacementSensor(sensor_location))

    # apply displacement load and solve
    total_time = 1.0
    while problem.time <= total_time:
        problem.experiment.apply_displ_load(problem.time * displacement)
        problem.solve()
        problem.pv_plot()
        print("computed disp", problem.time, problem.fields.displacement.x.array[:].max())

    disp_result = problem.sensors["DisplacementSensor"].get_last_entry().magnitude
    strain_result = problem.sensors["StrainSensor"].get_last_entry().magnitude
    stress_result = problem.sensors["StressSensor"].get_last_entry().magnitude
    print("results", disp_result, strain_result, stress_result)


if __name__ == "__main__":

    test_fc(3)
