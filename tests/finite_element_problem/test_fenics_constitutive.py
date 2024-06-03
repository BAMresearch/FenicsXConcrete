import os
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

# for know copy material law from fenics-constitutive to tests/finite_element_problem should be a module later
from linear_elasticity_model import LinearElasticityModel
from mises_plasticity_isotropic_hardening import VonMises3D
from spring_kelvin_model import SpringKelvinModel

from fenicsxconcrete.experimental_setup.simple_cube import SimpleCube
from fenicsxconcrete.finite_element_problem.fenics_constitutive import FenicsConstitutive
from fenicsxconcrete.sensor_definition.displacement_sensor import DisplacementSensor
from fenicsxconcrete.sensor_definition.reaction_force_sensor import ReactionForceSensor
from fenicsxconcrete.sensor_definition.stress_sensor import StressSensor
from fenicsxconcrete.util import ureg


@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("mat", ["linear_elastic", "visco_Kelvin", "mises"])
def test_fc(dim: int, mat: Literal["linear_elastic", "mises"]) -> None:
    """easy cube test for checking interface fenicsxconcrete - fencis_constitutive
    uniaxial tension test displacment controlled"""

    # setup paths and directories
    data_dir = "data_files"
    data_path = Path(__file__).parent / data_dir

    # define file name and path for paraview output
    file_name = f"test_mat_fc_uniaxial_{mat}_{dim}d"
    files = [data_path / (file_name + ".xdmf"), data_path / (file_name + ".h5")]
    # delete file if it exists (only relevant for local tests)
    for file in files:
        if file.is_file():
            os.remove(file)

    # experiment:
    parameters = {}

    parameters["dim"] = dim * ureg("")
    parameters["num_elements_length"] = 2 * ureg("")
    parameters["num_elements_height"] = 2 * ureg("")
    parameters["num_elements_width"] = 2 * ureg("")

    experiment = SimpleCube(parameters)

    # material:
    if mat == "linear_elastic":
        material_law = LinearElasticityModel
        parameters["E"] = 42000 * ureg("Pa")  # young's modulus
        parameters["nu"] = 0.3 * ureg("")  # poisson ratio
    elif mat == "visco_Kelvin":
        material_law = SpringKelvinModel
        parameters["E0"] = 550 * ureg("Pa")
        parameters["E1"] = 190 * ureg("Pa")
        parameters["tau"] = 10 * ureg("s")
        parameters["nu"] = 0.3 * ureg("")  # poisson ratio
    elif mat == "mises":
        material_law = VonMises3D
        parameters["p_ka"] = 175000 * ureg("Pa")  # bulk modulus
        parameters["p_mu"] = 80769 * ureg("Pa")  # shear modulus
        parameters["p_y0"] = 1200 * ureg("Pa")  # initial yield stress
        parameters["p_y00"] = 2500 * ureg("Pa")  # final yield stress
        parameters["p_w"] = 200 * ureg("")  # saturation parameter
    else:
        raise ValueError("material not supported")

    # problem:
    parameters["rho"] = 2000 * ureg("kg/m^3")
    parameters["strain_state"] = "uniaxial" * ureg("")
    parameters["dt"] = 0.1 * ureg("s")
    parameters["q_degree"] = 4 * ureg("")

    problem = FenicsConstitutive(experiment, parameters, material_law, pv_name=file_name, pv_path=data_path)

    # sensors:
    sensor_location = [0.5, 0.5, 0.5]
    problem.add_sensor(StressSensor(sensor_location))
    problem.add_sensor(DisplacementSensor(sensor_location))
    problem.add_sensor(ReactionForceSensor())

    # apply displacement load and solve
    displacement = 0.005 * ureg("m")
    total_time = 1.0
    while problem.time <= total_time:
        problem.experiment.apply_displ_load(problem.time * displacement)

        # update material parameters globally in time
        if mat == "linear_elastic":
            problem.mechanics_problem.laws[0][0].factor = 1.0 + 0.1 * problem.time
        elif mat == "visco_Kelvin":
            problem.mechanics_problem.laws[0][0].E0 = (1.0 + 0.1 * problem.time) * parameters["E0"]
            problem.mechanics_problem.laws[0][0].factor_E0 = 1.0 + 0.1 * problem.time

        problem.solve()
        problem.pv_plot()
        print("computed disp", problem.time, problem.fields.displacement.x.array[:].max())

    delta_stress_z = np.array(problem.sensors["StressSensor"].data)[:, 2]
    delta_disp_z = np.array(problem.sensors["DisplacementSensor"].data)[:, 2]

    print(np.diff(delta_disp_z).mean(), parameters["dt"].magnitude * displacement.magnitude)
    # check
    assert np.isclose(abs(problem.fields.displacement.x.array[:]).max(), abs(displacement.magnitude), rtol=1e-2)
    # displacement in middle 1/5 of total per time step
    assert np.isclose(np.diff(delta_disp_z).mean(), parameters["dt"].magnitude * displacement.magnitude / 2, rtol=1e-2)
    # for changing material params over time
    if mat == "linear_elastic" or mat == "visco_Kelvin":
        assert np.diff(delta_stress_z)[-1] > np.diff(delta_stress_z)[1]


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)

    # test_fc(3, "linear_elastic")
    # test_fc(3, "mises")
    test_fc(3, "visco_Kelvin")
