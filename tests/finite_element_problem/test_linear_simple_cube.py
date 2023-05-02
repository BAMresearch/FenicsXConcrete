import os
from pathlib import Path

import numpy as np
import pytest

from fenicsxconcrete.experimental_setup.simple_cube import SimpleCube
from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity
from fenicsxconcrete.sensor_definition.displacement_sensor import DisplacementSensor
from fenicsxconcrete.sensor_definition.strain_sensor import StrainSensor
from fenicsxconcrete.sensor_definition.stress_sensor import StressSensor
from fenicsxconcrete.unit_registry import ureg


@pytest.mark.parametrize("dim", [2, 3])
def test_disp(dim: int) -> None:
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
    parameters["strain_state"] = "uniaxial" * ureg("")

    if dim == 2:
        # change default stress_state
        parameters["stress_state"] = "plane_stress" * ureg("")

    # setting up the problem
    experiment = SimpleCube(parameters)
    problem = LinearElasticity(experiment, parameters, pv_name=file_name, pv_path=data_path)

    if dim == 2:
        sensor_location = [0.5, 0.5, 0.0]
    elif dim == 3:
        sensor_location = [0.5, 0.5, 0.5]

    # add sensors
    problem.add_sensor(StressSensor(sensor_location))
    problem.add_sensor(StrainSensor(sensor_location))

    # apply displacement load and solve
    problem.experiment.apply_displ_load(displacement)
    problem.solve()
    problem.pv_plot()

    # checks
    analytic_eps = (displacement.to_base_units() / (1.0 * ureg("m"))).magnitude

    strain_result = problem.sensors["StrainSensor"].get_last_entry().magnitude
    stress_result = problem.sensors["StressSensor"].get_last_entry().magnitude
    if dim == 2:
        # strain in yy direction
        assert strain_result[-1] == pytest.approx(analytic_eps)
        # strain in xx direction
        assert strain_result[0] == pytest.approx(-problem.parameters["nu"].magnitude * analytic_eps)
        # strain in xy and yx direction
        assert strain_result[1] == pytest.approx(strain_result[2])
        assert strain_result[1] == pytest.approx(0.0)
        # stress in yy direction
        assert stress_result[-1] == pytest.approx((analytic_eps * problem.parameters["E"]).magnitude)

    elif dim == 3:
        # strain in zz direction
        assert strain_result[-1] == pytest.approx(analytic_eps)
        # strain in yy direction
        assert strain_result[4] == pytest.approx(-problem.parameters["nu"].magnitude * analytic_eps)
        # strain in xx direction
        assert strain_result[0] == pytest.approx(-problem.parameters["nu"].magnitude * analytic_eps)
        # shear strains
        sum_mixed_strains = (
            strain_result[1]  # xy
            - strain_result[3]  # yx
            - strain_result[2]  # xz
            - strain_result[6]  # zx
            - strain_result[5]  # yz
            - strain_result[7]  # zy
        )
        assert sum_mixed_strains == pytest.approx(0.0)

        # stress in zz direction
        assert stress_result[-1] == pytest.approx((analytic_eps * problem.parameters["E"].magnitude))


@pytest.mark.parametrize("dim", [2, 3])
def test_strain_state_error(dim: int) -> None:
    setup_parameters = SimpleCube.default_parameters()
    setup_parameters["dim"] = dim * ureg("")
    setup_parameters["strain_state"] = "wrong" * ureg("")
    setup = SimpleCube(setup_parameters)
    default_setup, fem_parameters = LinearElasticity.default_parameters()
    with pytest.raises(ValueError):
        fem_problem = LinearElasticity(setup, fem_parameters)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("degree", [1, 2])
def test_multiaxial_strain(dim: int, degree: int) -> None:
    setup_parameters = SimpleCube.default_parameters()
    setup_parameters["dim"] = dim * ureg("")
    setup_parameters["degree"] = degree * ureg("")
    setup_parameters["strain_state"] = "multiaxial" * ureg("")
    setup = SimpleCube(setup_parameters)
    default_setup, fem_parameters = LinearElasticity.default_parameters()
    fem_problem = LinearElasticity(setup, fem_parameters)

    displ = -0.01
    fem_problem.experiment.apply_displ_load(displ * ureg("m"))

    if dim == 2:
        target = np.array([displ, displ])
        sensor_location_corner = [fem_problem.p["length"], fem_problem.p["height"], 0.0]
        sensor_location_center = [fem_problem.p["length"] / 2, fem_problem.p["height"] / 2, 0.0]
    elif dim == 3:
        target = np.array([displ, displ, displ])
        sensor_location_corner = [fem_problem.p["length"], fem_problem.p["width"], fem_problem.p["height"]]
        sensor_location_center = [
            fem_problem.p["length"] / 2,
            fem_problem.p["height"] / 2,
            fem_problem.p["height"] / 2,
        ]

    sensor_corner = DisplacementSensor(where=sensor_location_corner, name="displacement_corner")
    sensor_center = DisplacementSensor(where=sensor_location_center, name="displacement_center")

    fem_problem.add_sensor(sensor_corner)
    fem_problem.add_sensor(sensor_center)

    fem_problem.solve()
    result_corner = fem_problem.sensors.displacement_corner.get_last_entry().magnitude
    result_center = fem_problem.sensors.displacement_center.get_last_entry().magnitude

    assert result_corner == pytest.approx(target)
    assert result_center == pytest.approx(target / 2)
