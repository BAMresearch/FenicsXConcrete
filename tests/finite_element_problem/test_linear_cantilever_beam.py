import os
from pathlib import Path

import pytest

from fenicsxconcrete.experimental_setup.cantilever_beam import CantileverBeam
from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity
from fenicsxconcrete.sensor_definition.displacement_sensor import DisplacementSensor
from fenicsxconcrete.util import ureg


@pytest.mark.parametrize(
    "dimension,results",
    [
        [2, [-1.10366991e-06, -6.02823499e-06]],
        [3, [-1.18487757e-06, 3.58357285e-10, -6.42126235e-06]],
    ],
)
def test_linear_cantilever_beam(dimension: int, results: list[float]) -> None:
    # setup paths and directories
    data_dir = "data_files"
    data_path = Path(__file__).parent / data_dir

    # define file name and path for paraview output
    file_name = f"test_linear_cantilever_beam_{dimension}d"
    files = [data_path / (file_name + ".xdmf"), data_path / (file_name + ".h5")]
    # delete file if it exisits (only relevant for local tests)
    for file in files:
        if file.is_file():
            os.remove(file)

    setup_parameters = {}
    setup_parameters["length"] = 1 * ureg("m")
    setup_parameters["height"] = 0.3 * ureg("m")
    setup_parameters["width"] = 0.3 * ureg("m")  # only relevant for 3D case
    setup_parameters["dim"] = dimension * ureg("")
    setup_parameters["num_elements_length"] = 10 * ureg("")
    setup_parameters["num_elements_height"] = 3 * ureg("")
    setup_parameters["num_elements_width"] = 3 * ureg("")  # only relevant for 3D case

    fem_parameters = {}
    fem_parameters["rho"] = 7750 * ureg("kg/m^3")
    fem_parameters["E"] = 210e9 * ureg("N/m^2")
    fem_parameters["nu"] = 0.28 * ureg("")

    # Defining sensor positions
    # TODO: why do I need the third coordinate for a 2D problem?!?
    sensor_location = [setup_parameters["length"].magnitude, 0.0, 0.0]
    sensor = DisplacementSensor(sensor_location)

    # setting up the problem
    experiment = CantileverBeam(setup_parameters)  # Specifies the domain, discretises it and apply Dirichlet BCs

    problem = LinearElasticity(experiment, fem_parameters, pv_name=file_name, pv_path=data_path)
    problem.add_sensor(sensor)

    # solving and plotting
    problem.solve()
    problem.pv_plot()

    # check if files are created
    for file in files:
        assert file.is_file()

    # check sensor output
    displacement_data = problem.sensors["DisplacementSensor"].get_last_entry()
    assert displacement_data.magnitude == pytest.approx(results)

    # Second test
    # test linearity of material problem
    increase = 3
    fem_parameters["E"] = fem_parameters["E"] * increase
    problem2 = LinearElasticity(experiment, fem_parameters)
    problem2.add_sensor(sensor)
    problem2.solve()
    displacement_data2 = problem2.sensors["DisplacementSensor"].get_last_entry()

    assert displacement_data2.magnitude * increase == pytest.approx(displacement_data.magnitude)
