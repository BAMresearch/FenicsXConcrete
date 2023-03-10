import os
import pytest
from pathlib import Path

from fenicsxconcrete.helper import Parameters
from fenicsxconcrete.unit_registry import ureg
from fenicsxconcrete.experimental_setup.simple_beam import SimpleBeam
from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity
from fenicsxconcrete.sensor_definition.displacement_sensor import DisplacementSensor


@pytest.mark.parametrize("dimension", [[2, [0.00105057, -0.01310806]],
                                       [3, [1.13946512e-03,  1.42368783e-05, -1.42250453e-02]]])
def test_linear_simple_beam(dimension):
    dim = dimension[0]
    result = dimension[1]

    # setup paths and directories
    data_dir = 'data_files'
    data_path = Path(__file__).parent / data_dir

    # define file name and path for paraview output
    file_name = f'test_linear_simple_beam_{dim}d'
    files = [data_path / (file_name + '.xdmf'),data_path / (file_name + '.h5')]
    # delete file if it exisits (only relevant for local tests)
    for file in files:
        if file.is_file():
            os.remove(file)

    setup_parameters = Parameters()
    setup_parameters['length'] = 10 * ureg('m')
    setup_parameters['height'] = 0.5 * ureg('m')
    setup_parameters['width'] = 0.3 * ureg('m')  # only relevant for 3D case
    setup_parameters['dim'] = dim * ureg('')
    setup_parameters['num_elements_length'] = 10 * ureg('')
    setup_parameters['num_elements_height'] = 3 * ureg('')
    setup_parameters['num_elements_width'] = 3 * ureg('')  # only relevant for 3D case
    setup_parameters['load'] = 200 * ureg('kN/m^2')

    fem_parameters = Parameters()
    fem_parameters['rho'] = 7750 * ureg('kg/m^3')
    fem_parameters['E'] = 210e9 * ureg('N/m^2')
    fem_parameters['nu'] = 0.28 * ureg('')

    # Defining sensor positions
    sensor_location = [setup_parameters['length'].magnitude/2, 0.0, 0.0]
    sensor = DisplacementSensor([sensor_location])

    # setting up the problem
    experiment = SimpleBeam(setup_parameters)         # Specifies the domain, discretises it and apply Dirichlet BCs
    problem = LinearElasticity(experiment, fem_parameters, pv_name=file_name, pv_path=data_path)
    problem.add_sensor(sensor)

    # solving and plotting
    problem.solve()
    problem.pv_plot()

    # check if files are created
    for file in files:
        assert file.is_file()

    # check sensor output
    displacement_data = problem.sensors['DisplacementSensor'].data[-1]
    assert displacement_data.magnitude == pytest.approx(result, 1e-5)
